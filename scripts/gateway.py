#!/usr/bin/env python3
"""AI 模型网关（常驻能力，非临时补救）

暴露三种兼容 API（端口 8000）：
• OpenAI   — /v1/models, /v1/chat/completions（原有）
• Anthropic — /v1/messages, /v1/messages/count_tokens, /v1/messages/{model}（新增）
• 管理     — /v1/gateway/status（状态）、/v1/gateway/unload_all（腾显存）、
             /v1/gateway/switch（远程切换文本/视频，等价 start.sh --switch）
• 协商     — /v1/gateway/capabilities（说明接口，供客户端 agent 首次连接时学习）、
             /v1/gateway/gpus（GPU 发现：数量/容量/占用/利用率）、
             /v1/gateway/load（把模型加载到指定 GPU，wait=false 异步返回 request_id）、
             /v1/gateway/unload（按模型细粒度卸载）、/v1/gateway/ops[/{id}]（操作状态轮询）。
             推理请求还可带 X-Gateway-Load-Async:1（+可选 X-Gateway-GPUs:0）走 202 异步协商，
             网关把 loading→ready / failed 状态机（transitions）返回给客户端。
             详见 GET /v1/gateway/capabilities 的 negotiation 一节。

内部规则：
1. /v1/models 暴露全部已下载模型，而不是只暴露当前加载的那个
2. 客户端按模型名请求：未加载 → 自动加载（请求挂起等待就绪）
3. 显存放得下就并存（不限一卡一个，按模型名路由）：每卡可用空间按已加载模型的
   完整预算 util×total 扣减（含 KV cache 预留），新模型不侵占老模型的 KV 空间
4. 任何卡都放不下时：卸载其它【空闲】的模型（in_flight==0，文本按最久未用）；
   正忙的模型绝不驱逐，直接返回 503 让客户端稍后重试
5. 视频模型与文本模型平等：文本请求缺显存时，空闲的 ComfyUI（队列无任务，含模型
   加载中判定）会被自动卸载其已加载模型（POST /free）让位；ComfyUI 有任务在跑则
   503 等任务结束。卸载只释放模型权重，容器保留，下次视频任务自动重载。
6. 上下文长度一律按模型原生上限（models.json 的 gw.max_len）配满，不降档
7. 卸载只发生在"需要腾显存"时，不设空闲定时器

运行环境：ai-gateway 容器，挂载 docker.sock 管理后端容器，--gpus all 提供 nvidia-smi。
后端容器命名 ai-vllm-<model> / ai-llamacpp-<model>，仅加入 ai_default 内网，不暴露宿主机端口。
所有 docker/nvidia-smi 同步调用均经 asyncio.to_thread 包裹，不阻塞事件循环。
"""
import asyncio, fnmatch, json, os, re, subprocess, time, urllib.request
from contextlib import asynccontextmanager

import docker
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS_JSON = os.environ.get("MODELS_JSON", "/opt/models.json")
HOST_ROOT = os.environ.get("HOST_ROOT", "/home/pc/AI")
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest")
LLAMACPP_IMAGE = os.environ.get("LLAMACPP_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
NETWORK = os.environ.get("GW_NETWORK", "ai_default")
PRELOAD = [m for m in os.environ.get("PRELOAD", "").split(",") if m]
NEED_MARGIN_MB = 512  # vLLM 要求 空闲 >= util×total，再加安全余量，避免放行后启动崩溃
COMFY_HOST = os.environ.get("COMFY_HOST", "host.docker.internal")  # 网关访问 ComfyUI 的宿主地址

log = lambda *a: print("[gateway]", *a, flush=True)

dcli = docker.DockerClient(base_url="unix:///var/run/docker.sock")
http = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))

ops_lock = asyncio.Lock()      # 加载/卸载串行化
backends = {}                  # name -> {container, gpus, util, in_flight, last_used, status}
token_metrics_cache = {}       # name -> {time, prompt_total, generation_total, rate, metrics_ok, live}
_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 异步操作状态机（load/switch/unload）：客户端拿到 request_id 后轮询 GET /v1/gateway/ops/{id}，
# 直到 status 收敛到 success/failed。transitions 记录每个 stage 变更，供客户端展示"切换/加载"进度。
ops = {}          # request_id -> op dict
ops_seq = 0
OP_TTL = 3600     # 最近 1h 的操作保留（轮询结束即弃），防内存无限增长


def _new_op(kind, model=None):
    global ops_seq
    ops_seq += 1
    rid = f"{int(time.time() * 1000):x}-{ops_seq:04x}"
    now = time.time()
    op = {"request_id": rid, "kind": kind, "model": model, "status": "queued",
          "stage": "queued", "detail": "已排队", "target_gpus": None,
          "error": None, "error_type": None,
          "created_at": now, "updated_at": now,
          "transitions": [{"stage": "queued", "detail": "已排队", "at": now}]}
    ops[rid] = op
    _prune_ops()
    return op


def _prune_ops():
    now = time.time()
    for rid in [r for r, o in ops.items() if now - o["created_at"] > OP_TTL]:
        ops.pop(rid, None)


def _op_update(op, status=None, stage=None, detail="", error=None, error_type=None,
               target_gpus=None):
    op["status"] = status or op["status"]
    op["stage"] = stage or op["stage"]
    if detail:
        op["detail"] = detail
    if error is not None:
        op["error"] = error
    if error_type is not None:
        op["error_type"] = error_type
    if target_gpus is not None:
        op["target_gpus"] = sorted(target_gpus)
    op["updated_at"] = time.time()
    op["transitions"].append({"stage": op["stage"], "detail": op["detail"],
                              "at": op["updated_at"]})


def _op_snapshot(op):
    """操作快照（同步）：带当前后端实况；gpu_free 由端点另取（避免在事件循环内跑 nvidia-smi）。"""
    snap = dict(op)
    snap["transitions"] = list(op["transitions"])
    name = op.get("model")
    b = backends.get(name)
    if name and b:
        snap["backend"] = {"model": name, "container": b["container"],
                           "gpus": list(b["gpus"]), "engine": b.get("engine"),
                           "status": b["status"], "in_flight": b["in_flight"],
                           "slots": b.get("slots")}
    else:
        snap["backend"] = None
    return snap


# ---------- 同步原语（全部经 to_thread 调用） ----------

def load_manifest():
    with open(MODELS_JSON, encoding="utf-8") as f:
        return json.load(f)["llm"]


def model_dir(name):
    """官方 FP8 目录优先，否则用主目录"""
    for cand in (f"{name}-fp8", name):
        p = os.path.join(HOST_ROOT, "models/llm", cand)
        if os.path.isdir(p) and os.listdir(p):
            return cand
    return None


def gpu_free():
    """返回 {gpu_index: {"free": MB, "total": MB}}"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    r = {}
    for line in out.stdout.strip().splitlines():
        i, free, total = [int(x) for x in line.split(",")]
        r[i] = {"free": free, "total": total}
    return r


def gpu_info():
    """GPU 发现：{index: {index, total_mb, free_mb, used_mb, util_pct}}（含利用率）"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    r = {}
    for line in out.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            i, total, free, used, util = (int(p) for p in parts[:5])
        except ValueError:
            continue
        r[i] = {"index": i, "total_mb": total, "free_mb": free,
                "used_mb": used, "util_pct": util}
    return r


def comfy_containers():
    """所有已知 ComfyUI 实例实况（含 stopped / 不存在）：
    {container: {exists, status, port, gpus}}。port 为宿主端口；gpus 为该实例占用的 GPU。"""
    out = {}
    for cname in ("ai-comfyui", "ai-comfyui-gpu1"):
        try:
            c = dcli.containers.get(cname)
        except docker.errors.NotFound:
            out[cname] = {"exists": False, "status": None, "port": None, "gpus": []}
            continue
        port = None
        for binds in (c.attrs.get("NetworkSettings", {}).get("Ports") or {}).values():
            if binds:
                port = binds[0].get("HostPort")
                break
        gpus = set()
        for r in (c.attrs.get("HostConfig", {}).get("DeviceRequests") or []):
            gpus.update(int(x) for x in (r.get("DeviceIDs") or []))
        out[cname] = {"exists": True, "status": c.status, "port": port,
                      "gpus": sorted(gpus)}
    return out


def comfy_instances():
    """返回运行中的 ComfyUI 实例 [{container, port, gpus}]。
    port 为宿主端口（网关经 COMFY_HOST 访问）；gpus 为该实例占用的 GPU。"""
    out = []
    for cname, info in comfy_containers().items():
        if not info["exists"] or info["status"] != "running":
            continue
        out.append({"container": cname, "port": info["port"],
                    "gpus": set(info["gpus"])})
    return out


def comfy_gpus():
    """视频服务占用的 GPU 集合（仅用于错误提示）"""
    return {g for inst in comfy_instances() for g in inst["gpus"]}


def comfy_queue_busy(port):
    """ComfyUI 队列有任务（running/pending，含模型加载中）→ True。
    访问不到（如容器刚起）按 busy 处理，避免误卸。"""
    if not port:
        return True
    try:
        with urllib.request.urlopen(f"http://{COMFY_HOST}:{port}/queue", timeout=5) as r:
            d = json.loads(r.read())
        return bool(d.get("queue_running") or d.get("queue_pending"))
    except Exception:
        return True


def comfy_unload(port):
    """卸载 ComfyUI 已加载模型（POST /free），进程保留，下次视频任务自动重载"""
    if not port:
        return
    try:
        req = urllib.request.Request(
            f"http://{COMFY_HOST}:{port}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        log(f"已卸载视频模型（{COMFY_HOST}:{port}/free）")
    except Exception as e:
        log(f"卸载视频模型失败（{COMFY_HOST}:{port}）: {e}")


def _remove_container(name):
    try:
        dcli.containers.get(name).remove(force=True)
    except Exception:
        pass


def _container_state(name):
    """返回 (status, logs_tail)；容器不存在返回 (None, '')"""
    try:
        c = dcli.containers.get(name)
        tail = ""
        if c.status != "running":
            tail = c.logs(tail=30).decode(errors="replace")[-2000:]
        return c.status, tail
    except docker.errors.NotFound:
        return None, ""


def _http_get_text(url, timeout=0.8):
    with _no_proxy_opener.open(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


_PROM_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"')
_PREFILL_RE = re.compile(
    r"prompt processing, n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+),\s*"
    r"t\s*=\s*([\d.]+)\s*s\s*/\s*([\d.]+) tokens per second"
)


def _llamacpp_prefill(backend):
    """从 llama-server 日志尾部解析 prompt prefill 实时进度与吞吐。

    prefill 阶段 /metrics 的 prompt/generation counter 都要等请求结束才更新，
    但日志每秒打印 'prompt processing, n_tokens=..., progress=..., x tokens/second'。
    """
    try:
        c = dcli.containers.get(backend["container"])
        logs = c.logs(tail=300, since=int(time.time()) - 30).decode(errors="replace")
    except Exception:
        return None
    matches = list(_PREFILL_RE.finditer(logs))
    if not matches:
        return None
    m = matches[-1]
    return {"tokens": int(m.group(1)), "progress": float(m.group(2)),
            "elapsed": float(m.group(3)), "rate": float(m.group(4))}



def _parse_prometheus_metrics(text, wanted):
    """提取 Prometheus counter 中每个 label 组合的最大值。"""
    wanted = set(wanted)
    samples = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            labels_text, value_text = rest.rsplit("}", 1)
            labels = dict(_PROM_LABEL_RE.findall(labels_text))
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value_text, labels = parts[0], parts[1], {}
        if name not in wanted:
            continue
        try:
            value = float(value_text.split()[0])
        except (ValueError, IndexError):
            continue
        key = (name, tuple(sorted(labels.items())))
        samples[key] = max(samples.get(key, value), value)
    return list(samples.values())


def _metric_total(text, names):
    values = _parse_prometheus_metrics(text, names)
    return sum(values) if values else None


def _token_state(name):
    now = time.time()
    st = token_metrics_cache.setdefault(name, {"time": now, "prompt_total": 0.0,
                                               "generation_total": 0.0,
                                               "generation_rate": 0.0,
                                               "prompt_rate": 0.0,
                                               "metrics_ok": False, "live_seen": 0.0,
                                               "live_start": None})
    if st.get("live_start") and now > st["live_start"]:
        st["generation_rate"] = max(0.0, st.get("live_seen", 0.0) /
                                    max(now - st["live_start"], 1e-6))
    return st


def begin_live_tokens(name):
    st = _token_state(name)
    st["live_base_total"] = st.get("generation_total") or 0.0
    st["live_start"] = time.time()
    st["live_seen"] = 0.0
    st["generation_rate"] = 0.0


def add_live_tokens(name, count):
    if count <= 0:
        return
    st = _token_state(name)
    if st.get("live_start") is None:
        st["live_start"] = time.time()
    st["live_seen"] = st.get("live_seen", 0.0) + count
    st["metrics_ok"] = True


def finish_live_tokens(name, prompt_delta=None, generation_delta=None):
    st = _token_state(name)
    if generation_delta is not None:
        st["generation_total"] = (st.get("live_base_total",
                                         st.get("generation_total") or 0.0)
                                  + float(generation_delta))
    if prompt_delta is not None:
        st["prompt_total"] = (st.get("prompt_total") or 0.0) + float(prompt_delta)
    st["live_start"] = None
    st["live_seen"] = 0.0
    st["generation_rate"] = 0.0
    st["metrics_ok"] = True


def _sse_json_events(chunk):
    text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except Exception:
            continue


def count_sse_tokens(name, chunk):
    """从 OpenAI SSE 增量中实时估算 token；最终 usage chunk 负责校准。"""
    for event in _sse_json_events(chunk):
        usage = event.get("usage") or {}
        completion = usage.get("completion_tokens")
        if isinstance(completion, (int, float)):
            st = _token_state(name)
            confirmed = max(float(completion), st.get("live_seen", 0.0))
            finish_live_tokens(name, usage.get("prompt_tokens"), confirmed)
            continue
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                add_live_tokens(name, 1)
            if delta.get("reasoning_content"):
                add_live_tokens(name, 1)
            for tool in delta.get("tool_calls") or []:
                fn = tool.get("function") or {}
                if fn.get("arguments"):
                    add_live_tokens(name, 1)


def _fetch_backend_metrics(backend):
    text = _http_get_text(f"http://{backend['container']}:8000/metrics")
    if backend.get("engine") == "llamacpp":
        prompt = _metric_total(text, (
            "llamacpp:prompt_tokens_total", "llama:prompt_tokens_total",
            "llamacpp:preprocessed_tokens_total", "llama:preprocessed_tokens_total",
            "llamacpp:tokens_processed_total", "llama:tokens_processed_total",
        ))
        generated = _metric_total(text, (
            "llamacpp:tokens_predicted_total", "llama:tokens_predicted_total",
            "llamacpp:predicted_tokens_total", "llama:predicted_tokens_total",
            "llamacpp:generation_tokens_total", "llama:generation_tokens_total",
            "llamacpp:tokens_generated_total", "llama:tokens_generated_total",
        ))
    else:
        prompt = _metric_total(text, (
            "vllm:prompt_tokens_total", "vllm:prompt_tokens",
        ))
        generated = _metric_total(text, (
            "vllm:generation_tokens_total", "vllm:generation_tokens",
        ))
    if generated is None:
        raise ValueError("generation token not found")
    prefill = None
    if backend.get("engine") == "llamacpp":
        prefill = _llamacpp_prefill(backend)
    return prompt, generated, prefill


def _apply_backend_metrics(name, backend, metrics=None, error=False):
    now = time.time()
    item = _token_state(name)
    if error:
        item.update({"time": now, "metrics_ok": False})
        if not item.get("live_start"):
            item["generation_rate"] = 0.0
        return item
    prompt, generated, prefill = metrics
    if prompt is not None:
        item["prompt_total"] = prompt
    item["metrics_ok"] = True
    if prefill:
        prev_tokens = item.get("prefill_tokens")
        prev_seen = item.get("prefill_seen_at")
        growing = prev_seen and prefill["tokens"] > (prev_tokens or 0)
        first_seen_active = prev_seen is None and backend.get("in_flight", 0) > 0
        active = bool(growing or first_seen_active)
        if prefill["progress"] >= 0.999 and backend.get("in_flight", 0) == 0:
            active = False  # prefill 已完成且请求结束：最后一帧不再显示 prefill
        item["prefill_tokens"] = prefill["tokens"]
        item["prefill_progress"] = prefill["progress"]
        item["prefill_rate"] = prefill["rate"]
        item["prefill_elapsed"] = prefill["elapsed"]
        item["prefill_active"] = active
        item["prefill_seen_at"] = now
        # llama.cpp：prefill 速率来自 slot 日志（parallel=1 时与 decode 互斥）
        item["prompt_rate"] = float(prefill["rate"] or 0.0) if active else 0.0
    elif item.get("prefill_active") and (now - item.get("prefill_seen_at", 0)) > 10:
        item["prefill_active"] = False
        item["prefill_rate"] = 0.0
        item["prompt_rate"] = 0.0
    elif backend.get("engine") == "llamacpp":
        item.setdefault("prompt_rate", 0.0)
        if not item.get("prefill_active"):
            item["prompt_rate"] = 0.0
    if backend.get("engine") == "llamacpp":
        total = max(float(generated), item.get("generation_total") or 0.0)
        if item.get("live_start"):
            item["live_base_total"] = max(item.get("live_base_total", 0.0),
                                          float(generated))
            total = max(total, item["live_base_total"] + item.get("live_seen", 0.0))
        item["generation_total"] = total
        if not item.get("live_start"):
            item["generation_rate"] = 0.0
    else:
        # vLLM continuous batching：prefill 与 decode 可并发，
        # 用 prompt_tokens_total / generation_tokens_total 各自差分得到独立速率
        last_time = item.get("counter_time")
        last_generated = item.get("counter_total")
        last_prompt = item.get("counter_prompt")
        item["counter_time"] = now
        item["counter_total"] = generated
        item["counter_prompt"] = prompt
        if last_generated is not None and generated >= last_generated and last_time:
            dt = now - last_time
            item["generation_rate"] = (generated - last_generated) / dt if dt > 0 else 0.0
        else:
            item["generation_rate"] = 0.0
        if last_prompt is not None and prompt is not None and prompt >= last_prompt and last_time:
            dt = now - last_time
            item["prompt_rate"] = (prompt - last_prompt) / dt if dt > 0 else 0.0
        else:
            item["prompt_rate"] = 0.0
        item["generation_total"] = generated
    item["time"] = now
    return item


async def sample_all_backend_tokens():
    alive = set(backends)
    for dead in set(token_metrics_cache) - alive:
        token_metrics_cache.pop(dead, None)
    names = [n for n, b in backends.items() if b.get("status") in ("ready", "adopted")]
    if not names:
        return {}

    async def one(name):
        try:
            metrics = await asyncio.to_thread(_fetch_backend_metrics, backends[name])
            return name, _apply_backend_metrics(name, backends[name], metrics=metrics)
        except Exception:
            return name, _apply_backend_metrics(name, backends[name], error=True)

    return dict(await asyncio.gather(*(one(n) for n in names)))


def _find_gguf(dirname, pattern=None):
    """在模型目录里找 GGUF 主文件（多分片返回第一个 shard，llama.cpp 会自动续载）"""
    root = os.path.join(HOST_ROOT, "models/llm", dirname)
    cands = []
    for dp, _, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(".gguf"):
                continue
            if f.startswith("mmproj-") or f.startswith("imatrix"):
                continue
            cands.append(os.path.join(dp, f))
    if not cands:
        raise RuntimeError(f"模型 {dirname} 目录下找不到 .gguf 文件")
    if pattern:
        for c in cands:
            if fnmatch.fnmatch(os.path.basename(c), pattern) \
                    or fnmatch.fnmatch(c.replace(root + os.sep, ""), pattern):
                return c
    return cands[0]


_SHARD_RE = re.compile(r"^(?P<prefix>.*)-(\d+)-of-(?P<n>\d+)\.gguf$", re.IGNORECASE)


def _gguf_weights_gb(dirname, gguf_include=None):
    """模型权重总占用（GiB，1024 进制）：GGUF 全分片 + mmproj。

    只做文件大小求和（网关挂载了 /models），不解析 GGUF 二进制。
    多分片按 `-NNNNN-of-MMMMM.gguf` 命名约定求同一组全部分片。
    """
    first = _find_gguf(dirname, gguf_include)
    d = os.path.dirname(first)
    base = os.path.basename(first)
    m = _SHARD_RE.match(base)
    if m:
        n = int(m.group("n"))
        prefix = m.group("prefix")
        want = {f"{prefix}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)}
        files = [os.path.join(d, f) for f in os.listdir(d) if f in want]
        if not files:  # 兜底：前缀下所有匹配分片
            files = [os.path.join(d, f) for f in os.listdir(d)
                     if _SHARD_RE.match(f) and _SHARD_RE.match(f).group("prefix") == prefix]
    else:
        files = [first]
    total = sum(os.path.getsize(f) for f in files)
    mmproj = _find_mmproj(dirname)
    if mmproj and os.path.exists(mmproj):
        total += os.path.getsize(mmproj)
    return total / 1024 ** 3


def plan_llamacpp(gpus, weights_gb, max_len, gw=None):
    """按当前硬件为 llama.cpp 后端规划并发参数（纯函数，不启动容器）。

    返回 {parallel, ctx_size, per_slot_ctx, kv_budget_gb, weights_gb, cache_type}。
    依据：KV cache 是 llama.cpp 用"权重占满后剩余显存"自动封顶的共享池，
    --parallel N 不增加显存，只把 ctx-size 逻辑平分到 N 个 slot。
    将来加卡：gpus 含多张卡时 total 取各卡之和，KV 预算近似翻倍 → slot 自动增多。
    """
    gw = gw or {}
    util = float(gw.get("util", 0.92))
    reserved_gb = float(gw.get("reserved_gb", 1.5))
    slot_kv_gb = float(gw.get("slot_kv_gb", 3.0))
    max_parallel = max(1, int(gw.get("max_parallel", 4)))
    min_slot_ctx = int(gw.get("min_slot_ctx", 65536))

    free = gpu_free()
    total_gb = sum(free[g]["total"] for g in gpus if g in free) / 1024.0
    kv_budget_gb = max(0.0, total_gb * util - float(weights_gb) - reserved_gb)

    # 硬覆盖：parallel=N 指定则用之；parallel_auto=false 强制单 slot
    forced = gw.get("parallel")
    if forced is not None:
        n = max(1, int(forced))
    elif gw.get("parallel_auto") is False:
        n = 1
    else:
        n = int(kv_budget_gb // slot_kv_gb) if slot_kv_gb > 0 else 1
        n = max(1, min(max_parallel, n))
        # 上下文地板：平分后每个 slot 不得小于 min_slot_ctx，否则下调 n
        while n > 1 and max_len // n < min_slot_ctx:
            n -= 1

    cache_type = gw.get("kv_quant")  # 如 "q8_0"；默认 None = FP16，不自动量化
    return {
        "parallel": n,
        "ctx_size": int(max_len),
        "per_slot_ctx": int(max_len) // n,
        "kv_budget_gb": round(kv_budget_gb, 2),
        "weights_gb": round(float(weights_gb), 2),
        "total_gb": round(total_gb, 2),
        "cache_type": cache_type,
        "tight": kv_budget_gb < slot_kv_gb,
    }

def _run_backend(name, dirname, gpus, util, max_len, think, extra_args=None, env=None,
                 engine="vllm", gguf_include=None, plan=None):
    kwargs = {}
    if len(gpus) > 1:
        kwargs["ipc_mode"] = "host"
    if engine == "llamacpp":
        _run_llamacpp(name, dirname, gpus, max_len, think, gguf_include, extra_args, kwargs,
                      plan=plan)
        return
    tp = len(gpus)
    cmd = [f"/models/{dirname}", "--served-model-name", name,
           "--tensor-parallel-size", str(tp),
           "--gpu-memory-utilization", str(util),
           "--max-model-len", str(max_len),
           "--max-num-seqs", "128",
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]
    if think:
        # 仅思考模型可加 reasoning parser；非思考模型强开会把正文吞进 reasoning_content
        cmd += ["--reasoning-parser", "qwen3"]
    if extra_args:
        # 如 YaRN 扩上下文的 --hf-overrides（models.json 里按模型配置）
        cmd += extra_args
    log(f"加载 {name}: dir={dirname} gpus={gpus} tp={tp} util={util} max_len={max_len} think={think}"
        + (f" extra_args={extra_args}" if extra_args else ""))
    _remove_container(f"ai-vllm-{name}")
    # 编译/JIT 缓存持久化到宿主机：容器每次重建，不挂载则 torch.compile 每次重跑（~30s）
    cache_host = f"{HOST_ROOT}/data/vllm-cache"
    os.makedirs(cache_host, exist_ok=True)
    environment = {"TRITON_CACHE_DIR": "/root/.cache/triton"}
    if env:
        # 如 YaRN 超长所需的 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1（models.json 里按模型配置）
        environment.update(env)
    dcli.containers.run(
        VLLM_IMAGE, command=cmd, name=f"ai-vllm-{name}", detach=True,
        device_requests=[docker.types.DeviceRequest(
            device_ids=[str(g) for g in gpus], capabilities=[["gpu"]])],
        network=NETWORK, shm_size="16g",
        environment=environment,
        volumes={f"{HOST_ROOT}/models/llm": {"bind": "/models", "mode": "rw"},
                 cache_host: {"bind": "/root/.cache", "mode": "rw"}},
        **kwargs)


def _find_mmproj(dirname):
    """在模型目录顶层找多模态投影器 mmproj-*.gguf，没有返回 None"""
    root = os.path.join(HOST_ROOT, "models/llm", dirname)
    if not os.path.isdir(root):
        return None
    for f in sorted(os.listdir(root)):
        if f.startswith("mmproj-") and f.endswith(".gguf"):
            return os.path.join(root, f)
    return None


def _run_llamacpp(name, dirname, gpus, max_len, think, gguf_include, extra_args=None,
                  kwargs=None, plan=None):
    """启动 llama.cpp 后端（llama-server，OpenAI 兼容）。GGUF 多分片只传第一个文件，
    llama.cpp 按命名约定自动续载其余分片。flag 以 2026 版 llama-server --help 实测为准：
    -ngl 默认 auto（按显存自动分层）；工具调用由默认启用的 Jinja 聊天解析自动提取；
    思考模型用 --reasoning-format deepseek 把思考放进 message.reasoning_content；
    YaRN 扩上下文（如 --rope-scaling yarn --rope-scale 4.0）经 extra_args 透传。
    --parallel / KV 量化由 plan（plan_llamacpp 按硬件自动规划）决定。"""
    gguf = _find_gguf(dirname, gguf_include)
    plan = plan or {"parallel": 1, "ctx_size": int(max_len), "per_slot_ctx": int(max_len),
                    "cache_type": None, "kv_budget_gb": None, "weights_gb": None,
                    "total_gb": None, "tight": False}
    cmd = ["-m", f"/models/{dirname}/{os.path.relpath(gguf, os.path.join(HOST_ROOT, 'models/llm', dirname))}",
           "--host", "0.0.0.0", "--port", "8000",
           "--metrics",
           "--ctx-size", str(plan["ctx_size"]),
           "--parallel", str(plan["parallel"])]
    if plan.get("cache_type"):
        # KV cache 量化（默认不开；models.json gw.kv_quant 显式指定才加）
        cmd += ["--cache-type-k", str(plan["cache_type"]),
                "--cache-type-v", str(plan["cache_type"])]
    if think:
        # 思考模型：正文与推理分开返回（非思考模型不需要，强制反而可能吞正文）
        cmd += ["--reasoning-format", "deepseek"]
    mmproj = _find_mmproj(dirname)
    if mmproj:
        # 多模态投影器：图像/视频走同一 mtmd 管线（--video-fps 控制取帧率）
        cmd += ["--mmproj",
                f"/models/{dirname}/{os.path.relpath(mmproj, os.path.join(HOST_ROOT, 'models/llm', dirname))}"]
    if extra_args:
        # 如 YaRN 扩上下文（models.json 里按模型配置，与 vLLM 的 --hf-overrides 同源字段）
        cmd += extra_args
    log(f"加载 {name}（llamacpp）: gguf={gguf} mmproj={mmproj} gpus={gpus} max_len={max_len} think={think}"
        + f" | 自动规划 parallel={plan['parallel']} 每slot_ctx={plan['per_slot_ctx']}"
        + (f" 显存={plan.get('total_gb')}GB 权重={plan.get('weights_gb')}GB"
           f" KV预算={plan.get('kv_budget_gb')}GB" if plan.get("total_gb") else "")
        + (f" KV量化={plan['cache_type']}" if plan.get("cache_type") else "")
        + (f" extra_args={extra_args}" if extra_args else ""))
    if plan.get("tight") and plan["parallel"] <= 1 and not plan.get("cache_type"):
        log(f"{name}: KV 预算紧张仅够单 slot；如需更高并发，可在 models.json 的 "
            f"gw 设 kv_quant=q8_0（KV 显存减半，轻微精度影响）或加卡")
    _remove_container(f"ai-llamacpp-{name}")
    # 镜像自带的 HEALTHCHECK 探测 8080，而我们跑 8000——覆盖为 8000，避免误报 unhealthy
    hc = docker.types.Healthcheck(
        test=["CMD", "curl", "-f", "http://localhost:8000/health"],
        interval=30000000000, timeout=5000000000, retries=3, start_period=30000000000)
    dcli.containers.run(
        LLAMACPP_IMAGE, command=cmd, name=f"ai-llamacpp-{name}", detach=True,
        device_requests=[docker.types.DeviceRequest(
            device_ids=[str(g) for g in gpus], capabilities=[["gpu"]])],
        network=NETWORK, shm_size="16g",
        environment={},
        healthcheck=hc,
        volumes={f"{HOST_ROOT}/models/llm": {"bind": "/models", "mode": "rw"}},
        **kwargs)


# ---------- 调度 ----------

def evict_idle(need):
    """need: {gpu: 需要空闲MB}。卸载空闲文本后端（LRU）与空闲 ComfyUI 的已加载模型
    直到满足；返回仍不满足的 gpu 列表。busy（in_flight>0 / 加载中 / 视频队列有任务）
    的模型绝不驱逐。"""
    evicted = False
    unloaded_comfy = set()  # 本轮回已调过 /free 的实例，避免重复卸载刷日志
    deadline = time.time() + 90  # 强杀容器/卸载模型后显存释放有延迟，最多等 90s
    while True:
        free = gpu_free()
        lacking = [g for g, mb in need.items() if free.get(g, {}).get("free", 0) < mb]
        if not lacking:
            return []
        cands = sorted(
            [(k, b) for k, b in backends.items()
             if b["in_flight"] == 0 and b["status"] in ("ready", "adopted")
             and any(g in lacking for g in b["gpus"])],
            key=lambda kv: kv[1]["last_used"])
        if cands:
            vname, victim = cands[0]
            log(f"卸载空闲模型 {victim['container']}（释放 GPU {victim['gpus']}）")
            _remove_container(victim["container"])
            backends.pop(vname, None)
            evicted = True
            continue
        # 文本都卸完还不够：卸载占用 lacking 卡的空闲 ComfyUI 模型（与文本平等）
        freed = False
        for inst in comfy_instances():
            if not any(g in lacking for g in inst["gpus"]):
                continue
            if comfy_queue_busy(inst["port"]):
                continue  # 视频有任务在跑/加载中：绝不驱逐
            if inst["port"] in unloaded_comfy:
                continue
            comfy_unload(inst["port"])
            unloaded_comfy.add(inst["port"])
            freed = True
        evicted = evicted or freed
        if evicted and time.time() < deadline:
            time.sleep(2)
            continue
        return lacking


async def wait_ready(name, cname, timeout=900):
    url = f"http://{cname}:8000/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, tail = await asyncio.to_thread(_container_state, cname)
        if st is None:
            raise RuntimeError("后端容器消失")
        if st != "running":
            raise RuntimeError(f"后端容器退出，日志尾部：\n{tail}")
        try:
            r = await http.get(url, timeout=5.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(5)
    raise RuntimeError(f"等待就绪超时（{timeout}s）")


class ResourceError(Exception):
    pass


def _est_tokens(obj):
    """按解析后的字符串内容粗估 tokens：ASCII ~3字符/token，非 ASCII ~1字符/token。
    对中英文都是偏保守的高估（高估只会少留输出预算，不会放行超限请求）。"""
    if isinstance(obj, str):
        ascii_n = sum(1 for c in obj if ord(c) < 128)
        return ascii_n // 3 + (len(obj) - ascii_n)
    if isinstance(obj, dict):
        return sum(_est_tokens(k) + _est_tokens(v) for k, v in obj.items())
    if isinstance(obj, list):
        return sum(_est_tokens(x) for x in obj)
    return 1


def clamp_max_tokens(name, man, body):
    """钳制 max_tokens，保证 prompt + max_tokens ≤ 模型上下文上限。

    vLLM 对 prompt_tokens + max_tokens > max_model_len 一律 400；
    部分客户端（如 Kimi CLI 的 openai provider）默认带巨大的 max_tokens，
    prompt 刚过半就被 400 误当"上下文溢出"触发 compact（256K 模型 67K 就压缩）。

    钳制下限 min_out：allowed 低于 min_out 时不再继续下调，改按 min_out 转发——
    估算偏保守时 vLLM 正常受理；真放不下时由 vLLM 返回原生 400 透传给客户端
    触发 compact。绝不能钳到极小值（如 1024）：思考模型会把预算全烧在推理上，
    以 finishReason=length 收尾且只有 thinking、没有正文/工具调用，客户端直接报错。
    """
    try:
        payload = json.loads(body)
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    d = man.get(name, {})
    gw = d.get("gw", {})
    max_len = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
    min_out = 16384 if d.get("think") else 4096  # 思考模型推理动辄上万 token
    est_prompt = max(1, _est_tokens(payload))
    allowed = max_len - est_prompt - 1024  # 再留 1024 余量覆盖模板/特殊 token 与估算误差
    for key in ("max_tokens", "max_completion_tokens"):
        mt = payload.get(key)
        if isinstance(mt, int) and mt > allowed:
            payload[key] = max(min_out, allowed)
            log(f"{name}: {key} {mt} -> {payload[key]}（prompt 估 {est_prompt} tokens）"
                + ("［保底预算，超出由后端 400 触发 compact］" if allowed < min_out else ""))
    return json.dumps(payload).encode()


async def ensure_loaded(name, man, want_gpus=None, op=None):
    b = backends.get(name)
    if b and b["status"] == "ready":
        return b
    async with ops_lock:
        b = backends.get(name)
        if b and b["status"] == "ready":
            return b
        if b and b["status"] in ("loading", "adopted"):
            # 认领的/加载中的后端：先等健康确认；确认不了才重建
            if op:
                _op_update(op, status="running", stage="loading",
                           detail=f"{name} 已在加载/认领中，等待健康确认…",
                           target_gpus=b["gpus"])
            try:
                await wait_ready(name, b["container"], timeout=180)
                b["status"] = "ready"
                b["last_used"] = time.time()
                log(f"{name} 健康确认通过")
                return b
            except Exception:
                await asyncio.to_thread(_remove_container, b["container"])
                backends.pop(name, None)
        d = man[name]
        engine = d.get("engine", "vllm")
        gw = d.get("gw", {})
        gpus = [int(x) for x in gw.get("gpus", "1").split(",")]
        util = float(gw.get("util", 0.92))
        max_len = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
        think = bool(d.get("think"))
        cname = f"ai-{engine}-{name}"
        dirname = model_dir(name)
        if not dirname:
            raise RuntimeError(f"模型 {name} 未下载")

        free = await asyncio.to_thread(gpu_free)
        if want_gpus:
            # 客户端指定 GPU：硬性放置，不做静默改道；卡不存在/张数不足直接报错
            missing = [g for g in want_gpus if g not in free]
            if missing:
                raise RuntimeError(
                    f"请求的 GPU {missing} 不存在，本机只有 {sorted(free)}")
            min_gpus = int(d.get("min_gpus", 1))
            if len(want_gpus) < min_gpus:
                raise RuntimeError(
                    f"{name} 至少需要 {min_gpus} 张 GPU（min_gpus={min_gpus}），"
                    f"收到 {len(want_gpus)} 张 {want_gpus}")
            gpus = want_gpus
        elif any(g not in free for g in gpus):
            # 卡数适配：models.json 按双卡布局配置（LLM 在 GPU1），
            # 单卡机上配置的卡不存在时回落到现有卡，否则全部请求 503
            if len(gpus) == 1 and free:
                fb = sorted(free)[-1]
                log(f"{name}: 配置的 GPU{gpus[0]} 不存在，改用 GPU{fb}")
                gpus = [fb]
            else:
                raise RuntimeError(
                    f"{name} 需要 GPU {gpus}，但本机只有 {sorted(free)}")

        # 放置策略：能装就并存（不限一卡一个），装不下才卸空闲模型；正忙的绝不驱逐。
        # 每张卡的可用空间按"已加载模型的完整预算 util×total"扣减，而不是按
        # nvidia-smi 瞬时 free：给已加载模型完整保留其 KV cache 空间（vLLM 的
        # KV pool 就在 util 预算内），新模型不得侵占；同时取实际 free 下限，
        # 不遗漏 ComfyUI 等非网关管理的占用。
        # 客户端指定 GPU 时只尝试该组卡（硬约束）；否则允许回落到其它空闲卡并存。
        cand_sets = [gpus]
        if want_gpus is None and len(gpus) == 1:
            cand_sets += [[g] for g in sorted(free) if g not in gpus]

        def need_of(gs):
            return {g: int(util * free[g]["total"]) + NEED_MARGIN_MB for g in gs}

        def budget_avail(gs):
            r = {}
            for g in gs:
                occ = sum(int(b.get("util", 0.92) * free[g]["total"])
                          for b in backends.values() if g in b["gpus"])
                r[g] = min(free[g]["total"] - occ, free[g]["free"])
            return r

        chosen = None     # (gpus, need)：不动任何已加载模型就放得下
        evictable = None  # (gpus, need)：卸下空闲模型后放得下
        for gs in cand_sets:
            need = need_of(gs)
            avail = budget_avail(gs)
            if all(avail[g] >= need[g] for g in gs):
                chosen = (gs, need)
                break
            pot = dict(avail)
            for b in backends.values():
                if b["in_flight"] == 0 and b["status"] in ("ready", "adopted"):
                    for g in b["gpus"]:
                        if g in pot:
                            pot[g] += int(b.get("util", 0.92) * free[g]["total"])
            # 视频与文本平等：占用目标卡且队列空闲的 ComfyUI 可被卸载，
            # 卸载后该卡可用 = 总量 - 文本预算占用
            for inst in await asyncio.to_thread(comfy_instances):
                if any(g in pot for g in inst["gpus"]) \
                        and not await asyncio.to_thread(comfy_queue_busy, inst["port"]):
                    for g in inst["gpus"]:
                        if g in pot:
                            occ = sum(int(b.get("util", 0.92) * free[g]["total"])
                                      for b in backends.values() if g in b["gpus"])
                            pot[g] = max(pot[g], free[g]["total"] - occ)
            if evictable is None and all(pot[g] >= need[g] for g in gs):
                evictable = (gs, need)

        if chosen is None and evictable is not None:
            gs, need = evictable
            if not await asyncio.to_thread(evict_idle, need):
                chosen = (gs, need)

        if chosen is None:
            all_gs = sorted({g for gs in cand_sets for g in gs})
            busy = [k for k, b in backends.items()
                    if b["in_flight"] > 0 and any(g in all_gs for g in b["gpus"])]
            if busy:
                bn = busy[0]
                raise ResourceError(
                    f"模型 {bn} 正在处理请求（{backends[bn]['in_flight']} 个进行中），"
                    f"各卡显存已按预算分配给已加载模型，装不下 {name}。"
                    f"请等待 {bn} 完成后重试")
            avail = budget_avail(all_gs)
            detail = ", ".join(
                f"GPU{g} 可用 {avail[g]//1024}GB / 需要 {need_of([g])[g]//1024}GB"
                for g in all_gs)
            # 视频任务在跑/排队（含模型加载中）且占目标卡：等它结束，不驱逐繁忙服务
            busy_comfy = [inst for inst in await asyncio.to_thread(comfy_instances)
                          if inst["gpus"] & set(all_gs)
                          and await asyncio.to_thread(comfy_queue_busy, inst["port"])]
            if busy_comfy:
                b = busy_comfy[0]
                raise ResourceError(
                    f"显存不足：{detail}。GPU {sorted(b['gpus'])} 上视频服务 "
                    f"{b['container']} 有任务在运行/排队（含模型加载中），"
                    f"按平等原则不驱逐繁忙服务。请等待视频任务结束后重试，"
                    f"视频空闲时网关会自动卸载其模型让位")
            raise ResourceError(
                f"显存不足：{detail}。已卸载所有可卸载的空闲模型（文本与空闲视频）"
                f"仍不够——真实显存不足，请换更小的模型/变体")

        if chosen[0] != gpus:
            log(f"{name}: 配置的 GPU{gpus} 放不下，改放 GPU{chosen[0]}（与已加载模型并存）")
        gpus = chosen[0]
        if op:
            _op_update(op, status="running", stage="loading",
                       detail=f"放置 GPU {sorted(gpus)}，启动后端容器…",
                       target_gpus=gpus)

        # llama.cpp：按当前硬件（GPU 数/显存、权重体积）自动规划 --parallel 等
        plan = None
        if engine == "llamacpp":
            weights_gb = await asyncio.to_thread(_gguf_weights_gb, dirname,
                                                 d.get("gguf_include"))
            plan = await asyncio.to_thread(plan_llamacpp, gpus, weights_gb, max_len, gw)

        await asyncio.to_thread(_run_backend, name, dirname, gpus, util, max_len,
                                think, gw.get("extra_args"), gw.get("env"),
                                engine, d.get("gguf_include"), plan)
        backends[name] = {"container": cname, "gpus": gpus, "util": util,
                          "in_flight": 0, "last_used": time.time(), "status": "loading",
                          "engine": engine,
                          "slots": plan["parallel"] if plan else None}
        try:
            await wait_ready(name, cname)
        except Exception:
            backends.pop(name, None)
            raise
        backends[name]["status"] = "ready"
        backends[name]["last_used"] = time.time()
        log(f"{name} 就绪")
        if op:
            _op_update(op, status="running", stage="ready",
                       detail=f"{name} 后端就绪（GPU {sorted(gpus)}）",
                       target_gpus=gpus)
        return backends[name]


# ---------- HTTP ----------

@asynccontextmanager
async def lifespan(app):
    adopt_existing()
    for m in PRELOAD:
        asyncio.create_task(_preload(m))
    yield

app = FastAPI(lifespan=lifespan)


def adopt_existing():
    """网关重启后认领还活着的后端容器（标 adopted，可被卸载；首次请求时先等健康检查）"""
    for c in dcli.containers.list(all=True, filters={"status": ["running"]}):
        engine = None
        if c.name.startswith("ai-vllm-"):
            engine, name = "vllm", c.name[len("ai-vllm-"):]
        elif c.name.startswith("ai-llamacpp-"):
            engine, name = "llamacpp", c.name[len("ai-llamacpp-"):]
        else:
            continue
        try:
            req = c.attrs["HostConfig"].get("DeviceRequests") or []
            gpus = [int(x) for r in req for x in (r.get("DeviceIDs") or [])]
            slots = None
            if engine == "llamacpp":
                # 从已运行容器命令行恢复 --parallel，便于 sysmon 显示并发上限
                argv = c.attrs.get("Config", {}).get("Cmd") or []
                for i, a in enumerate(argv):
                    if a in ("--parallel", "-np") and i + 1 < len(argv):
                        try:
                            slots = int(argv[i + 1])
                        except ValueError:
                            pass
                        break
            backends[name] = {"container": c.name, "gpus": gpus, "util": 0.92,
                              "in_flight": 0, "last_used": time.time(),
                              "status": "adopted", "engine": engine, "slots": slots}
            log(f"认领现有后端 {name} gpus={gpus} engine={engine} slots={slots}（待健康确认）")
        except Exception as e:
            log(f"认领 {c.name} 失败（忽略，不影响网关启动）: {e}")


@app.get("/v1/models")
async def list_models():
    man = await asyncio.to_thread(load_manifest)
    data = []
    for name in man:
        if not model_dir(name):
            continue
        data.append({"id": name, "object": "model", "created": 0,
                     "owned_by": "ai-gateway",
                     "loaded": backends.get(name, {}).get("status") == "ready"})
    return {"object": "list", "data": data}


@app.get("/v1/gateway/status")
async def status(request: Request):
    token_stats = await sample_all_backend_tokens()
    out_backends = {}
    for name, backend in backends.items():
        item = dict(backend)
        stat = token_stats.get(name)
        if stat:
            item["tokens"] = {
                "prompt_total": stat.get("prompt_total"),
                "generation_total": stat.get("generation_total"),
                "generation_rate": stat.get("generation_rate", 0.0),
                "prompt_rate": stat.get("prompt_rate", 0.0),
                "updated_at": stat.get("time"),
                "metrics_ok": stat.get("metrics_ok", False),
                "prefill_active": stat.get("prefill_active", False),
                "prefill_tokens": stat.get("prefill_tokens"),
                "prefill_progress": stat.get("prefill_progress"),
                "prefill_rate": stat.get("prefill_rate", 0.0),
                "prefill_elapsed": stat.get("prefill_elapsed"),
            }
        else:
            item["tokens"] = None
        out_backends[name] = item
    model = request.query_params.get("model")
    if model:
        # 单模型过滤：客户端轮询某个模型是否加载完成（status ready / unloaded）
        out_backends = {model: out_backends[model]} if model in out_backends else {}
    return {"backends": out_backends,
            "gpu_free": await asyncio.to_thread(gpu_free),
            "gpu_count": len(await asyncio.to_thread(gpu_free)),
            "comfy_gpus": sorted(await asyncio.to_thread(comfy_gpus))}


async def _unload_idle_backends():
    """卸载所有空闲文本后端（in_flight==0），正忙的跳过。
    与 evict_idle 同一原则——绝不驱逐在处理的模型；视频模型由 evict 流程按需卸载。
    返回 (unloaded, busy)。"""
    unloaded, busy = [], []
    async with ops_lock:
        for name, b in list(backends.items()):
            if b["in_flight"] > 0 or b["status"] == "loading":
                # loading 中的后端容器可能还没建完，强删会让 ensure_loaded 在半空报错
                busy.append({"model": name, "in_flight": b["in_flight"],
                             "status": b["status"]})
                continue
            log(f"手动卸载 {b['container']}（释放 GPU {b['gpus']}）")
            await asyncio.to_thread(_remove_container, b["container"])
            backends.pop(name, None)
            unloaded.append(name)
    return unloaded, busy


@app.post("/v1/gateway/unload_all")
async def unload_all():
    """卸载所有空闲文本后端（in_flight==0），正忙的跳过。
    网关保持常驻：下一个请求仍会按需自动加载对应模型。"""
    unloaded, busy = await _unload_idle_backends()
    return {"unloaded": unloaded, "busy": busy,
            "gpu_free": await asyncio.to_thread(gpu_free)}


async def _do_switch(target, model):
    """执行切换（同步可等/异步后台共用）：llm=确认文本模式，video=拉起 ComfyUI+卸载空闲文本后端。
    校验（target 合法性、llm 的 model 是否存在、video 的 comfy 容器是否就绪）由调用方先做。"""
    if target in ("llm", "text"):
        return {"target": "llm", "model": model or None,
                "backends": {k: dict(v) for k, v in backends.items()},
                "comfy": await asyncio.to_thread(comfy_containers),
                "gpu_free": await asyncio.to_thread(gpu_free)}

    def _start_container(cname):
        dcli.containers.get(cname).start()

    comfy = await asyncio.to_thread(comfy_containers)
    started, already, skipped = [], [], []
    for cname, info in comfy.items():
        if not info["exists"]:
            continue  # ai-comfyui-gpu1 缺失不擅自创建（与 start.sh 同规则）
        if info["status"] == "running":
            already.append(cname)
            continue
        try:
            await asyncio.to_thread(_start_container, cname)
            log(f"switch video: 已启动 {cname}")
            started.append(cname)
        except Exception as e:
            log(f"switch video: 启动 {cname} 失败: {e}")
            skipped.append({"container": cname, "error": str(e)})

    # 卸载空闲文本后端，把显存让给视频（正忙的跳过并列出）
    unloaded, busy = await _unload_idle_backends()
    return {"target": "video",
            "started": started, "already_running": already, "skipped": skipped,
            "unloaded": unloaded, "busy": busy,
            "comfy": await asyncio.to_thread(comfy_containers),
            "gpu_free": await asyncio.to_thread(gpu_free)}


@app.post("/v1/gateway/switch")
async def gateway_switch(request: Request):
    """远程切换"显存归谁"（等价 start.sh --switch，供局域网客户端直接调用）。

    body: {"target": "video" | "llm", "model": "...", "wait": true}
      video: 启动 ComfyUI（已存在即拉起；ai-comfyui 容器缺失返回 409，提示先在
             宿主机用 start.sh 初始化视频配置）+ 卸载全部空闲文本后端把显存让给视频
      llm:   文本模式就绪确认（懒加载：不预热不占显存，模型等请求到达自动加载）；
             model 可选，传了会校验是否在 models.json 清单中
      wait:  true（默认）= 同步执行完返回结果；false = 立即 202 返回 request_id，
             客户端轮询 GET /v1/gateway/ops/{request_id} 直到 status=success/failed
    网关与 Open WebUI 常驻不动，切换只决定"显存归谁"，不需要重启网关。
    返回体含 _op（切换操作状态机：switching → done / failed）。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON",
                                       "type": "invalid_request_error"}}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "请求体必须是 JSON 对象",
                                       "type": "invalid_request_error"}}, status_code=400)
    target = body.get("target", "")
    if target not in ("video", "llm", "text"):
        return JSONResponse({"error": {"message": "target 必须是 video | llm",
                                       "type": "invalid_request_error"}}, status_code=400)
    wait = body.get("wait", True)
    model = body.get("model") or ""

    # 同步预校验：llm 的 model 是否存在；video 的 comfy 容器是否就绪
    if target in ("llm", "text") and model:
        man = await asyncio.to_thread(load_manifest)
        if model not in man:
            return JSONResponse({"error": {"message": f"模型 '{model}' 不存在",
                                           "type": "invalid_model_error"}}, status_code=400)
    if target == "video":
        comfy = await asyncio.to_thread(comfy_containers)
        if not comfy.get("ai-comfyui", {}).get("exists"):
            return JSONResponse({"error": {"message":
                "ai-comfyui 容器不存在，无法切到视频。请先在宿主机执行 "
                "./scripts/start.sh --llm <模型> --video <视频模型> -y "
                "初始化视频配置（--clean 后同样需要）",
                "type": "switch_error"}}, status_code=409)

    op = _new_op("switch", model or None)
    _op_update(op, status="running", stage="switching", detail=f"切换目标 {target}")

    async def _run_switch_op(op, target, model):
        try:
            result = await _do_switch(target, model)
            _op_update(op, status="success", stage="done", detail=f"已切换到 {target}")
            return result
        except Exception as e:
            _op_update(op, status="failed", stage="failed", detail=str(e), error=str(e),
                       error_type="switch_error")
            return None

    if wait is False:
        asyncio.create_task(_run_switch_op(op, target, model))
        return JSONResponse({"status": "accepted", "target": target,
                             "op": await _op_response(op)}, status_code=202)

    result = await _run_switch_op(op, target, model)
    result["op"] = await _op_response(op)
    if op["status"] == "failed":
        return JSONResponse({"error": {"message": op["error"], "type": "switch_error"},
                             "op": result["op"]}, status_code=409)
    return result


async def _run_load(op, model, man, want_gpus):
    """执行一次加载并把状态写回 op（同步可等/异步后台共用）。"""
    _op_update(op, status="running", stage="loading",
               detail=f"准备加载 {model}", target_gpus=want_gpus)
    try:
        b = await ensure_loaded(model, man, want_gpus=want_gpus, op=op)
        _op_update(op, status="success", stage="ready",
                   detail=f"{model} 就绪 on GPU {sorted(b['gpus'])}",
                   target_gpus=b["gpus"])
        return b
    except ResourceError as e:
        _op_update(op, status="failed", stage="failed", detail=str(e), error=str(e),
                   error_type="insufficient_resources")
    except Exception as e:
        _op_update(op, status="failed", stage="failed", detail=f"加载失败: {e}", error=str(e),
                   error_type="backend_error")
    return None


async def _op_response(op):
    """操作快照（异步）：附上实时 GPU 空闲情况，供客户端判断下一步放哪。"""
    snap = _op_snapshot(op)
    snap["gpu_free"] = await asyncio.to_thread(gpu_free)
    return snap


async def _gpu_discovery():
    """GPU 拓扑 + 每卡占用明细（谁占了这张卡）。"""
    info = await asyncio.to_thread(gpu_info)
    free = await asyncio.to_thread(gpu_free)
    by_gpu = {}
    for idx in sorted(info):
        d = info[idx]
        d["free_mb"] = free.get(idx, {}).get("free", d["free_mb"])
        d["llm_backends"] = [
            {"model": k, "container": v["container"], "gpus": list(v["gpus"]),
             "engine": v.get("engine"), "status": v["status"], "in_flight": v["in_flight"]}
            for k, v in backends.items() if idx in v["gpus"]]
        by_gpu[idx] = d
    comfy = {}
    for cname, cinfo in (await asyncio.to_thread(comfy_containers)).items():
        comfy[cname] = {"exists": cinfo["exists"], "status": cinfo["status"],
                        "port": cinfo["port"], "gpus": list(cinfo["gpus"])}
    return {"gpu_count": len(by_gpu), "gpus": by_gpu, "comfy": comfy,
            "updated_at": time.time()}


@app.get("/v1/gateway/gpus")
async def gateway_gpus():
    """GPU 发现：数量、每卡容量/实时占用/利用率，以及谁占用了这张卡。
    客户端据此决定把模型加载到哪张卡（配合 POST /v1/gateway/load 的 gpus 字段）。"""
    return await _gpu_discovery()


@app.get("/v1/gateway/capabilities")
async def gateway_capabilities():
    """说明接口：供客户端 agent 首次连接时学习网关能力与协商协议。
    返回支持的 API 风格、协商步骤、GPU 拓扑、模型目录（含加载状态）与注意事项。"""
    man = await asyncio.to_thread(load_manifest)
    models = {}
    for name in man:
        if not model_dir(name):
            continue
        d = man[name]
        gw = d.get("gw", {})
        b = backends.get(name)
        models[name] = {
            "engine": d.get("engine", "vllm"),
            "min_gpus": int(d.get("min_gpus", 1)),
            "context_window": int(gw.get("max_len", d.get("max_ctx",
                                                          d.get("ctx_default", 32768)))),
            "think": bool(d.get("think")),
            "desc": d.get("desc", ""),
            "loaded": bool(b and b["status"] == "ready"),
            "backend_status": b["status"] if b else "unloaded",
            "gpus": list(b["gpus"]) if b else None,
        }
    gpus = await _gpu_discovery()
    return {
        "name": "ai-gateway",
        "version": "2.0",
        "description": "局域网 AI 模型网关：OpenAI / Anthropic / Responses 兼容推理 + 按 GPU 协商加载。",
        "protocol": "gateway-negotiation-v1",
        "capabilities": {
            "gpu_discovery": True,
            "gpu_selection": True,
            "async_load": True,
            "load_status_polling": True,
            "per_model_unload": True,
            "video_switch": True,
            "api_styles": ["openai-chat", "anthropic-messages", "openai-responses"],
        },
        "endpoints": [
            {"method": "GET", "path": "/v1/gateway/capabilities",
             "purpose": "本说明接口：网关能力、协商协议、GPU 拓扑与模型目录"},
            {"method": "GET", "path": "/v1/gateway/gpus",
             "purpose": "GPU 发现：数量、每卡容量/占用/利用率、谁占用了卡"},
            {"method": "POST", "path": "/v1/gateway/load",
             "purpose": "加载模型到指定 GPU；wait=false 异步返回 request_id 供轮询"},
            {"method": "GET", "path": "/v1/gateway/ops/{request_id}",
             "purpose": "查询加载/切换/卸载操作状态（轮询直到 status=success/failed）"},
            {"method": "GET", "path": "/v1/gateway/ops",
             "purpose": "最近操作列表（load/switch/unload）"},
            {"method": "POST", "path": "/v1/gateway/unload",
             "purpose": "按模型细粒度卸载释放显存"},
            {"method": "POST", "path": "/v1/gateway/switch",
             "purpose": "切换显存归属 text/video；wait=false 异步"},
            {"method": "GET", "path": "/v1/gateway/status",
             "purpose": "实时状态：各后端 in_flight/tokens、GPU 空闲、ComfyUI（支持 ?model= 过滤）"},
            {"method": "GET", "path": "/v1/models",
             "purpose": "OpenAI 兼容模型列表（含 loaded 标记）"},
            {"method": "POST", "path": "/v1/chat/completions",
             "purpose": "OpenAI 推理（未加载自动加载；带 X-Gateway-Load-Async 可 202 异步协商）"},
            {"method": "POST", "path": "/v1/messages",
             "purpose": "Anthropic Messages 推理"},
            {"method": "POST", "path": "/v1/responses",
             "purpose": "Responses API（自动识别 Anthropic / OpenAI 格式）"},
        ],
        "negotiation": {
            "discover_gpus": "GET /v1/gateway/gpus  → 得到 gpu_count 与每卡 free_mb / 占用者",
            "load_on_gpu": "POST /v1/gateway/load  body={'model':..., 'gpus':[0], 'wait':false}",
            "poll_status": "GET /v1/gateway/ops/{request_id}  → 直到 status=success/failed",
            "inference": "模型 ready 后正常调 /v1/chat/completions（GPU 已固化在该模型后端）",
            "async_on_demand": "推理请求带 header X-Gateway-Load-Async:1（可选 X-Gateway-GPUs:0）→ 202 + request_id，轮询后再发",
            "switch": "POST /v1/gateway/switch {'target':'video'|'llm','wait':false} 异步切换显存归属",
        },
        "gpus": gpus,
        "models": models,
        "notes": [
            "gpus 字段传数组，如 [0] 或 [0,1]；多卡模型必须给足 min_gpus 张卡",
            "load 的 gpus 为硬性指定：指定卡放不下时返回 503 并说明各卡可用量，不做静默改道",
            "显存不足时网关会先卸载空闲模型让位（in_flight==0 的最久未用；正忙的绝不驱逐）",
            "后端状态取值：unloaded / adopted / loading / ready；op.status：queued/running/success/failed",
        ],
    }


@app.post("/v1/gateway/load")
async def gateway_load(request: Request):
    """把指定模型加载到指定 GPU（细粒度协商入口）。

    body:
      model: 模型名（models.json 中的 key）
      gpus:  [0] 或 [0,1]；可选，缺省按 models.json gw.gpus 自动放置（可能回落其它空闲卡）
      wait:  true（默认）= 同步等待就绪后返回最终状态；
             false = 立即 202 返回 request_id，客户端轮询 GET /v1/gateway/ops/{request_id}
    说明：
      gpus 为硬性指定：指定卡放不下时返回 503 并给出各卡可用量，不会静默改道。
      显存不足时网关会先卸载空闲模型让位（in_flight==0 的最久未用；正忙的绝不驱逐）。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON",
                                       "type": "invalid_request_error"}}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "请求体必须是 JSON 对象",
                                       "type": "invalid_request_error"}}, status_code=400)
    model = body.get("model", "")
    if not model:
        return JSONResponse({"error": {"message": "缺少 model 字段",
                                       "type": "invalid_request_error"}}, status_code=400)
    man = await asyncio.to_thread(load_manifest)
    if model not in man:
        return JSONResponse({"error": {"message": f"模型 '{model}' 不存在",
                                       "type": "invalid_model_error"}}, status_code=404)
    if not model_dir(model):
        return JSONResponse({"error": {"message": f"模型 '{model}' 未下载",
                                       "type": "invalid_model_error"}}, status_code=404)

    gpus = body.get("gpus")
    if gpus is not None:
        if (not isinstance(gpus, list) or not gpus
                or not all(isinstance(g, int) and g >= 0 for g in gpus)):
            return JSONResponse({"error": {"message":
                "gpus 必须是非负整数数组，如 [0] 或 [0,1]",
                "type": "invalid_request_error"}}, status_code=400)
        gpus = sorted(set(gpus))
        min_gpus = int(man[model].get("min_gpus", 1))
        if len(gpus) < min_gpus:
            return JSONResponse({"error": {"message":
                f"{model} 至少需要 {min_gpus} 张 GPU（min_gpus={min_gpus}），"
                f"收到 {len(gpus)} 张 {gpus}",
                "type": "invalid_request_error"}}, status_code=400)
        free = await asyncio.to_thread(gpu_free)
        missing = [g for g in gpus if g not in free]
        if missing:
            return JSONResponse({"error": {"message":
                f"请求的 GPU {missing} 不存在，本机只有 {sorted(free)}",
                "type": "invalid_request_error"}}, status_code=400)
    wait = body.get("wait", True)

    # 已加载：直接返回当前状态（幂等）
    b = backends.get(model)
    if b and b["status"] == "ready":
        op = _new_op("load", model)
        _op_update(op, status="success", stage="ready",
                   detail=f"{model} 已就绪 on GPU {sorted(b['gpus'])}",
                   target_gpus=b["gpus"])
        return JSONResponse(await _op_response(op), status_code=200)

    op = _new_op("load", model)
    if wait is False:
        asyncio.create_task(_run_load(op, model, man, gpus))
        return JSONResponse({"status": "accepted", "model": model,
                             "op": await _op_response(op)}, status_code=202)

    await _run_load(op, model, man, gpus)
    if op["status"] == "success":
        return JSONResponse(await _op_response(op), status_code=200)
    code = 503 if op.get("error_type") == "insufficient_resources" else 502
    return JSONResponse({"error": {"message": op["error"],
                                   "type": op.get("error_type", "backend_error")},
                         "op": await _op_response(op)}, status_code=code)


@app.post("/v1/gateway/unload")
async def gateway_unload(request: Request):
    """按模型细粒度卸载，释放其占用显存（unload_all 的单模型版本）。
    body: {"model": "..."}。仅卸载空闲后端（in_flight==0）；正忙/加载中返回 409。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON",
                                       "type": "invalid_request_error"}}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": {"message": "请求体必须是 JSON 对象",
                                       "type": "invalid_request_error"}}, status_code=400)
    model = body.get("model", "")
    if not model:
        return JSONResponse({"error": {"message": "缺少 model 字段",
                                       "type": "invalid_request_error"}}, status_code=400)
    b = backends.get(model)
    if not b:
        return JSONResponse({"status": "unloaded", "model": model,
                             "detail": "模型未加载，无需卸载",
                             "gpu_free": await asyncio.to_thread(gpu_free)}, status_code=200)
    if b["in_flight"] > 0 or b["status"] == "loading":
        return JSONResponse({"error": {"message":
            f"{model} 正忙（in_flight={b['in_flight']}, status={b['status']}），拒绝卸载。"
            f"等请求结束或加载完成后重试",
            "type": "busy_error"}}, status_code=409)

    op = _new_op("unload", model)
    _op_update(op, status="running", stage="unloading",
               detail=f"卸载 {b['container']}（释放 GPU {sorted(b['gpus'])}）")
    async with ops_lock:
        b = backends.get(model)
        if b and b["in_flight"] == 0 and b["status"] != "loading":
            await asyncio.to_thread(_remove_container, b["container"])
            backends.pop(model, None)
            _op_update(op, status="success", stage="unloaded", detail=f"{model} 已卸载")
        else:
            _op_update(op, status="failed", stage="failed",
                       detail=f"{model} 状态已变化（in_flight={b and b['in_flight']}, "
                              f"status={b and b['status']}），卸载取消", error="busy")
    return JSONResponse(await _op_response(op), status_code=200)


@app.get("/v1/gateway/ops")
async def gateway_ops_list():
    """最近操作列表（load/switch/unload），按创建时间倒序，保留最近 1h。"""
    recent = sorted(ops.values(), key=lambda o: o["created_at"], reverse=True)[:20]
    return {"ops": [await _op_response(o) for o in recent]}


@app.get("/v1/gateway/ops/{request_id}")
async def gateway_ops_get(request_id: str):
    """轮询操作状态：GET /v1/gateway/ops/{request_id}
    返回 op：{request_id, kind, model, status, stage, detail, target_gpus, error,
              transitions[], backend, gpu_free}。
    status 收敛到 success / failed 即终态；transitions 记录了每次 stage 变更。"""
    op = ops.get(request_id)
    if not op:
        return JSONResponse({"error": {"message":
            f"未知操作 {request_id}（可能已超过保留期 {int(OP_TTL // 60)} 分钟）",
            "type": "not_found"}}, status_code=404)
    return await _op_response(op)


async def proxy(name, request: Request):
    man = await asyncio.to_thread(load_manifest)
    if name not in man or not model_dir(name):
        return JSONResponse({"error": {"message": f"模型 '{name}' 不存在或未下载",
                                       "type": "invalid_request_error"}}, status_code=404)
    # 异步协商：客户端可带 X-Gateway-Load-Async:1（+ 可选 X-Gateway-GPUs:0）让未加载模型的
    # 首请求不挂起等待，改为 202 返回 request_id；轮询 GET /v1/gateway/ops/{id} 就绪后再发请求。
    if not backends.get(name) or backends.get(name, {}).get("status") != "ready":
        if (request.headers.get("x-gateway-load-async", "").lower() in ("1", "true")
                or request.query_params.get("gateway_load_async") == "1"):
            want_gpus = None
            hg = request.headers.get("x-gateway-gpus", "")
            if hg.strip():
                try:
                    want_gpus = sorted({int(x) for x in hg.split(",") if x.strip()})
                except ValueError:
                    return JSONResponse({"error": {"message":
                        f"X-Gateway-GPUs 非法: '{hg}'（应为逗号分隔的 GPU 序号，如 0 或 0,1）",
                        "type": "invalid_request_error"}}, status_code=400)
            op = _new_op("load", name)
            _op_update(op, status="running", stage="loading",
                       detail=f"异步加载 {name}", target_gpus=want_gpus)
            asyncio.create_task(_run_load(op, name, man, want_gpus))
            return JSONResponse({"status": "accepted", "model": name,
                                 "message": f"{name} 正在异步加载，轮询 "
                                            "GET /v1/gateway/ops/{request_id} 直到 "
                                            "status=success 后再请求。",
                                 "op": await _op_response(op)}, status_code=202)
    try:
        b = await ensure_loaded(name, man)
    except ResourceError as e:
        return JSONResponse({"error": {"message": str(e), "type": "insufficient_resources"}},
                            status_code=503)
    except Exception as e:
        return JSONResponse({"error": {"message": f"模型加载失败: {e}",
                                       "type": "backend_error"}}, status_code=502)

    b["in_flight"] += 1
    b["last_used"] = time.time()
    body = await request.body()
    body = clamp_max_tokens(name, man, body)
    qs = request.url.query
    url = f"http://{b['container']}:8000{request.url.path}" + (f"?{qs}" if qs else "")
    headers = {"content-type": request.headers.get("content-type", "application/json")}
    client_stream = False
    live_tokens = False
    if request.method == "POST" and request.url.path == "/v1/chat/completions" \
            and b.get("engine") == "llamacpp":
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                client_stream = bool(payload.get("stream"))
                payload["stream"] = True
                opts = payload.setdefault("stream_options", {})
                if isinstance(opts, dict):
                    opts["include_usage"] = True
                body = json.dumps(payload, ensure_ascii=False).encode()
                live_tokens = True
                begin_live_tokens(name)
        except Exception:
            live_tokens = False
    try:
        req = http.build_request(request.method, url, content=body, headers=headers)
        r = await http.send(req, stream=True)
    except httpx.ConnectError:
        # 后端半死（如网关重启后认领的容器里 vLLM 已挂）：剔除后走重新加载流程重试一次
        b["in_flight"] -= 1
        log(f"后端 {name} 连接失败，剔除并重新加载")
        async with ops_lock:
            if backends.get(name) is b:
                await asyncio.to_thread(_remove_container, b["container"])
                backends.pop(name, None)
        try:
            b = await ensure_loaded(name, man)
        except Exception as e:
            return JSONResponse({"error": {"message": f"后端重建失败: {e}",
                                           "type": "backend_error"}}, status_code=502)
        b["in_flight"] += 1
        try:
            req = http.build_request(request.method, url, content=body, headers=headers)
            r = await http.send(req, stream=True)
        except Exception as e:
            b["in_flight"] -= 1
            return JSONResponse({"error": {"message": f"转发失败: {e}", "type": "proxy_error"}},
                                status_code=502)
    except Exception as e:
        b["in_flight"] -= 1
        return JSONResponse({"error": {"message": f"转发失败: {e}", "type": "proxy_error"}},
                            status_code=502)

    if r.status_code >= 400:
        # 上游错误：读出错误体记入网关日志（vLLM 自身不记 400 原因），再原样透传给客户端
        err = await r.aread()
        await r.aclose()
        b["in_flight"] -= 1
        log(f"{name} 上游 {r.status_code}: {err.decode(errors='replace')[:500]}")
        return StreamingResponse(iter([err]), status_code=r.status_code,
                                 headers={"content-type": r.headers.get(
                                     "content-type", "application/json")})

    if live_tokens and not client_stream:
        try:
            return await _buffered_openai_response(name, r)
        finally:
            b["in_flight"] -= 1
            b["last_used"] = time.time()

    async def gen():
        try:
            async for chunk in r.aiter_raw():
                if live_tokens:
                    count_sse_tokens(name, chunk)
                yield chunk
        finally:
            await r.aclose()
            if live_tokens and _token_state(name).get("live_start"):
                finish_live_tokens(name)
            b["in_flight"] -= 1
            b["last_used"] = time.time()
    return StreamingResponse(gen(), status_code=r.status_code,
                             headers={"content-type": r.headers.get("content-type", "application/json")})


async def _buffered_openai_response(name, r):
    if r.status_code >= 400:
        err = await r.aread()
        await r.aclose()
        try:
            payload = json.loads(err or b"{}")
        except Exception:
            payload = {"error": {"message": err.decode(errors="replace")[:500]}}
        return JSONResponse(payload, status_code=r.status_code,
                            headers={"content-type": r.headers.get("content-type", "application/json")})

    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    tool_parts = {}
    finish_reason = None
    response_id = model = fingerprint = None
    usage = {}
    try:
        async for raw in r.aiter_raw():
            count_sse_tokens(name, raw)
            for event in _sse_json_events(raw):
                response_id = event.get("id", response_id)
                model = event.get("model", model)
                fingerprint = event.get("system_fingerprint", fingerprint)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        message["content"] += delta["content"]
                    if delta.get("reasoning_content"):
                        message["reasoning_content"] += delta["reasoning_content"]
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        part = tool_parts.setdefault(idx, {"id": "", "type": "function",
                                                           "function": {"name": "", "arguments": ""}})
                        part["id"] = tc.get("id", part["id"])
                        fn = tc.get("function") or {}
                        part["function"]["name"] += fn.get("name") or ""
                        part["function"]["arguments"] += fn.get("arguments") or ""
    finally:
        await r.aclose()

    if message["reasoning_content"]:
        message["reasoning_content"] = message.pop("reasoning_content")
    else:
        message.pop("reasoning_content", None)
    if tool_parts:
        message["tool_calls"] = [tool_parts[k] for k in sorted(tool_parts)]
    data = {
        "id": response_id or f"chatcmpl-{name}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or name,
        "system_fingerprint": fingerprint,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop",
                     "logprobs": None}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(data, status_code=200,
                        headers={"content-type": "application/json"})


# ============================================================
# Anthropic Messages API 兼容层
# ============================================================

def _anthropic_to_openai(body: dict, manifest: dict) -> tuple[str, dict]:
    """将 Anthropic 请求体转为 OpenAI 格式。

    返回 (model_name, openai_body)。
    """
    name = body.get("model", "")
    system = body.get("system", "")
    messages_in = body.get("messages", [])
    tools_in = body.get("tools")
    # --- 构建 OpenAI messages ---
    oa_msgs: list[dict] = []
    if system:
        oa_msgs.append({"role": "system", "content": system})

    # 将 Anthropic messages 转为 OpenAI messages，同时处理 tool_use/tool_result 的历史
    # Anthropic 要求 tool_use 和 tool_result 成对出现，我们按顺序组装
    i = 0
    while i < len(messages_in):
        msg = messages_in[i]
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            # 合并连续的 assistant 文本和 tool_use blocks
            parts: list[dict] = []
            while i < len(messages_in) and messages_in[i].get("role") == "assistant":
                msg = messages_in[i]
                c = msg.get("content", "")
                if isinstance(c, str):
                    if c.strip():
                        parts.append({"type": "text", "text": c})
                elif isinstance(c, list):
                    parts.extend(c)
                i += 1
            # 组装 assistant 消息
            text_blocks = [p for p in parts if p.get("type") == "text"]
            tool_blocks = [p for p in parts if p.get("type") == "tool_use"]
            if text_blocks:
                content_text = "\n".join(p["text"] for p in text_blocks)
            else:
                content_text = None

            if tool_blocks:
                tool_calls = []
                for tb in tool_blocks:
                    tool_calls.append({
                        "id": tb.get("id", f"call_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": tb.get("name", ""),
                            "arguments": json.dumps(tb.get("input", {})),
                        },
                    })
                # 如果有文本内容 + tool_calls，OpenAI 要求分开
                if content_text and tool_calls:
                    # 文本作为前文，tool_calls 作为另一个 assistant block
                    oa_msgs.append({"role": "assistant", "content": content_text})
                    oa_msgs.append({"role": "assistant", "tool_calls": tool_calls})
                elif content_text:
                    oa_msgs.append({"role": "assistant", "content": content_text})
                else:
                    oa_msgs.append({"role": "assistant", "tool_calls": tool_calls})
            elif content_text:
                oa_msgs.append({"role": "assistant", "content": content_text})
            continue

        if role == "user":
            c = msg.get("content", "")
            if isinstance(c, list):
                # 查找是否有 tool_result
                text_parts = []
                has_tool_result = False
                for block in c:
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_result":
                        has_tool_result = True
                        break
                if has_tool_result:
                    # 有 tool_result 时，直接透传 content blocks 给 OpenAI
                    # OpenAI 的 assistant 消息中 tool_call 已被提取，tool_result 由 user 承载
                    # 将 tool_result 转成 OpenAI tool_message
                    tool_results = [b for b in c if b.get("type") == "tool_result"]
                    other_parts = [b for b in c if b.get("type") != "tool_result"]
                    if other_parts:
                        texts = [b.get("text", "") for b in other_parts if b.get("type") == "text"]
                        if texts:
                            oa_msgs.append({"role": "user", "content": "\n".join(texts)})
                    for tr in tool_results:
                        oa_msgs.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", f"call_{len(oa_msgs)}"),
                            "content": tr.get("content", ""),
                        })
                elif text_parts:
                    oa_msgs.append({"role": "user", "content": "\n".join(text_parts)})
                else:
                    oa_msgs.append({"role": "user", "content": str(c)})
            else:
                oa_msgs.append({"role": "user", "content": c})
            i += 1

    # --- 转换 tools ---
    oa_tools = None
    if tools_in:
        oa_tools = []
        for t in tools_in:
            schema = t.get("input_schema", {})
            if "parameters" not in schema:
                # Anthropic 的 input_schema 可能直接是参数 schema
                schema = {"type": "object", **schema}
            oa_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": schema,
                },
            })

    # --- 转换 tool_choice ---
    tc = body.get("tool_choice")
    oa_tc = None
    if tc is not None:
        if isinstance(tc, str):
            # "auto", "any", "none"
            oa_tc = tc if tc != "any" else "required"
        elif isinstance(tc, dict):
            if tc.get("type") == "tool":
                oa_tc = {"type": "function", "function": {"name": tc["name"]}}

    # --- 构建 body ---
    oa_body: dict = {
        "model": name,
        "messages": oa_msgs,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
    }
    if body.get("temperature") is not None:
        oa_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oa_body["top_p"] = body["top_p"]
    if body.get("top_k") is not None:
        # OpenAI 不直接支持 top_k，但有些后端支持，保留
        oa_body["top_k"] = body["top_k"]
    if oa_tools:
        oa_body["tools"] = oa_tools
    if oa_tc is not None:
        oa_body["tool_choice"] = oa_tc

    # 处理 thinking 字段 — 转为 system prompt 指令
    thinking = body.get("thinking")
    if thinking:
        budget = thinking.get("type") == "enabled" and thinking.get("budget_tokens") or 1024
        thinking_hint = (
            f"You will engage in extended thinking before responding. "
            f"Plan your response carefully. Think through the problem step by step. "
            f"Maximum thinking budget: {budget} tokens."
        )
        # Prepend thinking hint to system prompt
        if oa_msgs and oa_msgs[0].get("role") == "system":
            oa_msgs[0]["content"] = thinking_hint + "\n\n" + oa_msgs[0]["content"]
        else:
            oa_msgs.insert(0, {"role": "system", "content": thinking_hint})

    return name, oa_body


def _openai_to_anthropic_content(openai_msg: dict) -> list[dict]:
    """将 OpenAI assistant message 转为 Anthropic content blocks。"""
    blocks: list[dict] = []
    content = openai_msg.get("content")
    if content is not None and content != "":
        blocks.append({"type": "text", "text": content})

    tool_calls = openai_msg.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            input_dict = json.loads(fn.get("arguments", "{}"))
        except Exception:
            input_dict = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"call_{len(blocks)}"),
            "name": fn.get("name", ""),
            "input": input_dict,
        })

    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _build_anthropic_response(data: dict) -> dict:
    """将 OpenAI 非流式响应转为 Anthropic 响应体。"""
    choice = data["choices"][0]
    msg = choice["message"]

    stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
    stop_reason = stop_map.get(choice.get("finish_reason", "stop"), "end_turn")

    content = _openai_to_anthropic_content(msg)

    # 尝试从 reasoning 中提取 thinking 内容
    thinking_content = msg.get("reasoning_content", "")

    resp = {
        "id": data.get("id", "msg_fake"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": data.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,  # vLLM 不直接返回
            "output_tokens": 0,
        },
    }

    # 如果有 reasoning，添加 thinking block
    if thinking_content:
        # 在 content 最前面插入 thinking block
        content.insert(0, {"type": "thinking", "thinking": thinking_content})
        resp["content"] = content

    return resp


def _stream_anthropic(data_iter, model_name: str):
    """Generator: 将 OpenAI SSE 流转为 Anthropic SSE 流。

    每 yield 一行 "data: ..." JSON。
    """
    # 状态跟踪
    block_id = 0
    input_json_accum = ""
    message_id = f"msg_{model_name}_{int(time.time() * 1000)}_{block_id:04d}"
    # 首行: message_start
    yield f'data: {json.dumps({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })}\n\n'

    text_accum = ""
    thinking_accum = ""
    has_tool_use = False
    current_tool_id = None
    current_tool_name = None

    try:
        for raw in data_iter:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            if not line.startswith("data: "):
                continue
            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                break
            try:
                payload = json.loads(payload_str)
            except Exception:
                continue

            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            # --- text delta ---
            text_delta = delta.get("content")
            reasoning = delta.get("reasoning_content", "")
            if text_delta:
                if not text_accum and not has_tool_use:
                    # 开启 content_block_start
                    block_id += 1
                    yield f'data: {json.dumps({
                        "type": "content_block_start",
                        "index": block_id - 1,
                        "content_block": {"type": "text", "text": ""},
                    })}\n\n'
                text_accum += text_delta
                yield f'data: {json.dumps({
                    "type": "content_block_delta",
                    "index": block_id - 1,
                    "delta": {"type": "text_delta", "text": text_delta},
                })}\n\n'
            if reasoning:
                if not thinking_accum and not has_tool_use:
                    block_id += 1
                    yield f'data: {json.dumps({
                        "type": "content_block_start",
                        "index": block_id - 1,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })}\n\n'
                thinking_accum += reasoning
                yield f'data: {json.dumps({
                    "type": "content_block_delta",
                    "index": block_id - 1,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                })}\n\n'

            # --- tool call delta ---
            tool_calls = delta.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    if not has_tool_use or block_id == idx:
                        has_tool_use = True
                    fn = tc.get("function", {})
                    tc_id = tc.get("id", f"call_{block_id}")
                    tc_name = fn.get("name", "")
                    tc_args = fn.get("arguments", "")

                    if not has_tool_use or (not current_tool_id):
                        # 开启 content_block_start for tool_use
                        block_id += 1
                        yield f'data: {json.dumps({
                            "type": "content_block_start",
                            "index": block_id - 1,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name},
                        })}\n\n'
                        current_tool_id = tc_id
                        current_tool_name = tc_name
                        input_json_accum = ""

                    if tc_args:
                        input_json_accum += tc_args
                        yield f'data: {json.dumps({
                            "type": "content_block_delta",
                            "index": block_id - 1,
                            "delta": {"type": "input_json_delta", "partial_json": tc_args},
                        })}\n\n'

            # --- finish ---
            if finish:
                stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
                stop_reason = stop_map.get(finish, "end_turn")
                # 关闭 text block
                if text_accum:
                    yield f'data: {json.dumps({
                        "type": "content_block_stop",
                        "index": 0 if not has_tool_use else (1 if thinking_accum else 0),
                    })}\n\n'
                # 关闭 thinking block
                if thinking_accum:
                    yield f'data: {json.dumps({
                        "type": "content_block_stop",
                        "index": 0 if not text_accum and not has_tool_use else 1,
                    })}\n\n'
                # 关闭 tool_use block
                if has_tool_use:
                    yield f'data: {json.dumps({
                        "type": "content_block_stop",
                        "index": block_id - 1,
                    })}\n\n'
                # message_stop
                yield f'data: {json.dumps({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 1},  # 粗略估计
                })}\n\n'
    finally:
        pass

    yield "data: [DONE]\n\n"


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Anthropic token 计数接口（虚拟实现，返回粗略值）。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "请求体不是合法 JSON"}},
            status_code=400)

    name = body.get("model", "")
    # 用 messages 字段粗略估算 token 数
    messages = body.get("messages", [])
    total_tokens = 0
    for msg in messages:
        c = msg.get("content", "")
        if isinstance(c, str):
            total_tokens += len(c) // 4  # 粗略估算
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict):
                    total_tokens += len(json.dumps(block)) // 4
    return {
        "input_tokens": total_tokens,
        "output_tokens": 0,
    }


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API 入口。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "请求体不是合法 JSON"}},
            status_code=400)

    name = body.get("model", "")
    if not name:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "缺少 model 字段"}},
            status_code=400)

    man = await asyncio.to_thread(load_manifest)
    if name not in man:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_model_error", "message": f"模型 '{name}' 不存在"}},
            status_code=400)

    # 转换请求体
    _, oa_body = _anthropic_to_openai(body, man)
    is_stream = oa_body.get("stream", False)

    # 确保模型已加载
    try:
        b = await ensure_loaded(name, man)
    except ResourceError as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "insufficient_resources", "message": str(e)}},
            status_code=503)
    except Exception as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "backend_error", "message": f"模型加载失败: {e}"}},
            status_code=502)

    b["in_flight"] += 1
    b["last_used"] = time.time()

    qs = request.url.query
    url = f"http://{b['container']}:8000/v1/chat/completions" + (f"?{qs}" if qs else "")
    headers = {"content-type": "application/json"}
    try:
        req = http.build_request("POST", url, content=json.dumps(oa_body).encode(), headers=headers)
        r = await http.send(req, stream=True)
    except Exception as e:
        b["in_flight"] -= 1
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": f"转发失败: {e}"}},
            status_code=502)

    # 非流式：读取完整响应并转换
    if not is_stream:
        try:
            oa_resp = json.loads(await r.aread())
        except Exception:
            b["in_flight"] -= 1
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": "读取响应失败"}},
                status_code=502)
        b["in_flight"] -= 1
        anth_resp = _build_anthropic_response(oa_resp)
        return JSONResponse(anth_resp)

    # 流式：转换 SSE 流
    gen = _stream_anthropic(r.aiter_raw(), name)
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"x-request-id": r.headers.get("x-request-id", "")})


@app.get("/v1/messages/{model_name}")
async def anthropic_model_info(model_name: str):
    """Anthropic 单模型信息接口。"""
    man = await asyncio.to_thread(load_manifest)
    if model_name not in man:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_model_error", "message": f"模型 '{model_name}' 不存在"}},
            status_code=404)

    d = man[model_name]
    return {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "ai-gateway",
        "context_window": d.get("max_ctx", d.get("ctx_default", 32768)),
        "description": d.get("desc", ""),
    }


# ============================================================
# Anthropic Responses API ( /v1/responses )
# Anthropic 2025 新一代接口，替代 /v1/messages
# ============================================================

def _responses_input_to_openai_items(input_items: list) -> list[dict]:
    """将 Responses API input 数组转为 OpenAI messages。

    Responses 的 input 是扁平的混合项：messages + function_call + function_call_output
    需要重构成 OpenAI 的 role/content 格式。
    """
    items = list(input_items)  # 拷贝避免修改原数据
    oa_msgs: list[dict] = []
    i = 0
    while i < len(items):
        item = items[i]
        item_type = item.get("type", "")

        if item_type == "message":
            # {"type": "message", "role": "user|assistant", "content": "..." | [...]}
            role = item.get("role")
            content = item.get("content", "")
            if isinstance(content, str):
                oa_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                # 内容块列表 — 转换为 OpenAI 格式
                texts = []
                for block in content:
                    if block.get("type") == "input_text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "input_image":
                        # OpenAI vision 格式
                        img_url = block.get("image_url", block.get("image", ""))
                        texts.append(f"[图片: {img_url}]")
                    elif block.get("type") == "document":
                        doc_url = block.get("document", "")
                        texts.append(f"[文档: {doc_url}]")
                oa_msgs.append({"role": role, "content": "\n".join(texts) or ""})
            i += 1

        elif item_type == "function_call":
            # 独立的 tool_use item — 与之前的 assistant message 合并
            call_id = item.get("call_id", f"call_{len(oa_msgs)}")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            # 查找最后一个 assistant 消息并追加 tool_calls
            found = False
            for m in reversed(oa_msgs):
                if m.get("role") == "assistant":
                    tc = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                    m.setdefault("tool_calls", []).append(tc)
                    found = True
                    break
            if not found:
                # 没有前文 assistant message，新建一个
                oa_msgs.append({
                    "role": "assistant",
                    "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
                    "content": "",
                })
            i += 1

        elif item_type == "function_call_output":
            # 独立的 tool_result item — 转成 tool role message
            oa_msgs.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
            i += 1

        else:
            # 未知类型，跳过
            i += 1

    return oa_msgs


def _responses_tools_to_openai(tools_in: list) -> list[dict]:
    """转换 Responses tools 为 OpenAI tools 格式。"""
    oa_tools = []
    for t in tools_in:
        if t.get("type", "").startswith("web_search") or t.get("name") == "web_search":
            continue  # web_search 暂不支持
        schema = t.get("input_schema", {}) or t.get("parameters", {})
        if "parameters" not in schema:
            schema = {"type": "object", **schema}
        oa_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return oa_tools


def _responses_tool_choice_to_openai(tc):
    """转换 Responses tool_choice 为 OpenAI 格式。"""
    if isinstance(tc, str):
        if tc == "auto":
            return "auto"
        elif tc == "required":
            return "required"
    if isinstance(tc, dict):
        tc_type = tc.get("type", "")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "required":
            return "required"
        elif tc_type == "function":
            return {"type": "function", "function": {"name": tc["name"]}}
    return None


def _responses_reasoning_to_think(reasoning):
    """转换 Responses reasoning 为 thinking 字符串。"""
    if not reasoning:
        return None
    effort = reasoning.get("effort", "minimal")
    budget_tokens = reasoning.get("max_media_inputs", 0)
    return {
        "type": "enabled",
        "budget_tokens": {"high": 16000, "medium": 8000, "low": 4000, "minimal": 1024}.get(effort, 1024),
    }


def _detect_responses_format(body: dict) -> str:
    """自动检测 Responses API 请求体格式。

    返回 "anthropic" | "openai" | "unknown"。
    通过字段特征判断，不依赖 HTTP 头。
    """
    # --- Anthropic 独有字段 ---
    if body.get("anthropic_version"):
        return "anthropic"
    if "output_config" in body:
        return "anthropic"

    # --- OpenAI 独有字段 ---
    if "output_schema" in body:
        return "openai"
    if "parallel_tool_calls" in body:
        return "openai"
    if "store" in body:
        return "openai"

    # --- tools 数组结构区分 ---
    tools = body.get("tools")
    if tools and isinstance(tools, list) and len(tools) > 0:
        for t in tools:
            if isinstance(t, dict):
                if "input_schema" in t:
                    return "anthropic"
                fn = t.get("function")
                if isinstance(fn, dict) and "parameters" in fn:
                    return "openai"

    # --- input 结构区分 ---
    inp = body.get("input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                itype = item.get("type", "")
                if itype == "input_message":
                    return "anthropic"
                if itype == "function_call" or itype == "function_call_output":
                    # 两者都用，但 Anthropic 的 function_call 有 call_id, OpenAI 也有
                    # 看其他字段：Anthropic 用 input_message 包裹，OpenAI 直接在 input 数组里
                    continue
                if itype == "file":
                    return "openai"  # OpenAI file item

    return "unknown"


def _responses_to_anthropic_messages(body: dict, manifest: dict) -> tuple[str, dict]:
    """将 Responses API 请求体转为 Anthropic Messages 格式（再经现有 _anthropic_to_openai 转 OpenAI）。

    返回 (model_name, anthropic_body)。
    """
    name = body.get("model", "")

    # --- input → messages ---
    input_items = body.get("input", [])
    oa_msgs = _responses_input_to_openai_items(input_items)

    # --- instructions → system ---
    instructions = body.get("instructions", "")

    # --- 构建 Anthropic body ---
    anth_body = {
        "model": name,
        "messages": [],
        "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 4096)),
        "stream": body.get("stream", False),
    }

    if body.get("temperature") is not None:
        anth_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        anth_body["top_p"] = body["top_p"]

    # system → system
    if instructions:
        anth_body["system"] = instructions

    # 把 OpenAI 格式 messages 转回 Anthropic messages 格式
    # 因为我们复用 _anthropic_to_openai，所以需要将 OpenAI messages 转回 Anthropic 格式
    anth_msgs = []
    for m in oa_msgs:
        role = m["role"]
        content = m.get("content", "")
        if role == "tool":
            # tool message → 需要先有 tool_use，再跟 tool_result
            anth_msgs.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": content}],
            })
        elif role == "assistant" and m.get("tool_calls"):
            # assistant 有 tool_calls
            parts = []
            if content and content != "":
                parts.append({"type": "text", "text": content})
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                try:
                    input_dict = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    input_dict = {}
                parts.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": input_dict,
                })
            anth_msgs.append({"role": "assistant", "content": parts if parts else ""})
        else:
            anth_msgs.append({"role": role, "content": content})
    anth_body["messages"] = anth_msgs

    # tools
    tools_in = body.get("tools", [])
    if tools_in:
        anth_tools = []
        for t in tools_in:
            schema = t.get("input_schema", {}) or t.get("parameters", {})
            anth_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": schema if schema else {"type": "object"},
            })
        anth_body["tools"] = anth_tools

    # tool_choice
    tc = body.get("tool_choice")
    oa_tc = _responses_tool_choice_to_openai(tc) if tc else None
    if oa_tc:
        if isinstance(oa_tc, str):
            anth_body["tool_choice"] = "auto" if oa_tc == "auto" else "any" if oa_tc == "required" else "none"
        elif isinstance(oa_tc, dict):
            anth_body["tool_choice"] = {"type": "tool", "name": oa_tc["function"]["name"]}

    # reasoning → thinking
    reasoning = body.get("reasoning")
    if reasoning:
        think = _responses_reasoning_to_think(reasoning)
        if think:
            budget = think["budget_tokens"]
            thinking_hint = (
                f"You will engage in extended thinking before responding. "
                f"Plan your response carefully. Think through the problem step by step. "
                f"Maximum thinking budget: {budget} tokens."
            )
            if anth_body.get("system"):
                anth_body["system"] = thinking_hint + "\n\n" + anth_body["system"]
            else:
                # 插入 thinking 作为第一条 user 消息前的 system
                # Anthropic: thinking 通过 system prompt 传递
                if anth_msgs and anth_msgs[0].get("role") == "system":
                    pass  # 已有 system
                else:
                    pass  # 没有 system，thinking 已在 body 层

    # 返回后需经 _anthropic_to_openai 转 OpenAI
    # 但 anth_body 已经是 Anthropic 格式了，调用 _anthropic_to_openai 即可
    return name, anth_body


def _build_responses_output(openai_data: dict) -> dict:
    """将 OpenAI 非流式响应转为 Responses API 格式。"""
    choice = openai_data["choices"][0]
    msg = choice["message"]

    output_items: list[dict] = []

    # --- text content → output_text ---
    content = msg.get("content", "")
    if content:
        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })

    # --- tool_calls → function_call ---
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        output_items.append({
            "type": "function_call",
            "call_id": tc.get("id", f"call_{len(output_items)}"),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })

    # --- reasoning_content → reasoning ---
    reasoning_content = msg.get("reasoning_content", "")
    reasoning_items = None
    if reasoning_content:
        reasoning_items = [{"type": "summary", "text": reasoning_content}]

    stop_map = {"stop": "completed", "tool_calls": "tool_calls", "length": "max_output_tokens"}
    status = stop_map.get(choice.get("finish_reason", "stop"), "completed")

    # 尝试从 OpenAI id 派生 response id
    response_id = openai_data.get("id", f"resp_{int(time.time() * 1000)}")

    resp = {
        "id": response_id,
        "model": openai_data.get("model", ""),
        "output": output_items,
        "status": status,
        "stop_reason": choice.get("finish_reason"),
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }

    if reasoning_items:
        resp["reasoning"] = reasoning_items

    return resp


def _stream_responses(openai_iter, model_name: str):
    """Generator: 将 OpenAI SSE 流转换为 Responses API 流式 SSE。

    Responses API 流式格式与 Messages API类似，但使用 response_* 前缀。
    """
    response_id = f"resp_{model_name}_{int(time.time() * 1000)}"
    item_index = 0
    accumulated_text = ""
    accumulated_json = ""
    current_call_id = None
    current_name = ""

    # 首行: response.created
    yield f'data: {json.dumps({
        "type": "response.created",
        "response": {
            "id": response_id,
            "model": model_name,
            "status": "in_progress",
            "output": [],
            "usage": {},
        },
    })}\n\n'

    try:
        for raw in openai_iter:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            if not line.startswith("data: "):
                continue
            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                # response.done
                yield f'data: {json.dumps({
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "model": model_name,
                        "status": "completed",
                        "output": [],
                        "usage": {"output_tokens": 1},
                    },
                })}\n\n'
                break
            try:
                payload = json.loads(payload_str)
            except Exception:
                continue

            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            text = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            tool_calls = delta.get("tool_calls", [])

            if text:
                # response.output_text.delta
                yield f'data: {json.dumps({
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": 0,
                    "delta": text,
                })}\n\n'
                accumulated_text += text

            if reasoning:
                # Responses API 暂不处理 reasoning 流式
                pass

            if tool_calls:
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    fn = tc.get("function", {})
                    tc_id = tc.get("id", f"call_{item_index}")
                    tc_name = fn.get("name", "")
                    tc_args = fn.get("arguments", "")

                    if not current_call_id:
                        # function_call.created + .partial_create
                        item_index += 1
                        yield f'data: {json.dumps({
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "output_index": idx,
                            "call_id": tc_id,
                            "delta": "",
                        })}\n\n'
                        current_call_id = tc_id
                        current_name = tc_name

                    if tc_args:
                        yield f'data: {json.dumps({
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "output_index": idx,
                            "call_id": tc_id,
                            "delta": tc_args,
                        })}\n\n'
                        accumulated_json += tc_args

            if finish:
                stop_map = {"stop": "completed", "tool_calls": "tool_calls", "length": "max_output_tokens"}
                if finish == "tool_calls" and current_call_id:
                    # function_call.done
                    yield f'data: {json.dumps({
                        "type": "response.function_call_arguments.done",
                        "response_id": response_id,
                        "output_index": item_index,
                        "call_id": current_call_id,
                        "arguments": accumulated_json,
                    })}\n\n'
                # response.done
                yield f'data: {json.dumps({
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "model": model_name,
                        "status": stop_map.get(finish, "completed"),
                        "output": [],
                        "usage": {"output_tokens": 1},
                    },
                })}\n\n'
    finally:
        pass

    yield "data: [DONE]\n\n"


# ============================================================
# OpenAI Responses API 转换器
# Responses → Chat Completions 已在上面处理
# 这里是 Chat Completions → Responses 输出转换
# ============================================================

def _openai_chat_to_responses_output(oa_data: dict) -> dict:
    """将 OpenAI Chat Completions 非流式响应转为 OpenAI Responses 格式。"""
    choice = oa_data["choices"][0]
    msg = choice["message"]

    output_items: list[dict] = []
    accumulated_text = ""

    # text content → response/output_text
    content = msg.get("content", "")
    if content:
        output_items.append({
            "type": "response.output_text.delta",
            "delta": content,
        })
        accumulated_text = content

    # reasoning → response/reasoning
    reasoning = msg.get("reasoning_content", "")
    if reasoning:
        output_items.append({
            "type": "response.reasoning.delta",
            "delta": reasoning,
        })

    # tool_calls → response/function_call
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        output_items.append({
            "type": "response.function_call",
            "call_id": tc.get("id", f"call_{len(output_items)}"),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })

    stop_map = {"stop": "completed", "tool_calls": "tool_calls", "length": "max_output_tokens"}
    status = stop_map.get(choice.get("finish_reason", "stop"), "completed")

    resp_id = oa_data.get("id", f"resp_{int(time.time() * 1000)}")

    return {
        "id": resp_id,
        "object": "response",
        "model": oa_data.get("model", ""),
        "status": status,
        "output": output_items,
        "stop_tokens": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def _openai_chat_stream_to_responses(data_iter, model_name: str):
    """Generator: 将 OpenAI Chat Completions SSE 流转换为 OpenAI Responses 流。"""
    response_id = f"resp_{model_name}_{int(time.time() * 1000)}"
    item_index = 0
    accumulated_text = ""
    accumulated_json = ""
    current_call_id = None

    yield f'data: {json.dumps({
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "model": model_name,
            "status": "in_progress",
            "output": [],
        },
    })}\n\n'

    try:
        for raw in data_iter:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            if not line.startswith("data: "):
                continue
            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                yield f'data: {json.dumps({
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model_name,
                        "status": "completed",
                        "output": [],
                        "usage": {"output_tokens": 1},
                    },
                })}\n\n'
                break
            try:
                payload = json.loads(payload_str)
            except Exception:
                continue

            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            finish = choices[0].get("finish_reason")

            text = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")
            tool_calls = delta.get("tool_calls", [])

            if text:
                yield f'data: {json.dumps({
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": item_index,
                    "delta": text,
                })}\n\n'
                accumulated_text += text

            if reasoning:
                yield f'data: {json.dumps({
                    "type": "response.reasoning.delta",
                    "response_id": response_id,
                    "output_index": item_index,
                    "delta": reasoning,
                })}\n\n'

            if tool_calls:
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    fn = tc.get("function", {})
                    tc_id = tc.get("id", f"call_{item_index}")
                    tc_name = fn.get("name", "")
                    tc_args = fn.get("arguments", "")

                    if not current_call_id:
                        # function_call.start
                        item_index += 1
                        current_call_id = tc_id
                        accumulated_json = ""

                    if tc_args:
                        yield f'data: {json.dumps({
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "output_index": item_index - 1,
                            "call_id": tc_id,
                            "delta": tc_args,
                        })}\n\n'
                        accumulated_json += tc_args

            if finish:
                if finish == "tool_calls" and current_call_id:
                    yield f'data: {json.dumps({
                        "type": "response.function_call_arguments.done",
                        "response_id": response_id,
                        "output_index": item_index - 1,
                        "call_id": current_call_id,
                        "arguments": accumulated_json,
                    })}\n\n'

                stop_map = {"stop": "completed", "tool_calls": "tool_calls", "length": "max_output_tokens"}
                yield f'data: {json.dumps({
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "model": model_name,
                        "status": stop_map.get(finish, "completed"),
                        "output": [],
                        "usage": {"output_tokens": 1},
                    },
                })}\n\n'
    finally:
        pass

    yield "data: [DONE]\n\n"


@app.post("/v1/responses")
async def responses_api(request: Request):
    """统一 Responses API 入口，按请求体结构自动检测格式。

    通过 body 字段特征判断，不依赖 HTTP 头：
    • Anthropic: anthropic_version / output_config / input_schema / input_message
    • OpenAI: output_schema / parallel_tool_calls / store / parameters / file
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "请求体不是合法 JSON"}},
            status_code=400)

    # 类型检查：必须是 JSON object，不能是 array / number / string / null
    if not isinstance(body, dict):
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                "message": "请求体必须是 JSON 对象，不是 " + type(body).__name__}},
            status_code=400)

    # 空对象也直接拒绝
    if not body:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                "message": "请求体不能为空"}},
            status_code=400)

    fmt = _detect_responses_format(body)
    if fmt == "anthropic":
        return await _handle_anthropic_responses(request, body)
    elif fmt == "openai":
        return await _handle_openai_responses(request, body)
    else:
        # 无法识别：先尝试 Anthropic，失败再试 OpenAI
        r = await _handle_anthropic_responses(request, body)
        if isinstance(r, JSONResponse) and r.status_code == 400:
            # 400 可能是格式不匹配，不立即返回，给 OpenAI 一次机会
            pass
        else:
            return r
        return await _handle_openai_responses(request, body)


async def _handle_anthropic_responses(request: Request, body: dict):
    """Anthropic Responses API 处理函数。

    body 已由路由层解析并传入。
    """
    name = body.get("model", "")
    if not name:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "缺少 model 字段"}},
            status_code=400)

    # 校验 input 必须存在且非空
    inp = body.get("input")
    if not inp:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error",
                "message": "缺少 input 字段，Responses API 需要提供输入内容"}},
            status_code=400)

    man = await asyncio.to_thread(load_manifest)
    if name not in man:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_model_error", "message": f"模型 '{name}' 不存在"}},
            status_code=400)

    try:
        anth_body = _responses_to_anthropic_messages(body, man)
        model_name, oa_body = _anthropic_to_openai(anth_body, man)
    except Exception as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": f"请求格式错误: {e}"}},
            status_code=400)

    is_stream = oa_body.get("stream", False)

    try:
        b = await ensure_loaded(model_name, man)
    except ResourceError as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "insufficient_resources", "message": str(e)}},
            status_code=503)
    except Exception as e:
        return JSONResponse(
            {"type": "error", "error": {"type": "backend_error", "message": f"模型加载失败: {e}"}},
            status_code=502)

    b["in_flight"] += 1
    b["last_used"] = time.time()

    url = f"http://{b['container']}:8000/v1/chat/completions"
    qs = request.url.query
    if qs:
        url += f"?{qs}"
    headers = {"content-type": "application/json"}
    try:
        req = http.build_request("POST", url, content=json.dumps(oa_body).encode(), headers=headers)
        r = await http.send(req, stream=True)
    except Exception as e:
        b["in_flight"] -= 1
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": f"转发失败: {e}"}},
            status_code=502)

    if not is_stream:
        try:
            oa_resp = json.loads(await r.aread())
        except Exception:
            b["in_flight"] -= 1
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": "读取响应失败"}},
                status_code=502)
        b["in_flight"] -= 1
        resp = _build_responses_output(oa_resp)
        return JSONResponse(resp)

    gen = _stream_responses(r.aiter_raw(), model_name)
    return StreamingResponse(gen, media_type="text/event-stream")


async def _handle_openai_responses(request: Request, body: dict):
    """OpenAI Responses API 处理函数。

    body 已由路由层解析并传入。
    将 OpenAI Responses 格式 → OpenAI Chat Completions → OpenAI Responses 格式。
    """
    name = body.get("model", "")
    if not name:
        return JSONResponse(
            {"error": {"message": "缺少 model 字段", "type": "invalid_request_error"}},
            status_code=400)

    # 校验 input 必须存在且非空
    inp = body.get("input")
    if not inp:
        return JSONResponse(
            {"error": {"message": "缺少 input 字段，Responses API 需要提供输入内容",
                       "type": "invalid_request_error"}},
            status_code=400)

    man = await asyncio.to_thread(load_manifest)
    if name not in man:
        return JSONResponse(
            {"error": {"message": f"模型 '{name}' 不存在", "type": "invalid_model_error"}},
            status_code=400)

    is_stream = body.get("stream", False)
    max_output_tokens = body.get("max_output_tokens")
    output_schema = body.get("output_schema")

    # --- input → OpenAI messages ---
    input_items = body.get("input", [])
    oa_msgs = _responses_input_to_openai_items(input_items)

    # --- instructions → system ---
    instructions = body.get("instructions", "")

    # --- output_schema → response_format ---
    oa_response_format = None
    if output_schema:
        oa_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.get("name", "structured_output"),
                "schema": output_schema.get("schema"),
                "strict": output_schema.get("strict", False),
            },
        }

    # --- 构建 OpenAI Chat Completions body ---
    oa_body = {
        "model": name,
        "messages": oa_msgs,
        "stream": is_stream,
    }
    if max_output_tokens is not None:
        oa_body["max_tokens"] = max_output_tokens
    if body.get("temperature") is not None:
        oa_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        oa_body["top_p"] = body["top_p"]
    if oa_response_format:
        oa_body["response_format"] = oa_response_format

    # tools (OpenAI Responses 与 Chat Completions 格式一致)
    tools_in = body.get("tools", [])
    if tools_in:
        oa_tools = []
        for t in tools_in:
            if t.get("type", "").startswith("web_search") or t.get("name") == "web_search":
                continue
            schema = t.get("input_schema", {}) or t.get("parameters", {})
            oa_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": schema if schema else {"type": "object"},
                },
            })
        if oa_tools:
            oa_body["tools"] = oa_tools

    # tool_choice
    tc = body.get("tool_choice")
    oa_tc = _responses_tool_choice_to_openai(tc) if tc else None
    if oa_tc is not None:
        oa_body["tool_choice"] = oa_tc

    # --- 确保模型加载 ---
    try:
        b = await ensure_loaded(name, man)
    except ResourceError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "insufficient_resources"}},
            status_code=503)
    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"模型加载失败: {e}", "type": "backend_error"}},
            status_code=502)

    b["in_flight"] += 1
    b["last_used"] = time.time()

    url = f"http://{b['container']}:8000/v1/chat/completions"
    qs = request.url.query
    if qs:
        url += f"?{qs}"
    headers = {"content-type": "application/json"}
    try:
        req = http.build_request("POST", url, content=json.dumps(oa_body).encode(), headers=headers)
        r = await http.send(req, stream=True)
    except Exception as e:
        b["in_flight"] -= 1
        return JSONResponse(
            {"error": {"message": f"转发失败: {e}", "type": "api_error"}},
            status_code=502)

    # --- 非流式：Chat Completions → Responses ---
    if not is_stream:
        try:
            oa_resp = json.loads(await r.aread())
        except Exception:
            b["in_flight"] -= 1
            return JSONResponse(
                {"error": {"message": "读取响应失败", "type": "api_error"}},
                status_code=502)
        b["in_flight"] -= 1
        resp = _openai_chat_to_responses_output(oa_resp)
        return JSONResponse(resp)

    # --- 流式：转发为 Responses SSE ---
    gen = _openai_chat_stream_to_responses(r.aiter_raw(), name)
    return StreamingResponse(gen, media_type="text/event-stream")


@app.get("/v1/responses/{response_id}")
async def anthropic_get_response(response_id: str):
    """获取 Responses API 的响应详情（仅返回元数据，无状态）。"""
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": "unknown",
    }


@app.post("/v1/responses/prefetch")
async def anthropic_prefetch():
    """Responses prefetch 接口（无需持久化，直接返回 200）。"""
    return {"id": "none", "status": "skipped"}


@app.api_route("/v1/{path:path}", methods=["POST"])
async def route(path: str, request: Request):
    try:
        body = await request.json()
        name = body.get("model", "")
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON",
                                       "type": "invalid_request_error"}}, status_code=400)
    return await proxy(name, request)


async def _preload(m):
    try:
        log(f"预加载 {m} …")
        await ensure_loaded(m, load_manifest())
    except Exception as e:
        log(f"预加载 {m} 失败: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
