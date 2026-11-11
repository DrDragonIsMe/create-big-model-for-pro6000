#!/usr/bin/env python3
"""AI 模型网关（常驻能力，非临时补救）

暴露三种兼容 API（端口 8000）：
• OpenAI   — /v1/models（含 capabilities/context_length）、/v1/chat/completions、/v1/responses
• Anthropic — /v1/messages、/v1/messages/count_tokens、/v1/messages/{model}，
             以及 Anthropic 风格 /v1/responses（anthropic-version 或 x-api-format 选择）
• 管理     — /v1/gateway/status（状态）、/v1/gateway/unload_all（腾显存）、
             /v1/gateway/switch（远程切换文本/视频，等价 start.sh --switch）
• 协商     — /v1/gateway/capabilities（说明接口，供客户端 agent 首次连接时学习）、
             /v1/gateway/gpus（GPU 发现：数量/容量/占用/利用率）、
             /v1/gateway/modes（布局模式目录：每个模型在本机的现行摆放与可切布局，2026-10-26 新增）、
             /v1/gateway/load（把模型加载到指定 GPU / 按布局模式切换，wait=false 异步返回 request_id）、
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
8. 同一模型可在每张卡上各起一个副本（models.json 的 gw.replicas），请求按副本在途数
   分流：单卡装得下的模型用多卡堆并发，而不是用 TP 把一层计算切薄

运行环境：ai-gateway 容器，挂载 docker.sock 管理后端容器，--gpus all 提供 nvidia-smi。
后端容器命名 ai-vllm-<model> / ai-llamacpp-<model>（多副本时带 -r0/-r1… 后缀），
仅加入 ai_default 内网，不暴露宿主机端口。
所有 docker/nvidia-smi 同步调用均经 asyncio.to_thread 包裹，不阻塞事件循环。
"""
import asyncio, calendar, fnmatch, json, os, re, subprocess, time, urllib.request
from contextlib import asynccontextmanager

import docker
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway_protocols import (
    ProtocolError, anthropic_to_chat, chat_to_anthropic, chat_to_response,
    responses_to_chat, stream_anthropic, stream_responses,
)

MODELS_JSON = os.environ.get("MODELS_JSON", "/opt/models.json")
HOST_ROOT = os.environ.get("HOST_ROOT", "/home/pc/AI")
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest")
LLAMACPP_IMAGE = os.environ.get("LLAMACPP_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
NETWORK = os.environ.get("GW_NETWORK", "ai_default")
PRELOAD = [m for m in os.environ.get("PRELOAD", "").split(",") if m]
NEED_MARGIN_MB = 512  # vLLM 要求 空闲 >= util×total，再加安全余量，避免放行后启动崩溃
COMFY_HOST = os.environ.get("COMFY_HOST", "host.docker.internal")  # 网关访问 ComfyUI 的宿主地址
STATE_FILE = os.environ.get("GATEWAY_STATE_FILE", "/data/gateway/state.json")

log = lambda *a: print("[gateway]", *a, flush=True)

def remember_loaded(name):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_loaded_llm": name, "loaded_at": time.time()}, f)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        log(f"记录最近加载模型失败（不影响推理）: {e}")

dcli = docker.DockerClient(base_url="unix:///var/run/docker.sock")
http = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))

ops_lock = asyncio.Lock()      # 加载/卸载串行化
backends = {}                  # name -> {container, gpus, util, in_flight, last_used, status,
                               #          replicas, peers:[{container, gpus, in_flight}]}
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


def model_dir(name, d=None):
    """按模型条目的 gw.variant 选权重目录（目录名相对 $HOST_ROOT/models/llm）。

    不配 variant 时保持历史行为：官方 FP8 目录优先，回落主目录。
    variant 显式指定时严格取该精度——精度是配置意图，不能静默换成另一种，
    目录缺失就报未下载，由宿主机 menu.py / manage.sh 去补。"""
    variant = ((d or {}).get("gw") or {}).get("variant")
    if d is None:
        # 4 处校验型调用点（/v1/models、capabilities、load、proxy）没带条目：
        # 自己按同一份清单取，避免它们用默认口径判"已下载"、而 ensure_loaded 按
        # variant 口径判"未下载"，模型在列表里却调不通。
        variant = (load_manifest().get(name) or {}).get("gw", {}).get("variant")
    if variant == "bf16":
        cands = (name,)
    elif variant:
        cands = (f"{name}-{variant}",)
    else:
        cands = (f"{name}-fp8", name)
    for cand in cands:
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
# slot 日志行的前缀 'slot print_timing: id  0 | task 12 | ...'，--parallel N 时用它分槽
_SLOT_TAG_RE = re.compile(r"\bid\s*(-?\d+)\s*\|\s*task\s*(-?\d+)")
# vLLM 引擎自己的窗口统计（loggers.py，默认每 10s 一条）：
#   Engine 000: Avg prompt throughput: 781.8 tokens/s, Avg generation throughput: 83.8 tokens/s,
#               Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 43.5%, ...
_VLLM_STATS_RE = re.compile(r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s")
_VLLM_ENGINE_RE = re.compile(r"Engine\s+(\d+)")
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?"
                        r"(Z|[+-]\d{2}:?\d{2})?\s+(.*)$")

PREFILL_LOG_WINDOW = 40      # 读后端容器日志尾部的秒数
LLAMA_PREFILL_FRESH = 4.0    # 距最后一条 prefill 日志多久内算"该 slot 仍在预填充"
VLLM_STATS_WINDOW = 10.0     # vLLM 统计行的打印周期（速率是该窗口的平均值）
VLLM_PREFILL_FRESH = VLLM_STATS_WINDOW + 3.0
TOKEN_SAMPLE_INTERVAL = 2.0  # 后台吞吐采样周期
TOKEN_SAMPLE_MIN_DT = 1.0    # 同一后端两次采样的最小间隔（轮询再快也只是读缓存）
PREFILL_MIN_RATE = 20.0      # 低于此 p/s 不算"正在预填充"：step 差分在纯 decode 期只有一两个 token 的错位噪声
PREFILL_HOLD_SEC = 4.0       # 突发 prefill 的最近一次实测速率保留这么久，免得监控端只看到一闪
PREFILL_HOLD_MIN_TOKENS = 2048   # 只有这么大规模的突发 prefill 才保持显示


def _log_epoch(m):
    """docker 日志时间戳（RFC3339，可能带小数秒与时区）-> epoch 秒。"""
    sec = float(calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")))
    if m.group(2):
        sec += float("0." + m.group(2)[:6])
    off = m.group(3)
    if off and off != "Z":
        digits = off[1:].replace(":", "")
        delta = int(digits[:2]) * 3600 + int(digits[2:4] or 0) * 60
        sec += -delta if off[0] == "+" else delta
    return sec


def _container_log(backend, window):
    """后端容器最近 window 秒的日志，返回 [(epoch|None, 文本)]。

    取不到（网关没挂 docker.sock、容器消失等）抛异常，由调用方退回 /metrics 差分兜底。
    """
    c = dcli.containers.get(backend["container"])
    raw = c.logs(tail=800, since=int(time.time()) - window - 1, timestamps=True)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    out = []
    for line in str(raw).splitlines():
        m = _LOG_TS_RE.match(line)
        out.append((_log_epoch(m) if m else None, m.group(4) if m else line))
    return out


def _llamacpp_prefill(backend, now):
    """llama.cpp 的实时预填充进度与吞吐（来自 slot 日志，/metrics 要等请求结束才动）。

    这行日志只在某个 slot 预填充期间每秒打印，所以"最近打印过"就是活跃判据；
    n_tokens 是每个请求各自从 0 重新计数的，拿它跨请求/跨 slot 比增长只会失真。
    --parallel N 并发预填充时按 slot 各取最新一行，速率与已处理 token 相加。
    absolute=True：日志给的就是该请求的绝对进度，监控端可以直接画进度条。
    """
    try:
        lines = _container_log(backend, PREFILL_LOG_WINDOW)
    except Exception:
        return None
    slots = {}
    for ts, text in lines:
        m = _PREFILL_RE.search(text)
        if not m:
            continue
        tag = _SLOT_TAG_RE.search(text)
        slots[tag.group(1) if tag else "0"] = {
            "tokens": int(m.group(1)), "progress": float(m.group(2)),
            "elapsed": float(m.group(3)), "rate": float(m.group(4)),
            "seen": ts if ts is not None else now}   # 无时间戳时保守当作刚发生
    last_seen = max((s["seen"] for s in slots.values()), default=0.0)
    live = [s for s in slots.values() if now - s["seen"] <= LLAMA_PREFILL_FRESH]
    if not live:
        return {"available": True, "absolute": True, "rate": 0.0, "tokens": None,
                "progress": None, "elapsed": None, "seen_at": last_seen}
    done = float(sum(s["tokens"] for s in live))
    # progress = 该 slot 已处理 / 该请求 prompt 总长，反推总长才能把多槽合并成一个总进度
    total = sum(s["tokens"] / s["progress"] for s in live if s["progress"] > 0)
    return {"available": True, "absolute": True,
            "rate": sum(s["rate"] for s in live), "tokens": done,
            "progress": done / total if total > 0 else None,
            "elapsed": max(s["elapsed"] for s in live), "seen_at": now}


def _vllm_prefill(backend, now):
    """vLLM 预填充吞吐的兜底来源：引擎统计行（step 计数缺失的 vLLM 才用）。

    首选是 /metrics 的 iteration_tokens_total 差分——实时、且只统计真正算过的 token。
    统计行的 Avg prompt throughput 也能用，但它是 10s 窗口均值（滞后最多一个窗口）、
    没有单请求进度。vllm:prompt_tokens_total 则根本不能拿来算速率：它按请求一次性入账，
    还把 prefix cache 命中算进去（本机累计 2.69e9 里 93% 是命中），虚高几个数量级。
    """
    try:
        lines = _container_log(backend, PREFILL_LOG_WINDOW)
    except Exception:
        return None
    engines = {}
    for ts, text in lines:
        m = _VLLM_STATS_RE.search(text)
        if not m:
            continue
        eid = _VLLM_ENGINE_RE.search(text)
        engines[eid.group(1) if eid else "0"] = {
            "rate": float(m.group(1)), "seen": ts if ts is not None else now}
    seen_at = max((e["seen"] for e in engines.values()), default=0.0)
    live = [e for e in engines.values()
            if now - e["seen"] <= VLLM_PREFILL_FRESH and e["rate"] > 0]
    return {"available": True, "absolute": False,
            "rate": sum(e["rate"] for e in live), "tokens": None,
            "progress": None, "elapsed": None, "seen_at": seen_at}



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


def refresh_live_counters(name):
    """查询时就地重算在飞请求的速率与累计。

    SSE 增量是逐 chunk 连续进账的，没必要等后台 2s 采样节拍；否则 1s 刷新的监控端
    有一半的帧读到旧值，生成中的 t/s 会跳成 0、Σgen 也会一格一格地跳。
    """
    st = token_metrics_cache.get(name)
    if not st or not st.get("live_start"):
        return None
    now = time.time()
    elapsed = max(now - st["live_start"], 1e-6)
    st["generation_rate"] = max(0.0, st.get("live_seen", 0.0) / elapsed)
    st["generation_total"] = max(st.get("generation_total") or 0.0,
                                 st.get("live_base_total", 0.0) + st.get("live_seen", 0.0))
    return st


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
    step = None
    if backend.get("engine") == "vllm":
        # 每个 step 实际计算的 token 数（含 prefill 与 decode）：prefix cache 命中不产生
        # step，所以这个计数只统计真算过的 token，是 vLLM 侧唯一实时的 prefill 信号
        step = _metric_total(text, ("vllm:iteration_tokens_total_sum",))
    # prompt 速率的另一条来源是引擎日志：llama.cpp 的 slot 行带单请求进度；
    # vLLM 的统计行是 10s 窗口均值、且没有单请求进度，只在 step 计数缺失时兜底
    if backend.get("engine") == "llamacpp":
        prefill = _llamacpp_prefill(backend, time.time())
    elif backend.get("engine") == "vllm" and step is None:
        prefill = _vllm_prefill(backend, time.time())
    else:
        prefill = None
    return {"prompt": prompt, "generated": generated, "step": step, "prefill": prefill}


def _apply_backend_metrics(name, backend, metrics=None, error=False):
    now = time.time()
    item = _token_state(name)
    if error:
        # 后端不可达：阶段与速率一并清零，免得监控端停在最后一次的 Prompt 画面
        item.update({"time": now, "metrics_ok": False, "prefill_active": False,
                     "prefill_rate": 0.0, "prompt_rate": 0.0, "prefill_tokens": None,
                     "prefill_progress": None, "prefill_elapsed": None})
        if not item.get("live_start"):
            item["generation_rate"] = 0.0
        return item
    prompt, generated, step, prefill = (metrics.get(k) for k in ("prompt", "generated",
                                                                 "step", "prefill"))
    if prompt is not None:
        item["prompt_total"] = prompt
    item["metrics_ok"] = True
    # ── counter 差分 ──
    # iteration 计数 = 每个 step 真正计算的 token（prefill + decode），减掉同期 generation
    # 增量就是纯预填充吞吐。prompt_tokens_total 不能用来算 p/s：它按请求一次性入账，
    # 还把 prefix cache 命中算进去（本机累计 2.69e9 里 93% 是命中），
    # 既看不见进行中的预填充，请求结束瞬间又虚高几个数量级。
    prev_at, prev_step = item.get("counter_time"), item.get("counter_step")
    prev_gen = item.get("counter_total")
    dt = (now - prev_at) if prev_at and now > prev_at else 0.0
    item["counter_time"], item["counter_step"], item["counter_total"] = now, step, generated
    computed = 0.0
    if (step is not None and prev_step is not None and dt > 0
            and prev_gen is not None and step >= prev_step and generated >= prev_gen):
        computed = max(0.0, (step - prev_step) - (generated - prev_gen))

    # ── p/s 取源：llama.cpp 走 slot 日志（带单请求进度）；vLLM 走 step 差分，
    #    引擎统计行只在缺 step 计数的 vLLM 上兜底（10s 窗口均值，且没有单请求进度）──
    src = None
    if prefill is not None and prefill.get("available"):
        rate = float(prefill.get("rate") or 0.0)
        src = {"rate": rate, "tokens": prefill.get("tokens"),
               "progress": prefill.get("progress"), "elapsed": prefill.get("elapsed"),
               "delta": None if prefill.get("absolute") else rate * dt, "gated": True}
    elif step is not None and prev_step is not None and dt > 0:
        src = {"rate": computed / dt, "tokens": None, "progress": None,
               "elapsed": None, "delta": computed, "gated": False}
    # gated：日志类来源要多一道"网关侧确有在飞请求"的佐证——vLLM 统计行是 10s 窗口
    # 均值、slot 日志最长可能晚了 LLAMA_PREFILL_FRESH 秒，光看值会把已结束的阶段画成在进行。
    # step 差分不受此限：它统计的就是上一个采样间隔，整段 prefill 一步算完也能抓到。
    active = bool(src and src["rate"] >= PREFILL_MIN_RATE
                  and (not src["gated"] or backend.get("in_flight", 0) > 0))
    held = False
    if (not active and src is not None and not src["gated"] and item.get("prefill_active")
            and (item.get("prefill_tokens") or 0) >= PREFILL_HOLD_MIN_TOKENS
            and now - item.get("prefill_measured_at", 0) < PREFILL_HOLD_SEC):
        # 整段 prefill 一个 step 就完成时，非零值只存在一次采样：把最近一次实测值
        # 保留 PREFILL_HOLD_SEC 秒，否则 p/s 一闪而过，监控端看着还是"没值"。
        # 只有足够大的 prefill 才保持——几十 token 的小预填充没必要停留显示
        active, held = True, True
        src = dict(src, rate=item.get("prefill_rate", 0.0), delta=None,
                   tokens=item.get("prefill_tokens"))
    if active and not held:
        item["prefill_measured_at"] = now
    if active and not item.get("prefill_active"):
        item["prefill_streak_at"], item["prefill_streak_tokens"] = now, 0.0
    if active:
        if src["delta"]:      # 窗口均值/step 差分：积分出这轮已处理多少 token
            item["prefill_streak_tokens"] = (item.get("prefill_streak_tokens") or 0.0) + src["delta"]
        tokens = src["tokens"]
        if tokens is None:
            tokens = item.get("prefill_streak_tokens")
        elapsed = src["elapsed"]
        if elapsed is None:
            elapsed = now - item.get("prefill_streak_at", now)
    else:
        tokens = elapsed = None
    item["prefill_active"] = active
    item["prefill_rate"] = src["rate"] if active else 0.0
    item["prompt_rate"] = src["rate"] if active else 0.0
    item["prefill_tokens"] = tokens
    item["prefill_progress"] = src["progress"] if active else None
    item["prefill_elapsed"] = elapsed
    item["prefill_seen_at"] = now
    if backend.get("engine") == "llamacpp":
        prev_total = item.get("generation_total") or 0.0
        # 容器重建后 llama.cpp 的 counter 从 0 重新累计：新值若远小于旧值
        # （正常回落最多小几个 token，不是砍半）说明后端重启过，重置基线，
        # 否则 max() 会把 generation_total 永久冻结在旧容器的累计值上，
        # 监控端差分 t/s 恒为 0
        if generated < prev_total * 0.5:
            prev_total = 0.0
        total = max(float(generated), prev_total)
        if item.get("live_start"):
            item["live_base_total"] = max(item["live_base_total"], 0.0,
                                          float(generated))
            total = max(total, item["live_base_total"] + item.get("live_seen", 0.0))
        item["generation_total"] = total
        if not item.get("live_start"):
            item["generation_rate"] = 0.0
    else:
        # vLLM continuous batching：generation counter 每步都涨，差分即实时 t/s
        item["generation_total"] = generated
        if prev_gen is not None and generated >= prev_gen and dt > 0:
            item["generation_rate"] = (generated - prev_gen) / dt
        else:
            item["generation_rate"] = 0.0
    item["time"] = now
    return item


async def sample_all_backend_tokens(force=False):
    """采样各后端的吞吐与阶段。

    速率是差分/窗口算出来的，两次采样挨太近只剩噪声，所以默认跳过
    TOKEN_SAMPLE_MIN_DT 内刚采过的后端、直接回缓存：/v1/gateway/status 可以被
    多个监控端高频轮询，后端的采样节奏不受影响。
    """
    now = time.time()
    alive = set(backends)
    for dead in set(token_metrics_cache) - alive:
        token_metrics_cache.pop(dead, None)
    live = [n for n, b in backends.items() if b.get("status") in ("ready", "adopted")]
    names = [n for n in live
             if force or now - (token_metrics_cache.get(n) or {}).get("time", 0) >= TOKEN_SAMPLE_MIN_DT]

    async def one(name):
        b = backends.get(name)
        if not b:
            return name, None      # 卸载/mode 切换随时可能摘掉登记（采样集是快照）
        try:
            metrics = await asyncio.to_thread(_fetch_backend_metrics, b)
            return name, _apply_backend_metrics(name, b, metrics=metrics)
        except Exception:
            return name, _apply_backend_metrics(name, b, error=True)

    if names:
        await asyncio.gather(*(one(n) for n in names))
    return {n: token_metrics_cache[n] for n in live if n in token_metrics_cache}


async def _token_metrics_loop():
    """后台定时采样：以前只有 status 被请求时才采一次，dt 等于客户端轮询间隔，
    没人看监控就不推进，速率忽有忽无。改成定时采、监控端只读缓存。"""
    while True:
        try:
            await sample_all_backend_tokens(force=True)
        except Exception as e:
            log(f"吞吐采样失败: {e}")
        await asyncio.sleep(TOKEN_SAMPLE_INTERVAL)


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


def _physical_cores():
    """宿主物理核数：/proc/cpuinfo 按 (physical id, core id) 去重，并尊重 cpuset 亲和性。
    容器内 /proc 反映的是宿主；SMT 兄弟核对 ggml 常为负收益，按物理核取。"""
    try:
        pairs, cur = {}, None
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("processor"):
                    cur = int(line.split(":", 1)[1])
                    pairs[cur] = [None, None]
                elif line.startswith("physical id") and cur is not None:
                    pairs[cur][0] = line.split(":", 1)[1].strip()
                elif line.startswith("core id") and cur is not None:
                    pairs[cur][1] = line.split(":", 1)[1].strip()
        try:
            allowed = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            allowed = set(pairs)
        cores = {tuple(v) for k, v in pairs.items() if k in allowed and None not in v}
        if cores:
            return len(cores)
    except (OSError, ValueError):
        pass
    return max(1, (os.cpu_count() or 8) // 2)


def _host_cpu_flags():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return None


def _host_gpu_arch_families():
    """宿主 GPU 的 CUDA 架构族集合（如 {'120'}，由 compute_cap '12.0' 去点号得来）；
    拿不到返回 None（不做判断）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout
        fams = {ln.strip().replace(".", "") for ln in out.splitlines()
                if ln.strip() and ln.strip()[0].isdigit()}
        return fams or None
    except Exception:
        return None


def _cuda_arch_hint(cap):
    """compute_cap 去点号（'120'）→ CMAKE_CUDA_ARCHITECTURES 建议值。"""
    return f"{cap}a-real" if cap.startswith(("90", "100", "120")) else cap


_BUILD_INFO_CACHE: dict = {}


def _llamacpp_build_info(image):
    """读本地编译镜像里 /etc/ai-llamacpp-build.env 的构建元数据（带缓存）。
    旧镜像/预构建镜像没有该文件 → 空 dict（调用方视为不拦）。
    用 docker SDK 起一次性容器读文件（网关容器里没有 docker CLI）。"""
    if image in _BUILD_INFO_CACHE:
        return _BUILD_INFO_CACHE[image]
    info = {}
    try:
        out = dcli.containers.run(
            image, remove=True, network_mode="none",
            entrypoint=["sh", "-c", "cat /etc/ai-llamacpp-build.env 2>/dev/null || true"])
        for ln in out.decode("utf-8", "ignore").splitlines():
            k, _, v = ln.partition("=")
            if k.startswith("AI_LLAMACPP_"):
                info[k] = v.strip()
    except Exception:
        pass
    _BUILD_INFO_CACHE[image] = info
    return info


def check_llamacpp_image(image):
    """本地编译的 llamacpp 镜像与当前机器的适配检查 → (ok, reason, rebuild_hint)。
    只在能确定不匹配时拦截；缺元数据/探测不到宿主信息一律放行（宁漏勿错）。
    两类不匹配：GPU 架构族不同（换卡必重编）；编译机 CPU 指令集超出本机
    （换 CPU 后典型症状是 invalid opcode 退出 132）。"""
    info = _llamacpp_build_info(image)
    if not info:
        return True, "", ""
    arch = info.get("AI_LLAMACPP_CUDA_ARCH", "")
    fams = _host_gpu_arch_families()
    if arch and fams:
        build_fams = {m.group() for a in re.split(r"[;\s,]+", arch)
                      for m in [re.match(r"\d+", a)] if m}
        if build_fams and not (build_fams & fams):
            hint_arch = ";".join(sorted({_cuda_arch_hint(f) for f in fams}))
            return False, (f"镜像 CUDA 架构 {arch} 与本机 GPU compute_cap 族"
                           f"（{','.join(sorted(fams))}）不匹配"), \
                (f"docker build --no-cache --build-arg "
                 f"CMAKE_CUDA_ARCHITECTURES={hint_arch} -t {image} ./docker/llamacpp")
    bflags = set(info.get("AI_LLAMACPP_CPU_FLAGS", "").split())
    hflags = _host_cpu_flags()
    if bflags and hflags:
        missing = sorted(bflags - hflags)
        if missing:
            return False, (f"镜像编译机指令集超出本机（缺 {', '.join(missing[:8])}"
                           f"{'…' if len(missing) > 8 else ''}），运行可能 invalid opcode"), \
                f"docker build --no-cache -t {image} ./docker/llamacpp"
    return True, "", ""


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

    # 多卡 llama.cpp：按层跨卡拆分（84GB 权重摊到 2×48GB 这类"整池放得下、
    # 单卡放不下"的场景）。比例默认按各卡总显存推导；gw.tensor_split 可显式
    # 覆盖（menu 的共存模式会写 "0.948,0.052" 这类权重），"none" 表示禁用拆分。
    split_mode = gw.get("split_mode")
    tensor_split = gw.get("tensor_split")
    if len(gpus) > 1:
        if split_mode in (None, ""):
            split_mode = "layer"
        if split_mode != "none" and tensor_split in (None, "", "auto"):
            per = [max(1, round(free[g]["total"] / 1024.0)) for g in gpus if g in free]
            tensor_split = ",".join(str(x) for x in per) if per else None
    else:
        split_mode = tensor_split = None

    return {
        "parallel": n,
        "ctx_size": int(max_len),
        "per_slot_ctx": int(max_len) // n,
        "kv_budget_gb": round(kv_budget_gb, 2),
        "weights_gb": round(float(weights_gb), 2),
        "total_gb": round(total_gb, 2),
        "cache_type": cache_type,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "tight": kv_budget_gb < slot_kv_gb,
    }

def _replica_groups(gpus, replicas):
    """把选中的卡切成 replicas 组，每组一个后端容器（组内卡数即 tensor-parallel-size）。
    replicas<=1 时整组交给一个容器（保持现有 TP 语义）；卡不够就自动降到每卡一组。"""
    n = max(1, int(replicas or 1))
    if n == 1 or len(gpus) <= 1:
        return [list(gpus)]
    n = min(n, len(gpus))          # 副本数不超过可用卡数
    return [[g] for g in gpus[:n]]


def _pick_peer(b):
    """多副本路由：取在途请求最少的副本。没有 peers（单副本/老数据）时退回自身。"""
    peers = b.get("peers") or []
    if not peers:
        return b
    return min(peers, key=lambda p: p.get("in_flight", 0))


def _remove_backend_containers(b):
    """删除该后端的全部副本容器；单副本时等价于 _remove_container(b["container"])"""
    peers = b.get("peers") or [b]
    for p in peers:
        c = p.get("container")
        if c:
            _remove_container(c)


def _run_backend(name, dirname, gpus, util, max_len, think, extra_args=None, env=None,
                 engine="vllm", gguf_include=None, plan=None, cname=None, image=None,
                 max_num_seqs=None):
    kwargs = {}
    if len(gpus) > 1:
        kwargs["ipc_mode"] = "host"
    if engine == "llamacpp":
        # llama.cpp 不做多副本：单容器吃整组卡，cname 忽略
        _run_llamacpp(name, dirname, gpus, max_len, think, gguf_include, extra_args, kwargs,
                      plan=plan, env=env, image=image)
        return
    tp = len(gpus)
    cmd = [f"/models/{dirname}", "--served-model-name", name,
           "--tensor-parallel-size", str(tp),
           "--gpu-memory-utilization", str(util),
           "--max-model-len", str(max_len),
           "--max-num-seqs", str(max_num_seqs or 128),
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]
    # vLLM 默认不报 usage.prompt_tokens_details.cached_tokens，Anthropic 协议的
    # cache_read_input_tokens 就没数据可回填（Claude Code 状态栏恒 0/256k）。
    # extra_args 里已手传同名 flag（含 --no- 形式）时不重复注入。
    _ea = [str(a) for a in (extra_args or [])]
    if not any("enable-prompt-tokens-details" in a for a in _ea):
        cmd += ["--enable-prompt-tokens-details"]
    if think:
        # 仅思考模型可加 reasoning parser；非思考模型强开会把正文吞进 reasoning_content
        cmd += ["--reasoning-parser", "qwen3"]
    if extra_args:
        # 如 YaRN 扩上下文的 --hf-overrides（models.json 里按模型配置）
        cmd += extra_args
    log(f"加载 {name}: dir={dirname} gpus={gpus} tp={tp} util={util} max_len={max_len} think={think}"
        + (f" extra_args={extra_args}" if extra_args else ""))
    _remove_container(cname or f"ai-vllm-{name}")
    # 编译/JIT 缓存持久化到宿主机：容器每次重建，不挂载则 torch.compile 每次重跑（~30s）
    cache_host = f"{HOST_ROOT}/data/vllm-cache"
    os.makedirs(cache_host, exist_ok=True)
    environment = {"TRITON_CACHE_DIR": "/root/.cache/triton"}
    if env:
        # 如 YaRN 超长所需的 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1（models.json 里按模型配置）
        environment.update(env)
    dcli.containers.run(
        VLLM_IMAGE, command=cmd, name=cname or f"ai-vllm-{name}", detach=True,
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
                  kwargs=None, plan=None, env=None, image=None):
    """启动 llama.cpp 后端（llama-server，OpenAI 兼容）。GGUF 多分片只传第一个文件，
    llama.cpp 按命名约定自动续载其余分片。flag 以 2026 版 llama-server --help 实测为准：
    -ngl 默认 auto（按显存自动分层）；工具调用由默认启用的 Jinja 聊天解析自动提取；
    思考模型用 --reasoning-format deepseek 把思考放进 message.reasoning_content；
    YaRN 扩上下文（如 --rope-scaling yarn --rope-scale 4.0）经 extra_args 透传。
    --parallel / KV 量化 / 多卡 --split-mode+--tensor-split 由 plan（plan_llamacpp
    按硬件自动规划）决定：一组卡且权重单卡放不下时按层跨卡摊分。"""
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
    # 多卡分层：由 plan 拼 --split-mode/--tensor-split（单卡时 plan 里就是 None）。
    # extra_args 里已手传同名 flag 时以手传为准，不重复注入。
    _ea = [str(a) for a in (extra_args or [])]
    if not any(a in ("-t", "--threads") for a in _ea):
        # 未显式指定 -t 时按宿主物理核注入：写死值随机器而异（曾在 models.json 写 -t 16
        # 拖慢 offload 模型近半），运行时自适应；models.json 显式给了 -t 则以显式值为准
        _nthread = _physical_cores()
        cmd += ["-t", str(_nthread), "-tb", str(_nthread)]
    if plan.get("split_mode") and not any(a.startswith("--split-mode") for a in _ea):
        cmd += ["--split-mode", str(plan["split_mode"])]
        if plan.get("tensor_split") and not any(a.startswith("--tensor-split") for a in _ea):
            cmd += ["--tensor-split", str(plan["tensor_split"])]
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
        + (f" 分层={plan['split_mode']}:{plan['tensor_split']}" if plan.get("split_mode") else "")
        + (f" extra_args={extra_args}" if extra_args else ""))
    if plan.get("tight") and plan["parallel"] <= 1 and not plan.get("cache_type"):
        log(f"{name}: KV 预算紧张仅够单 slot；如需更高并发，可在 models.json 的 "
            f"gw 设 kv_quant=q8_0（KV 显存减半，轻微精度影响）或加卡")
    image = image or LLAMACPP_IMAGE
    ok, why, hint = check_llamacpp_image(image)
    if not ok:
        raise RuntimeError(f"llama.cpp 镜像与当前机器不匹配：{why}。请重编后重试：{hint}")
    _remove_container(f"ai-llamacpp-{name}")
    # 镜像自带的 HEALTHCHECK 探测 8080，而我们跑 8000——覆盖为 8000，避免误报 unhealthy
    hc = docker.types.Healthcheck(
        test=["CMD", "curl", "-f", "http://localhost:8000/health"],
        interval=30000000000, timeout=5000000000, retries=3, start_period=30000000000)
    dcli.containers.run(
        image or LLAMACPP_IMAGE, command=cmd, name=f"ai-llamacpp-{name}", detach=True,
        device_requests=[docker.types.DeviceRequest(
            device_ids=[str(g) for g in gpus], capabilities=[["gpu"]])],
        network=NETWORK, shm_size="16g",
        environment=dict(env or {}),
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
            _remove_backend_containers(victim)
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
                remember_loaded(name)
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


async def ensure_loaded(name, man, want_gpus=None, op=None, want_replicas=None):
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
                await asyncio.to_thread(_remove_backend_containers, b)
                backends.pop(name, None)
        d = man[name]
        engine = d.get("engine", "vllm")
        gw = d.get("gw", {})
        gpus = [int(x) for x in gw.get("gpus", "1").split(",")]
        util = float(gw.get("util", 0.92))
        max_len = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
        think = bool(d.get("think"))
        cname = f"ai-{engine}-{name}"
        dirname = model_dir(name, d)
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
            elif engine == "llamacpp" and free:
                # llama.cpp 按层拆分与"哪几张卡"无关：配置的卡有缺失时改用本机全部卡，
                # 显存够不够由下面放置预算把关（vLLM 的 TP 语义不同，保持严格报错）
                fb = sorted(free)
                log(f"{name}: 配置的 GPU{gpus} 与本机 {fb} 不符，"
                    f"llama.cpp 跨卡分层改用全部卡 {fb}")
                gpus = fb
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

        # 多副本：放置定了才切副本组——每组一个容器，请求按在途数分流。
        # replicas 缺省/为 1 时 groups 就是 [gpus]，容器名无后缀，与旧行为完全一致。
        # llama.cpp 不支持多副本（单容器吃整组卡，起两个会同名互删），整组交给一个容器。
        # want_replicas：客户端 mode/replicas 的运行时覆盖；缺省沿用 gw 落盘值。
        # 网关对 models.json 是只读挂载，覆盖只影响本次拉起，持久化归宿主 menu.py。
        replicas = int(want_replicas if want_replicas else gw.get("replicas", 1))
        groups = _replica_groups(gpus, replicas) if engine != "llamacpp" else [list(gpus)]
        cnodes = ([cname] if len(groups) == 1
                  else [f"{cname}-r{i}" for i in range(len(groups))])
        if op:
            _op_update(op, status="running", stage="loading",
                       detail=f"放置 GPU {sorted(gpus)}，启动后端容器"
                              + (f" ×{len(groups)}（每副本 GPU {groups}）" if len(groups) > 1 else "")
                              + "…",
                       target_gpus=gpus)

        # llama.cpp：按当前硬件（GPU 数/显存、权重体积）自动规划 --parallel 等
        plan = None
        if engine == "llamacpp":
            weights_gb = await asyncio.to_thread(_gguf_weights_gb, dirname,
                                                 d.get("gguf_include"))
            plan = await asyncio.to_thread(plan_llamacpp, gpus, weights_gb, max_len, gw)

        launched = []
        try:
            for grp, node in zip(groups, cnodes):
                await asyncio.to_thread(_run_backend, name, dirname, grp, util, max_len,
                                        think, gw.get("extra_args"), gw.get("env"),
                                        engine, d.get("gguf_include"), plan, node,
                                        image=gw.get("image"),
                                        max_num_seqs=gw.get("max_num_seqs"))
                launched.append(node)
        except Exception:
            # 某副本连容器都没拉起来：删掉已拉起的副本，别留登记之外占着显存的孤儿容器
            for node in launched:
                await asyncio.to_thread(_remove_container, node)
            raise
        backends[name] = {"container": cnodes[0], "gpus": gpus, "util": util,
                          "in_flight": 0, "last_used": time.time(), "status": "loading",
                          "engine": engine,
                          "slots": plan["parallel"] if plan else None,
                          "replicas": len(groups),
                          "peers": [{"container": c, "gpus": g, "in_flight": 0}
                                    for c, g in zip(cnodes, groups)]}
        # 所有副本都拉起后再并发等就绪：ops_lock 全程持有，串行等会把冷启动墙钟翻倍
        res = await asyncio.gather(*(wait_ready(name, node, timeout=900) for node in cnodes),
                                   return_exceptions=True)
        ready, ready_groups, dead = [], [], []
        for node, grp, r in zip(cnodes, groups, res):
            if isinstance(r, BaseException):
                dead.append((node, r))
            else:
                ready.append(node)
                ready_groups.append(grp)
        if not ready:
            # 全起不来：与单副本旧行为一致——摘掉登记并抛出，容器留着让人 docker logs 排查
            backends.pop(name, None)
            raise dead[0][1]
        for node, r in dead:
            # 部分副本失败不拖垮其它副本：删掉它释放显存，剩下的照常服务
            await asyncio.to_thread(_remove_container, node)
            log(f"{name} 副本 {node} 就绪失败，已删除（不影响其它 {len(ready)} 个副本）: {r}")
        backends[name] = {"container": ready[0], "gpus": gpus, "util": util,
                          "in_flight": 0, "last_used": time.time(), "status": "loading",
                          "engine": engine, "slots": plan["parallel"] if plan else None,
                          "replicas": len(ready),
                          "peers": [{"container": c, "gpus": g, "in_flight": 0}
                                    for c, g in zip(ready, ready_groups)]}
        backends[name]["status"] = "ready"
        backends[name]["last_used"] = time.time()
        log(f"{name} 就绪"
            + (f"：{len(ready)} 副本 {list(zip(ready, ready_groups))}"
               if len(groups) > 1 else ""))
        if op:
            _op_update(op, status="running", stage="ready",
                       detail=f"{name} 后端就绪（GPU {sorted(gpus)}）"
                              + (f"，{len(ready)} 副本每副本 GPU {ready_groups}"
                                 if len(groups) > 1 else ""),
                       target_gpus=gpus)
        return backends[name]


# ---------- HTTP ----------

@asynccontextmanager
async def lifespan(app):
    adopt_existing()
    sampler = asyncio.create_task(_token_metrics_loop())
    for m in PRELOAD:
        asyncio.create_task(_preload(m))
    yield
    sampler.cancel()

app = FastAPI(lifespan=lifespan)


def adopt_existing():
    """网关重启后认领还活着的后端容器（标 adopted，可被卸载；首次请求时先等健康检查）。
    同一模型的多个副本（<model>-r0/-r1…）合并成一个后端条目 + peers：
    按原名认领会把 "xxx-r0" 当成一个模型注册出来，且后面的副本会覆盖前面的。"""
    try:
        man = load_manifest()
    except Exception as e:
        log(f"认领后端：读模型清单失败（忽略，按容器原名认领）: {e}")
        man = {}
    found = {}   # name -> [(副本序号, engine, gpus, container, slots)]；无后缀记 -1 排最前
    for c in dcli.containers.list(all=True, filters={"status": ["running"]}):
        engine = None
        if c.name.startswith("ai-vllm-"):
            engine, raw = "vllm", c.name[len("ai-vllm-"):]
        elif c.name.startswith("ai-llamacpp-"):
            engine, raw = "llamacpp", c.name[len("ai-llamacpp-"):]
        else:
            continue
        try:
            # 副本后缀只对"原名不在清单里"的容器生效——模型名本身带 -rN 时以清单为准
            name, seq = raw, -1
            if raw not in man:
                m = re.match(r"^(.+)-r(\d+)$", raw)
                if m and m.group(1) in man:
                    name, seq = m.group(1), int(m.group(2))
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
            found.setdefault(name, []).append((seq, engine, gpus, c.name, slots))
        except Exception as e:
            log(f"认领 {c.name} 失败（忽略，不影响网关启动）: {e}")
    for name, items in found.items():
        try:
            items.sort(key=lambda t: t[0])
            gpus = sorted({g for _, _, gs, _, _ in items for g in gs})
            slots = next((s for _, _, _, _, s in items if s is not None), None)
            backends[name] = {"container": items[0][3], "gpus": gpus, "util": 0.92,
                              "in_flight": 0, "last_used": time.time(),
                              "status": "adopted", "engine": items[0][1], "slots": slots,
                              "replicas": len(items),
                              "peers": [{"container": cn, "gpus": gs, "in_flight": 0}
                                        for _, _, gs, cn, _ in items]}
            log(f"认领现有后端 {name} gpus={gpus} engine={items[0][1]} slots={slots}"
                + (f" 副本×{len(items)}" if len(items) > 1 else "") + "（待健康确认）")
        except Exception as e:
            log(f"认领 {name} 失败（忽略，不影响网关启动）: {e}")


def model_context_length(d):
    gw = d.get("gw", {})
    n = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
    if d.get("engine") == "llamacpp":
        # llama.cpp --parallel N 把 ctx 总池平分为 N 个槽，单请求只占一槽；
        # 广告总池会让按 context_length 决定 compact 时机的客户端超限被拒
        # （实测 263629 tokens 撞 262144 槽上限）。广告值 = 单请求真实上限。
        n //= max(1, int(gw.get("parallel", 1)))
    return n


def model_capabilities(d):
    capabilities = {
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "streaming": True,
        "tool_calls": True,
        "structured_outputs": True,
        "reasoning": bool(d.get("think")),
        "vision": False,
        "video_input": False,
    }
    capabilities.update(d.get("capabilities") or {})
    capabilities["api_endpoints"] = [
        "/v1/chat/completions", "/v1/responses", "/v1/messages",
    ]
    return capabilities


def model_descriptor(name, d):
    return {
        "id": name,
        "object": "model",
        "created": 0,
        "owned_by": "ai-gateway",
        "loaded": backends.get(name, {}).get("status") == "ready",
        "context_length": model_context_length(d),
        "native_context_length": int(d.get("max_ctx", d.get("ctx_default", 32768))),
        "capabilities": model_capabilities(d),
    }


@app.get("/v1/models")
async def list_models():
    man = await asyncio.to_thread(load_manifest)
    data = []
    for name, descriptor in man.items():
        if not model_dir(name):
            continue
        data.append(model_descriptor(name, descriptor))
    return {"object": "list", "data": data}


@app.get("/v1/gateway/status")
async def status(request: Request):
    token_stats = await sample_all_backend_tokens()
    out_backends = {}
    for name, backend in backends.items():
        item = dict(backend)
        stat = refresh_live_counters(name)
        if stat is None:
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
            log(f"手动卸载 {b['container']}"
                + (f" 等 {b.get('replicas') or 1} 个副本" if (b.get("replicas") or 1) > 1 else "")
                + f"（释放 GPU {b['gpus']}）")
            await asyncio.to_thread(_remove_backend_containers, b)
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


async def _run_load(op, model, man, want_gpus, want_replicas=None):
    """执行一次加载并把状态写回 op（同步可等/异步后台共用）。"""
    _op_update(op, status="running", stage="loading",
               detail=f"准备加载 {model}"
                      + (f"（副本×{want_replicas}）" if want_replicas else ""),
               target_gpus=want_gpus)
    try:
        b = await ensure_loaded(model, man, want_gpus=want_gpus, op=op,
                                want_replicas=want_replicas)
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
        b = backends.get(name)
        models[name] = {
            **model_descriptor(name, d),
            "engine": d.get("engine", "vllm"),
            "min_gpus": int(d.get("min_gpus", 1)),
            "think": bool(d.get("think")),
            "desc": d.get("desc", ""),
            "backend_status": b["status"] if b else "unloaded",
            "gpus": list(b["gpus"]) if b else None,
        }
    gpus = await _gpu_discovery()
    return {
        "name": "ai-gateway",
        "version": "3.0",
        "description": "局域网 AI 模型网关：OpenAI / Anthropic / Responses 兼容推理 + 按 GPU 协商加载。",
        "protocol": "gateway-negotiation-v1",
        "capabilities": {
            "gpu_discovery": True,
            "gpu_selection": True,
            "async_load": True,
            "load_status_polling": True,
            "per_model_unload": True,
            "video_switch": True,
            "layout_modes": True,        # GET /v1/gateway/modes + load 的 mode/replicas
            "runtime_replica_override": True,
            "api_styles": ["openai-chat", "anthropic-messages", "openai-responses",
                           "anthropic-responses"],
            "response_style_selection": "默认 OpenAI；Anthropic 使用 anthropic-version、x-api-format: anthropic 或 ?format=anthropic",
        },
        "endpoints": [
            {"method": "GET", "path": "/v1/gateway/capabilities",
             "purpose": "本说明接口：网关能力、协商协议、GPU 拓扑与模型目录"},
            {"method": "GET", "path": "/v1/gateway/gpus",
             "purpose": "GPU 发现：数量、每卡容量/占用/利用率、谁占用了卡"},
            {"method": "GET", "path": "/v1/gateway/modes",
             "purpose": "布局模式目录：各模型现行摆放/可切布局（dual/tp/replica/pool/auto）、"
                        "ctx 估算与宿主持久化命令；?model= 查单个"},
            {"method": "POST", "path": "/v1/gateway/load",
             "purpose": "加载模型到指定 GPU 或按 mode 切布局（布局不符会先卸后装；"
                        "正忙 409）；wait=false 异步返回 request_id 供轮询"},
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
             "purpose": "模型列表：capabilities、实际/原生上下文长度与 loaded 标记"},
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
            "layout_modes": "GET /v1/gateway/modes 查本机布局目录 → "
                            "POST /v1/gateway/load {'model':...,'mode':'tp'|'replica'|'dual'|'auto'} "
                            "当场切摆放（max_len 增益需先跑目录里的 host_command 落 gw，网关不写配置）",
        },
        "gpus": gpus,
        "models": models,
        "notes": [
            "gpus 字段传数组，如 [0] 或 [0,1]；多卡模型必须给足 min_gpus 张卡",
            "load 的 gpus 为硬性指定：指定卡放不下时返回 503 并说明各卡可用量，不做静默改道",
            "显存不足时网关会先卸载空闲模型让位（in_flight==0 的最久未用；正忙的绝不驱逐）",
            "后端状态取值：unloaded / adopted / loading / ready；op.status：queued/running/success/failed",
            "布局模式边界：网关对 models.json 只读——摆放（哪几张卡/几副本/TP）当场执行，"
            "ctx 持久化（gw.max_len/gw.replicas 落盘）必须宿主 menu.py；"
            "modes 目录同时给 enforced_max_len（现在切过去的实际值）与 est_ctx_ceiling（持久化后的上限）",
        ],
    }


# ---------- 布局模式（gateway-layout-v1） ----------
# 与 models.json 的只读挂载死死对齐：网关能当场执行"纯摆放"（哪几张卡、几个副本、
# TP=每引擎卡数——拉起参数全在运行时），绝不写回配置；凡涉及把 max_len/gpus 持久化
# 的新布局，返回 host_command 指路宿主侧 menu.py——权威 ctx 值永远出自本机实算，
# 网关报的是同一公式的估算值（menu.py 与这里各写一份公式才是真隐患，估算仅供选型）。

def _weights_gb(d, dirname):
    """实际加载目录的权重体量近似值（GB），仅用于目录 ctx 估算。"""
    v = d.get("variants") or {}
    if not v:
        return 0.0
    if dirname and dirname.endswith("-fp8") and "fp8" in v:
        return float(v["fp8"])
    if "bf16" in v:
        return float(v["bf16"])
    return float(min(v.values()))


def _yarn_cap(d, gw):
    """YaRN 扩展上限：从 gw.extra_args 的 rope_parameters JSON 解析 factor ×
    original_max_position_embeddings，与原生 max_ctx 取大者。与宿主侧
    check_llm_config.yarn_override_limit 同一规则。"""
    cap = int(d.get("max_ctx") or d.get("ctx_default") or 32768)
    for a in gw.get("extra_args") or []:
        s = str(a)
        m = re.search(r'"factor":\s*([\d.]+)', s)
        o = re.search(r'"original_max_position_embeddings":\s*(\d+)', s)
        if m and o:
            try:
                cap = max(cap, int(int(o.group(1)) * float(m.group(1))))
            except ValueError:
                pass
    return cap


def _ctx_est(d, gw, weights_gb, gmem_gb, cards):
    """vLLM 单请求 ctx 估算（与 menu.py 的 vllm_ctx 同式）：KV 预算 × 实测密度，
    封顶 YaRN/原生。weights_gb 为 0（未下载/未声明）时退回 gw.max_len。"""
    if not weights_gb:
        return int(gw.get("max_len", 0) or 0)
    density = int(d.get("kv_tokens_per_gb") or 15300)
    total = gmem_gb * float(gw.get("util", 0.92)) * cards
    kv = max(0.0, total - weights_gb - 4)
    est = int(kv * density)
    if d.get("kv_tokens_per_gb"):
        est = int(est * 0.98)   # 与 menu.py vllm_ctx 同口径：实测密度没含 CUDA-graph 预留
    return max(32768, min(_yarn_cap(d, gw), est))


def _active_layout(name, d):
    """当前布局：优先运行中后端，退回 gw 落盘值。"""
    gw = d.get("gw", {})
    b = backends.get(name)
    if b and b["status"] in ("ready", "adopted", "loading"):
        gpus = sorted(b["gpus"])
        reps = int(b.get("replicas") or 1)
        src = "backend"
    else:
        gpus = sorted(int(x) for x in gw.get("gpus", "1").split(","))
        reps = int(gw.get("replicas") or 1)
        src = "manifest"
    return {"gpus": gpus, "replicas": reps,
            "tp": max(1, round(len(gpus) / reps)), "source": src}


def _mode_catalog(name, d):
    """某模型在本机可用的布局模式目录（advisory + 可执行参数）。

    enforced_max_len 是"现在走 load 切过去，实际会按哪个值起后端"——网关永远读
    gw.max_len；est_ctx_ceiling 是该摆放全兑现后 menu.py 会落的权威值。
    两者不等时 ctx_gain_requires_host=true：先跑宿主命令改写 gw，再 load 才有增益。"""
    gw = d.get("gw", {})
    engine = d.get("engine", "vllm")
    info = gpu_info()
    n = len(info)
    if not n:
        return {"engine": engine, "error": "nvidia-smi 无输出", "modes": []}
    gmem = round(min(x["total_mb"] for x in info.values()) / 1024)
    dirname = model_dir(name, d) or ""
    wgb = _weights_gb(d, dirname)
    max_len_gw = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
    all_cards = sorted(info)
    act = _active_layout(name, d)
    min_gpus = int(d.get("min_gpus", 1))
    modes = []

    def add(mid, label, gpus, replicas, note, host_cmd=None, runnable=True,
            ceiling=None, video=False):
        cards = len(gpus) if gpus else 0
        modes.append({
            "id": mid, "label": label, "gpus": gpus, "replicas": replicas,
            "tp": (max(1, round(cards / replicas))
                   if engine == "vllm" and replicas else None),
            "video_coexist": video,
            "enforced_max_len": max_len_gw,
            "est_ctx_ceiling": ceiling or max_len_gw,
            "ctx_gain_requires_host": bool(ceiling and ceiling > max_len_gw),
            "runnable_via_load": runnable,
            "active": (gpus == act["gpus"] and replicas == act["replicas"]),
            "note": note, "host_command": host_cmd})

    if engine == "llamacpp":
        add("pool", f"独占 {n} 卡跨卡分层（llama.cpp）", all_cards, 1,
            f"GGUF 按层摊到全部卡（ctx 按 gw.max_len={max_len_gw} 已含 YaRN 配置）；视频让位",
            ceiling=max_len_gw)
    elif min_gpus > n:
        add("tp", f"需 {min_gpus} 卡，本机仅 {n} 卡", all_cards, 1,
            "卡数不足，任何摆放都起不来；先释放/换机", runnable=False)
    elif min_gpus >= 2:
        add("tp", f"原生 {n} 卡 TP（旗舰摆放）", all_cards, 1,
            "min_gpus≥2 的模型只有整池 TP 一种形态；视频让位",
            ceiling=_ctx_est(d, gw, wgb, gmem, n))
    else:
        # dual 的基准卡优先取 gw 的单卡配置（宿主 menu.py 落的意图，如 111 的 GPU1
        # =LLM 位、GPU0=视频位）；gw 卡号在本机不存在（单卡机继承蓝本残留）才退回
        # 运行态或首卡——正在跑多副本时 act["gpus"] 是整池，不能拿来定单卡基准。
        gw_single = [int(x) for x in gw.get("gpus", "").split(",")
                     if x.strip().lstrip("-").isdigit()]
        gw_single = [g for g in gw_single if g in info]
        if len(gw_single) == 1:
            single_card = [gw_single[0]]
        elif len(act["gpus"]) == 1 and act["gpus"][0] in info:
            single_card = list(act["gpus"])
        else:
            single_card = [all_cards[0]]
        add("dual", "单卡 LLM，其余卡给视频/第二模型（默认）", single_card, 1,
            f"ctx={max_len_gw}（按单卡 KV 预算落的权威值）；可与 ComfyUI 真并行",
            ceiling=_ctx_est(d, gw, wgb, gmem, 1), video=True)
        if n >= 2 and engine == "vllm":
            tp_c = _ctx_est(d, gw, wgb, gmem, n)
            add("tp", f"{n} 卡单引擎 TP（整池换单请求上下文）", all_cards, 1,
                f"解码速度不涨（无 NVLink 时 all-reduce 净亏）；视频让位。"
                f"ctx 上限 {tp_c}：先跑 host_command 落 gw.max_len，再 load 才吃到",
                host_cmd=f"./scripts/start.sh --llm {name} --tp {n} -y",
                ceiling=tp_c)
            add("replica", f"每卡一份副本（聚合吞吐≈×{n}）", all_cards, n,
                f"每副本一卡一引擎，零 TP 通信；每流 ctx 仍 {max_len_gw}；视频让位。"
                "持久副本数走 host_command 写 gw.replicas",
                host_cmd=f"./scripts/start.sh --llm {name} --replicas {n} -y",
                ceiling=_ctx_est(d, gw, wgb, gmem, 1))
    add("auto", "按 models.json gw 缺省摆放（清除覆盖）", None, None,
        "不带任何摆放覆盖的加载：完全沿用 gw.gpus/replicas/max_len",
        runnable=True)
    return {"engine": engine, "weights_gb": wgb,
            "cards": {"count": n, "gmem_gb": gmem, "pool_gb": int(gmem * n * 0.92)},
            "active": act, "enforced_max_len": max_len_gw, "modes": modes}


@app.get("/v1/gateway/modes")
async def gateway_modes(model: str = ""):
    """布局模式目录（gateway-layout-v1）：各模型的现行摆放、可切布局、ctx 估算与
    宿主侧持久化命令。带 ?model=<名> 只查一个模型。

    选型建议按"你要什么"读：吞吐→replica；单请求超长→tp（先跑 host_command）；
    并行跑视频→dual；旗舰质量长上下文→pool 型 GGUF。所有摆放冲突（busy/显存不足）
    最终由 POST /v1/gateway/load 的同一套校验兜底，目录不做乐观承诺。"""
    man = await asyncio.to_thread(load_manifest)
    names = [model] if model else list(man)
    out = {}
    for nm in names:
        if nm not in man:
            return JSONResponse({"error": {"message": f"模型 '{nm}' 不存在",
                                           "type": "invalid_model_error"}}, status_code=404)
        if not await asyncio.to_thread(model_dir, nm):
            continue
        out[nm] = await asyncio.to_thread(_mode_catalog, nm, man[nm])
    return {"protocol": "gateway-layout-v1",
            "boundary": "网关对 models.json 只读：摆放当场执行，max_len/replicas 持久化走 host_command",
            "models": out}


@app.post("/v1/gateway/load")
async def gateway_load(request: Request):
    """把指定模型加载到指定 GPU（细粒度协商入口）。

    body:
      model: 模型名（models.json 中的 key）
      gpus:  [0] 或 [0,1]；可选，缺省按 models.json gw.gpus 自动放置（可能回落其它空闲卡）
      mode:  可选布局模式（dual/tp/replica/pool/auto），gpus+replicas 的语义糖，
             本机可用值见 GET /v1/gateway/modes?model=…；显式 gpus/replicas 优先于 mode
      replicas: 可选副本数（每副本至少一卡），运行时覆盖 gw.replicas，不落盘
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

    # 布局模式糖：mode 解析成 (gpus, replicas)，显式参数优先；auto = 不带覆盖
    mode = (body.get("mode") or "").strip().lower()
    want_replicas = body.get("replicas")
    if mode and mode != "auto":
        cat = await asyncio.to_thread(_mode_catalog, model, man[model])
        hit = next((m for m in cat["modes"] if m["id"] == mode), None)
        if hit is None:
            return JSONResponse({"error": {"message":
                f"{model} 本机没有模式 '{mode}'，可用: "
                + " / ".join(m["id"] for m in cat["modes"]),
                "type": "invalid_request_error"}}, status_code=400)
        if not hit["runnable_via_load"]:
            return JSONResponse({"error": {"message":
                f"模式 {mode} 当前不可执行：{hit['note']}", "type": "mode_unavailable"},
                "mode": hit}, status_code=409)
        if body.get("gpus") is None and hit["gpus"] is not None:
            body["gpus"] = hit["gpus"]
        if want_replicas is None:
            want_replicas = hit["replicas"]
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
    if want_replicas is not None and (not isinstance(want_replicas, int)
                                      or want_replicas < 1):
        return JSONResponse({"error": {"message": "replicas 必须是正整数",
                                       "type": "invalid_request_error"}}, status_code=400)
    if want_replicas and gpus and want_replicas > len(gpus):
        return JSONResponse({"error": {"message":
            f"replicas={want_replicas} 超过指定卡数 {gpus}（每副本至少一卡）",
            "type": "invalid_request_error"}}, status_code=400)
    wait = body.get("wait", True)

    # 已加载且布局一致：直接返回当前状态（幂等）。布局不一致（gpus/replicas/mode
    # 要求了别的摆放）：先卸旧后端再起新布局——"切模式"要当场生效，不是"确保已加载"。
    # loading 不抢切：容器正被 create 中，摘登记只会留下没人收的孤儿，让终态收敛后再切。
    b = backends.get(model)
    if b:
        want_r = want_replicas or int(man[model].get("gw", {}).get("replicas") or 1)
        same = (gpus is None or sorted(b["gpus"]) == sorted(gpus)) \
            and int(b.get("replicas") or 1) == want_r
        if b["status"] == "ready" and same:
            op = _new_op("load", model)
            _op_update(op, status="success", stage="ready",
                       detail=f"{model} 已就绪 on GPU {sorted(b['gpus'])}",
                       target_gpus=b["gpus"])
            return JSONResponse(await _op_response(op), status_code=200)
        if mode or gpus is not None or want_replicas:
            if b["status"] == "loading":
                return JSONResponse({"error": {"message":
                    f"{model} 正在加载（旧布局），拒绝插队改摆放；等其就绪或失败后再切",
                    "type": "busy_error"}}, status_code=409)
            if b["in_flight"] > 0:
                return JSONResponse({"error": {"message":
                    f"{model} 正在处理请求（{b['in_flight']} 个在途），改摆放需先等其完成",
                    "type": "busy_error"}}, status_code=409)
            prev_free = {g: x.get("free", 0)
                         for g, x in (await asyncio.to_thread(gpu_free)).items()
                         if g in b["gpus"]}
            await asyncio.to_thread(_remove_backend_containers, b)
            backends.pop(model, None)
            log(f"{model}: 布局切换（{mode or (str(gpus) + f' x{want_r}')}），旧后端已卸载")
            # 强删的显存释放有数秒延迟；同卡立刻起新后端会把显存算少、CUDA graph 预算
            # 挤爆（本机实测：adopted 后端被删后立刻 replica×2，两副本双双起不来）。
            # 轮询旧卡的 free 回还到删除前水平，最多 15s。
            for _ in range(30):
                f2 = await asyncio.to_thread(gpu_free)
                if all(f2.get(g, {}).get("free", 0) >= fr - 256
                       for g, fr in prev_free.items()):
                    break
                await asyncio.sleep(0.5)

    op = _new_op("load", model)
    if wait is False:
        asyncio.create_task(_run_load(op, model, man, gpus, want_replicas))
        return JSONResponse({"status": "accepted", "model": model,
                             "op": await _op_response(op)}, status_code=202)

    await _run_load(op, model, man, gpus, want_replicas)
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
               detail=f"卸载 {b['container']}"
                      + (f" 等 {b.get('replicas') or 1} 个副本"
                         if (b.get("replicas") or 1) > 1 else "")
                      + f"（释放 GPU {sorted(b['gpus'])}）")
    async with ops_lock:
        b = backends.get(model)
        if b and b["in_flight"] == 0 and b["status"] != "loading":
            await asyncio.to_thread(_remove_backend_containers, b)
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
            want_replicas = None
            hm = request.headers.get("x-gateway-mode", "").strip().lower()
            if hm and hm != "auto":
                cat = await asyncio.to_thread(_mode_catalog, name, man[name])
                hit = next((m for m in cat["modes"] if m["id"] == hm), None)
                if hit is None or not hit["runnable_via_load"]:
                    return JSONResponse({"error": {"message":
                        f"X-Gateway-Mode '{hm}' 不可用，可选: "
                        + " / ".join(m["id"] for m in cat["modes"] if m["runnable_via_load"]),
                        "type": "invalid_request_error"}}, status_code=400)
                if want_gpus is None and hit["gpus"] is not None:
                    want_gpus = hit["gpus"]
                want_replicas = hit["replicas"]
            op = _new_op("load", name)
            _op_update(op, status="running", stage="loading",
                       detail=f"异步加载 {name}", target_gpus=want_gpus)
            asyncio.create_task(_run_load(op, name, man, want_gpus, want_replicas))
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
    p = _pick_peer(b)          # 多副本：挑在途最少的副本；单副本时 peers 只 1 个元素，分流退化为自己
    p["in_flight"] = p.get("in_flight", 0) + 1   # 与 b 的自增同一原子段，中间不 await
    b["last_used"] = time.time()
    body = await request.body()
    body = clamp_max_tokens(name, man, body)
    qs = request.url.query
    url = f"http://{p['container']}:8000{request.url.path}" + (f"?{qs}" if qs else "")
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
        p["in_flight"] -= 1
        log(f"后端 {name} 连接失败，剔除并重新加载")
        async with ops_lock:
            if backends.get(name) is b:
                # 整组副本一起删：只删一个会让其它副本占着显存变成孤儿
                await asyncio.to_thread(_remove_backend_containers, b)
                backends.pop(name, None)
        try:
            b = await ensure_loaded(name, man)
        except Exception as e:
            return JSONResponse({"error": {"message": f"后端重建失败: {e}",
                                           "type": "backend_error"}}, status_code=502)
        b["in_flight"] += 1
        p = _pick_peer(b)
        p["in_flight"] = p.get("in_flight", 0) + 1
        url = f"http://{p['container']}:8000{request.url.path}" + (f"?{qs}" if qs else "")
        try:
            req = http.build_request(request.method, url, content=body, headers=headers)
            r = await http.send(req, stream=True)
        except Exception as e:
            b["in_flight"] -= 1
            p["in_flight"] -= 1
            return JSONResponse({"error": {"message": f"转发失败: {e}", "type": "proxy_error"}},
                                status_code=502)
    except Exception as e:
        b["in_flight"] -= 1
        p["in_flight"] -= 1
        return JSONResponse({"error": {"message": f"转发失败: {e}", "type": "proxy_error"}},
                            status_code=502)

    if r.status_code >= 400:
        # 上游错误：读出错误体记入网关日志（vLLM 自身不记 400 原因），再原样透传给客户端
        err = await r.aread()
        await r.aclose()
        b["in_flight"] -= 1
        p["in_flight"] -= 1
        log(f"{name} 上游 {r.status_code}: {err.decode(errors='replace')[:500]}")
        return StreamingResponse(iter([err]), status_code=r.status_code,
                                 headers={"content-type": r.headers.get(
                                     "content-type", "application/json")})

    if live_tokens and not client_stream:
        try:
            return await _buffered_openai_response(name, r)
        finally:
            b["in_flight"] -= 1
            p["in_flight"] -= 1
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
            p["in_flight"] -= 1
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
# OpenAI Responses / Anthropic Messages 兼容层
# ============================================================

response_cache = {}
RESPONSE_CACHE_SIZE = 128


class UpstreamError(Exception):
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload


def _protocol_error(message, style="openai", status=400, error_type="invalid_request_error"):
    if style == "anthropic":
        return JSONResponse({"type": "error", "error": {"type": error_type, "message": str(message)}},
                            status_code=status)
    return JSONResponse({"error": {"message": str(message), "type": error_type}}, status_code=status)


def _upstream_error(error, style):
    payload = error.payload
    if style == "anthropic":
        source = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(source, dict):
            message = source.get("message", str(source))
        else:
            message = str(payload)
        return _protocol_error(message, style, error.status, "api_error")
    if isinstance(payload, dict) and "error" in payload:
        return JSONResponse(payload, status_code=error.status)
    return _protocol_error(str(payload), style, error.status, "api_error")


def _release_chat_backend(backend, peer):
    backend["in_flight"] -= 1
    peer["in_flight"] -= 1
    backend["last_used"] = time.time()


async def _send_chat(name, manifest, body):
    if not name:
        raise ProtocolError("缺少 model 字段")
    if name not in manifest or not await asyncio.to_thread(model_dir, name):
        raise ProtocolError(f"模型 '{name}' 不存在或未下载")
    body = json.loads(clamp_max_tokens(name, manifest, json.dumps(body, ensure_ascii=False).encode()))
    if body.get("stream"):
        options = body.setdefault("stream_options", {})
        if isinstance(options, dict):
            options["include_usage"] = True
    try:
        backend = await ensure_loaded(name, manifest)
    except ResourceError as e:
        raise UpstreamError(503, {"error": {"message": str(e), "type": "insufficient_resources"}})
    except Exception as e:
        raise UpstreamError(502, {"error": {"message": f"模型加载失败: {e}", "type": "backend_error"}})
    backend["in_flight"] += 1
    peer = _pick_peer(backend)
    peer["in_flight"] = peer.get("in_flight", 0) + 1
    backend["last_used"] = time.time()
    url = f"http://{peer['container']}:8000/v1/chat/completions"
    try:
        req = http.build_request("POST", url, content=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"content-type": "application/json"})
        response = await http.send(req, stream=True)
    except Exception as e:
        _release_chat_backend(backend, peer)
        raise UpstreamError(502, {"error": {"message": f"转发失败: {e}", "type": "api_error"}})
    if response.status_code >= 400:
        raw = await response.aread()
        await response.aclose()
        _release_chat_backend(backend, peer)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode(errors="replace")[:1000], "type": "api_error"}}
        log(f"{name} 兼容接口上游 {response.status_code}: {raw.decode(errors='replace')[:500]}")
        raise UpstreamError(response.status_code, payload)
    return backend, peer, response


async def _read_chat_response(backend, peer, response):
    try:
        raw = await response.aread()
        return json.loads(raw)
    except Exception as e:
        raise UpstreamError(502, {"error": {"message": f"读取后端响应失败: {e}", "type": "api_error"}})
    finally:
        await response.aclose()
        _release_chat_backend(backend, peer)


def _converted_stream(backend, peer, response, converter):
    async def generate():
        try:
            async for chunk in converter:
                yield chunk
        finally:
            await response.aclose()
            _release_chat_backend(backend, peer)
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


def _cache_response(response):
    response_cache[response["id"]] = response
    while len(response_cache) > RESPONSE_CACHE_SIZE:
        response_cache.pop(next(iter(response_cache)))


async def _json_request(request, style):
    try:
        body = await request.json()
    except Exception:
        raise ProtocolError("请求体不是合法 JSON")
    if not isinstance(body, dict) or not body:
        raise ProtocolError("请求体必须是非空 JSON 对象")
    return body


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    try:
        body = await _json_request(request, "anthropic")
        text = json.dumps({k: v for k, v in body.items() if k != "stream"}, ensure_ascii=False)
        ascii_count = sum(1 for char in text if ord(char) < 128)
        return {"input_tokens": ascii_count // 3 + len(text) - ascii_count}
    except ProtocolError as e:
        return _protocol_error(e, "anthropic")


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try:
        body = await _json_request(request, "anthropic")
        manifest = await asyncio.to_thread(load_manifest)
        chat_body = anthropic_to_chat(body)
        backend, peer, upstream = await _send_chat(body.get("model", ""), manifest, chat_body)
        if chat_body.get("stream"):
            return _converted_stream(backend, peer, upstream,
                                     stream_anthropic(upstream.aiter_lines(), body["model"]))
        data = await _read_chat_response(backend, peer, upstream)
        return JSONResponse(chat_to_anthropic(data))
    except ProtocolError as e:
        return _protocol_error(e, "anthropic")
    except UpstreamError as e:
        return _upstream_error(e, "anthropic")


@app.get("/v1/messages/{model_name}")
async def anthropic_model_info(model_name: str):
    manifest = await asyncio.to_thread(load_manifest)
    if model_name not in manifest:
        return _protocol_error(f"模型 '{model_name}' 不存在", "anthropic", 404, "invalid_model_error")
    descriptor = model_descriptor(model_name, manifest[model_name])
    descriptor["description"] = manifest[model_name].get("desc", "")
    descriptor["context_window"] = descriptor["context_length"]
    return descriptor


def _responses_style(request, body):
    explicit = (request.query_params.get("format")
                or request.headers.get("x-api-format")
                or request.headers.get("x-response-format"))
    if explicit:
        value = explicit.lower()
        if value in ("anthropic", "claude"):
            return "anthropic"
        if value in ("openai", "openai-responses"):
            return "openai"
    if request.headers.get("anthropic-version") or body.get("anthropic_version") \
            or "output_config" in body:
        return "anthropic"
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and "input_schema" in tool:
            return "anthropic"
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "input_message":
            return "anthropic"
    return "openai"


@app.post("/v1/responses")
async def responses_api(request: Request):
    style = "openai"
    try:
        body = await _json_request(request, style)
        style = _responses_style(request, body)
        manifest = await asyncio.to_thread(load_manifest)
        chat_body = responses_to_chat(body)
        backend, peer, upstream = await _send_chat(body.get("model", ""), manifest, chat_body)
        if chat_body.get("stream"):
            return _converted_stream(backend, peer, upstream,
                                     stream_responses(upstream.aiter_lines(), body["model"]))
        data = await _read_chat_response(backend, peer, upstream)
        response = chat_to_response(data)
        _cache_response(response)
        return JSONResponse(response)
    except ProtocolError as e:
        return _protocol_error(e, style)
    except UpstreamError as e:
        return _upstream_error(e, style)


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    response = response_cache.get(response_id)
    if response is None:
        return _protocol_error(f"响应 '{response_id}' 不存在或已过期", status=404, error_type="not_found")
    return response


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    if response_cache.pop(response_id, None) is None:
        return _protocol_error(f"响应 '{response_id}' 不存在或已过期", status=404, error_type="not_found")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


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
