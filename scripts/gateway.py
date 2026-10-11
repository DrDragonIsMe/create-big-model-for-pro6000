#!/usr/bin/env python3
"""AI 模型网关（常驻能力，非临时补救）

暴露两种兼容 API（端口 8000）：
• OpenAI   — /v1/models, /v1/chat/completions（原有）
• Anthropic — /v1/messages, /v1/messages/count_tokens, /v1/messages/{model}（新增）

内部规则：
1. /v1/models 暴露全部已下载模型，而不是只暴露当前加载的那个
2. 客户端按模型名请求：未加载 → 自动加载（请求挂起等待就绪）
3. 显存放得下就并存（不限一卡一个，按模型名路由）：每卡可用空间按已加载模型的
   完整预算 util×total 扣减（含 KV cache 预留），新模型不侵占老模型的 KV 空间
4. 任何卡都放不下时：卸载其它【空闲】的文本模型（in_flight==0，按最久未用）；
   正忙的模型绝不驱逐，直接返回 503 让客户端稍后重试
5. 永远不碰视频服务（ai-comfyui）与 WebUI；被视频挡住时返回明确 503，不静默降配
6. 上下文长度一律按模型原生上限（models.json 的 gw.max_len）配满，不降档
7. 卸载只发生在"需要腾显存"时，不设空闲定时器

运行环境：ai-gateway 容器，挂载 docker.sock 管理后端容器，--gpus all 提供 nvidia-smi。
后端容器命名 ai-vllm-<model>，仅加入 ai_default 内网，不暴露宿主机端口。
所有 docker/nvidia-smi 同步调用均经 asyncio.to_thread 包裹，不阻塞事件循环。
"""
import asyncio, json, os, subprocess, time
from contextlib import asynccontextmanager

import docker
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS_JSON = os.environ.get("MODELS_JSON", "/opt/models.json")
HOST_ROOT = os.environ.get("HOST_ROOT", "/home/pc/AI")
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:latest")
NETWORK = os.environ.get("GW_NETWORK", "ai_default")
PRELOAD = [m for m in os.environ.get("PRELOAD", "").split(",") if m]
NEED_MARGIN_MB = 512  # vLLM 要求 空闲 >= util×total，再加安全余量，避免放行后启动崩溃

log = lambda *a: print("[gateway]", *a, flush=True)

dcli = docker.DockerClient(base_url="unix:///var/run/docker.sock")
http = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))

ops_lock = asyncio.Lock()      # 加载/卸载串行化
backends = {}                  # name -> {container, gpus, util, in_flight, last_used, status}


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


def comfy_gpus():
    """视频服务占用的 GPU（仅用于错误提示，绝不卸载它）"""
    try:
        c = dcli.containers.get("ai-comfyui")
        if c.status != "running":
            return set()
        req = c.attrs["HostConfig"].get("DeviceRequests") or []
        ids = set()
        for r in req:
            ids.update(int(x) for x in (r.get("DeviceIDs") or []))
        return ids
    except Exception:
        return set()


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


def _run_backend(name, dirname, gpus, util, max_len, think, extra_args=None, env=None):
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
    kwargs = {}
    if tp > 1:
        kwargs["ipc_mode"] = "host"
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


# ---------- 调度 ----------

def evict_idle(need):
    """need: {gpu: 需要空闲MB}。卸载空闲文本后端直到满足；返回仍不满足的 gpu 列表"""
    evicted = False
    deadline = time.time() + 60  # 强杀容器后显存释放有延迟，最多等 60s
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
        if not cands:
            if evicted and time.time() < deadline:
                time.sleep(2)
                continue
            return lacking
        vname, victim = cands[0]
        log(f"卸载空闲模型 {victim['container']}（释放 GPU {victim['gpus']}）")
        _remove_container(victim["container"])
        backends.pop(vname, None)
        evicted = True


async def wait_ready(name, timeout=900):
    url = f"http://ai-vllm-{name}:8000/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, tail = await asyncio.to_thread(_container_state, f"ai-vllm-{name}")
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


async def ensure_loaded(name, man):
    b = backends.get(name)
    if b and b["status"] == "ready":
        return b
    async with ops_lock:
        b = backends.get(name)
        if b and b["status"] == "ready":
            return b
        if b and b["status"] in ("loading", "adopted"):
            # 认领的/加载中的后端：先等健康确认；确认不了才重建
            try:
                await wait_ready(name, timeout=180)
                b["status"] = "ready"
                b["last_used"] = time.time()
                log(f"{name} 健康确认通过")
                return b
            except Exception:
                await asyncio.to_thread(_remove_container, b["container"])
                backends.pop(name, None)
        d = man[name]
        gw = d.get("gw", {})
        gpus = [int(x) for x in gw.get("gpus", "1").split(",")]
        util = float(gw.get("util", 0.92))
        max_len = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
        think = bool(d.get("think"))
        dirname = model_dir(name)
        if not dirname:
            raise RuntimeError(f"模型 {name} 未下载")

        free = await asyncio.to_thread(gpu_free)
        # 卡数适配：models.json 按双卡布局配置（LLM 在 GPU1），
        # 单卡机上配置的卡不存在时回落到现有卡，否则全部请求 503
        if any(g not in free for g in gpus):
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
        cand_sets = [gpus]
        if len(gpus) == 1:
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
            vg = await asyncio.to_thread(comfy_gpus)
            vg &= set(all_gs)
            avail = budget_avail(all_gs)
            detail = ", ".join(
                f"GPU{g} 可用 {avail[g]//1024}GB / 需要 {need_of([g])[g]//1024}GB"
                for g in all_gs)
            msg = f"显存不足：{detail}。已卸载所有空闲文本模型仍不够。"
            if vg:
                msg += (f" GPU {sorted(vg)} 被视频服务 ai-comfyui 占用（规则：不动视频）。"
                        f" 如需运行 {name}，请先手动停止视频：docker stop ai-comfyui")
            raise ResourceError(msg)

        if chosen[0] != gpus:
            log(f"{name}: 配置的 GPU{gpus} 放不下，改放 GPU{chosen[0]}（与已加载模型并存）")
        gpus = chosen[0]

        await asyncio.to_thread(_run_backend, name, dirname, gpus, util, max_len,
                                think, gw.get("extra_args"), gw.get("env"))
        backends[name] = {"container": f"ai-vllm-{name}", "gpus": gpus, "util": util,
                          "in_flight": 0, "last_used": time.time(), "status": "loading"}
        try:
            await wait_ready(name)
        except Exception:
            backends.pop(name, None)
            raise
        backends[name]["status"] = "ready"
        backends[name]["last_used"] = time.time()
        log(f"{name} 就绪")
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
        if not c.name.startswith("ai-vllm-"):
            continue
        try:
            name = c.name[len("ai-vllm-"):]
            req = c.attrs["HostConfig"].get("DeviceRequests") or []
            gpus = [int(x) for r in req for x in (r.get("DeviceIDs") or [])]
            backends[name] = {"container": c.name, "gpus": gpus, "util": 0.92,
                              "in_flight": 0, "last_used": time.time(), "status": "adopted"}
            log(f"认领现有后端 {name} gpus={gpus}（待健康确认）")
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
async def status():
    return {"backends": {k: dict(v) for k, v in backends.items()},
            "gpu_free": await asyncio.to_thread(gpu_free),
            "comfy_gpus": sorted(await asyncio.to_thread(comfy_gpus))}


@app.post("/v1/gateway/unload_all")
async def unload_all():
    """卸载所有空闲文本后端（in_flight==0），正忙的跳过。
    网关保持常驻：下一个请求仍会按需自动加载对应模型。
    与 evict_idle 同一原则——绝不驱逐在处理的模型，绝不碰视频服务。"""
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
    return {"unloaded": unloaded, "busy": busy,
            "gpu_free": await asyncio.to_thread(gpu_free)}


async def proxy(name, request: Request):
    man = await asyncio.to_thread(load_manifest)
    if name not in man or not model_dir(name):
        return JSONResponse({"error": {"message": f"模型 '{name}' 不存在或未下载",
                                       "type": "invalid_request_error"}}, status_code=404)
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

    async def gen():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            b["in_flight"] -= 1
            b["last_used"] = time.time()
    return StreamingResponse(gen(), status_code=r.status_code,
                             headers={"content-type": r.headers.get("content-type", "application/json")})


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
