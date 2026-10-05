#!/usr/bin/env python3
"""AI 模型网关（常驻能力，非临时补救）

规则（与用户需求一一对应）：
1. /v1/models 暴露全部已下载模型，而不是只暴露当前加载的那个
2. 客户端按模型名请求：未加载 → 自动加载（请求挂起等待就绪）
3. 显存不足时：优先卸载其它【空闲】的文本模型（in_flight==0，按最久未用）
4. 永远不碰视频服务（ai-comfyui）与 WebUI；被视频挡住时返回明确 503，不静默降配
5. 上下文长度一律按模型原生上限（models.json 的 gw.max_len）配满，不降档
6. 卸载只发生在"需要腾显存"时，不设空闲定时器

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


def _run_backend(name, dirname, gpus, util, max_len, think):
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
    log(f"加载 {name}: dir={dirname} gpus={gpus} tp={tp} util={util} max_len={max_len} think={think}")
    _remove_container(f"ai-vllm-{name}")
    kwargs = {}
    if tp > 1:
        kwargs["ipc_mode"] = "host"
    # 编译/JIT 缓存持久化到宿主机：容器每次重建，不挂载则 torch.compile 每次重跑（~30s）
    cache_host = f"{HOST_ROOT}/data/vllm-cache"
    os.makedirs(cache_host, exist_ok=True)
    dcli.containers.run(
        VLLM_IMAGE, command=cmd, name=f"ai-vllm-{name}", detach=True,
        device_requests=[docker.types.DeviceRequest(
            device_ids=[str(g) for g in gpus], capabilities=[["gpu"]])],
        network=NETWORK, shm_size="16g",
        environment={"TRITON_CACHE_DIR": "/root/.cache/triton"},
        volumes={f"{HOST_ROOT}/models/llm": {"bind": "/models", "mode": "rw"},
                 cache_host: {"bind": "/root/.cache", "mode": "rw"}},
        **kwargs)


# ---------- 调度 ----------

def evict_idle(need):
    """need: {gpu: 需要空闲MB}。卸载空闲文本后端直到满足；返回仍不满足的 gpu 列表"""
    while True:
        free = gpu_free()
        lacking = [g for g, mb in need.items() if free.get(g, {}).get("free", 0) < mb]
        if not lacking:
            return []
        cands = sorted(
            [(k, b) for k, b in backends.items()
             if b["in_flight"] == 0 and b["status"] == "ready"
             and any(g in lacking for g in b["gpus"])],
            key=lambda kv: kv[1]["last_used"])
        if not cands:
            return lacking
        vname, victim = cands[0]
        log(f"卸载空闲模型 {victim['container']}（释放 GPU {victim['gpus']}）")
        _remove_container(victim["container"])
        backends.pop(vname, None)


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
    est_prompt = max(1, _est_tokens(payload))
    allowed = max_len - est_prompt - 1024  # 再留 1024 余量覆盖模板/特殊 token 与估算误差
    for key in ("max_tokens", "max_completion_tokens"):
        mt = payload.get(key)
        if isinstance(mt, int) and mt > allowed:
            payload[key] = max(1024, allowed)
            log(f"{name}: {key} {mt} -> {payload[key]}（prompt 估 {est_prompt} tokens）")
    return json.dumps(payload).encode()


async def ensure_loaded(name, man):
    b = backends.get(name)
    if b and b["status"] == "ready":
        return b
    async with ops_lock:
        b = backends.get(name)
        if b and b["status"] == "ready":
            return b
        if b and b["status"] == "loading":
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
                fallback = sorted(free)[-1]
                log(f"{name}: 配置的 GPU{gpus[0]} 不存在，改用 GPU{fallback}")
                gpus = [fallback]
            else:
                raise RuntimeError(
                    f"{name} 需要 GPU {gpus}，但本机只有 {sorted(free)}")
        need = {g: int(util * free.get(g, {}).get("total", 0)) + NEED_MARGIN_MB
                for g in gpus}
        # 预检：即使卸载全部空闲文本后端也不够（典型：视频占着 GPU0），直接报错，不做无谓卸载
        potential = {g: free.get(g, {}).get("free", 0) for g in gpus}
        for b in backends.values():
            if b["in_flight"] == 0 and b["status"] == "ready":
                for g in b["gpus"]:
                    if g in potential:
                        potential[g] += int(b.get("util", 0.92)
                                            * free.get(g, {}).get("total", 0))
        hopeless = [g for g in gpus if potential[g] < need[g]]
        if not hopeless:
            hopeless = await asyncio.to_thread(evict_idle, need)
        if hopeless:
            vg = await asyncio.to_thread(comfy_gpus)
            vg &= set(hopeless)
            free = await asyncio.to_thread(gpu_free)
            detail = ", ".join(
                f"GPU{g} 空闲 {free.get(g,{}).get('free',0)//1024}GB / 需要 {need[g]//1024}GB"
                for g in hopeless)
            msg = f"显存不足：{detail}。已卸载所有空闲文本模型仍不够。"
            if vg:
                msg += (f" GPU {sorted(vg)} 被视频服务 ai-comfyui 占用（规则：不动视频）。"
                        f" 如需运行 {name}，请先手动停止视频：docker stop ai-comfyui")
            raise ResourceError(msg)

        await asyncio.to_thread(_run_backend, name, dirname, gpus, util, max_len,
                                think)
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
    """网关重启后认领还活着的后端容器（标 loading，首次请求时先等健康检查）"""
    for c in dcli.containers.list(all=True, filters={"status": ["running"]}):
        if not c.name.startswith("ai-vllm-"):
            continue
        try:
            name = c.name[len("ai-vllm-"):]
            req = c.attrs["HostConfig"].get("DeviceRequests") or []
            gpus = [int(x) for r in req for x in (r.get("DeviceIDs") or [])]
            backends[name] = {"container": c.name, "gpus": gpus, "util": 0.92,
                              "in_flight": 0, "last_used": time.time(), "status": "loading"}
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
