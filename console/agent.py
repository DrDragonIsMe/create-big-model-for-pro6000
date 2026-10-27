#!/usr/bin/env python3
"""AI 栈管理后台（console agent，端口 9001）

一台同构节点跑一个：把本机已有的四套能力聚成 GUI 需要的 JSON API——
  • 网关协商（8000）：状态/GPU/布局目录/mode 切换——全部 HTTP 转发，本 agent 不复制判断逻辑
  • sysmonitor（8088）：/api/snapshot 原样代理（sysmonitor 保持可独立分发、零改动）
  • models.json + 文件系统：模型目录与下载状态（与 manage.sh list 同一口径：includes 逐条命中）
  • manage.sh / menu.py：下载、实测、持久布局——以"任务（job）"形式后台跑子进程，日志落盘可尾随
设计边界：
  • 本 agent 是宿主脚本的"带鉴权语义的遥控器"：摆放/ctx 的权威计算仍在 menu.py（经 start.sh 同参数），
    运行态切换走网关 gateway-layout-v1；agent 自己不写 models.json/.env（写的人只有一个：menu.py）。
  • 监听 0.0.0.0:9001 且默认无鉴权——与 8088/8188 同一内网信任级；CONSOLE_TOKEN 非空时校验 Bearer。
"""
import asyncio, fnmatch, json, os, re, shlex, shutil, sys, time, uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.environ.get("HOST_ROOT", "/home/pc/AI")
os.chdir(ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
GW = os.environ.get("CONSOLE_GATEWAY", "http://127.0.0.1:8000")
SYSMON = os.environ.get("CONSOLE_SYSMON", "http://127.0.0.1:8088")
TOKEN = os.environ.get("CONSOLE_TOKEN", "")
JOBS_DIR = os.path.join(ROOT, "data", "console-jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# 转发到网关的白名单：只允许这些管理路径，方法对齐；防止 agent 变成任意 SSRF 跳板
GW_POST = {"load", "unload", "unload_all", "switch"}
GW_GET = re.compile(r"^/(status|gpus|modes|capabilities|ops(/[0-9a-fA-F-]+)?)$")
# 只读代理别台机器的 HTTP（异构节点如 dgx-spark-pair 没有本 agent，前端借最近的口看它）
try:
    with open(os.path.join(HERE, "console.json"), encoding="utf-8") as f:
        RELAY = json.load(f).get("relay", {})
except Exception:
    RELAY = {}

app = FastAPI(title="ai-console")
jobs = {}                      # id -> dict(proc 信息 + 元数据)，agent 重启后进程灭、日志留档


# ---------- 基础 ----------

def run_as_ids():
    """manage.sh 下载容器以 --user 跑；本 agent 在容器里是 root，直接继承会把宿主
    models/ 里的文件全变成 root-owned。按 models 目录的真实属主覆盖（MANAGE_RUN_AS）。"""
    try:
        st = os.stat(os.path.join(ROOT, "models"))
        return f"{st.st_uid}:{st.st_gid}"
    except OSError:
        return "0:0"


def env_for_scripts():
    env = dict(os.environ)
    env["MANAGE_RUN_AS"] = env.get("MANAGE_RUN_AS") or run_as_ids()
    # menu.py 重写 .env 时若不带上原 key 会把 LLM_API_KEY 洗成空（真机踩过）：
    # 从现行 .env 里取出来传给子进程，agent 自己的任何响应都不回显它。
    try:
        with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("LLM_API_KEY="):
                    env["LLM_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    return env


def load_manifest():
    with open(os.path.join(ROOT, "models.json"), encoding="utf-8") as f:
        return json.load(f)


def gw_ok(v):
    return bool(TOKEN) and v.split(" ", 1)[-1] != TOKEN


@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.url.path.startswith("/api") \
            and request.headers.get("authorization", "") != f"Bearer {TOKEN}":
        return JSONResponse({"error": "invalid token"}, status_code=401)
    resp = await call_next(request)
    # 页面与静态资源一律 no-cache：上一轮修了 app.js 的 bug，浏览器却抱着旧缓存
    # 不放，用户看到的还是坏版本（"模型消失"的错觉来自旧 JS 在详情页抛异常）
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

# 后注册 = 更外层：跨源预检（OPTIONS+Authorization）必须先进 CORS 手里答掉，
# 再轮到鉴权——浏览器从任意一台 9001 加载页面后要直连另外三台的 agent，没有这两
# 个响应头全是 "TypeError: Failed to fetch"（首轮部署实测踩过）。
# 允许头里必须显式含 authorization：allow_headers=["*"] 在带凭据语义的头集合里
# 规范上不覆盖 authorization，Chrome/Firefox 会预检失败。
from fastapi.middleware.cors import CORSMiddleware
try:
    # starlette≥0.38 才有 allow_private_network：Chrome 的本地网络访问(LNA/PNA)
    # 预检带 Access-Control-Request-Private-Network，不答 true 直接 ERR_FAILED
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type", "authorization"],
        allow_private_network=True,
    )
except TypeError:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type", "authorization"],
    )


# ---------- 下载状态（与 manage.sh list / menu.py 同口径：includes 逐条命中） ----------

def llm_downloaded(name):
    for suffix in ("", "-fp8"):
        d = os.path.join(ROOT, "models/llm", name + suffix)
        if os.path.isdir(d) and os.listdir(d):
            return True
    return False


def comfy_downloaded(entry):
    """口径与 menu.py video_downloaded / manage.sh vhas 三处同源（改动须同批）：
    模式取 basename（common 段 includes 带 repo 路径，落盘是裸文件名）+
    大小写不敏感（官方文件有 InfiniTetalk 这类大小写笔误，敏感匹配会漏判）。"""
    d = os.path.join(ROOT, "models/comfy", entry.get("dest", ""))
    if not os.path.isdir(d):
        return False
    files = [f.lower() for f in os.listdir(d)]
    pats = [os.path.basename(p).lower() for p in (entry.get("includes") or ["*"])]
    return all(any(fnmatch.fnmatch(f, p) for f in files) for p in pats)


def image_downloaded(entry):
    dest = entry.get("dest") or {}
    return bool(dest) and all(os.path.isfile(os.path.join(ROOT, "models/comfy", dd, ff))
                              for dd, ff in dest.items())


def catalog():
    man = load_manifest()
    out = {"llm": [], "video": [], "image": [], "common": []}
    for n, d in man.get("llm", {}).items():
        out["llm"].append({
            "name": n, "engine": d.get("engine", "vllm"),
            "variants": d.get("variants", {}), "min_gpus": d.get("min_gpus", 1),
            "downloaded": llm_downloaded(n), "deprecated": bool(d.get("deprecated")),
            "desc": d.get("desc", ""), "brief": d.get("menu_brief", ""),
            "ctx": int(d.get("gw", {}).get("max_len") or d.get("max_ctx") or 0)})
    for n, d in man.get("video", {}).items():
        out["video"].append({
            "name": n, "downloaded": comfy_downloaded(d), "dest": d.get("dest"),
            "vram_480p": d.get("vram_480p_fp8"), "vram_720p": d.get("vram_720p_bf16"),
            "vram_480p_peak": d.get("vram_480p_fp8_peak_gb"),
            "vram_720p_peak": d.get("vram_720p_bf16_peak_gb"), "desc": d.get("desc", "")})
    for n, d in man.get("image", {}).items():
        out["image"].append({"name": n, "size_gb": d.get("size_gb"),
                             "downloaded": image_downloaded(d), "desc": d.get("desc", "")})
    for n, d in man.get("common", {}).items():
        out["common"].append({"name": n, "downloaded": comfy_downloaded(d),
                              "desc": d.get("desc", n)})
    return out


# ---------- 任务 ----------

def spawn_job(kind, argv, note=""):
    jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:4]
    log = os.path.join(JOBS_DIR, jid + ".log")
    f = open(log, "w", encoding="utf-8")
    f.write(f"$ {' '.join(argv)}\n{note}\n" + "=" * 60 + "\n")
    f.flush()
    proc = subprocess_spawn(argv, f)
    # proc 对象必须存进登记：job_state 靠 poll() 判 running/success/failed，
    # 只存 pid 的话任务永远显示 unknown（e2e 实测踩过）。JSON 出口统一 pop 掉。
    jobs[jid] = {"id": jid, "kind": kind, "cmd": " ".join(argv), "note": note,
                 "proc": proc, "pid": proc.pid, "started": time.time(), "log": log}
    return jid


def subprocess_spawn(argv, logfile):
    import subprocess
    return subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=ROOT, env=env_for_scripts(),
                            start_new_session=True)


def job_state(j):
    st = dict(j)
    st.pop("log", None)
    st.pop("proc", None)
    if "proc" in j:
        rc = j["proc"].poll()
        if rc is None:
            st["status"] = "running"
        else:
            st["status"] = "success" if rc == 0 else f"failed(exit {rc})"
            st["exit"] = rc
            j.pop("proc", None)
    else:
        st["status"] = "unknown"
    st["log_size"] = os.path.getsize(j["log"])
    return st


def env_summary():
    """.env 里的布局现状（AI_MODE/LLM_NAME/LLM_VARIANT/LLM_MAX_LEN/LLM_DEVICES/
    LLM_TP_SIZE/VIDEO_NAME/COMFY_ENABLED…）供 GUI 展示；LLM_API_KEY 一律不回显。"""
    out = {}
    try:
        with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                k, _, v = line.partition("=")
                k = k.strip()
                if k in ("AI_MODE", "AI_GPU_COUNT", "LLM_NAME", "LLM_ENGINE", "LLM_VARIANT",
                         "LLM_VRAM_GB", "LLM_TP_SIZE", "LLM_MEM_UTIL", "LLM_REPLICAS",
                         "LLM_MAX_LEN", "LLM_DEVICES", "COMFY_ENABLED", "COMFY_DEVICES",
                         "VIDEO_NAME", "VIDEO_MAX_RES"):
                    out[k] = v.strip().strip('"')
    except OSError:
        pass
    return out


# ---------- 可行组合（与 menu.py 互推面板同一口径的展示性复刻） ----------
# 边界声明：这里是"组合建议"的估算复刻（不写盘、不改摆放），判档规则与
# menu.video_tier / hwprofile.budgets 一致：实测峰值优先，slack 只打折声明值。
# 权威落盘永远由宿主 menu.py 完成；组合被采纳时给的仍是宿主命令。

def video_fit(vd, free_gb, slack=12):
    """返回 (档位 or None, 需求GB, 是否实测值)"""
    meas720 = vd.get("vram_720p_bf16_peak_gb")
    meas480 = vd.get("vram_480p_fp8_peak_gb")
    n720 = meas720 or vd["vram_720p_bf16"]
    n480 = meas480 or vd["vram_480p_fp8"]
    if free_gb >= n720:
        return "720p", n720, meas720 is not None
    if free_gb >= n480:
        return "480p", n480, meas480 is not None
    if not meas480 and free_gb >= n480 - slack:
        return "480p+offload", n480, False
    return None, n480, meas480 is not None


def combos():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from hwprofile import budgets          # 预算派生与 menu.py 同一实现，绝不另写一份
    man = load_manifest()
    import subprocess as _sp
    out = _sp.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True)
    cards = [int(float(x)) for x in out.stdout.split()]
    if not cards:
        return {"error": "nvidia-smi 无输出"}
    gmem = round(min(cards) / 1024)
    n = len(cards)
    bud = budgets(man["meta"], gmem)
    env = env_summary()
    free_video = bud["video_card"]
    half = round(gmem * float(man["meta"].get("dual_llm_util", 0.88)))

    def llm_state(nm, d):
        """网关实际会加载的变体（-fp8 目录优先，同 menu/model_dir 口径）+ 体量"""
        v = d.get("variants") or {}
        chosen = None
        if os.path.isdir(os.path.join(ROOT, "models/llm", nm + "-fp8")) and "fp8" in v:
            chosen = "fp8"
        else:
            for cand in ("bf16",):
                if cand in v:
                    chosen = cand
                    break
            chosen = chosen or (min(v, key=lambda k: v[k]) if v else None)
        return chosen, (v.get(chosen) if chosen else 0), llm_downloaded(nm)

    res = []
    for nm, d in man.get("llm", {}).items():
        variant, wgb, have = llm_state(nm, d)
        if not variant:
            continue
        engine = d.get("engine", "vllm")
        entry = {"llm": nm, "variant": variant, "weights_gb": wgb, "downloaded": have,
                 "engine": engine,
                 "ctx_single": int(d.get("gw", {}).get("max_len") or 0),
                 "combos": []}
        fits_single = wgb and wgb <= (half if n >= 2 else bud["single_llm"])
        if engine == "vllm" and n >= 2 and fits_single:
            slot = {"layout": f"单卡（GPU[{env.get('LLM_DEVICES','1')}]）+ 另一卡给视频/生图",
                    "kind": "dual", "video_budget_gb": free_video}
            vids = []
            for vn, vd in man.get("video", {}).items():
                tier, need, measured = video_fit(vd, free_video)
                vids.append({"model": vn, "tier": tier, "need_gb": need,
                             "measured": measured,
                             "downloaded": comfy_downloaded(vd), "desc": vd.get("desc", "")})
            slot["video"] = sorted(vids, key=lambda x: (x["tier"] is None, -(x["need_gb"] or 0)))
            imgs = []
            for inn, idd in man.get("image", {}).items():
                need = float(idd.get("size_gb") or 0)
                imgs.append({"model": inn, "fits": 0 < need <= free_video,
                             "need_gb": need, "downloaded": image_downloaded(idd),
                             "desc": idd.get("desc", "")})
            slot["image"] = imgs
            entry["combos"].append(slot)
        elif engine == "llamacpp" and wgb <= bud.get("pool_gb", gmem * n * 0.92) or \
                (engine == "vllm" and wgb and wgb > half):
            entry["combos"].append({"layout": "独占全部卡（无视频组合）",
                                    "kind": "exclusive", "video_budget_gb": 0,
                                    "video": [], "image": []})
        if n == 1:
            fv = max(0, gmem - wgb - float(man["meta"].get("safety_gap_gb", 4)) - 2)
            entry["combos"].append({"layout": f"单卡机分时共享（LLM 后余 ~{fv:.0f}G）",
                                    "kind": "single", "video_budget_gb": round(fv),
                                    "video": [{"model": vn, "tier": video_fit(vd, fv)[0],
                                               "need_gb": video_fit(vd, fv)[1],
                                               "measured": video_fit(vd, fv)[2],
                                               "downloaded": comfy_downloaded(vd),
                                               "desc": vd.get("desc", "")}
                                              for vn, vd in man.get("video", {}).items()],
                                    "image": []})
        res.append(entry)
    return {"cards": {"count": n, "gmem_gb": gmem, "video_card_gb": free_video,
                      "half_budget_gb": half},
            "current": {"llm": env.get("LLM_NAME"), "mode": env.get("AI_MODE"),
                        "video": env.get("VIDEO_NAME", "none"),
                        "comfy": env.get("COMFY_ENABLED")},
            "note": "档位=按另一卡预算判档（实测峰值优先）；采纳组合执行对应 "
                    "start.sh --llm X --video Y，独占布局组合为空",
            "nodes": res}


# ---------- API ----------

@app.get("/api/combos")
async def api_combos():
    return await asyncio.to_thread(combos)


@app.get("/")
async def index():
    # 不用 FileResponse（依赖可选的 aiofiles）；index 就 2KB，直接吐字节
    with open(os.path.join(WEB, "index.html"), "rb") as f:
        return Response(f.read(), media_type="text/html; charset=utf-8")


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/api/health")
async def health():
    return {"node": os.uname().nodename, "kind": "ai-stack",
            "host_root": ROOT, "ts": time.time()}


@app.get("/api/snapshot")
async def snapshot():
    """一屏数据：sysmon 快照 + 网关状态/版本 + 磁盘 + 任务摘要 + 模型目录。
    任何一源挂了只置 null 并带 error，不影响整屏（GUI 常对着半死的节点渲染）。"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(4.0)) as c:
        async def grab(url):
            try:
                r = await c.get(url)
                return r.json() if r.status_code == 200 else {"__error__": f"HTTP {r.status_code}"}
            except Exception as e:
                return {"__error__": str(e)[:200]}
        sm, gwst, gcap = await asyncio.gather(
            grab(f"{SYSMON}/api/snapshot"), grab(f"{GW}/v1/gateway/status"),
            grab(f"{GW}/v1/gateway/capabilities"))
    du = shutil.disk_usage(ROOT)
    return {"ts": time.time(), "node": os.uname().nodename, "env": env_summary(),
            "sysmon": None if "__error__" in sm else sm,
            "sysmon_error": sm.get("__error__"),
            "gateway": None if "__error__" in gwst else gwst,
            "gateway_error": gwst.get("__error__"),
            "gateway_version": (gcap.get("version") if "__error__" not in gcap else None),
            "gateway_layout_modes": bool((gcap.get("capabilities") or {}).get("layout_modes"))
                                    if "__error__" not in gcap else False,
            "disk": {"total_gb": round(du.total / 2**30), "used_gb": round(du.used / 2**30),
                     "free_gb": round(du.free / 2**30)},
            "jobs": [job_state(j) for j in list(jobs.values())][-12:]}


@app.get("/api/catalog")
async def api_catalog():
    return await asyncio.to_thread(catalog)


@app.get("/api/modes")
async def api_modes(model: str = ""):
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(f"{GW}/v1/gateway/modes", params={"model": model})
        return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/gateway/{action}")
async def api_gateway_post(action: str, request: Request):
    if action not in GW_POST:
        return JSONResponse({"error": {"message": f"action 不在白名单: {action}"}}, 404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(f"{GW}/v1/gateway/{action}", json=body)
        try:
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception:
            return JSONResponse({"error": {"message": r.text[:300]}}, 502)


@app.get("/api/gateway/{path:path}")
async def api_gateway_get(path: str):
    p = "/" + path
    if not GW_GET.match(p):
        return JSONResponse({"error": {"message": f"path 不在白名单: {p}"}}, 404)
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.get(f"{GW}/v1/gateway{p}")
        return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/api/relay")
async def relay(node: str, path: str):
    """异构节点（dgx-spark-pair 等没有本 agent 的机器）的只读代理：
    上游与路径都在 console.json 白名单里，不接受任意 URL。"""
    peer = RELAY.get(node)
    if not peer:
        return JSONResponse({"error": f"未知 relay 节点: {node}"}, 404)
    allow = peer.get("allow_paths", ["/v1/models", "/api/snapshot"])
    if not any(path.startswith(a) for a in allow):
        return JSONResponse({"error": f"path 不在白名单: {path}"}, 404)
    base = peer.get("gateway" if path.startswith("/v1") else "sysmon", "")
    if not base:
        return JSONResponse({"error": "该节点未配置对应上游"}, 400)
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(base + path)
            return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, 502)


@app.post("/api/download")
async def api_download(request: Request):
    b = await request.json()
    name = b.get("name", "")
    variant = b.get("variant", "")
    man = await asyncio.to_thread(load_manifest)
    known = any(name in man.get(k, {}) for k in ("llm", "video", "image", "common"))
    if not known:
        return JSONResponse({"error": {"message": f"清单里没有 '{name}'"}}, 400)
    argv = ["bash", "scripts/manage.sh", "download", name] + ([variant] if variant else [])
    if variant and variant != "fp8":
        return JSONResponse({"error": {"message": "variant 仅支持 fp8"}}, 400)
    jid = spawn_job("download", argv, note=f"模型 {name}" + (f"（{variant}）" if variant else ""))
    return {"job": jid, "status": "started"}


@app.post("/api/verify")
async def api_verify(request: Request):
    b = await request.json()
    argv = ["bash", "scripts/manage.sh", "verify"]
    model = b.get("model", "")
    if model:
        argv.append(model)
    return {"job": spawn_job("verify", argv, note=f"网关实测 {model or '全部'}"), "status": "started"}


@app.post("/api/apply")
async def api_apply(request: Request):
    """持久布局：等价 `start.sh --llm X [--video Y] [--tp N | --replicas N] -y`，
    跑本机 menu.py（.env/models.json 的唯一写者）。dry=1 只回命令与现状，不执行。"""
    b = await request.json()
    llm = b.get("llm", "")
    man = await asyncio.to_thread(load_manifest)
    if llm not in man.get("llm", {}):
        return JSONResponse({"error": {"message": f"LLM '{llm}' 不在清单"}}, 400)
    argv = ["python3", "scripts/menu.py", "--llm", llm]
    if "video" in b:
        argv += ["--video", str(b["video"]) if b["video"] else "none"]
    for flag, key in (("--tp", "tp"), ("--replicas", "replicas"), ("--variant", "variant")):
        if b.get(key):
            v = str(b[key])
            if key != "variant" and not v.isdigit():
                return JSONResponse({"error": {"message": f"{key} 必须是整数"}}, 400)
            if key == "variant" and not re.fullmatch(r"[A-Za-z0-9._-]+", v):
                return JSONResponse({"error": {"message": "variant 含非法字符"}}, 400)
            argv += [flag, v]
    argv.append("-y")
    plan = {"command": " ".join(shlex.quote(x) for x in argv),
            "cwd": ROOT,
            "effect": "重写 .env / compose.selected.yaml 并按本机显存落 gw（gpus/util/max_len/…）；"
                      "已加载后端不变，网关下个请求按新配置拉起"}
    if b.get("dry"):
        return plan
    return {"job": spawn_job("apply", argv, note=plan["command"]),
            "status": "started", "plan": plan}


@app.post("/api/init")
async def api_init(request: Request):
    """整机初始化/一键部署——重操作，前端必须二次确认；模式白名单，不透传任意参数。"""
    b = await request.json()
    mode = b.get("mode", "")
    flag = {"minimal": "--minimal", "full": "--full", "start_only": "--no-start"}.get(mode)
    if not flag:
        return JSONResponse({"error": {"message": "mode ∈ minimal|full|start_only"}}, 400)
    return {"job": spawn_job("init", ["bash", "scripts/init.sh", flag],
                             note=f"init.sh {flag}（新机初始化，可能触发大批下载）"),
            "status": "started"}


@app.get("/api/jobs")
async def api_jobs():
    return [job_state(j) for j in list(jobs.values())][-30:]


@app.get("/api/jobs/{jid}/log")
async def api_job_log(jid: str, tail: int = 4000):
    if not re.fullmatch(r"[\w-]+", jid or ""):
        return JSONResponse({"error": "bad id"}, 400)
    j = jobs.get(jid)
    path = os.path.join(JOBS_DIR, jid + ".log")
    if not os.path.isfile(path):
        return JSONResponse({"error": "无此任务（可能属于 agent 重启前的历史）" if not j else "日志缺失"}, 404)
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max(200, min(200000, tail))))
        data = f.read().decode("utf-8", "replace")
    state = job_state(j) if j else {"id": jid, "status": "detached"}
    return {"job": state, "log": data, "size": size}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("CONSOLE_PORT", "9001")), log_level="warning")
