#!/usr/bin/env python3
"""本机硬件探测 + 派生显存预算：替代 models.json 里那批"参考机调出来的绝对 GB"。

为什么要有这个文件：网关容器只 bind-mount 了 models.json，读不到宿主 .env，所以任何
按硬件定的决定最后都必须落到 models.json；而 meta 里的 gpu_mem_gb=96、video_reserve_gb=8
是在参考机（2×RTX PRO 6000，单卡 96GB）上定的量纲。同一份数搬到 2×RTX4090（单卡 48GB）
上原样生效，video_tier() 就按 48-8=40GB 的预算，拒绝了一个实测单卡峰值 44.8GB 已经跑通
的任务。规则改成：能派生的一律从探测结果派生，meta 只在"人类显式写过、且与本机量纲不
冲突"时作为覆盖项——细节见 budgets()。

本模块只做只读探测和纯计算，不写文件、不起容器；写 models.json 由调用方决定。
"""
import json
import os
import re
import subprocess

# meta 里的绝对 GB 来自单卡 96GB 的参考机。判断某个 reserve 到底是"人类给本机刻意留的
# 安全垫"还是"大卡机的遗留值"，用派生 margin 的 2 倍作分界：
#   96GB → margin 5、可信上限 10：meta 的 8 采纳（参考机行为一点不变，视频预算仍是 88）
#   48GB → margin 2、可信上限  4：meta 的 8 占单卡 16.7%，是遗留值，退回 margin（视频预算 46）
# 不这么做的话，48GB 机上 8GB reserve 会把实测 44.8GB 就跑通的任务判成"放不下"——
# 那正是本仓库真机踩过的误判。被忽略的 meta 键会列在 budgets() 返回的 ignored 里，
# 由调用方打印出来：安全垫可以按本机重算，但不能悄悄重算。
_RESERVE_TRUST_RATIO = 2


def write_manifest_inplace(path, man):
    """就地截断写 models.json：不 tmp+rename，也不补结尾换行。

    compose.yaml 是以"单个文件"bind-mount ./models.json → /opt/models.json，rename 会换
    inode，运行中的容器就一直挂在被弃用的旧 inode 上——宿主改完、网关永远读不到（实测
    宿主 186048 / 容器 900000）。仓库原有的写者（check_llm_config.py:151、manage.sh add、
    menu.py 的 gw 回写）都是 open(path,"w") 截断写，这里把那份约定收拢成一处；
    不补换行是免得 git 把整份文件记成改动。
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return path


def run(cmd, timeout=20):
    """跑一条命令拿 stdout；命令不存在/超时/非零退出都返回 None，由调用方决定降级方式。"""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _total_ram_gb():
    """MemTotal（kB）→ GB；容器/非 Linux 环境下拿不到就 0.0，不参与显存判定。"""
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except (OSError, ValueError):
        pass
    out = run(["sysctl", "-n", "hw.memsize"])  # 开发机（macOS）兜底，只为让 detect() 可跑通
    if out and out.strip().isdigit():
        return int(out.strip()) / 1024.0 ** 3
    return 0.0


def _cores():
    out = run(["nproc"])
    if out and out.strip().isdigit():
        return int(out.strip())
    return os.cpu_count() or 0


def detect():
    """nvidia-smi + nproc + /proc/meminfo 的只读画像。

    同型号假设沿用 menu 原来的口径（"同型号卡取首卡"）；型号不一致（混插）时取最小的
    那张卡并标 mixed=True——预算按最小的卡算才不会放行一个只有大卡放得下的模型。
    nvidia-smi 缺失时抛 RuntimeError，怎么退出由调用方定（menu 是 sys.exit，calib 是报错）。
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
    except OSError:
        raise RuntimeError("找不到 nvidia-smi：NVIDIA 驱动未安装或不在 PATH。"
                           "摆放决策要按本机显存算，缺它没法算。")
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi 查询失败（rc={proc.returncode}）："
                           f"{(proc.stderr or '').strip()[:200]}")
    cards = []
    for line in (proc.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        # 卡名里可能带逗号，所以两头按位置取、中间全部并回名字
        try:
            cards.append({"index": int(parts[0]),
                          "name": " ".join(parts[1:-2]),
                          "total": int(parts[-2]), "free": int(parts[-1])})
        except ValueError:
            continue
    if not cards:
        raise RuntimeError("nvidia-smi 有输出但解析不到 GPU 行，无法确定本机显存")
    names = sorted({c["name"] for c in cards})
    return {
        "gpu_count": len(cards),
        "gmem_mib": min(c["total"] for c in cards),
        "gmem_gb": min(c["total"] for c in cards) / 1024.0,
        "free_mib": {c["index"]: c["free"] for c in cards},
        "names": names,
        "mixed": len(names) > 1,
        "ram_gb": _total_ram_gb(),
        "cores": _cores(),
    }


def card_margin_gb(gmem_gb, ratio=0.05):
    """单卡必须留给"别的东西"的 GB：驱动 + CUDA context + 碎片，按 5% 派生，最低 2GB。

    实测：48GB 卡上 vLLM 用 --gpu-memory-utilization 0.92（留 3.8GB）跑得好好的，
    ComfyUI 峰值 44.8GB 也还在 48GB 内；4090 上驱动+X 大约吃 0.5~1GB，2GB 是下限而不是估计。
    """
    return max(2, int(round(gmem_gb * ratio)))


def budgets(meta, gmem_gb):
    """按单卡容量派生各角色预算（GB）。meta 只提供"显式覆盖"，缺省就派生。

    - llm_card：独占一整张卡给 LLM 时可用的显存 = 卡容量 - margin。
      对应现在 dual 分支硬编码的 mem_util 0.92（96GB → 88.3，派生 91 更贴实测）。
    - video_card：独占一整张卡给 ComfyUI 时可用的显存。参考机沿用 meta 的 8GB reserve
      得到 88（与今天完全一致，不收紧）；48GB 机派生 margin 2 得到 46，
      于是 minimax-h3 实测 45.7GB 的单卡峰值变成"可 480p"而不是"放不下"。
    - single_llm：单卡机上 LLM 与视频共存时给 LLM 的预算，沿用现有语义
      （meta.single_llm_budget_gb 存在就是人类定的门槛，照用），否则派生 llm_card。
    返回 dict：llm_card / video_card / single_llm / margin + overridden（取自 meta 的键）
    + ignored（因量纲不符被退回派生值的 meta 键，调用方要打印，不能静默）。
    """
    meta = meta or {}
    margin = card_margin_gb(gmem_gb)
    derived_card = gmem_gb - margin
    out = {"margin": margin, "llm_card": derived_card, "video_card": derived_card,
           "single_llm": derived_card}
    overridden, ignored = [], []

    reserve = meta.get("video_reserve_gb")
    if reserve is not None and reserve > margin:
        if reserve <= margin * _RESERVE_TRUST_RATIO:
            out["video_card"] = gmem_gb - reserve
            overridden.append("video_card")
        else:
            ignored.append("video_reserve_gb")

    if meta.get("single_llm_budget_gb") is not None:
        out["single_llm"] = meta["single_llm_budget_gb"]
        overridden.append("single_llm")

    out["overridden"] = overridden
    out["ignored"] = ignored
    return out


OWN_PREFIXES = ("ai-", "bench-")
# 系统辅助容器白名单：manage.sh 的下载容器（用完即走、不占显存）。
# 只列确实由本系统创建/复用的镜像，别把"来路不明的容器"洗进白名单。
HELPER_IMAGES = {"python:3.12-slim"}


def _owned_pids():
    """本系统自己起的容器（ai-/bench- 前缀）里的进程 PID 集合。拿不到 docker 返回 None。"""
    names = run(["docker", "ps", "--format", "{{.Names}}"])
    if names is None:
        return None
    pids = set()
    for nm in names.split():
        if not nm.startswith(OWN_PREFIXES):
            continue
        top = run(["docker", "top", nm, "-o", "pid"])
        if top is None:
            continue
        for tok in top.split()[1:]:  # 首行是表头 "PID"
            if tok.isdigit():
                pids.add(int(tok))
    return pids


def _proc_info(pid):
    """ps -o pid,user,etime,args -p <pid> → dict；进程已消失返回 None。"""
    out = run(["ps", "-o", "pid=,user=,etime=,args=", "-p", str(pid)])
    if not out or not out.strip():
        return {}
    parts = out.strip().split(None, 3)
    if len(parts) < 4:
        return {}
    _pid, user, etime, args = parts
    return {"user": user, "etime": etime, "args": args,
            "comm": os.path.basename(args.split()[0]) if args.split() else args}


def foreign_occupancy():
    """审计"本系统之外的显存持有者和容器"。只检测，绝不 kill / 不 stop。

    为什么值得专门做一次：菜单是按"整张卡减预算"决定放哪个模型的，卡上要是有别的
    进程先咬走 20GB，算出来的 max_len/util 就全错——真机上这正是那次"模型放不下"误判
    的成因（其实是别人的容器在跑）。
    返回 list[dict]，每项都有 owned 字段：
      {"kind":"gpu_process", pid, user, etime, comm, args, mem_mib, owned:False}
      {"kind":"container",   container, image, status, owned:False}
      {"kind":"notice", msg, owned:True}   # docker 不可用之类的降级说明（不算违规项）
    已停掉的容器不入列：不占显存的东西列出来只是噪声。
    """
    res = []
    apps = run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"])
    if apps is None:
        res.append({"kind": "notice", "owned": True,
                    "msg": "nvidia-smi --query-compute-apps 不可用，跳过 GPU 进程审计"})
        return res
    gpu_procs = []
    for line in apps.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            gpu_procs.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue

    owned = _owned_pids()
    if owned is None:
        res.append({"kind": "notice", "owned": True,
                    "msg": "docker CLI 不可用或 /var/run/docker.sock 无权限："
                           "只做了 GPU 进程审计，容器归属无法判断，GPU 进程全部按外来处理"})
    for pid, mib in gpu_procs:
        if owned and pid in owned:
            continue
        info = _proc_info(pid)
        res.append({"kind": "gpu_process", "pid": pid, "mem_mib": mib, "owned": False,
                    "user": info.get("user", "?"), "etime": info.get("etime", "?"),
                    "comm": info.get("comm", "?"), "args": info.get("args", "?")})

    if owned is not None:
        names = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"])
        for line in (names or "").splitlines():
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            cname, image, status = cols[0], cols[1], cols[2]
            if cname.startswith(OWN_PREFIXES) or image in HELPER_IMAGES:
                continue
            res.append({"kind": "container", "container": cname, "image": image,
                        "status": status, "owned": False})
    return res


_KV_MEM_RE = re.compile(r"Available KV cache memory:\s*([\d.,]+)\s*GiB", re.I)
# vLLM 把这两句打在相邻的两行日志上（gpu_worker.py / kv_cache_utils.py），
# 所以分开匹配、别指望一个正则跨行；数字带千位分隔符，统一去掉再转数。
_KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d.,]+)\s*tokens", re.I)
_KV_CONC_RE = re.compile(
    r"Maximum concurrency for\s*([\d.,]+)\s*tokens per request:\s*([\d.,]+)\s*x", re.I)


def _last_num(rx, text):
    hits = rx.findall(text or "")
    if not hits:
        return None
    tok = hits[-1]
    if isinstance(tok, tuple):
        tok = tok[0]
    try:
        return float(str(tok).replace(",", ""))
    except ValueError:
        return None


def _last_num2(rx, text):
    hits = rx.findall(text or "")
    if not hits:
        return None
    return tuple(float(str(t).replace(",", "")) for t in hits[-1])


def kv_density_from_log(container, logs=None):
    """从后端容器日志里量出"每 GB KV 能装多少 token"。

    logs 参数是给测试注入文本用的口子，日常只传容器名。日志里没有这两行（模型还没
    加载完 / 不是 vLLM）返回 None，调用方自己决定是重试还是放弃猜测。
    取最后一次命中：容器重启过就有好几轮加载，最新那轮才对应现在的配置。
    """
    if logs is None:
        # 只 tail 尾部：KV 那两行在加载时打，但加载后可能已经跑了几千行请求日志，
        # 全量 docker logs 在一个长期运行的容器上能拉出上百 MB
        logs = run(["docker", "logs", "--tail", "20000", container], timeout=120) or ""
    kv = _last_num(_KV_MEM_RE, logs)
    tokens = _last_num(_KV_TOKENS_RE, logs)
    if not kv or not tokens:
        return None
    out = {"tokens_per_gb": int(round(tokens / kv)), "kv_gib": kv,
           "kv_tokens": int(tokens)}
    conc = _last_num2(_KV_CONC_RE, logs)
    if conc:
        out["max_model_len"] = int(conc[0])
        out["concurrency"] = conc[1]
    return out


def video_peak_from_samples(samples, seconds=None):
    """把"每秒一次的每卡显存采样"收成单卡峰值。

    samples: {卡号: [MiB, ...]} 或 [[MiB...], [MiB...]]（外层一维是卡）。
    peak_gb 只做 round(0.1)；写盘那个"向上取整到 0.1"由调用方定，因为要不要留余量
    是准入策略，不是测量结果。
    """
    if isinstance(samples, dict):
        series = {int(k): [int(v) for v in vals] for k, vals in samples.items() if vals}
    else:
        series = {i: [int(v) for v in vals] for i, vals in enumerate(samples) if vals}
    peak_mib = {i: max(vals) for i, vals in series.items()}
    return {"peak_mib": peak_mib,
            "peak_gb": {i: round(m / 1024.0, 1) for i, m in peak_mib.items()},
            "seconds": int(seconds) if seconds is not None
                       else max((len(v) for v in series.values()), default=0)}
