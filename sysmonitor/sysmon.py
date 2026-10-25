#!/usr/bin/env python3
"""nvtop-style 图形化系统资源监控 - curses TUI（当前值仪表版）。

只呈现「此刻」的压力与负载，不画时间曲线：所有百分比都是即采即画的水平仪表条，
▰=已占用 / ▱=余量；压力类指标（算力/带宽/CPU/交换/磁盘）按值分级着色
绿 <60% ≤ 黄 <85% ≤ 红，容量类（显存/内存）用固定青蓝，一眼区分"忙"与"占"。
仪表动效（A 键可全关，关闭即回退到精确静态值）：
  - 平滑过渡：采样 tick 间显示值以 ease-out 滑行（~0.5s 收敛），数字/条/颜色同步
  - 微光扫掠：已填充段有一个亮格自左向右漂移（btop 式 shimmer，窄条自动跳过）
  - 流向箭头：网速 ↓入/↑出 箭头在有流量时交替闪动，指示方向流动
数据采样与动画刷新分离（画面 ~8fps）：
  - 顶部：每块 GPU 一个面板，温度/功耗/风扇 + 算力/显存/带宽 三条当前值仪表
  - 中部：LLM 模型面板——每模型两行：阶段动画（Prompt 进度条 / 生成脉冲 / 加载转圈）、
          当前 t/s 与 p/s 速率数值、slot 占用表+排队、累计 tokens；
          面板头汇总全局 t/s、p/s、请求数与排队数
  - CPU 总体仪表 + 每核方块图 + 每核占用数字表；右侧内存/交换/磁盘仪表
  - 网络面板：各活动网卡实时 ↓入/↑出 速率 + 累计流量（按链路速率归一化仪表）
  - 磁盘 I/O 面板：物理盘的 util 繁忙度仪表 + 读写吞吐/IOPS（busy_time 差分）
  - 底部：全系统进程表（GPU/PID/USER/%CPU/%MEM/显存/RSS/显存仪表/模型/命令行），
          可用 `P` 键在 显存/CPU/MEM 三种排序间循环，占用高的行分级着色

快捷键：
    q / Esc    退出
    Space      暂停/继续数据采样（数据与动效一起冻结便于读数，墙钟继续）
    A          开/关动画（脉冲、转圈、spinner）
    P          进程表排序：显存 → %CPU → %MEM 循环
    T          下一主题    Shift+Tab  上一主题

用法：
    python3 sysmon.py          # TUI 图形界面
    python3 sysmon.py -i 0.5   # 自定义数据刷新间隔（秒）
"""

from __future__ import annotations

import argparse
import curses
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request

import psutil

# ─── 配置 ──────────────────────────────────────────────────────
MIN_COLS = 100
MIN_ROWS = 30
ANIM_TICK = 0.15          # 动画帧间隔（秒）
DEFAULT_GATEWAY = os.environ.get("SYSMON_GATEWAY", "http://localhost:8000")

# ─── 动画字形 ──────────────────────────────────────────────────
PULSE = "●○"                              # 生成中呼吸点
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"                # braille 转圈（Prompt 预填充）
LOAD_SPIN = "◐◓◑◒"                        # 加载转圈
METER_ON = "▰"                            # 当前值仪表：已占用
METER_OFF = "▱"                           # 当前值仪表：余量
ARROW_IN = "↓⇣"                           # 网速箭头：入（有流量时交替=流动感）
ARROW_OUT = "↑⇡"                          # 网速箭头：出
GLINT_STEP = 2                            # 仪表微光每帧推进格数（扫掠速度）
LERP = 0.4                                # 仪表值平滑系数（越大越跟手）
_FRAME = 0                                # 当前动画帧（draw() 每帧更新）
try:
    _KEY_BTAB = curses.KEY_BTAB          # Shift+Tab（部分终端/仿真环境可能缺失）
except AttributeError:
    _KEY_BTAB = None

# ─── 文本工具 ──────────────────────────────────────────────────

def disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)

def clip(s: str, width: int) -> str:
    """按显示宽度截断，超出部分用 … 表示。"""
    if disp_width(s) <= width:
        return s
    cut = 0; w = 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width - 1:
            break
        cut += 1; w += cw
    return s[:cut] + "…"

def pad(s: str, w: int) -> str:
    return s + " " * max(0, w - disp_width(s))

def rpad(s: str, w: int) -> str:
    return " " * max(0, w - disp_width(s)) + s

def fmt_bytes(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024

def fmt_mb(mb: float) -> str:
    return fmt_bytes(mb * 1024 * 1024)

def fmt_int(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{int(n):,}"

def fmt_tokens(n: float | None) -> str:
    """token 数量紧凑格式：单位 K=千 tokens、M=百万 tokens。"""
    if n is None:
        return "n/a"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"

def fmt_dur(secs: float | None) -> str:
    """空闲/运行时长紧凑格式：12s / 3m12s / 2h05 / 5d03h。"""
    if secs is None or secs < 0:
        return "-"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}"
    return f"{secs // 86400}d{(secs % 86400) // 3600:02d}h"

# ─── LLM token 指标 ───────────────────────────────────────────
_llm_baseline: dict[str, tuple[float, float]] = {}
_last_active: dict[str, float] = {}     # 每模型最后一次「有活动」的时刻
GATEWAY_CONNECTED = False


def get_llm_models(gateway: str = DEFAULT_GATEWAY, timeout: float = 0.6) -> list[dict]:
    """从网关读取已加载模型，本地差分当前速率并记录空闲时长。"""
    global GATEWAY_CONNECTED
    if not gateway:
        GATEWAY_CONNECTED = False
        return []
    try:
        with _no_proxy_opener.open(gateway.rstrip("/") + "/v1/gateway/status",
                                   timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        GATEWAY_CONNECTED = True
    except Exception:
        GATEWAY_CONNECTED = False
        return []

    now = time.time()
    models = []
    for name, b in (data.get("backends") or {}).items():
        tokens = b.get("tokens") or {}
        total = tokens.get("generation_total")
        rate = None
        if isinstance(total, (int, float)):
            prev = _llm_baseline.get(name)
            if prev:
                dt = now - prev[0]
                delta = total - prev[1]
                if dt > 0 and delta >= 0:
                    rate = delta / dt
            _llm_baseline[name] = (now, float(total))
        else:
            _llm_baseline.pop(name, None)
        gw_rate = tokens.get("generation_rate")
        if (rate is None or rate <= 0) and isinstance(gw_rate, (int, float)):
            rate = gw_rate
        rate = float(rate or 0.0)
        active = b.get("in_flight", 0) or 0
        prefill = bool(tokens.get("prefill_active"))
        p_rate = float(tokens.get("prompt_rate") or tokens.get("prefill_rate") or 0.0)
        # 防御：旧版网关会把已完成请求的 prefill_rate 永久锁存。in_flight=0 且
        # prefill_active=false 时，非零 p/s 只可能是残留读数，忽略它——否则空闲时长
        # 被顶在 0，面板一直画 Prompt。prefill_active=true 是网关侧的合法短期保持
        # （PREFILL_HOLD_SEC，突发 prefill 只存在一次采样），原样透传。
        if p_rate > 0 and not prefill and active <= 0:
            p_rate = 0.0
        if active > 0 or rate > 0 or p_rate > 0:
            _last_active[name] = now
        idle = now - _last_active[name] if name in _last_active else None

        gpus = b.get("gpus") or []
        models.append({
            "name": name,
            "engine": b.get("engine", "?"),
            "gpus": ",".join(str(g) for g in gpus) if gpus else "-",
            "active": active,
            "slots": b.get("slots"),
            "status": b.get("status", "?"),
            "prompt_total": tokens.get("prompt_total"),
            "generation_total": total,
            "rate": rate,
            "prompt_rate": p_rate,
            "prefill_active": prefill,
            "prefill_tokens": tokens.get("prefill_tokens"),
            "prefill_progress": tokens.get("prefill_progress"),
            "prefill_elapsed": tokens.get("prefill_elapsed"),
            "idle": idle,
            "metrics_ok": bool(tokens.get("metrics_ok")),
        })
    # 清理已卸载模型的跟踪状态
    live = {m["name"] for m in models}
    for gone in set(_llm_baseline) - live:
        _llm_baseline.pop(gone, None)
        _last_active.pop(gone, None)
    models.sort(key=lambda m: (m["active"] <= 0, -(m["rate"] or 0), m["name"]))
    return models

_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ─── 当前值仪表 ────────────────────────────────────────────────
# 所有百分比都画成一行式仪表：标签 + 数值 + ▰▱ 条 + 右侧明细，只呈现此刻的值。
# 压力类（算力/带宽/CPU/交换/磁盘）按值分级着色：绿 <60 ≤黄 <85 ≤红；
# 容量类（显存/内存）固定青蓝系——模型常驻显存是"占"不是"忙"，避免误读为过载。
GAUGE_LABEL_W = 12        # 标签+数值固定列宽（"算力 100.0%"）
GAUGE_METER_MIN = 8       # 仪表条最小保留宽度
_gauge_shown: dict[str, float] = {}   # 平滑显示值（key -> 上次画出的 %）

def put_meter(stdscr, y, x, w, frac, pair, glint=False):
    """水平仪表条：▰=当前占用 / ▱=余量（暗色）。

    glint：开动画时在已填充段让一个亮格自左向右扫掠（微光流动感），
    其余格略暗形成对比；窄条（<10 格）或填充 <3 格时自动跳过。"""
    if w <= 0:
        return
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * w))
    if glint and ANIM and n >= 3 and w >= 10:
        put(stdscr, y, x, METER_ON * n, cpair(pair) | curses.A_DIM, w)
        put(stdscr, y, x + (_FRAME * GLINT_STEP) % n, METER_ON, cpair(pair, True), 1)
    else:
        put(stdscr, y, x, METER_ON * n, cpair(pair, True), w)
    put(stdscr, y, x + n, METER_OFF * (w - n), cpair(PAIR_BORDER), w - n)

def put_gauge(stdscr, y, x, w, label, pct, detail="", pair=None, key=None):
    """一行当前值仪表：左=标签+百分比（着色），中=▰▱仪表条，右=暗色明细。

    key：开动画时显示值向采样值 ease-out 滑行（~0.5s 收敛），数字/条/颜色同步；
    动画关或暂停时直接跳到真实值，读数不失真。"""
    if key is not None and ANIM and not PAUSED:
        prev = _gauge_shown.get(key)
        v = pct if prev is None else prev + (pct - prev) * LERP
        if abs(pct - v) < 0.05:
            v = pct
        _gauge_shown[key] = v
    else:
        v = pct
    pair = pair if pair is not None else color_for(v)
    head = pad(label, 4) + f" {v:5.1f}%"
    put(stdscr, y, x, clip(head, GAUGE_LABEL_W), cpair(pair, True), GAUGE_LABEL_W)
    dw = disp_width(detail) if detail else 0
    dw = min(dw, max(0, w - GAUGE_LABEL_W - GAUGE_METER_MIN - 2))
    mw = max(0, w - GAUGE_LABEL_W - 2 - (dw + 1 if dw else 0))
    put_meter(stdscr, y, x + GAUGE_LABEL_W + 1, mw, v / 100, pair, glint=True)
    if dw:
        put(stdscr, y, x + GAUGE_LABEL_W + 2 + mw, clip(detail, dw),
            cpair(PAIR_DIM), dw)

# ─── 模型识别 ─────────────────────────────────────────────────
MODEL_FLAGS = (
    "--model", "--model-path", "--model-name", "--served-model-name",
    "--ckpt", "--checkpoint", "-m",
)

def guess_model(args: list[str]) -> str | None:
    """从命令行参数猜模型名：优先 --model 类参数，其次路径形态的位置参数。"""
    for i, a in enumerate(args):
        if a in MODEL_FLAGS and i + 1 < len(args):
            return args[i + 1].rstrip("/").split("/")[-1]
        for flag in MODEL_FLAGS:
            if a.startswith(flag + "="):
                return a.split("=", 1)[1].rstrip("/").split("/")[-1]
    for i, a in enumerate(args):
        if i == 0 or args[i - 1].startswith("-") or "=" in a:
            continue
        if "/" in a and not a.endswith(".py") and not a.startswith(("/usr", "/opt", "/bin", "/data")):
            return a.rstrip("/").split("/")[-1]
    return None

def describe_proc(pid: int) -> tuple[str, str | None]:
    """返回 (命令行, 模型名)。worker 类进程沿父进程链找完整命令。"""
    try:
        proc = psutil.Process(pid)
        args = [a for a in proc.cmdline() if a.strip()]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"<exited> PID{pid}", None

    if len(args) <= 1:  # 如 VLLM::Worker_TP0，命令行本身不含模型信息
        try:
            for parent in proc.parents():
                p_args = [a for a in parent.cmdline() if a.strip()]
                if len(p_args) > len(args):
                    return " ".join(p_args), guess_model(p_args)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return (" ".join(args) or f"PID{pid}"), guess_model(args)

# ─── 数据采集 ─────────────────────────────────────────────────

_proc_objs: dict[int, psutil.Process] = {}

def proc_stats(pid: int) -> dict:
    """进程累计指标。%CPU 依赖跨刷新周期采样，故缓存 Process 对象。"""
    out = {"cpu": 0.0, "rss": 0.0, "pmem": 0.0, "user": "?", "up": ""}
    proc = _proc_objs.get(pid)
    if proc is None:
        try:
            proc = psutil.Process(pid)
            proc.cpu_percent(None)          # 首次调用建立基线
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return out
        _proc_objs[pid] = proc
    try:
        out["cpu"] = proc.cpu_percent(None)
        info = proc.memory_info()
        out["rss"] = info.rss
        out["pmem"] = proc.memory_percent()
        out["user"] = proc.username()[:8]
        up = time.time() - proc.create_time()
        d, up = divmod(int(up), 86400)
        h, up = divmod(up, 3600)
        m, s = divmod(up, 60)
        out["up"] = f"{d}d{h:02d}h" if d else (f"{h}:{m:02d}" if h else f"{m}:{s:02d}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        _proc_objs.pop(pid, None)
    return out

def get_gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    fields = (
        "index,uuid,name,utilization.gpu,utilization.memory,memory.used,"
        "memory.total,temperature.gpu,power.draw,power.limit,fan.speed"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        return []

    def num(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        gpus.append({
            "index": int(parts[0]), "uuid": parts[1], "name": parts[2],
            "util": num(parts[3]) or 0.0,
            "mem_util": num(parts[4]) or 0.0,
            "mem_used": num(parts[5]) or 0.0,
            "mem_total": num(parts[6]) or 1.0,
            "temp": int(num(parts[7]) or 0),
            "power": num(parts[8]) or 0.0,
            "power_limit": num(parts[9]) or 0.0,
            "fan": num(parts[10]),
            "procs": [],
        })

    # 占用显存的进程
    by_uuid = {g["uuid"]: g for g in gpus}
    try:
        pout = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if pout.returncode == 0:
            for pl in pout.stdout.strip().splitlines():
                pp = [p.strip() for p in pl.split(",")]
                if len(pp) < 3:
                    continue
                g = by_uuid.get(pp[0])
                if not g:
                    continue
                try:
                    pid = int(pp[1])
                except ValueError:
                    continue
                mem_mb = num(pp[2])
                cmd, model = describe_proc(pid)
                g["procs"].append({
                    "pid": pid, "cmd": cmd, "model": model,
                    "mem": mem_mb, "gpu": g["index"], **proc_stats(pid),
                })
    except (subprocess.TimeoutExpired, OSError):
        pass

    # 清理已消失的进程缓存
    live = {p["pid"] for g in gpus for p in g["procs"]}
    for dead in set(_proc_objs) - live:
        _proc_objs.pop(dead, None)

    for g in gpus:
        g["procs"].sort(key=lambda p: p["mem"] or 0, reverse=True)
    return gpus


def get_cpu_info() -> dict:
    freq = psutil.cpu_freq()
    try:
        load = psutil.getloadavg()
    except (AttributeError, OSError):
        load = None
    return {
        "total": psutil.cpu_percent(interval=None),
        "per_core": psutil.cpu_percent(interval=None, percpu=True),
        "freq": freq.current if freq else 0,
        "load": load,
        "count": psutil.cpu_count(logical=True) or 1,
    }


def get_mem_info() -> dict:
    return {"vm": psutil.virtual_memory(), "sm": psutil.swap_memory()}


def get_disk_info() -> dict:
    try:
        du = psutil.disk_usage(os.environ.get("SYSMON_DISK_PATH", "/"))
    except OSError:
        du = None
    read_bps = write_bps = 0.0
    global _disk_cache
    try:
        io = psutil.disk_io_counters()
    except (AttributeError, OSError):
        io = None
    if io:
        now = time.time()
        cur = (io.read_bytes, io.write_bytes, now)
        if _disk_cache:
            dt = now - _disk_cache[2]
            if dt > 0:
                read_bps = (cur[0] - _disk_cache[0]) / dt
                write_bps = (cur[1] - _disk_cache[1]) / dt
        _disk_cache = cur
    return {"du": du, "read_bps": read_bps, "write_bps": write_bps}

_disk_cache: tuple = ()


def get_net_info() -> list[dict]:
    """各网卡当前收发速率与累计字节（差分自上次采样）。

    排除 lo/veth/br-/docker 等虚拟口；满格刻度 = 链路速率（未知时取本批最活跃口）。
    """
    now = time.time()
    try:
        per = psutil.net_io_counters(pernic=True)
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
    except (AttributeError, OSError):
        return []
    nics = []
    for name, io in per.items():
        if name.startswith(("lo", "veth", "br-", "docker", "virbr")):
            continue
        st = stats.get(name)
        if st and not st.isup:
            continue
        prev = _net_cache.get(name)
        rx = tx = 0.0
        if prev:
            dt = now - prev[2]
            if dt > 0:
                rx = max(0.0, (io.bytes_recv - prev[0]) / dt)
                tx = max(0.0, (io.bytes_sent - prev[1]) / dt)
        _net_cache[name] = (io.bytes_recv, io.bytes_sent, now)
        ip = ""
        for a in addrs.get(name, []):
            if a.family == socket.AF_INET:
                ip = a.address
                break
        nics.append({"name": name, "rx": rx, "tx": tx,
                     "rx_total": io.bytes_recv, "tx_total": io.bytes_sent,
                     "link_mbps": (st.speed if st else 0),   # 以太网为 Mb/s
                     "ip": ip})
    nics.sort(key=lambda n: -(n["rx"] + n["tx"]))
    return nics

_net_cache: dict[str, tuple[int, int, float]] = {}


_DISK_NAME_RE = re.compile(r"(nvme\d+n\d+|mmcblk\d+|sd[a-z]|vd[a-z]|hd[a-z])$")


def get_diskio_info() -> list[dict]:
    """物理盘当前 I/O：吞吐、IOPS、利用率%（busy_time 差分——真正的磁盘压力）。"""
    now = time.time()
    try:
        per = psutil.disk_io_counters(perdisk=True)
    except (AttributeError, OSError):
        return []
    out = []
    for name, v in (per or {}).items():
        if not _DISK_NAME_RE.fullmatch(name):
            continue
        prev = _diskio_cache.get(name)
        r = w = ri = wi = util = 0.0
        if prev:
            dt = now - prev[0]
            if dt > 0:
                r = max(0.0, (v.read_bytes - prev[1]) / dt)
                w = max(0.0, (v.write_bytes - prev[2]) / dt)
                ri = max(0.0, (v.read_count - prev[3]) / dt)
                wi = max(0.0, (v.write_count - prev[4]) / dt)
                util = min(100.0, max(0.0, (v.busy_time - prev[5]) / (dt * 1000) * 100))
        _diskio_cache[name] = (now, v.read_bytes, v.write_bytes,
                               v.read_count, v.write_count, v.busy_time)
        out.append({"name": name, "r_bps": r, "w_bps": w,
                    "r_iops": ri, "w_iops": wi, "util": util})
    out.sort(key=lambda d: -(d["util"] * 1e6 + d["r_bps"] + d["w_bps"]))
    return out

_diskio_cache: dict[str, tuple] = {}


def get_proc_list() -> list[dict]:
    """全系统进程当前指标（%CPU 需跨采样差分，首次出现记 0）。

    缓存 Process 对象与 create_time；后者变化 = pid 被复用，重建基线。
    跳过内核线程（无命令行）。
    """
    out = []
    seen = set()
    for pid in psutil.pids():
        try:
            p = _proc_all.get(pid)
            if p is None:
                p = psutil.Process(pid)
                cmd = [a for a in p.cmdline() if a.strip()]
                if not cmd:
                    continue
                _proc_all[pid] = p
                _proc_ctime[pid] = p.create_time()
                p.cpu_percent(None)          # 建立差分基线，本轮 %CPU 记 0
                cpu = 0.0
            else:
                if p.create_time() != _proc_ctime.get(pid):
                    _proc_all.pop(pid, None)     # pid 复用：下一轮重建
                    _proc_ctime.pop(pid, None)
                    continue
                cmd = [a for a in p.cmdline() if a.strip()]
                if not cmd:
                    continue
                cpu = p.cpu_percent(None)
            mi = p.memory_info()
            out.append({"pid": pid, "cpu": cpu, "rss": mi.rss,
                        "pmem": p.memory_percent(), "user": p.username()[:8],
                        "cmd": " ".join(cmd)})
            seen.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            _proc_all.pop(pid, None)
            _proc_ctime.pop(pid, None)
    for dead in set(_proc_all) - seen:
        _proc_all.pop(dead, None)
        _proc_ctime.pop(dead, None)
    return out

_proc_all: dict[int, psutil.Process] = {}
_proc_ctime: dict[int, float] = {}


def get_uptime() -> float:
    try:
        return time.time() - psutil.boot_time()
    except OSError:
        return 0.0

# ─── 颜色 ──────────────────────────────────────────────────────
# 语义角色（全程序统一，颜色只是这些角色的具体呈现）：
#   HEADER  面板标题条 / 顶栏          黑字 + 青色底
#   BORDER  分隔线 / 仪表条余量▱       暗灰（弱化结构线）
#   DIM     次要信息：单位、明细、注释  中灰
#   ACCENT  表头 / 标签                柔和青蓝
#   OK      正常 / 生成阶段 t/s / 低负载仪表   绿
#   WARN    注意 / Prompt 处理 p/s / 中负载仪表 琥珀（忙碌但非异常）
#   CRIT    告警：高温 / 利用率 ≥85% 仪表      红
#   GPU     GPU 面板标题（多卡按下面板循环）蓝紫青系
#   MEM     显存/内存仪表（容量占用类，非压力）青蓝
#   CPU     CPU 相关数值                          绿
#   DISK    磁盘/交换数值                         土黄
#   MODEL   GPU 上进程的模型名标签      紫
(PAIR_HEADER, PAIR_DIM, PAIR_BORDER, PAIR_OK, PAIR_WARN, PAIR_CRIT,
 PAIR_GPU, PAIR_MEM, PAIR_CPU, PAIR_DISK, PAIR_MODEL, PAIR_ACCENT) = range(1, 13)

_ROLE_PAIR = {
    "header": PAIR_HEADER, "border": PAIR_BORDER, "dim": PAIR_DIM,
    "accent": PAIR_ACCENT, "ok": PAIR_OK, "warn": PAIR_WARN, "crit": PAIR_CRIT,
    "mem": PAIR_MEM, "cpu": PAIR_CPU, "disk": PAIR_DISK, "model": PAIR_MODEL,
}
_LOGICAL_C = {
    "black": curses.COLOR_BLACK, "red": curses.COLOR_RED,
    "green": curses.COLOR_GREEN, "yellow": curses.COLOR_YELLOW,
    "blue": curses.COLOR_BLUE, "magenta": curses.COLOR_MAGENTA,
    "cyan": curses.COLOR_CYAN, "white": curses.COLOR_WHITE,
}

# 运行时状态（main/render 里填充）
THEME = None
ANIM = True              # 动画开关（A 键）
PAUSED = False           # 暂停数据采样（Space 键）
GPU_COLORS = (_LOGICAL_C["blue"], _LOGICAL_C["magenta"],
             _LOGICAL_C["cyan"], _LOGICAL_C["green"])

def init_colors(theme=None):
    import theme as _theme
    theme = theme or THEME or _theme.DEFAULT_THEME
    curses.start_color()
    curses.use_default_colors()
    hi = curses.COLORS is not None and curses.COLORS >= 256
    colors = theme["colors"]
    for role, pair in _ROLE_PAIR.items():
        spec = colors.get(role)
        if hi:
            if role == "header" and isinstance(spec, list):
                fg, bg = spec[0], spec[1]
            else:
                fg, bg = (spec if isinstance(spec, list) else spec), -1
        else:
            base = _theme.BASIC_COLORS.get(role, "white")
            if role == "header":
                bf, bb = base
                fg, bg = _LOGICAL_C[bf], _LOGICAL_C[bb]
            else:
                fg, bg = _LOGICAL_C[base], -1
        curses.init_pair(pair, fg, bg)
    # 动态配色：GPU 面板色（pair 20+i）与亮色版本（pair 40+i）
    global GPU_COLORS
    if hi:
        GPU_COLORS = tuple(theme.get("gpu") or _theme.DEFAULT_THEME["gpu"])
    else:
        GPU_COLORS = tuple(_LOGICAL_C[n] for n in _theme.BASIC_GPU)
    for i, fg in enumerate(GPU_COLORS):
        curses.init_pair(20 + i, fg, -1)
        curses.init_pair(40 + i, fg, -1)

def cpair(pair: int, bold: bool = False) -> int:
    try:
        a = curses.color_pair(pair)
        return curses.A_BOLD | a if bold else a
    except curses.error:
        return curses.A_NORMAL

def color_for(pct: float) -> int:
    if pct >= 85:
        return PAIR_CRIT
    if pct >= 60:
        return PAIR_WARN
    return PAIR_OK

def temp_color(t: int) -> int:
    if t >= 85:
        return PAIR_CRIT
    if t >= 70:
        return PAIR_WARN
    return PAIR_OK

# ─── 绘制辅助 ──────────────────────────────────────────────────

def put(stdscr, y, x, text, attr=curses.A_NORMAL, maxw=None):
    """安全写字符串，越界自动截断。"""
    rows, cols = stdscr.getmaxyx()
    if y < 0 or y >= rows or x >= cols:
        return
    if maxw is not None:
        cols = min(cols, x + maxw)
    if x < 0:
        text = text[-x:]; x = 0
    try:
        stdscr.addnstr(y, x, text, max(0, cols - x), attr)
    except curses.error:
        pass

def vline(stdscr, x, y0, y1, ch="│"):
    for y in range(y0, y1):
        put(stdscr, y, x, ch, cpair(PAIR_BORDER))

def anim_char(seq: str, frame: int) -> str:
    return seq[frame % len(seq)] if ANIM else seq[0]

# ─── GPU 面板 ─────────────────────────────────────────────────

def panel_gpu(stdscr, y, x, w, gpu):
    """单卡面板：标题 + 状态行 + 算力/显存/带宽 三条当前值仪表。共 5 行。"""
    gi = gpu["index"]
    gc = 20 + (gi % len(GPU_COLORS))

    # 标题行
    title = f"[GPU {gi}] {gpu['name']}"
    put(stdscr, y, x, clip(title, w), cpair(gc, True), w)

    # 状态行：温度(分级着色) / 功耗 / 风扇
    y += 1
    cx = x
    if gpu["temp"]:
        t = f"{gpu['temp']}°C"
        put(stdscr, y, cx, t, cpair(temp_color(gpu["temp"]), True), disp_width(t))
        cx += disp_width(t) + 2
    if gpu["power_limit"]:
        p = f"{gpu['power']:.0f}/{gpu['power_limit']:.0f}W"
        put(stdscr, y, cx, p, cpair(PAIR_DIM), len(p))
        cx += len(p) + 2
    if gpu.get("fan") is not None:
        f = f"风扇 {gpu['fan']:.0f}%"
        put(stdscr, y, cx, f, cpair(PAIR_DIM), disp_width(f))

    mem_pct = gpu["mem_used"] / gpu["mem_total"] * 100
    free = gpu["mem_total"] - gpu["mem_used"]

    # 当前值仪表：算力/带宽=压力分级色，显存=容量色（青蓝）
    y += 1
    put_gauge(stdscr, y, x, w, "算力", gpu["util"], key=f"gu{gi}u")
    y += 1
    put_gauge(stdscr, y, x, w, "显存", mem_pct,
              f"{fmt_mb(gpu['mem_used'])}/{fmt_mb(gpu['mem_total'])} "
              f"空闲{fmt_mb(free)}", pair=PAIR_MEM, key=f"gu{gi}m")
    y += 1
    put_gauge(stdscr, y, x, w, "带宽", gpu.get("mem_util") or 0.0, key=f"gu{gi}b")
    return y + 1

# ─── LLM token 面板 ───────────────────────────────────────────

def panel_llm(stdscr, y, x, w, models, max_rows, frame):
    """每模型两行：第一行 阶段动画+Prompt 进度；第二行 双速率曲线+速率数值+累计。
    面板头汇总全局吞吐；返回下一可用行。
    """
    total_t = sum(float(m.get("rate") or 0) for m in models)
    total_p = sum(float(m.get("prompt_rate") or 0) for m in models)
    total_req = sum(int(m["active"]) for m in models) if models else 0
    queued = sum(1 for m in models if m.get("slots") and m["active"] > m["slots"])

    # ── 面板头（聚合信息）──
    put(stdscr, y, x, " LLM ", cpair(PAIR_HEADER, True), 5)
    if models:
        cx = x + 6
        head = f"Σ {total_t:6.1f} t/s"
        put(stdscr, y, cx, head, cpair(PAIR_OK, True), disp_width(head))
        cx += disp_width(head) + 2
        if total_p > 0:
            s = f"{total_p:6.0f} p/s"
            put(stdscr, y, cx, s, cpair(PAIR_WARN, True), disp_width(s))
            cx += disp_width(s) + 2
        s = f"请求 {total_req}" + (f" ▲{queued}排队" if queued else "")
        put(stdscr, y, cx, s, cpair(PAIR_ACCENT, True), disp_width(s))
        pulse = anim_char(PULSE, frame // 2) if (total_t > 0 or total_p > 0) else "○"
        put(stdscr, y, x + max(0, w - 8), f"{pulse} 网关",
            cpair(PAIR_OK if GATEWAY_CONNECTED else PAIR_CRIT, True), 6)
    else:
        msg = (" 网关离线/不可达" if not GATEWAY_CONNECTED
               else " 已连接，暂无已加载模型")
        put(stdscr, y, x + 6, clip(msg, w - 6), cpair(PAIR_DIM), w - 6)
    y += 1

    if not models:
        return y + 1

    for m in models[:max(0, max_rows)]:
        prefill = bool(m.get("prefill_active"))
        p_rate = float(m.get("prompt_rate") or 0)
        t_rate = float(m.get("rate") or 0)
        decoding = t_rate > 0 or (m["active"] > 0 and not prefill)
        loading = m["status"] == "loading"
        slots = m.get("slots")
        queueing = bool(slots and m["active"] > slots)
        prog_raw = m.get("prefill_progress")   # None=引擎不报单请求进度（vLLM），只显速率不画条
        prog = (prog_raw or 0) * 100

        # ── 第一行：模型名 + 引擎/GPU + 阶段动画 + Prompt 进度 ──
        name = clip(m["name"], 30)
        put(stdscr, y, x + 2, name, cpair(PAIR_MODEL, True), disp_width(name))
        cx = x + 2 + disp_width(name) + 1
        meta = f"{('vLLM' if m['engine'] == 'vllm' else 'llama')} ·GPU{m['gpus']}"
        put(stdscr, y, cx, meta, cpair(PAIR_DIM), disp_width(meta))
        cx += disp_width(meta) + 2

        elapsed = m.get("prefill_elapsed")
        if prefill:
            sp = anim_char(SPINNER, frame)
            seg = f"{sp} Prompt "
            put(stdscr, y, cx, seg, cpair(PAIR_WARN, True), disp_width(seg))
            cx += disp_width(seg)
            head = ""
            if prog_raw is not None:
                barw = max(6, min(14, x + w - cx - 18))
                fn = int(round(max(0.0, min(1.0, prog_raw)) * barw))
                put(stdscr, y, cx, METER_ON * fn, cpair(PAIR_WARN, True), barw)
                put(stdscr, y, cx + fn, METER_OFF * (barw - fn), cpair(PAIR_BORDER),
                    barw - fn)
                cx += barw + 1
                head = f"{prog:3.0f}% "
            ptoks = m.get("prefill_tokens")
            tail = head + (f"{fmt_tokens(ptoks)}tok {float(elapsed):.0f}s"
                           if elapsed and ptoks is not None
                           else fmt_tokens(ptoks) if ptoks is not None else "")
            tw = disp_width(tail)
            put(stdscr, y, min(cx, x + w - tw - 1), tail, cpair(PAIR_WARN, True), tw)
        elif loading:
            sp = anim_char(LOAD_SPIN, frame)
            put(stdscr, y, cx, f"{sp} 模型加载中…", cpair(PAIR_ACCENT, True),
                disp_width("⠿ 模型加载中…"))
        elif decoding:
            dot = anim_char(PULSE, frame)
            put(stdscr, y, cx, f"{dot} 生成中", cpair(PAIR_OK, True),
                disp_width("● 生成中"))
        elif m["idle"] is not None:
            put(stdscr, y, cx, f"空闲 {fmt_dur(m['idle'])}", cpair(PAIR_DIM),
                disp_width("空闲 0m00s"))
        else:
            put(stdscr, y, cx, "空闲", cpair(PAIR_DIM), 4)
        y += 1

        # ── 第二行：当前速率数值 + slot 占用仪表 + 累计 ──
        cx = x + 2
        put(stdscr, y, cx, "t/s", cpair(PAIR_OK, True), 3)
        t_txt = f"{t_rate:7.1f}" if t_rate > 0 else "     --"
        put(stdscr, y, cx + 4, t_txt,
            cpair(PAIR_OK, True) if t_rate > 0 else cpair(PAIR_DIM), 7)
        cx += 13
        put(stdscr, y, cx, "p/s", cpair(PAIR_WARN, True), 3)
        p_txt = f"{p_rate:6.0f}" if p_rate > 0 else "    --"
        put(stdscr, y, cx + 4, p_txt,
            cpair(PAIR_WARN, True) if p_rate > 0 else cpair(PAIR_DIM), 6)
        cx += 12

        put(stdscr, y, cx, "ACT", cpair(PAIR_DIM, True), 3)
        cx += 4
        if slots:
            mw = 6
            pair = PAIR_WARN if queueing else (PAIR_OK if m["active"] > 0 else PAIR_MEM)
            put_meter(stdscr, y, cx, mw, m["active"] / slots if slots else 0.0, pair)
            cx += mw + 1
            put(stdscr, y, cx, f"{m['active']}/{slots}", cpair(pair, True), 5)
            cx += 6
            if queueing:
                q = anim_char(PULSE, frame // 3)
                seg = f"{q}+{m['active'] - slots}排队"
                put(stdscr, y, cx, seg, cpair(PAIR_WARN, True), disp_width(seg))
                cx += disp_width(seg) + 1
        else:
            put(stdscr, y, cx, str(m["active"]),
                cpair(PAIR_OK, True) if m["active"] > 0 else cpair(PAIR_MEM), 3)
            cx += 4

        put(stdscr, y, cx,
            f"Σprompt {fmt_tokens(m.get('prompt_total'))}", cpair(PAIR_DIM), 18)
        cx += 18
        put(stdscr, y, cx,
            f"Σgen {fmt_tokens(m.get('generation_total'))}", cpair(PAIR_DIM), 15)
        y += 1

    hidden = len(models) - max_rows
    if hidden > 0:
        put(stdscr, y, x, f"… 另有 {hidden} 个模型未显示", cpair(PAIR_DIM), w)
        y += 1

    # 图例
    lx = x + 2
    legend = (
        (cpair(PAIR_OK), "●"), (cpair(PAIR_DIM), "生成 "),
        (cpair(PAIR_WARN), "⠋"), (cpair(PAIR_DIM), "Prompt "),
        (cpair(PAIR_OK, True), "t/s"), (cpair(PAIR_DIM), "=生成速率 "),
        (cpair(PAIR_WARN, True), "p/s"), (cpair(PAIR_DIM), "=Prompt速率 "),
        (cpair(PAIR_MEM), "▰▱"), (cpair(PAIR_DIM), " ACT=slot占用(超则排队)  "),
        (cpair(PAIR_OK), "▰"), (cpair(PAIR_DIM), " 绿<60 "),
        (cpair(PAIR_WARN), "▰"), (cpair(PAIR_DIM), " 黄<85 "),
        (cpair(PAIR_CRIT), "▰"), (cpair(PAIR_DIM), " 红≥85  "),
        (cpair(PAIR_DIM), "Σ=加载以来累计  K=千 M=百万tok"),
    )
    for a, seg in legend:
        if lx - x + disp_width(seg) > w:
            break                                   # 窄屏：图例装不下就截断
        put(stdscr, y, lx, seg, a, disp_width(seg))
        lx += disp_width(seg)
    return y + 2

# ─── CPU 面板 ─────────────────────────────────────────────────

def panel_cpu(stdscr, y, x, w, cpu, room=99):
    """CPU 总体仪表 + 每核方块图（均为当前值）+ 每核占用数字表。"""
    label_w = 13
    chart_w = max(10, w - label_w - 1)
    top = y

    extra = ""
    if cpu["load"]:
        extra = f"load {cpu['load'][0]:.2f} {cpu['load'][1]:.2f} {cpu['load'][2]:.2f}"
    if cpu["freq"]:
        extra += f"  {cpu['freq'] / 1000:.2f}GHz"
    put_gauge(stdscr, y, x, w, "CPU", cpu["total"], extra, key="cpu")

    # 每核方块图：一列一核，当前值 8 级
    LEVELS = " ▁▂▃▄▅▆▇█"
    cores = cpu["per_core"]
    y += 1
    put(stdscr, y, x, f"每核 {len(cores)}:", cpair(PAIR_DIM), label_w)
    for i, c in enumerate(cores[:chart_w]):
        put(stdscr, y, x + label_w + i, LEVELS[min(8, int(round(c / 100 * 8)))],
            cpair(color_for(c)), 1)

    # 每核数字表：用整列宽度排布，填满剩余空间
    y += 1
    cell_w = 11
    per_row = max(1, w // cell_w)
    used = y - top
    grid_rows = max(0, min(room - used, math.ceil(len(cores) / per_row)))
    for r in range(grid_rows):
        ry = y + r
        for j in range(per_row):
            i = r * per_row + j
            if i >= len(cores):
                break
            txt = f"{i:>2}: {cores[i]:5.1f}%"
            put(stdscr, ry, x + j * cell_w, txt, cpair(color_for(cores[i])), cell_w - 1)
    y += grid_rows

    if grid_rows and grid_rows < math.ceil(len(cores) / per_row):
        put(stdscr, y, x, f"（余下 {len(cores) - grid_rows * per_row} 核未显示，放大终端）",
            cpair(PAIR_DIM), w)
        y += 1
    return y

# ─── 系统面板（内存/交换/磁盘）────────────────────────────────

def panel_mem(stdscr, y, x, w, mem, disk):
    """内存 / 交换 / 磁盘 一行式当前值仪表。"""
    vm, sm = mem["vm"], mem["sm"]

    def block(row, label, pct, detail, pair=None, key=None):
        put_gauge(stdscr, row, x, w, label, pct, detail, pair=pair, key=key)

    block(y, "内存", vm.percent,
          f"{fmt_bytes(vm.used)}/{fmt_bytes(vm.total)}  可用 {fmt_bytes(vm.available)}",
          pair=PAIR_MEM, key="ram")
    y += 1
    block(y, "交换", sm.percent, f"{fmt_bytes(sm.used)}/{fmt_bytes(sm.total)}",
          key="swap")
    y += 1
    if disk["du"]:
        block(y, "磁盘", disk["du"].percent,
              f"{fmt_bytes(disk['du'].used)}/{fmt_bytes(disk['du'].total)}"
              f"  剩余 {fmt_bytes(disk['du'].free)}"
              f"  读 {fmt_bytes(disk['read_bps'])}/s 写 {fmt_bytes(disk['write_bps'])}/s",
              key="disk")
        y += 1
    return y + 1

# ─── 进程表 ────────────────────────────────────────────────────

# ─── 网络面板 ─────────────────────────────────────────────────

def panel_net(stdscr, y, x, w, nics):
    """进出口网速：每网卡当前 ↓入 ↑出 + mini 仪表。满格=链路速率（未知取本批最活跃）。"""
    head = "网络  ↓入 ↑出"
    put(stdscr, y, x, head, cpair(PAIR_ACCENT, True), disp_width(head))
    hint = "满格=带宽"
    put(stdscr, y, x + max(0, w - disp_width(hint)), hint, cpair(PAIR_DIM), disp_width(hint))
    y += 1
    if not nics:
        put(stdscr, y, x, "（无活动网卡）", cpair(PAIR_DIM), w)
        return y + 2
    rel = max([max(n["rx"], n["tx"]) for n in nics] + [100_000.0])
    for n in nics[:3]:
        scale = (n["link_mbps"] * 1_000_000 / 8) if n.get("link_mbps") else rel
        cx = x
        name = clip(n["name"], 12)
        put(stdscr, y, cx, pad(name, 13), cpair(PAIR_DIM), 13)
        cx += 13
        for glyph, rate, in_color in ((ARROW_IN, n["rx"], PAIR_OK),
                                      (ARROW_OUT, n["tx"], PAIR_MEM)):
            pct = rate / scale * 100 if scale > 0 else 0.0
            arrow = (glyph[(_FRAME // 2) % 2] if (ANIM and rate > 0)
                     else glyph[0])
            put(stdscr, y, cx, arrow, cpair(in_color, True), 1)
            put(stdscr, y, cx + 1, rpad(fmt_bytes(rate) + "/s", 8),
                cpair(PAIR_OK if pct >= 1 else PAIR_DIM), 8)
            put_meter(stdscr, y, cx + 10, 6, pct / 100, color_for(pct))
            cx += 17
        ip = n.get("ip") or ""
        if ip and cx - x + disp_width(ip) < w:
            put(stdscr, y, cx, ip, cpair(PAIR_DIM), x + w - cx)
        y += 1
    tot_d = sum(n["rx_total"] for n in nics)
    tot_u = sum(n["tx_total"] for n in nics)
    put(stdscr, y, x, f"Σ累计 ↓{fmt_bytes(tot_d)}  ↑{fmt_bytes(tot_u)}",
        cpair(PAIR_DIM), w)
    return y + 2

# ─── 磁盘 I/O 面板 ────────────────────────────────────────────

def panel_diskio(stdscr, y, x, w, disks):
    """物理盘当前吞吐 / IOPS / util（busy_time 差分——磁盘真正的压力指标）。"""
    head = "磁盘 I/O  util=繁忙度"
    put(stdscr, y, x, head, cpair(PAIR_ACCENT, True), disp_width(head))
    y += 1
    if not disks:
        put(stdscr, y, x, "（无磁盘 I/O 数据）", cpair(PAIR_DIM), w)
        return y + 1
    for d in disks[:3]:
        name = clip(d["name"], 9)
        put(stdscr, y, x, pad(name, 10), cpair(PAIR_DIM), disp_width(name))
        put_gauge(stdscr, y, x + 10, w - 10, "", d["util"],
                  f"读{fmt_bytes(d['r_bps'])}/s 写{fmt_bytes(d['w_bps'])}/s "
                  f"{d['r_iops'] + d['w_iops']:,.0f}io/s", key=f"dio{d['name']}")
        y += 1
    return y

# ─── 进程表（全系统，可排序）──────────────────────────────────

PROC_SORTS = ("vram", "cpu", "mem")
PROC_SORT_NAMES = {"vram": "显存", "cpu": "%CPU", "mem": "%MEM"}
PROC_SORT = 0                   # 运行时索引（P 键循环）


def cycle_proc_sort() -> str:
    global PROC_SORT
    PROC_SORT = (PROC_SORT + 1) % len(PROC_SORTS)
    return PROC_SORTS[PROC_SORT]


def panel_procs(stdscr, y, x, w, gpus, procs, max_rows, bar_w=14):
    """全宽进程表：GPU/PID/USER/%CPU/%MEM/RSS/VRAM + 显存仪表 + 模型/命令行。
    排序方式 P 键循环（显存→%CPU→%MEM），覆盖全系统进程。"""
    vram_by_pid = {p["pid"]: p for g in gpus for p in g["procs"]}
    gpu_total = {g["index"]: g["mem_total"] for g in gpus}
    rows = []
    for p in procs:
        g = vram_by_pid.get(p["pid"])
        rows.append({**p, "vram": g["mem"] if g else None,
                     "gpu": g["gpu"] if g else None,
                     "model": g.get("model") if g else None,
                     "cmd": g["cmd"] if g else p["cmd"]})
    mode = PROC_SORTS[PROC_SORT]
    if mode == "vram":
        rows.sort(key=lambda r: (-(r["vram"] or 0), -r["cpu"]))
    elif mode == "cpu":
        rows.sort(key=lambda r: -r["cpu"])
    else:
        rows.sort(key=lambda r: -r["pmem"])

    cols = [
        ("GPU",   3, lambda r: f"{r['gpu']:>3}" if r["gpu"] is not None else " · "),
        ("PID",   7, lambda r: f"{r['pid']:>7}"),
        ("USER",  7, lambda r: f"{r.get('user', '?'):>7}"),
        ("%CPU",  5, lambda r: f"{r['cpu']:>5.1f}"),
        ("%MEM",  5, lambda r: f"{r['pmem']:>5.1f}"),
        ("RSS",   6, lambda r: f"{fmt_bytes(r['rss']):>6}"),
        ("VRAM",  7, lambda r: f"{fmt_mb(r['vram']):>7}" if r["vram"] else "     · "),
    ]
    if w < 116:
        cols = [c for c in cols if c[0] not in ("USER", "RSS")]
    while w < sum(c[1] + 1 for c in cols) + bar_w + 21 and len(cols) > 3:
        cols.pop(-2 if cols[-1][0] == "VRAM" else -1)

    xs, acc = [], 0
    for name, cw, _ in cols:
        xs.append(acc)
        acc += cw + 1
    bar_x, cmd_x = acc, acc + bar_w + 1

    sort_col = {"vram": "VRAM", "cpu": "%CPU", "mem": "%MEM"}[mode]
    for cx, (name, _, _) in zip(xs, cols):
        put(stdscr, y, x + cx, name, cpair(PAIR_ACCENT, name == sort_col),
            max(0, w - cx))
    put(stdscr, y, x + bar_x, "占用", cpair(PAIR_ACCENT, True), max(0, w - bar_x))
    tag = f"模型 / 命令行（P排序:{PROC_SORT_NAMES[mode]}）"
    put(stdscr, y, x + cmd_x, tag, cpair(PAIR_ACCENT, True), max(0, w - cmd_x))
    y += 1

    for r in rows[:max(0, max_rows)]:
        mem_pct = ((r["vram"] or 0) / gpu_total[r["gpu"]] * 100) \
            if r["gpu"] is not None and gpu_total.get(r["gpu"]) else 0.0
        for cx, (name, cw, fmt) in zip(xs, cols):
            a = cpair(PAIR_DIM)
            if name == "%CPU" and r["cpu"] >= 60:
                a = cpair(color_for(r["cpu"]), True)
            elif name == "%MEM" and r["pmem"] >= 5:
                a = cpair(color_for(min(100.0, r["pmem"] * 2)), True)
            elif name == "VRAM" and r["vram"]:
                a = cpair(color_for(mem_pct), True)
            put(stdscr, y, x + cx, fmt(r), a, cw)
        if r["vram"]:
            put_meter(stdscr, y, x + bar_x, bar_w, mem_pct / 100,
                      color_for(mem_pct), glint=True)
        else:
            put(stdscr, y, x + bar_x, " " * bar_w, cpair(PAIR_DIM), bar_w)
        mtag = f"[{r['model']}] " if r["model"] else ""
        if mtag:
            put(stdscr, y, x + cmd_x, mtag, cpair(PAIR_MODEL, True), max(0, w - cmd_x))
        put(stdscr, y, x + cmd_x + len(mtag),
            clip(r["cmd"], w - cmd_x - len(mtag)), cpair(PAIR_DIM),
            max(0, w - cmd_x - len(mtag)))
        y += 1
    n_vram = sum(1 for r in rows if r["vram"])
    if not rows:
        put(stdscr, y, x, "（无用户进程）", cpair(PAIR_DIM), w)
        y += 1
    else:
        rest = len(rows) - max_rows
        put(stdscr, y, x,
            f"共 {len(rows)} 进程 · 排序:{PROC_SORT_NAMES[mode]}（P 切换）· "
            f"{n_vram} 个占显存" + (f" · 另有 {rest} 行未显示" if rest > 0 else ""),
            cpair(PAIR_DIM), w)
        y += 1
    return y

# ─── 采样与主渲染 ──────────────────────────────────────────────

def collect_samples(gateway, interval):
    """采集一轮数据（每数据周期只调用一次；全部为当前值，不留历史）。"""
    return {
        "gpus": get_gpu_info(),
        "models": get_llm_models(gateway, timeout=min(0.8, max(0.2, interval / 2))),
        "cpu": get_cpu_info(),
        "mem": get_mem_info(),
        "disk": get_disk_info(),
        "net": get_net_info(),
        "diskio": get_diskio_info(),
        "procs": get_proc_list(),
    }


def draw(stdscr, samples, interval, frame, theme_warn=None):
    global _FRAME
    _FRAME = frame
    rows, cols = stdscr.getmaxyx()
    gpus = samples["gpus"]
    llm_models = samples["models"]
    cpu = samples["cpu"]
    mem = samples["mem"]
    disk = samples["disk"]

    stdscr.erase()
    if rows < MIN_ROWS or cols < MIN_COLS:
        put(stdscr, 0, 0, f"终端太小: {cols}x{rows}（需要 ≥ {MIN_COLS}x{MIN_ROWS}）",
            cpair(PAIR_CRIT))
        stdscr.refresh()
        return

    # ── 标题栏（含全局吞吐聚合）──
    host = socket.gethostname()
    total_t = sum(float(m.get("rate") or 0) for m in llm_models)
    total_p = sum(float(m.get("prompt_rate") or 0) for m in llm_models)
    up = fmt_dur(get_uptime())
    bar = pad(f" SYSMON  {host}" + (f"  运行{up}" if up != "-" else "")
              + f"  {time.strftime('%Y-%m-%d %H:%M:%S')} ", cols)
    put(stdscr, 0, 0, bar, cpair(PAIR_HEADER, True), cols)
    if llm_models:
        tag = f"Σ {total_t:5.1f} t/s" + (f" · {total_p:4.0f} p/s " if total_p > 0 else " ")
        x0 = cols - disp_width(tag) - 1
        put(stdscr, 0, x0, tag, cpair(PAIR_HEADER, True), disp_width(tag))
        dot = anim_char(PULSE, frame // 2) if (total_t > 0 or total_p > 0) else "○"
        put(stdscr, 0, max(0, x0 - 2), dot,
            cpair(PAIR_OK if (total_t > 0 or total_p > 0) else PAIR_DIM, True), 1)

    # 终端高度自适应：每模型 2 行 + 面板头 + 图例
    llm_rows = min(len(llm_models), max(1, (rows - 22) // 2)) if llm_models else 1
    y = 1

    # ── GPU 区域：多卡并排 ──
    if gpus:
        pw = cols // len(gpus)
        gpu_h = 5
        for gi, g in enumerate(gpus):
            gx = gi * pw
            gw = (cols - gx) if gi == len(gpus) - 1 else pw - 1
            panel_gpu(stdscr, y, gx, gw, g)
        for gi in range(1, len(gpus)):
            vline(stdscr, gi * pw - 1, y, y + gpu_h)
        y += gpu_h
    else:
        put(stdscr, y, 0, " 未检测到 NVIDIA GPU", cpair(PAIR_WARN, True) | curses.A_REVERSE, cols)
        y += 2

    # ── LLM token 吞吐 ──
    if y < rows - 8:
        put(stdscr, y, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
        y = panel_llm(stdscr, y + 1, 0, cols - 1, llm_models, llm_rows, frame)

    # ── CPU | 系统·网络·磁盘IO 双列 ──
    half = cols // 2
    put(stdscr, y, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
    sep_y = y + 1
    n_nic = min(len(samples["net"]), 3)
    n_disk = min(len(samples["diskio"]), 3)
    # 右列总高：mem(3+空1) + net(头+nics+Σ+空) + diskio(头+disks)；左列 CPU 数字表补到同高
    right_h = 4 + (n_nic + 4) + (1 + n_disk)
    room = max(3, min(right_h, rows - 6 - sep_y))

    y_cpu = panel_cpu(stdscr, sep_y, 0, half - 1, cpu, room)
    rw = cols - half - 2
    ry = panel_mem(stdscr, sep_y, half + 1, rw, mem, disk)
    ry = panel_net(stdscr, ry, half + 1, rw, samples["net"])
    ry = panel_diskio(stdscr, ry, half + 1, rw, samples["diskio"])
    vline(stdscr, half, sep_y, min(max(y_cpu, ry), rows - 2))

    # ── 进程表（剩余空间不足则省略；预留表头+排序脚注 2 行）──
    y_tbl = max(y_cpu, ry)
    avail = rows - 1 - y_tbl - 2          # 减去分隔线、表头、状态栏
    if avail >= 4:
        put(stdscr, y_tbl, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
        panel_procs(stdscr, y_tbl + 1, 0, cols - 1, gpus, samples["procs"], avail - 2)

    # ── 状态栏 ──
    theme_name = (THEME or {}).get("name", "catppuccin-mocha")
    pause_tag = " ⏸已暂停 " if PAUSED else ""
    anim_tag = "动效开" if ANIM else "动效关"
    status = (f" q 退出  Space 暂停  A 动画  P 进程排序  T/⇧T 主题"
              f" | {pause_tag}{anim_tag}"
              f" 刷新 {interval:.1f}s 主题 {theme_name}"
              f" · ▰▱=当前值仪表 绿黄红=压力分级 蓝=容量占用")
    put(stdscr, rows - 1, 0, pad(status, cols - 1), cpair(PAIR_HEADER, True), cols - 1)
    if theme_warn:
        put(stdscr, rows - 1, max(0, cols - 1 - disp_width(theme_warn)),
            clip(theme_warn, cols - 1), cpair(PAIR_WARN, True), cols - 1)

    stdscr.refresh()


def _theme_names():
    import theme as _theme
    names = _theme._list_available_themes()
    if _theme.DEFAULT_THEME_NAME not in names:
        names.insert(0, _theme.DEFAULT_THEME_NAME)   # 内置兜底时也参与循环
    return names


def render(stdscr, interval, gateway=DEFAULT_GATEWAY, theme_warn=None):
    global THEME, ANIM, PAUSED, samples, theme_warn_c
    init_colors(THEME)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(120)                     # 动画 tick 粒度
    psutil.cpu_percent(interval=None)       # 预热

    frame = 0
    theme_warn_c = theme_warn
    next_anim = time.monotonic() + ANIM_TICK
    next_sample = 0.0                       # 立即采一次
    samples = None

    while True:
        now = time.monotonic()
        if samples is None or (not PAUSED and now >= next_sample):
            samples = collect_samples(gateway, interval)
            next_sample = time.monotonic() + interval
        if samples is not None and (now >= next_anim or PAUSED):
            if ANIM and not PAUSED:
                frame += 1
            next_anim = time.monotonic() + ANIM_TICK
            draw(stdscr, samples, interval, frame, theme_warn_c)

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 3, 27):  # q / Ctrl+C / Esc
            return
        elif key == ord("T"):                    # 下一主题
            _cycle_theme(+1, theme_warn_c)
            theme_warn_c = THEME_WARN_C[0]
        elif _KEY_BTAB is not None and key == _KEY_BTAB:   # Shift+Tab 上一主题
            _cycle_theme(-1, theme_warn_c)
            theme_warn_c = THEME_WARN_C[0]
        elif key == 32:                          # Space 暂停/继续
            PAUSED = not PAUSED
        elif key in (ord("A"), ord("a")):           # 动画开关
            ANIM = not ANIM
        elif key in (ord("P"),):                 # 进程表排序循环：显存→CPU→MEM
            cycle_proc_sort()
        elif key in (curses.KEY_RESIZE,):
            curses.update_lines_cols()
            samples = None                       # 立即重绘适配新尺寸


THEME_WARN_C = [None]


def _cycle_theme(step, _unused=None):
    import theme as _theme
    global THEME
    names = _theme_names()
    current = (THEME or {}).get("name")
    base = current.split(" ")[0] if current else names[0]
    try:
        idx = names.index(base)
    except ValueError:
        idx = 0
    new_name = names[(idx + step) % len(names)]
    THEME, warn = _theme.load_theme(new_name)
    init_colors(THEME)
    THEME_WARN_C[0] = warn

# ─── 入口 ──────────────────────────────────────────────────────

def main():
    import theme as _theme
    ap = argparse.ArgumentParser(description="nvtop-style GPU/CPU/LLM token 资源监控")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="数据刷新间隔秒（默认 1.0）")
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY,
                    help="模型网关地址（默认 %(default)s，也可用 SYSMON_GATEWAY）")
    ap.add_argument("--theme", default=None,
                    help="配色主题：主题名（themes/<名>.json）或 JSON 文件路径；"
                         "也可用 SYSMON_THEME。默认 catppuccin-mocha")
    ap.add_argument("--no-anim", action="store_true", help="关闭动画（脉冲/转圈/spinner 静止）")
    ap.add_argument("--list-themes", action="store_true", help="列出可用主题后退出")
    args = ap.parse_args()

    if args.list_themes:
        print(f"默认主题：{_theme.DEFAULT_THEME_NAME}（themes/ 目录，缺失时回退内置配色）")
        d = _theme.THEMES_DIR
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    print(" ", os.path.splitext(f)[0], "->", os.path.join(d, f))
        print("\n用法：--theme <名> 或 --theme /path/to/theme.json（也可设 SYSMON_THEME）")
        return

    global THEME, ANIM
    ANIM = not args.no_anim
    THEME, warn = _theme.load_theme(args.theme)
    try:
        curses.wrapper(render, args.interval, args.gateway, warn)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
