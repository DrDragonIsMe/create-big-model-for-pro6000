#!/usr/bin/env python3
"""nvtop-style 图形化系统资源监控 - curses TUI。

借鉴 nvtop 布局，图形与数字并存：
  - 顶部：每块 GPU 一个面板，温度/功耗/风扇/显存数值 + 算力、显存面积波形图
  - 中部：CPU 面积波形图 + 每核方块图 + 每核占用数字表；右侧内存/交换/磁盘
  - 底部：GPU 进程表（GPU/PID/USER/%CPU/%MEM/显存/模型/命令行）

用法：
    python3 sysmon.py          # TUI 图形界面（q / Ctrl+C 退出）
    python3 sysmon.py -i 0.5   # 自定义刷新间隔（秒）
"""

from __future__ import annotations

import argparse
import curses
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request

import psutil

# ─── 配置 ──────────────────────────────────────────────────────
HISTORY_LEN = 60          # 波形图采样点数
MIN_COLS = 100
MIN_ROWS = 30
ch = 2           # 系统波形图高度（行）
DEFAULT_GATEWAY = os.environ.get("SYSMON_GATEWAY", "http://localhost:8000")

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

# ─── LLM token 指标 ───────────────────────────────────────────
_llm_baseline: dict[str, tuple[float, float]] = {}
_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_llm_models(gateway: str = DEFAULT_GATEWAY, timeout: float = 0.6) -> list[dict]:
    """从网关读取已加载模型，并用本地差分计算当前刷新窗口的 toks/s。"""
    if not gateway:
        return []
    try:
        with _no_proxy_opener.open(gateway.rstrip("/") + "/v1/gateway/status",
                                   timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
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
        gpus = b.get("gpus") or []
        models.append({
            "name": name,
            "engine": b.get("engine", "?"),
            "gpus": ",".join(str(g) for g in gpus) if gpus else "-",
            "active": b.get("in_flight", 0) or 0,
            "slots": b.get("slots"),
            "status": b.get("status", "?"),
            "prompt_total": tokens.get("prompt_total"),
            "generation_total": total,
            "rate": rate,
            "prefill_active": bool(tokens.get("prefill_active")),
            "prefill_tokens": tokens.get("prefill_tokens"),
            "prefill_progress": tokens.get("prefill_progress"),
            "prompt_rate": tokens.get("prompt_rate", tokens.get("prefill_rate")),
            "metrics_ok": bool(tokens.get("metrics_ok")),
        })
    models.sort(key=lambda m: (m["active"] <= 0, -(m["rate"] or 0), m["name"]))
    return models

# ─── 波形图字符 ────────────────────────────────────────────────
# nvtop 用半块字符做上下堆叠的面积图：每个字符位表示 2 个垂直像素
HALF = {(0, 0): " ", (0, 1): "▀", (1, 0): "▄", (1, 1): "█"}

def area_chart(history: list[float], width: int, height: int) -> list[str]:
    """把 0..1 的历史序列画成 height 行的面积图，返回自上而下的行字符串。
    最左列为最早样本，最右列为最新；不足 width 点时左侧留空。
    """
    pix_h = height * 2
    n = min(len(history), width)
    cols = [0.0] * width
    for i in range(n):
        cols[width - n + i] = history[len(history) - n + i]

    rows = [[] for _ in range(height)]
    for v in cols:
        filled = int(round(min(max(v, 0.0), 1.0) * pix_h))
        for r in range(height):
            lo = 2 * (height - 1 - r)        # 该字符格下半像素（自底向上计数）
            hi = lo + 1
            bot = 1 if filled > lo else 0
            top = 1 if filled > hi else 0
            rows[r].append(HALF[(top, bot)])
    return ["".join(r) for r in rows]

# ─── 历史数据 ─────────────────────────────────────────────────
_history: dict[str, list[float]] = {}

def push(key: str, value: float) -> list[float]:
    h = _history.setdefault(key, [])
    h.append(value)
    if len(h) > HISTORY_LEN:
        h.pop(0)
    return h

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
        du = psutil.disk_usage("/")
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

# ─── 颜色 ──────────────────────────────────────────────────────
# 语义角色（全程序统一，颜色只是这些角色的具体呈现）：
#   HEADER  面板标题条 / 顶栏          黑字 + 青色底
#   BORDER  分隔线                     暗灰（弱化结构线）
#   DIM     次要信息：单位、明细、注释  中灰
#   ACCENT  表头 / 标签                柔和青蓝
#   OK      正常 / 生成阶段 t/s        绿
#   WARN    注意 / Prompt 处理 p/s     琥珀（忙碌但非异常）
#   CRIT    告警：高温 / 利用率 ≥85%   红
#   GPU     GPU 曲线（多卡按下面板循环）蓝紫青系
#   MEM     内存曲线 / 数值            青蓝
#   CPU     CPU 曲线 / 数值            绿
#   DISK    磁盘曲线 / 数值            土黄
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

# 运行时主题（main 里由 theme.load_theme 填充）；None 时 init_colors 用内置默认
THEME = None
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
    # 动态配色：GPU 曲线（pair 20+i）与亮色版本（pair 40+i）
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

def put_chart(stdscr, y, x, w, history, color_pair, height=2, bold=False):
    """在 (y,x,w,height) 区域内绘制面积图。"""
    lines = area_chart(history, w, height)
    attr = cpair(color_pair, bold)
    for r, line in enumerate(lines):
        put(stdscr, y + r, x, line, attr, w)

def vline(stdscr, x, y0, y1, ch="│"):
    for y in range(y0, y1):
        put(stdscr, y, x, ch, cpair(PAIR_BORDER))

# ─── GPU 面板 ─────────────────────────────────────────────────

def panel_gpu(stdscr, y, x, w, gpu, ch):
    """单卡面板：标题 + 状态行 + Util/Mem 数值与面积图。共 4 + 2*ch 行。"""
    gi = gpu["index"]
    gc = 20 + (gi % len(GPU_COLORS))
    label_w = 25 if w >= 40 else max(13, w - 12)
    chart_w = max(8, w - label_w - 1)

    # 标题行
    title = f"[GPU {gi}] {gpu['name']}"
    put(stdscr, y, x, clip(title, w), cpair(gc, True), w)

    # 状态行：温度(分级着色) / 功耗 / 风扇 / 剩余显存
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
        cx += disp_width(f) + 2
    free = gpu["mem_total"] - gpu["mem_used"]
    put(stdscr, y, cx, clip(f"显存空闲 {fmt_mb(free)}", max(0, w - (cx - x))),
        cpair(PAIR_DIM), max(0, w - (cx - x)))
    mem_pct = gpu["mem_used"] / gpu["mem_total"] * 100

    # Utilization：数值 + 面积图
    y += 1
    hist_u = push(f"gu{gi}", gpu["util"] / 100)
    put(stdscr, y, x, f"算力 {gpu['util']:6.1f}%", cpair(color_for(gpu["util"]), True), 13)
    if gpu.get("mem_util") is not None:
        put(stdscr, y, x + 14, f"带宽 {gpu['mem_util']:3.0f}%", cpair(PAIR_DIM),
            max(0, label_w - 14))
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist_u, gc, ch, bold=True)

    # Memory：数值 + 面积图
    y += 1 + ch
    hist_m = push(f"gm{gi}", gpu["mem_used"] / gpu["mem_total"])
    put(stdscr, y, x, f"显存 {mem_pct:5.1f}%", cpair(color_for(mem_pct), True), 13)
    detail = f"{fmt_mb(gpu['mem_used'])}/{fmt_mb(gpu['mem_total'])}"
    put(stdscr, y, x + 14, clip(detail, max(0, label_w - 14)), cpair(PAIR_DIM),
        max(0, label_w - 14))
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist_m, PAIR_MEM, ch, bold=True)

    return y + 1 + ch

# ─── LLM token 面板 ───────────────────────────────────────────

def panel_llm(stdscr, y, x, w, models, max_rows=4):
    """显示已加载模型的累计生成 tokens 与当前窗口速率。"""
    if not models:
        put(stdscr, y, x, clip("LLM Tokens  网关未返回已加载模型或指标暂不可用", w),
            cpair(PAIR_DIM), w)
        return y + 2

    put(stdscr, y, x, "LLM Tokens", cpair(PAIR_HEADER, True), 12)
    # 列：前 4 列标识，阶段三列只在 prefill 有值；p/s 与 t/s 互相独立可同时非零（并发请求）
    cols = (("模型", 24, "l"), ("引擎", 6, "l"), ("GPU", 4, "l"), ("活动", 5, "r"),
            ("阶段", 6, "r"), ("阶段tok", 9, "r"), ("进度", 5, "r"),
            ("p/s", 8, "r"), ("t/s", 8, "r"),
            ("累计prompt", 11, "r"), ("累计生成", 9, "r"))
    hx = 14
    for name, cw, al in cols:
        text = rpad(name, cw) if al == "r" else pad(name, cw)
        hattr = (cpair(PAIR_WARN, True) if name == "p/s"
                 else cpair(PAIR_OK, True) if name == "t/s"
                 else cpair(PAIR_ACCENT, True))
        put(stdscr, y, x + hx, text, hattr, cw)
        hx += cw + 1
    y += 1

    for m in models[:max(0, max_rows)]:
        prefill = bool(m.get("prefill_active"))
        p_rate = float(m.get("prompt_rate") or 0)
        t_rate = float(m.get("rate") or 0)
        decoding = t_rate > 0 or (m["active"] > 0 and not prefill)
        prog = (m.get("prefill_progress") or 0) * 100
        if prefill:
            phase_text = "Prompt"
            token_text = fmt_tokens(m.get("prefill_tokens"))
            prog_text = f"{prog:3.0f}%"
        elif decoding:
            phase_text = "生成"
            token_text = "--"
            prog_text = " --"
        else:
            phase_text = "空闲"
            token_text = "--"
            prog_text = " --"
        base_attr = cpair(PAIR_DIM)
        slots = m.get("slots")
        active_txt = f"{m['active']}/{slots}" if slots else str(m["active"])
        if m["active"] > 0:
            active_attr = cpair(PAIR_WARN, True) if (slots and m["active"] > slots) \
                else cpair(PAIR_OK, True)
        else:
            active_attr = base_attr
        cells = (
            (pad(clip(m["name"], 24), 24), base_attr),
            (pad(clip("vLLM" if m["engine"] == "vllm" else "llama", 6), 6), base_attr),
            (pad(clip(m["gpus"], 4), 4), base_attr),
            (f"{active_txt:>5}", active_attr),
            (rpad(phase_text, 6), cpair(PAIR_WARN, True) if prefill
             else cpair(PAIR_OK, True) if decoding else base_attr),
            (f"{token_text:>9}", cpair(PAIR_WARN, True) if prefill else base_attr),
            (f"{prog_text:>5}", cpair(PAIR_WARN, True) if prefill else base_attr),
            (f"{p_rate:7.0f} " if p_rate > 0 else f"{'--':>8}",
             cpair(PAIR_WARN, True) if p_rate > 0 else base_attr),
            (f"{t_rate:7.1f} " if t_rate > 0 else f"{'--':>8}",
             cpair(PAIR_OK, True) if t_rate > 0 else base_attr),
            (f"{fmt_tokens(m.get('prompt_total')):>11}", base_attr),
            (f"{fmt_tokens(m.get('generation_total')):>9}", base_attr),
        )
        cx = 14
        for (_, cw, _), (text, attr) in zip(cols, cells):
            put(stdscr, y, x + cx, text, attr, cw)
            cx += cw + 1
        y += 1

    # 图例：两种速率单位的含义与颜色；累计列的口径
    legend = (
        (cpair(PAIR_DIM), "活动=进行中/slot上限(超slot在排队,显黄)  "),
        (cpair(PAIR_WARN, True), "p/s"),
        (cpair(PAIR_DIM), "=Prompt速率  "),
        (cpair(PAIR_OK, True), "t/s"),
        (cpair(PAIR_DIM), "=生成速率  K=千 M=百万tokens  累计=模型加载以来(重启清零)"),
    )
    lx = x + 14
    for a, seg in legend:
        put(stdscr, y, lx, seg, a, disp_width(seg))
        lx += disp_width(seg) + 1
    y += 1

    hidden = len(models) - max_rows
    if hidden > 0:
        put(stdscr, y, x, f"… 另有 {hidden} 个模型未显示", cpair(PAIR_DIM), w)
        y += 1
    return y + 1

# ─── CPU 面板 ─────────────────────────────────────────────────

def panel_cpu(stdscr, y, x, w, cpu, ch, room=99):
    """CPU 面积图 + 每核方块 + 每核数字表。"""
    label_w = 13
    chart_w = max(10, w - label_w - 1)
    top = y

    hist = push("cpu", cpu["total"] / 100)
    put(stdscr, y, x, f"CPU  {cpu['total']:5.1f}%", cpair(color_for(cpu["total"]), True), label_w)
    extra = ""
    if cpu["load"]:
        extra = f"load {cpu['load'][0]:.2f} {cpu['load'][1]:.2f} {cpu['load'][2]:.2f}"
    if cpu["freq"]:
        extra += f"  {cpu['freq'] / 1000:.2f}GHz"
    put(stdscr, y, x + label_w, clip(extra, chart_w), cpair(PAIR_DIM), chart_w)
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist, PAIR_CPU, ch)

    # 每核方块图：一列一核，▁▂▃▄▅▆▇█ 八级
    LEVELS = " ▁▂▃▄▅▆▇█"
    cores = cpu["per_core"]
    y += 1 + ch
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

def panel_mem(stdscr, y, x, w, mem, disk, ch):
    """内存 / 交换 / 磁盘 面积图。"""
    label_w = 12
    chart_w = max(10, w - label_w - 1)
    vm, sm = mem["vm"], mem["sm"]

    def block(row, label, pct, detail, hist_key, pair):
        h = push(hist_key, pct / 100)
        put(stdscr, row, x, f"{label} {pct:5.1f}%", cpair(color_for(pct), True), label_w)
        put(stdscr, row, x + label_w, clip(detail, chart_w), cpair(PAIR_DIM), chart_w)
        put_chart(stdscr, row + 1, x + label_w, chart_w, h, pair, ch)

    block(y, "内存", vm.percent,
          f"{fmt_bytes(vm.used)}/{fmt_bytes(vm.total)}  可用 {fmt_bytes(vm.available)}"
          f"  活动 {fmt_bytes(getattr(vm, 'active', 0))}", "ram", PAIR_MEM)
    y += 1 + ch
    block(y, "交换", sm.percent, f"{fmt_bytes(sm.used)}/{fmt_bytes(sm.total)}", "swap", PAIR_DISK)
    y += 1 + ch
    if disk["du"]:
        block(y, "磁盘", disk["du"].percent,
              f"{fmt_bytes(disk['du'].used)}/{fmt_bytes(disk['du'].total)}"
              f"  剩余 {fmt_bytes(disk['du'].free)}", "disk", PAIR_DISK)
        put(stdscr, y + 1 + ch, x + label_w,
            f"读 {fmt_bytes(disk['read_bps'])}/s  写 {fmt_bytes(disk['write_bps'])}/s",
            cpair(PAIR_DIM), chart_w)
        y += 1 + ch
    return y + 1

# ─── 进程表 ────────────────────────────────────────────────────

def proc_columns(w):
    """按终端宽度决定进程表列；窄屏优先砍掉 USER / %MEM，保住命令行。"""
    cols = [
        ("GPU",   4, lambda p: f"{p['gpu']:>3}"),
        ("PID",   8, lambda p: f"{p['pid']:>7}"),
        ("USER",  8, lambda p: f"{p.get('user', '?'):>7}"),
        ("%CPU",  6, lambda p: f"{p.get('cpu', 0.0):>5.1f}"),
        ("%MEM",  6, lambda p: f"{p.get('pmem', 0.0):>5.1f}"),
        ("RSS",   7, lambda p: f"{fmt_bytes(p.get('rss', 0)):>6}"),
        ("VRAM",  8, lambda p: f"{fmt_mb(p['mem']):>7}" if p["mem"] else f"{'?':>7}"),
    ]
    if w < 118:
        cols = [c for c in cols if c[0] not in ("USER", "%MEM", "RSS")]
    elif w < 132:
        cols = [c for c in cols if c[0] != "RSS"]
    return cols


def panel_procs(stdscr, y, x, w, gpus, max_rows=8, bar_w=16):
    """全宽 GPU 进程表：GPU/PID/USER/%CPU/%MEM/RSS/VRAM + 显存条 + 模型/命令。"""
    procs = [p for g in gpus for p in g["procs"]]
    procs.sort(key=lambda p: p["mem"] or 0, reverse=True)

    cols = proc_columns(w)
    # 预留：显存条 + 至少 20 列给命令
    while w < sum(c[1] + 1 for c in cols) + bar_w + 21 and len(cols) > 2:
        cols.pop(-2 if cols[-1][0] == "VRAM" else -1)

    xs, acc = [], 0
    for name, cw, _ in cols:
        xs.append(acc)
        acc += cw + 1
    bar_x, cmd_x = acc, acc + bar_w + 1

    for cx, (name, _, _) in zip(xs, cols):
        put(stdscr, y, x + cx, name, cpair(PAIR_ACCENT, True), max(0, w - cx))
    put(stdscr, y, x + bar_x, "占用", cpair(PAIR_ACCENT, True), max(0, w - bar_x))
    put(stdscr, y, x + cmd_x, "模型 / 命令行", cpair(PAIR_ACCENT, True), max(0, w - cmd_x))
    y += 1

    for p in procs[:max(0, max_rows)]:
        gpu = next((g for g in gpus if g["index"] == p["gpu"]), None)
        mem_pct = ((p["mem"] or 0) / gpu["mem_total"] * 100) if gpu and gpu["mem_total"] else 0.0
        attr = cpair(color_for(mem_pct))
        for cx, (_, cw, fmt) in zip(xs, cols):
            put(stdscr, y, x + cx, fmt(p), attr, cw)
        n = int(round(mem_pct / 100 * bar_w))
        put(stdscr, y, x + bar_x, "█" * n + "░" * (bar_w - n), attr, bar_w)
        tag = f"[{p['model']}] " if p["model"] else ""
        if tag:
            put(stdscr, y, x + cmd_x, tag, cpair(PAIR_MODEL, True), max(0, w - cmd_x))
        put(stdscr, y, x + cmd_x + len(tag),
            clip(p["cmd"], w - cmd_x - len(tag)), cpair(PAIR_DIM),
            max(0, w - cmd_x - len(tag)))
        y += 1
    if not procs:
        put(stdscr, y, x, "（GPU 无占用进程）", cpair(PAIR_DIM), w)
        y += 1
    elif len(procs) > max_rows:
        put(stdscr, y, x, f"… 另有 {len(procs) - max_rows} 个进程未显示", cpair(PAIR_DIM), w)
        y += 1
    else:
        put(stdscr, y, x, f"共 {len(procs)} 个 GPU 进程", cpair(PAIR_DIM), w)
        y += 1
    return y

# ─── 主渲染 ────────────────────────────────────────────────────

def render(stdscr, interval, gateway=DEFAULT_GATEWAY, theme_warn=None):
    global THEME
    init_colors(THEME)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(int(interval * 1000))
    psutil.cpu_percent(interval=None)  # 预热

    while True:
        rows, cols = stdscr.getmaxyx()
        gpus = get_gpu_info()
        llm_models = get_llm_models(gateway, timeout=min(0.8, max(0.2, interval / 2)))
        cpu = get_cpu_info()
        mem = get_mem_info()
        disk = get_disk_info()

        stdscr.erase()
        if rows < MIN_ROWS or cols < MIN_COLS:
            put(stdscr, 0, 0, f"终端太小: {cols}x{rows}（需要 ≥ {MIN_COLS}x{MIN_ROWS}）",
                cpair(PAIR_CRIT))
            stdscr.refresh()
        else:
            # ── 标题栏 ──
            host = socket.gethostname()
            bar = pad(f" SYSMON  {host}  {time.strftime('%Y-%m-%d %H:%M:%S')} ", cols)
            put(stdscr, 0, 0, bar, cpair(PAIR_HEADER, True), cols)

            # 依据终端高度自适应图表行数（预留 LLM/系统/进程表所需）
            llm_rows = min(len(llm_models), max(1, (rows - 19) // 2)) if llm_models else 1
            llm_h = llm_rows + (3 if llm_models else 2)
            ch = max(1, (rows - 16 - llm_h) // 5)
            y = 1

            # ── GPU 区域：多卡并排 ──
            if gpus:
                pw = cols // len(gpus)
                gpu_h = 2 + 2 * (1 + ch)
                for gi, g in enumerate(gpus):
                    gx = gi * pw
                    gw = (cols - gx) if gi == len(gpus) - 1 else pw - 1
                    panel_gpu(stdscr, y, gx, gw, g, ch)
                for gi in range(1, len(gpus)):
                    vline(stdscr, gi * pw - 1, y, y + gpu_h - 1)
                y += gpu_h
            else:
                put(stdscr, y, 0, " 未检测到 NVIDIA GPU", cpair(PAIR_WARN, True) | curses.A_REVERSE, cols)
                y += 2

            # ── LLM token 吞吐 ──
            if y < rows - 8:
                put(stdscr, y, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
                y = panel_llm(stdscr, y + 1, 0, cols - 1, llm_models, llm_rows)

            # ── CPU / 系统 双列 ──
            half = cols // 2
            put(stdscr, y, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
            sep_y = y + 1
            # CPU 面板可用行数：底部要留给进程表（分隔线 + 表头 + 至少 2 行 + 状态栏）
            room = max(4, rows - 5 - sep_y)

            y_cpu = panel_cpu(stdscr, sep_y, 0, half - 1, cpu, ch, room)
            y_mem = panel_mem(stdscr, sep_y, half + 1, cols - half - 2, mem, disk, ch)
            vline(stdscr, half, sep_y, min(max(y_cpu, y_mem), rows - 2))

            # ── 进程表（剩余空间不足则省略）──
            y_tbl = max(y_cpu, y_mem)
            avail = rows - 1 - y_tbl - 2          # 减去分隔线、表头、状态栏
            if avail >= 1:
                put(stdscr, y_tbl, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
                panel_procs(stdscr, y_tbl + 1, 0, cols - 1, gpus, avail)

            # ── 状态栏 ──
            theme_name = (THEME or {}).get("name", "tokyo-night")
            put(stdscr, rows - 1, 0, pad(" q 退出   刷新 %.1fs   采样 %d 点   主题 %s" %
                (interval, len(_history.get("cpu", [])), theme_name), cols - 1),
                cpair(PAIR_HEADER, True), cols - 1)
            if theme_warn:
                put(stdscr, rows - 1, max(0, cols - 1 - disp_width(theme_warn)),
                    clip(theme_warn, cols - 1), cpair(PAIR_WARN, True), cols - 1)

            stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 3, 27):  # q / Ctrl+C / Esc
            return
        elif key == ord("T"):  # 切换主题
            import theme as _theme
            all_themes = [_theme.DEFAULT_THEME_NAME] + _theme._list_available_themes()
            current_name = (THEME or {}).get("name")
            idx = (all_themes.index(current_name) + 1) % len(all_themes) if current_name in all_themes else 0
            new_theme = all_themes[idx]
            THEME, warn = _theme.load_theme(new_theme)
            init_colors(THEME)
        elif key in (curses.KEY_RESIZE,):
            time.sleep(0.1)

# ─── 入口 ──────────────────────────────────────────────────────

def main():
    import theme as _theme
    ap = argparse.ArgumentParser(description="nvtop-style GPU/CPU/LLM token 资源监控")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="刷新间隔秒（默认 1.0）")
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY,
                    help="模型网关地址（默认 %(default)s，也可用 SYSMON_GATEWAY）")
    ap.add_argument("--theme", default=None,
                    help="配色主题：主题名（themes/<名>.json）或 JSON 文件路径；"
                         "也可用 SYSMON_THEME。默认内置 Tokyo Night")
    ap.add_argument("--list-themes", action="store_true", help="列出可用主题后退出")
    args = ap.parse_args()

    if args.list_themes:
        print("内置默认：tokyo-night")
        d = _theme.THEMES_DIR
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    print(" ", os.path.splitext(f)[0], "->", os.path.join(d, f))
        print("\n用法：--theme <名> 或 --theme /path/to/theme.json（也可设 SYSMON_THEME）")
        return

    global THEME
    THEME, warn = _theme.load_theme(args.theme)
    try:
        curses.wrapper(render, args.interval, args.gateway, warn)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
