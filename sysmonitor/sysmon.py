#!/usr/bin/env python3
"""nvtop-style 图形化系统资源监控 - curses TUI。

借鉴 nvtop 布局，图形与数字并存；数据采样与动画刷新分离（画面 ~8fps）：
  - 顶部：每块 GPU 一个面板，温度/功耗/风扇/显存数值 + 算力、显存负载图
  - 中部：LLM 模型面板——每模型两行：阶段动画（Prompt 进度条 / 生成脉冲 / 加载转圈）、
          p/s 与 t/s 双速率曲线 sparkline、活动/排队 x/slot、空闲时长、累计 tokens；
          面板头汇总全局 t/s、p/s、请求数与排队数
  - CPU 面积图 + 每核方块图 + 每核占用数字表；右侧内存/交换/磁盘
  - 底部：GPU 进程表（GPU/PID/USER/%CPU/%MEM/显存/模型/命令行）

负载图 4 种风格（C 键循环，--chart 指定，默认 wave）：
  area      面积图（经典 nvtop 滚动波形，▀▄█ 半块）
  wave      波动直方图（每样本一根柱，柱顶水面波纹动画，最直观）
  spectrum  频谱柱（柱高=负载 + 绿→红分级着色 + ▔ 峰值落帽）
  heat      热力条（单行密度条 ░▒▓█，按值着色，最紧凑省行）

快捷键：
    q / Esc    退出
    Space      暂停/继续数据采样（动画不停）
    A          开/关动画（脉冲、转圈、spinner）
    C          切换负载图风格（area → wave → spectrum → heat）
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
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request
from collections import deque

import psutil

# ─── 配置 ──────────────────────────────────────────────────────
HISTORY_LEN = 60          # 波形图采样点数
MIN_COLS = 100
MIN_ROWS = 30
ch = 2           # 系统波形图高度（行）
ANIM_TICK = 0.15          # 动画帧间隔（秒）
DEFAULT_GATEWAY = os.environ.get("SYSMON_GATEWAY", "http://localhost:8000")

# ─── 动画字形 ──────────────────────────────────────────────────
PULSE = "●○"                              # 生成中呼吸点
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"                # braille 转圈（Prompt 预填充）
LOAD_SPIN = "◐◓◑◒"                        # 加载转圈
SPARK = "▁▂▃▄▅▆▇█"                        # sparkline 八级方块
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

def progress_bar(frac: float, width: int = 10) -> str:
    """Prompt 预填充进度条：▰ 已填充 / ▱ 剩余。"""
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "▰" * n + "▱" * (width - n)

def sparkline(hist, width: int) -> str:
    """把速率序列画成方块 sparkline（按序列内最大值归一）。"""
    vals = [float(v) for v in hist][-width:]
    if not vals:
        return " " * width
    m = max(vals)
    if m <= 0:
        return SPARK[0] * len(vals) + " " * (width - len(vals))
    out = "".join(SPARK[min(7, int(v / m * 8 + 0.5))] for v in vals)
    return out + " " * (width - len(out))

# ─── LLM token 指标 ───────────────────────────────────────────
_llm_baseline: dict[str, tuple[float, float]] = {}
_gen_hist: dict[str, deque] = {}        # 每模型 t/s 历史（sparkline）
_prompt_hist: dict[str, deque] = {}     # 每模型 p/s 历史（sparkline）
_last_active: dict[str, float] = {}     # 每模型最后一次「有活动」的时刻
GATEWAY_CONNECTED = False


def get_llm_models(gateway: str = DEFAULT_GATEWAY, timeout: float = 0.6) -> list[dict]:
    """从网关读取已加载模型，本地差分速率并维护 sparkline 历史与空闲时长。"""
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
        p_rate = float(tokens.get("prompt_rate") or tokens.get("prefill_rate") or 0.0)

        gh = _gen_hist.setdefault(name, deque(maxlen=HISTORY_LEN))
        gh.append(rate)
        ph = _prompt_hist.setdefault(name, deque(maxlen=HISTORY_LEN))
        ph.append(p_rate)
        active = b.get("in_flight", 0) or 0
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
            "prefill_active": bool(tokens.get("prefill_active")),
            "prefill_tokens": tokens.get("prefill_tokens"),
            "prefill_progress": tokens.get("prefill_progress"),
            "prefill_elapsed": tokens.get("prefill_elapsed"),
            "idle": idle,
            "metrics_ok": bool(tokens.get("metrics_ok")),
        })
    # 清理已卸载模型的历史
    live = {m["name"] for m in models}
    for gone in set(_gen_hist) - live:
        _gen_hist.pop(gone, None)
        _prompt_hist.pop(gone, None)
        _last_active.pop(gone, None)
    models.sort(key=lambda m: (m["active"] <= 0, -(m["rate"] or 0), m["name"]))
    return models

_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ─── 图表字符与风格 ────────────────────────────────────────────
# nvtop 半块面积图的像素合成表：每个字符位表示 2 个垂直像素
HALF = {(0, 0): " ", (0, 1): "▀", (1, 0): "▄", (1, 1): "█"}
WAVE_TOP = "▁▂▃▄▅▆▇"       # 波柱顶面：水位起伏动画字形（按列相位循环）
DENSITY = " ░▒▓█"           # 热力条密度字符（5 级）

# 图表风格（C 键循环）：area=面积图(经典nvtop) / wave=波动直方图 /
# spectrum=频谱柱(分级着色+落顶帽) / heat=热力条(单行密度)
CHART_STYLES = ("area", "wave", "spectrum", "heat")
CHART_STYLE_NAMES = {"area": "面积图", "wave": "波动直方图",
                     "spectrum": "频谱柱", "heat": "热力条"}
CHART_STYLE = os.environ.get("SYSMON_CHART", "wave")
if CHART_STYLE not in CHART_STYLES:
    CHART_STYLE = "area"
_FRAME = 0                  # 当前动画帧（draw() 每帧更新，波顶相位用）
_spectrum_caps: dict[str, list[float]] = {}   # 频谱顶帽（hkey -> 每列峰值）


def cycle_chart_style() -> str:
    """C 键：切到下一图表风格，返回新风格名。"""
    global CHART_STYLE
    CHART_STYLE = CHART_STYLES[(CHART_STYLES.index(CHART_STYLE) + 1)
                               % len(CHART_STYLES)]
    _spectrum_caps.clear()
    return CHART_STYLE


def _cols_from(history: list[float], width: int) -> list[float]:
    """历史序列右对齐到 width 列（不足左侧补 0），clamp 到 0..1。"""
    n = min(len(history), width)
    cols = [0.0] * width
    if n:
        src = history[len(history) - n:]
        cols[width - n:] = [min(max(float(v), 0.0), 1.0) for v in src]
    return cols


def _bar_grid(history: list[float], width: int, height: int,
              frame: int) -> list[list[tuple[str, str, float]]]:
    """波柱/频谱共用的柱网格：cell = (字符, 类型, 该行高度比例0..1)。
    类型：body 柱身 / top 波顶 / empty 底纹 / cap 顶帽（由 _spectrum_cap_grid 写入）。
    柱自底向上填充；顶行按列相位取波浪字形，形成水面起伏。
    """
    cols = _cols_from(history, width)
    pad = max(0, width - len(history))        # 左侧无数据列：留白（区别于真实 0%）
    grid = [[(" ", "", 0.0)] * width for _ in range(height)]
    for i, v in enumerate(cols):
        h = v * height                        # 连续柱高（行）
        if i < pad:
            continue
        if h <= 0.05:
            for r in range(height):
                grid[r][i] = ("░", "empty", (height - r) / height)
            continue
        for r in range(height):
            pb = height - 1 - r               # 距底行数（0=最底）
            frac = (pb + 1) / height
            if pb < h - 0.5:
                grid[r][i] = ("█", "body", frac)
            elif pb < h + 0.5:
                grid[r][i] = (WAVE_TOP[(frame + i) % len(WAVE_TOP)], "top", frac)
            else:
                grid[r][i] = ("░", "empty", frac)
    return grid


def _apply_spectrum_caps(grid, history: list[float], width: int, height: int,
                         hkey: str | None):
    """频谱顶帽：每列记忆峰值并缓慢下落（▔），高于当前柱顶时可见。"""
    if not hkey:
        return
    cols = _cols_from(history, width)
    cap = _spectrum_caps.get(hkey)
    if cap is None or len(cap) != width:
        cap = list(cols)
        _spectrum_caps[hkey] = cap
    pad = max(0, width - len(history))
    for i, v in enumerate(cols):
        cap[i] = max(cap[i] * 0.985, v)       # ~8fps 下约 6s 半衰
        if i < pad or cap[i] <= 0.02:
            continue                           # 无数据/零值列不画帽
        cap_row = height - 1 - int(round(cap[i] * height))
        bar_top = next((r for r in range(height)
                        if grid[r][i][1] in ("body", "top")), height)
        if 0 <= cap_row < bar_top:
            grid[cap_row][i] = ("▔", "cap", cap[i])

def draw_chart_area(history: list[float], width: int, height: int, frame: int = 0) -> list[str]:
    """nvtop 经典面积图（原始风格）。"""
    pix_h = height * 2
    n = min(len(history), width)
    cols = [0.0] * width
    for i in range(n):
        cols[width - n + i] = history[len(history) - n + i]
    rows = [[] for _ in range(height)]
    for v in cols:
        filled = int(round(min(max(v, 0.0), 1.0) * pix_h))
        for r in range(height):
            lo = 2 * (height - 1 - r)
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
#   BORDER  分隔线 / 峰值标记          暗灰（弱化结构线）
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
 PAIR_GPU, PAIR_MEM, PAIR_CPU, PAIR_DISK, PAIR_MODEL, PAIR_ACCENT,
 PAIR_SPARK_G, PAIR_SPARK_P, PAIR_PEAK) = range(1, 16)

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
    # sparkline / 峰值：跟随 ok/warn/border 角色
    for src, dst in ((PAIR_OK, PAIR_SPARK_G), (PAIR_WARN, PAIR_SPARK_P),
                     (PAIR_BORDER, PAIR_PEAK)):
        try:
            curses.init_pair(dst, *curses.pair_content(src))
        except (curses.error, ValueError):
            curses.init_pair(dst, curses.COLOR_WHITE, -1)
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

def put_chart(stdscr, y, x, w, history, color_pair, height=2, bold=False,
              peak=False, hkey=None):
    """按当前 CHART_STYLE 绘制负载可视化（C 键切换）：
    area=面积图(peak ▁ 标记) / wave=波动直方图(顶面水波动画) /
    spectrum=频谱柱(绿→红分级着色+▔落顶帽) / heat=热力条(单行密度+按值着色)。
    hkey 用于频谱顶帽的列峰值记忆。
    """
    style = CHART_STYLE if CHART_STYLE in CHART_STYLES else "area"
    if style == "area":
        lines = draw_chart_area(history, w, height)
        attr = cpair(color_pair, bold)
        if peak and history and max(history) > 0.02:
            lines = _mark_peak(lines, history, w)
        for r, line in enumerate(lines):
            put(stdscr, y + r, x, line, attr, w)
        return
    if style == "heat":
        cols = _cols_from(history, w)
        pad = max(0, w - len(history))
        for i, v in enumerate(cols):
            if i < pad:
                continue                          # 无数据留白
            if v <= 0.02:
                put(stdscr, y + height - 1, x + i, "░", cpair(PAIR_BORDER), 1)
            else:
                cell = DENSITY[min(4, int(v * 4.999) + 1)]
                put(stdscr, y + height - 1, x + i, cell,
                    cpair(color_for(v * 100), v >= 0.6), 1)
        return
    grid = _bar_grid(history, w, height, _FRAME)
    if style == "spectrum":
        _apply_spectrum_caps(grid, history, w, height, hkey)
    for r in range(height):
        for i, (glyph, kind, frac) in enumerate(grid[r]):
            if kind == "body":
                a = cpair(color_for(frac * 100)) if style == "spectrum" \
                    else cpair(color_pair, bold)
            elif kind == "top":
                a = cpair(color_for(frac * 100), True) if style == "spectrum" \
                    else cpair(color_pair, True)
            elif kind == "cap":
                a = cpair(PAIR_ACCENT, True)
            else:
                a = cpair(PAIR_BORDER)
            put(stdscr, y + r, x + i, glyph, a, 1)

def _mark_peak(lines, history, w):
    """在峰值所在列顶格画 ▁ 标记线（重绘为列表便于覆写）。"""
    idx = max(range(len(history)), key=lambda i: history[i])
    col = w - len(history) + idx
    if 0 <= col < w and len(lines) >= 1:
        top = list(lines[0])
        if col < len(top) and top[col] == " ":
            top[col] = "▁"
            lines = ["".join(top)] + list(lines[1:])
    return lines

def spark_cells(stdscr, y, x, w, hist, base_pair):
    """绘制 sparkline；有非零样本时最后一格用亮色突出当前值。"""
    vals = list(hist)[-w:]
    text = sparkline(vals, w)
    put(stdscr, y, x, text, cpair(base_pair), w)
    if vals and max(vals) > 0:
        put(stdscr, y, x + w - 1, text[-1], cpair(base_pair, True), 1)

def vline(stdscr, x, y0, y1, ch="│"):
    for y in range(y0, y1):
        put(stdscr, y, x, ch, cpair(PAIR_BORDER))

def anim_char(seq: str, frame: int) -> str:
    return seq[frame % len(seq)] if ANIM else seq[0]

# ─── GPU 面板 ─────────────────────────────────────────────────

def panel_gpu(stdscr, y, x, w, gpu, ch):
    """单卡面板：标题 + 状态行 + Util/Mem 数值与负载图（风格随 CHART_STYLE）。共 4 + 2*ch 行。"""
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

    # Utilization：数值 + 负载图（风格随 CHART_STYLE）
    y += 1
    hist_u = push(f"gu{gi}", gpu["util"] / 100)
    put(stdscr, y, x, f"算力 {gpu['util']:6.1f}%", cpair(color_for(gpu["util"]), True), 13)
    if gpu.get("mem_util") is not None:
        put(stdscr, y, x + 14, f"带宽 {gpu['mem_util']:3.0f}%", cpair(PAIR_DIM),
            max(0, label_w - 14))
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist_u, gc, ch, bold=True,
              peak=True, hkey=f"gu{gi}")

    # Memory：数值 + 图表（风格由 C 键切换）
    y += 1 + ch
    hist_m = push(f"gm{gi}", gpu["mem_used"] / gpu["mem_total"])
    put(stdscr, y, x, f"显存 {mem_pct:5.1f}%", cpair(color_for(mem_pct), True), 13)
    detail = f"{fmt_mb(gpu['mem_used'])}/{fmt_mb(gpu['mem_total'])}"
    put(stdscr, y, x + 14, clip(detail, max(0, label_w - 14)), cpair(PAIR_DIM),
        max(0, label_w - 14))
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist_m, PAIR_MEM, ch, bold=True,
              peak=True, hkey=f"gm{gi}")

    return y + 1 + ch

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
        prog = (m.get("prefill_progress") or 0) * 100

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
            seg = f"{sp} Prompt ▰"
            put(stdscr, y, cx, seg, cpair(PAIR_WARN, True), disp_width(seg))
            cx += disp_width(seg)
            barw = max(6, min(14, x + w - cx - 18))
            bar = progress_bar((m.get("prefill_progress") or 0), barw)
            put(stdscr, y, cx, bar, cpair(PAIR_WARN), barw)
            cx += barw + 1
            tail = f"{prog:3.0f}% " + (f"{fmt_tokens(m.get('prefill_tokens'))}tok "
                                       f"{float(elapsed):.0f}s" if elapsed else
                                       fmt_tokens(m.get("prefill_tokens")))
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

        # ── 第二行：双 sparkline + 速率 + 活动/排队 + 累计 + 显存 ──
        spark_w = 14 if w >= 150 else 10
        put(stdscr, y, x + 2, "t/s", cpair(PAIR_OK, True), 3)
        spark_cells(stdscr, y, x + 5, spark_w, _gen_hist.get(m["name"], ()),
                    PAIR_SPARK_G)
        t_txt = f"{t_rate:7.1f}" if t_rate > 0 else "     --"
        put(stdscr, y, x + 5 + spark_w + 1, t_txt,
            cpair(PAIR_OK, True) if t_rate > 0 else cpair(PAIR_DIM), 7)
        cx = x + 5 + spark_w + 9
        put(stdscr, y, cx, "p/s", cpair(PAIR_WARN, True), 3)
        spark_cells(stdscr, y, cx + 3, spark_w, _prompt_hist.get(m["name"], ()),
                    PAIR_SPARK_P)
        cx += 3 + spark_w + 1
        p_txt = f"{p_rate:6.0f}" if p_rate > 0 else "    --"
        put(stdscr, y, cx, p_txt,
            cpair(PAIR_WARN, True) if p_rate > 0 else cpair(PAIR_DIM), 6)
        cx += 8

        active_txt = f"{m['active']}/{slots}" if slots else str(m["active"])
        if queueing:
            q = anim_char(PULSE, frame // 3)
            seg = f"{q} {active_txt} 排队"
            put(stdscr, y, cx, seg, cpair(PAIR_WARN, True), disp_width(seg))
        elif m["active"] > 0:
            put(stdscr, y, cx, f"ACT {active_txt}", cpair(PAIR_OK, True), 9)
        else:
            put(stdscr, y, cx, f"ACT {active_txt}", cpair(PAIR_DIM), 9)
        cx += 12

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

    # 图例（含当前负载图风格说明）
    lx = x + 2
    _chart_legend = {
        "area": ((cpair(PAIR_BORDER), "▁"), (cpair(PAIR_DIM), "峰值 ")),
        "wave": ((cpair(PAIR_OK), "▃▅▂"), (cpair(PAIR_DIM), "柱顶=水位 ")),
        "spectrum": ((cpair(PAIR_ACCENT, True), "▔"),
                     (cpair(PAIR_DIM), "顶帽=峰值回落 ")),
        "heat": ((cpair(PAIR_DIM), "░▒▓█"), (cpair(PAIR_DIM), "密度=负载 ")),
    }.get(CHART_STYLE, ())
    legend = (
        (cpair(PAIR_OK), "●"), (cpair(PAIR_DIM), "生成 "),
        (cpair(PAIR_WARN), "◐"), (cpair(PAIR_DIM), "Prompt "),
        (cpair(PAIR_OK, True), "t/s"), (cpair(PAIR_DIM), "=生成速率 "),
        (cpair(PAIR_WARN, True), "p/s"), (cpair(PAIR_DIM), "=Prompt速率 "),
    ) + _chart_legend + (
        (cpair(PAIR_DIM), f"C键:图→{CHART_STYLES[(CHART_STYLES.index(CHART_STYLE)+1) % len(CHART_STYLES)]}  "
                          "ACT=活动/slot(超则排队)  Σ=累计(加载以来)  K=千 M=百万tok"),
    )
    for a, seg in legend:
        put(stdscr, y, lx, seg, a, disp_width(seg))
        lx += disp_width(seg)
    return y + 2

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
    put_chart(stdscr, y + 1, x + label_w, chart_w, hist, PAIR_CPU, ch,
              peak=True, hkey="cpu")

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
        put_chart(stdscr, row + 1, x + label_w, chart_w, h, pair, ch,
                  peak=True, hkey=hist_key)

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

# ─── 采样与主渲染 ──────────────────────────────────────────────

def collect_samples(gateway, interval):
    """采集一轮数据（含历史 push，每数据周期只调用一次）。"""
    return {
        "gpus": get_gpu_info(),
        "models": get_llm_models(gateway, timeout=min(0.8, max(0.2, interval / 2))),
        "cpu": get_cpu_info(),
        "mem": get_mem_info(),
        "disk": get_disk_info(),
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
    bar = pad(f" SYSMON  {host}  {time.strftime('%Y-%m-%d %H:%M:%S')} ", cols)
    put(stdscr, 0, 0, bar, cpair(PAIR_HEADER, True), cols)
    if llm_models:
        tag = f"Σ {total_t:5.1f} t/s "
        right = x0 = cols - disp_width(tag) - 1
        put(stdscr, 0, x0, tag, cpair(PAIR_HEADER, True), disp_width(tag))
        dot = anim_char(PULSE, frame // 2) if total_t > 0 else "○"
        put(stdscr, 0, max(0, x0 - 2), dot,
            cpair(PAIR_OK if total_t > 0 else PAIR_DIM, True), 1)

    # 依据终端高度自适应图表行数（每模型 2 行 + 面板头 + 图例）
    llm_rows = min(len(llm_models), max(1, (rows - 20) // 2)) if llm_models else 1
    llm_h = 2 * llm_rows + (3 if llm_models else 2)
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
            vline(stdscr, gi * pw - 1, y, y + gpu_h)
        y += gpu_h
    else:
        put(stdscr, y, 0, " 未检测到 NVIDIA GPU", cpair(PAIR_WARN, True) | curses.A_REVERSE, cols)
        y += 2

    # ── LLM token 吞吐 ──
    if y < rows - 8:
        put(stdscr, y, 0, "─" * (cols - 1), cpair(PAIR_BORDER))
        y = panel_llm(stdscr, y + 1, 0, cols - 1, llm_models, llm_rows, frame)

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
    pause_tag = " ⏸已暂停 " if PAUSED else ""
    anim_tag = "动效开" if ANIM else "动效关"
    status = (f" q 退出  Space 暂停  A 动画  C 图表  T/⇧T 主题"
              f" | {pause_tag}{anim_tag}"
              f" 图表 {CHART_STYLE_NAMES[CHART_STYLE]}"
              f" 刷新 {interval:.1f}s 采样 {len(_history.get('cpu', []))} 点"
              f" 主题 {theme_name}")
    put(stdscr, rows - 1, 0, pad(status, cols - 1), cpair(PAIR_HEADER, True), cols - 1)
    if theme_warn:
        put(stdscr, rows - 1, max(0, cols - 1 - disp_width(theme_warn)),
            clip(theme_warn, cols - 1), cpair(PAIR_WARN, True), cols - 1)

    stdscr.refresh()


def _theme_names():
    import theme as _theme
    return [_theme.DEFAULT_THEME_NAME] + _theme._list_available_themes()


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
        elif key in (ord("A"),):                 # 动画开关
            ANIM = not ANIM
        elif key in (ord("C"), ord("c")):        # 图表风格循环
            cycle_chart_style()
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
                         "也可用 SYSMON_THEME。默认内置 Tokyo Night")
    ap.add_argument("--no-anim", action="store_true", help="关闭动画（脉冲/转圈/spinner 静止）")
    ap.add_argument("--chart", default=None, choices=CHART_STYLES,
                    help="负载图表风格：area=面积图(经典) wave=波动直方图 "
                         "spectrum=频谱柱(分级色+顶帽) heat=热力条；"
                         "运行中按 C 循环切换（默认 SYSMON_CHART 或 wave）")
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

    global THEME, ANIM, CHART_STYLE
    ANIM = not args.no_anim
    if args.chart:
        CHART_STYLE = args.chart        # 默认 wave（模块级），--chart area 回经典
    THEME, warn = _theme.load_theme(args.theme)
    try:
        curses.wrapper(render, args.interval, args.gateway, warn)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
