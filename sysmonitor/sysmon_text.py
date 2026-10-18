#!/usr/bin/env python3
"""系统资源监控：CPU / 内存 / 磁盘 / GPU（利用率、显存、温度、功耗）。

用法：
    python3 sysmon.py            # 输出一次快照
    python3 sysmon.py -w         # 持续刷新（默认每 2 秒）
    python3 sysmon.py -w -i 1    # 每 1 秒刷新一次
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime

import psutil

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
DEFAULT_GATEWAY = os.environ.get("SYSMON_GATEWAY", "http://localhost:8000")
GATEWAY = DEFAULT_GATEWAY
_llm_baseline = {}
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ── 语义配色（主题可换；默认 Tokyo Night）──
# 颜色由 theme.load_theme() 解析为 ANSI 码，存入下方 COL_*；逻辑见 theme.py
import theme as _theme
_THEME, _THEME_WARN = _theme.load_theme()
_HI = (os.environ.get("COLORTERM") in ("truecolor", "24bit")
       or os.environ.get("TERM", "").endswith("256color")
       or "256" in os.environ.get("TERM", ""))

_BASIC_ANSI = {
    "black": "30", "red": "31;1", "green": "32", "yellow": "33",
    "blue": "34", "magenta": "35;1", "cyan": "36", "white": "37",
}

def _role_code(role, bold_default=False):
    if _HI:
        spec = _THEME["colors"].get(role)
        idx = spec[0] if isinstance(spec, list) else spec
        return f"38;5;{idx}"
    base = _theme.BASIC_COLORS.get(role, "white")
    return _BASIC_ANSI.get(base[0] if isinstance(base, tuple) else base, "37")

def _apply_theme(spec=None):
    """加载主题并刷新模块级 COL_*（main 在 render 前调用）。返回 warning。"""
    global _THEME, COL_ACCENT, COL_OK, COL_WARN, COL_CRIT, COL_MEM, COL_DISK, COL_MODEL, COL_DIM
    _THEME, warn = _theme.load_theme(spec)
    COL_ACCENT = _role_code("accent")
    COL_OK = _role_code("ok")
    COL_WARN = _role_code("warn")
    COL_CRIT = _role_code("crit")
    COL_MEM = _role_code("mem")
    COL_DISK = _role_code("disk")
    COL_MODEL = _role_code("model")
    COL_DIM = _role_code("dim") if _HI else "2"
    return warn

_apply_theme()


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t):
    return c("1", t)


def dim(t):
    return c(COL_DIM, t)


def pct_color(pct, warn=60, crit=85):
    if pct >= crit:
        return COL_CRIT
    if pct >= warn:
        return COL_WARN
    return COL_OK


def bar(pct, width=24):
    filled = int(round(width * min(pct, 100) / 100))
    color = pct_color(pct)
    return c(color, "█" * filled) + dim("░" * (width - filled))


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024


def fmt_mb(mb):
    return f"{mb / 1024:.1f}G" if mb >= 1024 else f"{mb:.0f}M"


def fmt_int(n):
    if n is None:
        return "n/a"
    return f"{int(n):,}"


def fmt_tokens(n):
    """token 数量紧凑格式：单位 K=千 tokens、M=百万 tokens。"""
    if n is None:
        return "n/a"
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n):,}"


def get_llm_models(timeout=0.8):
    """读取网关状态，并按刷新间隔计算本窗口生成速率。"""
    if not GATEWAY:
        return []
    try:
        with _opener.open(GATEWAY.rstrip("/") + "/v1/gateway/status",
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
            "engine": "vLLM" if b.get("engine") == "vllm" else "llama",
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


def disp_width(s):
    """终端显示宽度：CJK 等全角字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def pad(s, width):
    return s + " " * max(0, width - disp_width(s))


def section(title):
    return bold(c(COL_ACCENT, f"┌─ {title} " + "─" * max(0, 46 - disp_width(title))))


def row(label, pct, detail=""):
    pct_str = c(pct_color(pct), f"{pct:5.1f}%")
    line = f"│ {dim(pad(label, 14))} {bar(pct)} {pct_str}"
    if detail:
        line += f"   {dim(detail)}"
    return line


def get_cpu():
    pct_total = psutil.cpu_percent(interval=None)
    pct_cores = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()
    try:
        load = psutil.getloadavg()
    except (AttributeError, OSError):
        load = None
    return pct_total, pct_cores, freq, load


def get_gpus():
    """通过 nvidia-smi 读取 GPU 信息；无 NVIDIA 显卡时返回空列表。"""
    if not shutil.which("nvidia-smi"):
        return []
    fields = [
        "index", "uuid", "name", "utilization.gpu", "utilization.memory",
        "memory.used", "memory.total", "temperature.gpu",
        "power.draw", "power.limit", "fan.speed",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return []
    if out.returncode != 0:
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]

        def num(s):
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        gpus.append({
            "index": parts[0],
            "uuid": parts[1],
            "name": parts[2],
            "util": num(parts[3]),
            "mem_util": num(parts[4]),
            "mem_used": num(parts[5]),
            "mem_total": num(parts[6]),
            "temp": num(parts[7]),
            "power": num(parts[8]),
            "power_limit": num(parts[9]),
            "fan": num(parts[10]),
        })
    return gpus


MODEL_FLAGS = (
    "--model", "--model-path", "--model-name", "--served-model-name",
    "--ckpt", "--checkpoint", "-m",
)


def guess_model(args):
    """从命令行参数里猜模型名：优先 --model 类参数，其次路径形态的参数。"""
    for i, a in enumerate(args):
        if a in MODEL_FLAGS and i + 1 < len(args):
            return args[i + 1].rstrip("/").split("/")[-1]
        for flag in MODEL_FLAGS:
            if a.startswith(flag + "="):
                return a.split("=", 1)[1].rstrip("/").split("/")[-1]
    for i, a in enumerate(args):
        if i == 0 or args[i - 1].startswith("-") or "=" in a:
            continue  # 跳过 flag 的值和 key=value，只看位置参数
        if "/" in a and not a.endswith(".py") and not a.startswith(("/usr", "/opt", "/bin", "/data")):
            return a.rstrip("/").split("/")[-1]
    return None


def describe_proc(pid):
    """进程命令行描述：过滤空 argv；信息不足时沿父进程链找更有信息量的命令行。

    返回 (命令行字符串, 模型名或 None)。
    """
    try:
        proc = psutil.Process(pid)
        args = [a for a in proc.cmdline() if a.strip()]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"<PID {pid} 无权限或已退出>", None

    # 如 vLLM::EngineCore 这类 worker，自身命令行不含模型信息，向父进程找
    if len(args) <= 1:
        try:
            for parent in proc.parents():
                p_args = [a for a in parent.cmdline() if a.strip()]
                if len(p_args) > len(args):
                    shown = " ".join(args) + "（来自: " + " ".join(p_args) + "）"
                    return shown, guess_model(p_args)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return " ".join(args) or f"<PID {pid}>", guess_model(args)


def get_gpu_procs():
    """每块 GPU 上的显存占用进程：{gpu_uuid: [(pid, 命令行, 显存MiB)]}，按显存降序。"""
    if not shutil.which("nvidia-smi"):
        return {}
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return {}
    if out.returncode != 0:
        return {}

    procs = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            mem_mb = float(mem_s)
        except ValueError:
            mem_mb = None
        cmdline, model = describe_proc(pid)
        procs.setdefault(uuid, []).append((pid, cmdline, model, mem_mb))

    for uuid in procs:
        procs[uuid].sort(key=lambda p: p[3] or 0, reverse=True)
    return procs


def render_cpu():
    lines = [section("CPU")]
    pct_total, pct_cores, freq, load = get_cpu()

    info = f"{psutil.cpu_count()} 逻辑核"
    if freq:
        info += f" · {freq.current / 1000:.2f}GHz"
    if load:
        info += f" · 负载 {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"
    lines.append(row("总体占用", pct_total, info))

    if pct_cores:
        per_row = 6
        for i in range(0, len(pct_cores), per_row):
            cells = []
            for j, p in enumerate(pct_cores[i:i + per_row]):
                cells.append(dim(f"{i + j:>3}") + " " + c(pct_color(p), f"{p:5.1f}%"))
            lines.append("│   " + "  ".join(cells))
    return lines


def render_memory():
    lines = [section("内存")]
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    lines.append(row(
        "物理内存", vm.percent,
        f"{fmt_bytes(vm.used)} / {fmt_bytes(vm.total)}（可用 {fmt_bytes(vm.available)}）",
    ))
    lines.append(row(
        "交换分区", sm.percent,
        f"{fmt_bytes(sm.used)} / {fmt_bytes(sm.total)}",
    ))
    return lines


def render_disk():
    lines = [section("磁盘")]
    for path in ("/",):
        du = psutil.disk_usage(path)
        lines.append(row(
            path, du.percent,
            f"{fmt_bytes(du.used)} / {fmt_bytes(du.total)}",
        ))
    return lines


def temp_color(t):
    if t >= 85:
        return "31;1"
    if t >= 70:
        return "33"
    return "32"


def shorten(s, width):
    """超宽时保留开头（程序名/模型路径通常在前），末尾用 … 表示省略。"""
    if disp_width(s) <= width:
        return s
    out = ""
    w = 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width - 1:
            break
        out += ch
        w += cw
    return out.rstrip() + " …"


def render_gpus():
    lines = [section("GPU")]
    gpus = get_gpus()
    if not gpus:
        lines.append("│ " + dim("未检测到 NVIDIA GPU 或 nvidia-smi 不可用"))
        return lines

    gpu_procs = get_gpu_procs()

    for i, g in enumerate(gpus):
        if i:
            lines.append("│")
        lines.append("│ " + bold(f"GPU {g['index']}") + dim(f"  {g['name']}"))

        util = g["util"] or 0
        mem_pct = (g["mem_used"] / g["mem_total"] * 100) if g["mem_total"] else 0
        lines.append(row(
            "  算力", util,
            f"显存控制器 {g['mem_util']:.0f}%" if g["mem_util"] is not None else "",
        ))
        lines.append(row(
            "  显存", mem_pct,
            f"{fmt_mb(g['mem_used'])} / {fmt_mb(g['mem_total'])}"
            if g["mem_total"] else "",
        ))

        stats = []
        if g["temp"] is not None:
            stats.append("温度 " + c(temp_color(g["temp"]), f"{g['temp']:.0f}°C"))
        if g["power"] is not None:
            pl = f" / {g['power_limit']:.0f}W" if g["power_limit"] else "W"
            stats.append("功耗 " + bold(f"{g['power']:.0f}W") + dim(pl))
        if g["fan"] is not None:
            stats.append(f"风扇 {g['fan']:.0f}%")
        if stats:
            lines.append("│     " + dim(" · ").join(stats))

        procs = gpu_procs.get(g["uuid"], [])
        if procs:
            lines.append("│     " + dim("占用进程："))
            term_w = shutil.get_terminal_size((110, 24)).columns
            for j, (pid, cmdline, model, mem_mb) in enumerate(procs):
                branch = "└─" if j == len(procs) - 1 else "├─"
                mem_str = (
                    bold(c(pct_color(mem_mb / g["mem_total"] * 100), fmt_mb(mem_mb)))
                    if mem_mb is not None and g["mem_total"]
                    else dim("?")
                )
                model_str = c(COL_MODEL, f"[{model}] ") if model else ""
                # 前缀约占 30 列，模型名另算；剩余宽度留给命令行
                budget = max(30, term_w - 32 - (disp_width(model) + 3 if model else 0))
                lines.append(
                    f"│     {dim(branch)} {mem_str}"
                    f"  {c('33', f'PID {pid}')}  {model_str}{dim(shorten(cmdline, budget))}"
                )
    return lines


def render_llm():
    lines = [section("LLM 模型")]
    models = get_llm_models()
    if not models:
        lines.append("│ " + dim("网关不可达，或当前没有已加载模型；token 指标暂不可用"))
        return lines

    for m in models:
        slots = m.get("slots")
        active_txt = f"{m['active']}/{slots}" if slots else str(m["active"])
        if m["active"] > 0:
            active = c(COL_WARN if (slots and m["active"] > slots) else COL_OK,
                       f"ACT {active_txt}")
        else:
            active = dim(f"ACT {active_txt}")
        model = c(COL_MODEL, shorten(m["name"], 30))
        p_rate = float(m.get("prompt_rate") or 0)
        t_rate = float(m.get("rate") or 0)
        prefill = bool(m.get("prefill_active"))
        decoding = t_rate > 0 or (m["active"] > 0 and not prefill)
        ps = c(COL_WARN, f"{p_rate:7.0f} p/s") if p_rate > 0 else dim("    -- p/s")
        ts = c(COL_OK, f"{t_rate:7.1f} t/s") if t_rate > 0 else dim("    -- t/s")
        cum = dim(f"累计prompt {fmt_tokens(m.get('prompt_total'))} · "
                  f"累计生成 {fmt_tokens(m.get('generation_total'))}")
        if prefill:
            progress = (m.get("prefill_progress") or 0) * 100
            phase = c(COL_WARN, "Prompt处理")
            detail = c(COL_WARN, f"{fmt_tokens(m.get('prefill_tokens'))} tok {progress:.0f}%")
        elif decoding:
            phase = c(COL_OK, "生成中")
            detail = ""
        else:
            phase = dim("空闲")
            detail = ""
        line = (
            f"│ {model} {dim(pad(m['engine'], 5))} GPU{pad(m['gpus'], 4)} {active}"
            f"  {pad(phase, 9)} {pad(detail, 16)} {ps}  {ts}   {cum}"
        )
        lines.append(line)
    lines.append(
        "│ " + c(COL_WARN, "p/s") + dim("=Prompt处理速率  ")
        + c(COL_OK, "t/s") + dim("=生成速率（两列独立，并发时可同时有值）  "
                                 "ACT=进行中/slot上限  K=千tokens M=百万tokens  "
                                 "累计=模型加载以来(换模型/重启清零)")
    )
    return lines


def render():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = socket.gethostname()
    lines = [
        bold(c(COL_MODEL, f"  系统资源监控")) + dim(f"  {host} · {now}"),
        "",
    ]
    lines += render_cpu() + [""]
    lines += render_memory() + [""]
    lines += render_disk() + [""]
    lines += render_gpus() + [""]
    lines += render_llm()
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="CPU/内存/GPU 资源监控")
    ap.add_argument("-w", "--watch", action="store_true", help="持续刷新模式")
    ap.add_argument("-i", "--interval", type=float, default=2.0, help="刷新间隔秒数（默认 2）")
    ap.add_argument("--no-color", action="store_true", help="禁用彩色输出")
    ap.add_argument("--gateway", default=DEFAULT_GATEWAY,
                    help="模型网关地址（默认 %(default)s，也可用 SYSMON_GATEWAY）")
    ap.add_argument("--theme", default=None,
                    help="配色主题：主题名（themes/<名>.json）或 JSON 文件路径；"
                         "也可用 SYSMON_THEME。默认内置 Tokyo Night")
    ap.add_argument("--list-themes", action="store_true", help="列出可用主题后退出")
    args = ap.parse_args()

    global USE_COLOR, GATEWAY, _THEME_WARN
    GATEWAY = args.gateway
    if args.no_color:
        USE_COLOR = False

    if args.list_themes:
        print("内置默认：tokyo-night")
        d = _theme.THEMES_DIR
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    print(" ", os.path.splitext(f)[0], "->", os.path.join(d, f))
        print("\n用法：--theme <名> 或 --theme /path/to/theme.json（也可设 SYSMON_THEME）")
        return

    _THEME_WARN = _apply_theme(args.theme)
    if _THEME_WARN:
        print(dim("# " + _THEME_WARN), file=sys.stderr)

    psutil.cpu_percent(interval=None)  # 第一次调用总是 0，先预热

    if not args.watch:
        time.sleep(0.2)  # 给 cpu_percent 一个采样窗口
        print(render())
        return

    try:
        while True:
            time.sleep(args.interval)
            print("\033[2J\033[H", end="")  # 清屏并移动光标到左上角
            print(render())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
