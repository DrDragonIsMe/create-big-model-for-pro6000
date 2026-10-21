#!/usr/bin/env python3
"""离线渲染 sysmon TUI 并把屏幕内容存成文本，方便检查布局。"""
import sys, os, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

ROWS, COLS = int(os.environ.get("CAP_ROWS", 44)), int(os.environ.get("CAP_COLS", 140))
FRAMES = int(os.environ.get("CAP_FRAMES", 12))


class FakeCurses:
    pass


fc = FakeCurses()
for i, name in enumerate(("COLOR_BLACK", "COLOR_RED", "COLOR_GREEN", "COLOR_YELLOW",
                          "COLOR_BLUE", "COLOR_MAGENTA", "COLOR_CYAN", "COLOR_WHITE")):
    setattr(fc, name, i)
fc.A_BOLD = 1
fc.COLORS = 256
fc.A_NORMAL = 0
fc.A_REVERSE = 2
fc.KEY_RESIZE = -1
fc.ERR = -1
fc.start_color = lambda: None
fc.use_default_colors = lambda: None
fc.curs_set = lambda v: None
fc.init_pair = lambda *a: None
fc.color_pair = lambda n: n
fc.pair_content = lambda n: (n, -1)
fc.update_lines_cols = lambda: None
class Error(Exception):
    pass
fc.error = Error


class CaptureScratch:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.screen = [[' '] * cols for _ in range(rows)]
        self._keys = []
    def getmaxyx(self):
        return (self.rows, self.cols)
    def addnstr(self, y, x, text, n=0, attr=1):
        n = n if n else len(text)
        for i, ch in enumerate(text[:n]):
            if 0 <= y < self.rows and 0 <= x + i < self.cols:
                self.screen[y][x + i] = ch
    def addch(self, y, x, ch, attr=1):
        c = chr(ch) if isinstance(ch, int) else ch
        if 0 <= y < self.rows and 0 <= x < self.cols:
            self.screen[y][x] = c
    def erase(self):
        self.screen = [[' '] * self.cols for _ in range(self.rows)]
    def refresh(self):
        pass
    def nodelay(self, v):
        pass
    def timeout(self, v):
        pass
    def getch(self):
        # 让 render 主循环多跑几帧后再退出
        self._keys.append(1)
        return -1 if len(self._keys) < FRAMES else ord('q')
    def getstr(self, *a):
        return ''


def fake_wrapper(func, *args):
    scr = CaptureScratch(ROWS, COLS)
    import sysmon
    real_cpu = sysmon.psutil.cpu_percent
    # 让每核/CPU 数值有波动，便于观察波形图
    call_count = {'n': 0}
    def fake_cpu_percent(interval=None, percpu=False):
        v = real_cpu(interval=interval, percpu=percpu)
        call_count['n'] += 1
        if percpu:
            return [min(100.0, c + (i * 37 % 60)) for i, c in enumerate(v)]
        return min(100.0, v + (call_count['n'] * 13 % 55))
    sysmon.psutil.cpu_percent = fake_cpu_percent
    try:
        func(scr, *args)
    except Exception as e:
        import traceback
        traceback.print_exc()
    out_path = SCRIPT_DIR / "sysmon_screenshot.txt"
    with out_path.open('w', encoding='utf-8') as f:
        for row in scr.screen:
            f.write(''.join(row).rstrip() + '\n')
    print(f"Screenshot saved to {out_path} ({ROWS}x{COLS}, {FRAMES} frames)")
    return scr

fc.wrapper = fake_wrapper

import sysmon
sysmon.curses = fc          # 让 sysmon 模块内的所有 curses 调用都走假实现

def fake_llm_models(gateway="", timeout=0.6):
    import math, random
    sysmon.GATEWAY_CONNECTED = True
    models = [
        {"name": "qwen3.6-35b-a3b", "engine": "vllm", "gpus": "1", "active": 2,
         "status": "ready", "prompt_total": 12345, "generation_total": 67890,
         "rate": 148.2, "prompt_rate": 2200.0, "slots": None, "metrics_ok": True,
         "idle": None, "prefill_elapsed": None},
        {"name": "qwen3.8-flash-next-iq3s", "engine": "llamacpp", "gpus": "0", "active": 3,
         "status": "ready", "prompt_total": 4321, "generation_total": 1000,
         "rate": 0.0, "prefill_active": True, "prefill_tokens": 320000,
         "prefill_progress": 0.82, "prefill_elapsed": 12.4, "prompt_rate": 1712.0,
         "slots": 2, "metrics_ok": True, "idle": None},
        {"name": "qwen3-32b-idle", "engine": "vllm", "gpus": "2", "active": 0,
         "status": "ready", "prompt_total": 999, "generation_total": 5366,
         "rate": 0.0, "prompt_rate": 0.0, "slots": None, "metrics_ok": True,
         "idle": 1830.0, "prefill_elapsed": None},
    ]
    # 预置 sparkline 历史，便于观察曲线渲染
    for m in models:
        g = sysmon._gen_hist.setdefault(m["name"], __import__("collections").deque(maxlen=60))
        p = sysmon._prompt_hist.setdefault(m["name"], __import__("collections").deque(maxlen=60))
        if not g:
            n = 46
            for i in range(n):
                w = math.sin(i / 5) * 0.5 + 0.5
                g.append(max(0.0, m["rate"] * (0.5 + w) + random.random() * 6))
                p.append(max(0.0, m["prompt_rate"] * (0.4 + w)))
    return models

sysmon.get_llm_models = fake_llm_models
sysmon.main()
