#!/usr/bin/env python3
"""sysmonitor Web GUI —— 浏览器实时看板（纯标准库，无额外依赖）。

数据与 CLI 完全一致：直接复用 sysmon.py 的采集器（GPU/LLM/CPU/内存/磁盘/
网络/磁盘I/O/进程），后台线程按间隔采样并缓存最新快照，HTTP 端点即取即答。

用法：
    python3 sysmon_web.py                    # http://0.0.0.0:8088
    python3 sysmon_web.py -i 0.5 --port 9000
    浏览器端快捷键与 CLI 相同：Space 暂停  P 进程排序  T 切主题

API：
    GET /                   看板页面
    GET /api/snapshot       最新一帧全量数据（JSON）
    GET /api/themes         可用主题列表
    GET /api/theme?name=X   主题角色色（已转 hex，供前端设 CSS 变量）
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sysmon
import theme as theme_mod

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"

# ─── xterm 256 色 -> hex（与终端显示一致的换算）─────────────────
_BASE16 = ("000000", "800000", "008000", "808000", "000080", "800080",
           "008080", "c0c0c0", "808080", "ff0000", "00ff00", "ffff00",
           "0000ff", "ff00ff", "00ffff", "ffffff")


def x256(i: int) -> str:
    if 0 <= i <= 15:
        return "#" + _BASE16[i]
    if 16 <= i <= 231:
        i -= 16
        r, g, b = i // 36, (i // 6) % 6, i % 6
        lv = lambda v: 0 if v == 0 else 55 + v * 40
        return "#{:02x}{:02x}{:02x}".format(lv(r), lv(g), lv(b))
    if 232 <= i <= 255:
        v = 8 + (i - 232) * 10
        return "#{:02x}{:02x}{:02x}".format(v, v, v)
    return "#ffffff"


def _role_hex(spec):
    """主题角色值 -> hex（int）或 [fg,bg]（list/dict）。"""
    if isinstance(spec, list) and len(spec) == 2:
        return [x256(spec[0]), x256(spec[1])]
    if isinstance(spec, dict):
        return [x256(spec.get("fg", 15)), x256(spec.get("bg", 0))]
    return x256(spec if isinstance(spec, int) else 15)


# ─── 采样线程：缓存最新快照 ────────────────────────────────────
_LOCK = threading.Lock()
_SNAP: dict = {}


def _fmt_proc(p, vram_map, gpu_total) -> dict:
    g = vram_map.get(p["pid"])
    return {"pid": p["pid"], "user": p["user"], "cpu": round(p["cpu"], 1),
            "pmem": round(p["pmem"], 1), "rss": p["rss"],
            "cmd": p["cmd"][:220],
            "vram": round(g["mem"], 0) if g and g.get("mem") else None,
            "gpu": g["gpu"] if g else None,
            "model": g.get("model") if g else None}


def collect_once(gateway: str) -> dict:
    """一轮采样 -> 可直接 JSON 序列化的快照（口径与 CLI 画面一致）。"""
    s = sysmon.collect_samples(gateway, 1.0)
    gpus, models = s["gpus"], s["models"]
    cpu, mem, disk = s["cpu"], s["mem"], s["disk"]

    vram_map = {p["pid"]: p for g in gpus for p in g["procs"]}

    # 进程表：三个排序键各取前 25 的并集（与 CLI P 键的三种排序对齐）
    procs = sysmon.get_proc_list()
    picks = {}

    def take(key, rev=True, n=25, filt=None):
        rows = [p for p in procs if filt(p)] if filt else list(procs)
        for p in sorted(rows, key=key, reverse=rev)[:n]:
            picks[p["pid"]] = p

    take(lambda p: vram_map.get(p["pid"], {}).get("mem") or 0)
    take(lambda p: p["cpu"])
    take(lambda p: p["pmem"])
    proc_rows = [_fmt_proc(p, vram_map, None) for p in picks.values()]
    proc_rows.sort(key=lambda r: -(r["vram"] or 0) if r["vram"] else -1e9)

    vm, sm = mem["vm"], mem["sm"]
    du = disk["du"]
    agg = {
        "t": round(sum(float(m.get("rate") or 0) for m in models), 1),
        "p": round(sum(float(m.get("prompt_rate") or 0) for m in models), 1),
        "req": sum(int(m["active"]) for m in models),
        "queued": sum(1 for m in models if m.get("slots") and m["active"] > m["slots"]),
    }
    return {
        "ts": time.time(),
        "host": socket.gethostname(),
        "uptime": sysmon.get_uptime(),
        "gateway": gateway,
        "gateway_connected": bool(sysmon.GATEWAY_CONNECTED),
        "gpus": [{k: g[k] for k in ("index", "name", "util", "mem_util",
                                    "mem_used", "mem_total", "temp", "power",
                                    "power_limit", "fan")} for g in gpus],
        "models": models,
        "agg": agg,
        "cpu": {**cpu, "per_core": [round(c, 1) for c in cpu["per_core"]],
                "load": [round(x, 2) for x in (cpu["load"] or (0, 0, 0))]},
        "mem": {"pct": round(vm.percent, 1), "used": vm.used, "total": vm.total,
                "avail": vm.available,
                "swap_pct": round(sm.percent, 1), "swap_used": sm.used,
                "swap_total": sm.total},
        "disk": {"pct": round(du.percent, 1) if du else 0,
                 "used": du.used if du else 0, "total": du.total if du else 0,
                 "free": du.free if du else 0,
                 "read_bps": round(disk["read_bps"], 0),
                 "write_bps": round(disk["write_bps"], 0)},
        "net": s["net"],
        "disks": s["diskio"],
        "procs": proc_rows,
        "procs_total": len(procs),
    }


def sampler_loop(gateway: str, interval: float):
    global _SNAP
    while True:
        t0 = time.monotonic()
        try:
            snap = collect_once(gateway)
        except Exception as e:                      # 采集异常不能让线程死掉
            with _LOCK:
                _SNAP = {**_SNAP, "error": f"{type(e).__name__}: {e}"}
            snap = None
        if snap is not None:
            with _LOCK:
                _SNAP = snap
        time.sleep(max(0.05, interval - (time.monotonic() - t0)))


# ─── HTTP ──────────────────────────────────────────────────────
_STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
           "/app.js": ("app.js", "application/javascript; charset=utf-8"),
           "/style.css": ("style.css", "text/css; charset=utf-8")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                        # 安静模式

    def _send(self, code: int, body: bytes, ctype: str, cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in _STATIC:
            fn, ctype = _STATIC[u.path]
            p = WEB_DIR / fn
            if p.is_file():
                self._send(200, p.read_bytes(), ctype, cache="no-cache")
            else:
                self._send(404, f"{fn} 不存在（web/ 目录不完整？）".encode(),
                           "text/plain; charset=utf-8")
            return
        if u.path == "/api/snapshot":
            with _LOCK:
                self._json(_SNAP or {"error": "采样线程尚未完成第一轮"})
            return
        if u.path == "/api/themes":
            self._json({"themes": sysmon._theme_names(),
                        "default": theme_mod.DEFAULT_THEME_NAME})
            return
        if u.path == "/api/theme":
            name = urllib.parse.parse_qs(u.query).get("name", [None])[0]
            th, warn = theme_mod.load_theme(name)
            colors = {r: _role_hex(v) for r, v in th["colors"].items()}
            self._json({"name": th["name"], "colors": colors,
                        "gpu": [x256(i) for i in th.get("gpu", [])],
                        "warning": warn})
            return
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser(description="sysmonitor Web GUI（数据与 CLI 一致）")
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="后台采样间隔秒（默认 1.0）")
    ap.add_argument("--gateway", default=sysmon.DEFAULT_GATEWAY,
                    help="模型网关地址（默认 %(default)s）")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0 局域网可达）")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("SYSMON_WEB_PORT", 8088)),
                    help="监听端口（默认 8088，或 SYSMON_WEB_PORT）")
    args = ap.parse_args()

    t = threading.Thread(target=sampler_loop, args=(args.gateway, args.interval),
                         daemon=True)
    t.start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sysmonitor Web  →  http://{socket.gethostname() or 'localhost'}:{args.port}/"
          f"（网关 {args.gateway}，采样 {args.interval}s）")
    print("Ctrl-C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
