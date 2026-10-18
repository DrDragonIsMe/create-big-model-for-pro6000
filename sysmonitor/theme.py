#!/usr/bin/env python3
"""sysmonitor 配色主题加载（curses 与文本版共用）。

主题文件为 JSON，按 --theme 参数或 SYSMON_THEME 环境变量选择：
  - 给的是文件路径（绝对或相对）则直接加载；
  - 否则按主题名在本目录 themes/<name>.json 查找；
  - 都没有 / 解析失败时回退内置 Tokyo Night，并把原因写入 load_warning。

JSON 格式（值为 256 色索引 0..255；header 可给 fg/bg）：
{
  "name": "my-theme",
  "colors": {
    "header": {"fg": 0, "bg": 111},
    "border": 238, "dim": 60, "accent": 117, "ok": 149, "warn": 179,
    "crit": 210, "mem": 116, "cpu": 149, "disk": 215, "model": 141
  },
  "gpu": [111, 141, 117, 116]
}
只写想覆盖的角色即可，其余沿用默认；未知角色忽略。8 色终端的回退色
固定按语义角色映射（见 BASIC_*），不读文件。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES_DIR = os.path.join(HERE, "themes")
DEFAULT_THEME_NAME = "tokyo-night"

# 语义角色；curses/text 两端都用这些名字取色
ROLES = ("header", "border", "dim", "accent", "ok", "warn", "crit",
         "mem", "cpu", "disk", "model")

# ── 内置默认：Tokyo Night（256 色索引）──
# header 为 [fg, bg]；其余为前景索引
DEFAULT_THEME = {
    "name": "tokyo-night (built-in)",
    "colors": {
        "header": [0, 111],     # 黑字 #7aa2f7 蓝底
        "border": 238,          # #414868
        "dim": 60,              # #565f89
        "accent": 117,          # #7dcfff
        "ok": 149,              # #9ece6a
        "warn": 179,            # #e0af68
        "crit": 210,            # #f7768e
        "mem": 116,             # #73daca
        "cpu": 149,             # #9ece6a
        "disk": 215,            # #ff9e64
        "model": 141,           # #bb9af7
    },
    "gpu": [111, 141, 117, 116],  # 蓝/紫/青/碧
}

# 8 色终端回退：角色 -> 逻辑色名（curses / ANSI 各自再翻译）
BASIC_COLORS = {
    "header": ("black", "blue"),   # (fg, bg)
    "border": "white", "dim": "white", "accent": "cyan",
    "ok": "green", "warn": "yellow", "crit": "red",
    "mem": "cyan", "cpu": "green", "disk": "yellow", "model": "magenta",
}
BASIC_GPU = ["blue", "magenta", "cyan", "green"]


def _coerce_color(v):
    """把 JSON 里的颜色值规范成 int 或 [fg, bg]，非法返回 None。"""
    if isinstance(v, int) and 0 <= v <= 255:
        return v
    if isinstance(v, list) and len(v) == 2 \
            and all(isinstance(x, int) and 0 <= x <= 255 for x in v):
        return [v[0], v[1]]
    if isinstance(v, dict):
        fg, bg = v.get("fg"), v.get("bg")
        if isinstance(fg, int) and isinstance(bg, int) \
                and 0 <= fg <= 255 and 0 <= bg <= 255:
            return [fg, bg]
    return None


def _theme_path(spec):
    if os.path.isfile(spec):
        return spec
    cand = os.path.join(THEMES_DIR, spec if spec.endswith(".json") else f"{spec}.json")
    return cand if os.path.isfile(cand) else None


def load_theme(spec=None):
    """返回 (theme, warning)。theme 结构同 DEFAULT_THEME；warning 为 None 或说明串。"""
    import copy
    theme = copy.deepcopy(DEFAULT_THEME)
    if not spec:
        spec = os.environ.get("SYSMON_THEME")
    if not spec:
        return theme, None

    path = _theme_path(spec)
    if path is None:
        return theme, f"主题 {spec!r} 未找到（查找 {THEMES_DIR}），使用内置 Tokyo Night"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = data.get("colors") or {}
        unknown = []
        for role, val in colors.items():
            if role not in ROLES:
                unknown.append(role)
                continue
            cv = _coerce_color(val)
            if cv is not None:
                theme["colors"][role] = cv
        gpu = data.get("gpu")
        if isinstance(gpu, list):
            g = [x for x in gpu if isinstance(x, int) and 0 <= x <= 255]
            if g:
                theme["gpu"] = g
        if isinstance(data.get("name"), str):
            theme["name"] = data["name"]
        else:
            theme["name"] = os.path.splitext(os.path.basename(path))[0]
        warn = None
        if unknown:
            warn = f"主题 {path} 含未知角色（已忽略）: {', '.join(unknown)}"
        return theme, warn
    except Exception as e:
        return DEFAULT_THEME, f"主题 {path} 加载失败（{e}），使用内置 Tokyo Night"


def _list_available_themes():
    """返回主题列表（不含内置默认，仅 themes/ 目录中的文件）。"""
    if not os.path.isdir(THEMES_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(THEMES_DIR) if f.endswith(".json"))
