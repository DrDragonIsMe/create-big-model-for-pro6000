# sysmonitor — 本地 AI 工作站实时监控

终端里的 nvtop 式监控：**GPU 利用率/显存/温度/功耗 + CPU/内存/磁盘 + LLM 模型吞吐**
（t/s 生成速率、p/s Prompt 处理速率、Prompt 进度动画、slots 并发/排队、累计 tokens）。
数据源：`nvidia-smi`、`psutil`、网关 `GET /v1/gateway/status`（无需额外服务）。

## 快速开始

```bash
cd ~/AI
python3 sysmonitor/sysmon.py            # 图形界面（推荐，需要 256 色终端 ≥100x30）
python3 sysmonitor/sysmon_text.py -w    # 文本刷新版（适合 ssh / 窄终端 / 录日志）
```

依赖：Python 3 + `psutil`（`pip install psutil`）。无 NVIDIA 显卡时自动隐藏 GPU 面板；
网关不可达时 LLM 面板显示离线，其余照常。

## 三个脚本

| 脚本 | 用途 |
|---|---|
| `sysmon.py` | curses 交互式 TUI，全功能 + 快捷键 |
| `sysmon_text.py` | 纯文本快照/刷新，可管道、可重定向日志 |
| `sysmon_capture.py` | 离线渲染一屏存 `sysmon_screenshot.txt`（写文档/报告用；环境变量 `CAP_ROWS/CAP_COLS/CAP_FRAMES` 控制画布） |

## sysmon.py（curses 版）

```bash
python3 sysmon.py                       # 默认：数据 1s 刷新、wave 柱图、tokyo-night 主题
python3 sysmon.py -i 0.5                # 数据采样间隔（动画独立 ~8fps，不受 -i 影响）
python3 sysmon.py --chart spectrum      # 指定负载图风格
python3 sysmon.py --theme dracula       # 指定配色主题
python3 sysmon.py --gateway http://192.168.0.103:8000   # 远程网关
python3 sysmon.py --no-anim             # 静止模式（无动画）
python3 sysmon.py --list-themes         # 列出全部主题
```

### 快捷键

| 键 | 作用 |
|---|---|
| `C` | **切换负载图风格**：area → wave → spectrum → heat 循环（状态栏显示当前风格） |
| `Space` | 暂停/继续数据采样（画面冻结便于读数，动画不停） |
| `A` | 开/关动画（脉冲、转圈、波纹） |
| `T` | 下一个配色主题 |
| `Shift+Tab` | 上一个配色主题 |
| `q` / `Esc` | 退出 |

### 界面结构

```
 SYSMON  主机名  时间                                        ● Σ 317.3 t/s   ← 顶栏：全局吞吐
[GPU 0] RTX PRO 6000 ...                │[GPU 1] ...          ← 每卡：温度/功耗/风扇/显存空闲
算力  100.0%  带宽  53%                 │                     ← 柱图/面积图（C 键换风格）
显存  93.9%   89.8G/95.6G               │
─────────────────────────────────────────
 LLM  Σ  317.3 t/s  3912 p/s  请求 5 ▲1排队              ● 网关   ← 聚合头（活动有脉冲）
  qwen3.8-flash-next vLLM ·GPU0,1  ⠼ Prompt ▰▰▰▰▱▱ 82% 320Ktok 12s  ← 第一行：阶段动画
  t/s███▇▂ 317.3  p/s▁▁█▁ 2200  ACT 3/2 排队  Σprompt 823M  Σgen 2.9M ← 第二行：曲线+数值
─────────────────────────────────────────
CPU 面积图 + 每核方块/数字              │内存/交换/磁盘
─────────────────────────────────────────
GPU 进程表：PID/用户/CPU/内存/显存条/模型/命令行
```

### LLM 面板字段

| 字段 | 含义 |
|---|---|
| 阶段动画 | `⠋ Prompt ▰▱ %` 预填充中（转圈+进度条+已处理 tok+耗时）；`● 生成中` 呼吸脉冲；`◐ 加载` 转圈；`空闲 3m12s` 距上次活动 |
| `t/s` | 生成 tokens 速率（绿） |
| `p/s` | Prompt 处理 tokens 速率（琥珀）；与 t/s 独立，并发时可同时有值 |
| `▁▂▄█` 曲线 | 近 60 次采样速率 sparkline，最后一格亮色 = 当前值 |
| `ACT 3/2` | 进行中请求/slot 上限；超 slot 显示 `○ 3/2 排队`（黄） |
| `Σprompt` `Σgen` | 自模型加载以来累计（K=千 M=百万 tokens；换模型/重启清零） |
| `● 网关` | 网关连接状态；有吞吐时脉冲动画 |

## 负载图 4 种风格（`C` 键）

| 风格 | 图形 | 特点 | 适合 |
|---|---|---|---|
| `area` 面积图 | `▄▀█` 滚动波形 + `▁` 峰值标记 | 经典 nvtop，每字符 2px 垂直分辨率最高 | 看长期趋势 |
| `wave` 波动直方图（**默认**） | 每样本一根柱，柱顶 `▁▂▃▄▅▆▇` 水面波纹 | 柱高即负载；波纹出现 = 数值真的在波动 | 一眼判断忙/闲/抖动 |
| `spectrum` 频谱柱 | 柱按高度绿→黄→红分级着色 + `▔` 峰值落帽 | 颜色即告警等级；帽悬在柱上方 ~6s 缓落，冲高回落看得清 | 盯高负载/超频区间 |
| `heat` 热力条 | 单行 `░▒▓█` 密度+按值着色 | 最省行 | 矮终端、多面板 |

留白 = 尚无数据；`░` 底纹柱 = 真实 0% 采样，两者有意区分。

## 主题（7 套）

`tokyo-night`（默认）、`catppuccin-mocha`、`nord`、`dracula`、`gruvbox-dark`、`one-dark`、`solarized-dark`

- 运行时：`T` / `Shift+Tab` 循环切换（curses 版）
- 启动指定：`--theme <名>` 或环境变量 `SYSMON_THEME=<名>`，也可直接给 JSON 文件路径
- 自定义：复制 `themes/tokyo-night.json` 改色即可被自动发现。值为 256 色索引
  （0-255），`header` 可写 `{"fg":0,"bg":111}`；未写的角色沿用默认。
  语义角色：`header border dim accent ok warn crit mem cpu disk model` + `gpu`（曲线色数组）

## sysmon_text.py（文本版）

```bash
python3 sysmon_text.py                  # 单次快照（适合脚本/巡检）
python3 sysmon_text.py -w -i 2          # 持续刷新（默认 2s）
python3 sysmon_text.py -w -a            # 动画模式：数据间隔内 ~8fps 重绘 spinner/脉冲（需 tty）
python3 sysmon_text.py --chart spectrum # 速率曲线分级着色（spectrum/heat；默认 area 方块）
python3 sysmon_text.py --no-color       # 去色（重定向文件时用，或设 NO_COLOR=1）
python3 sysmon_text.py -w >> monitor.log  # 记录时间序列日志
```

文本版无交互按键；`-w` 下每轮重打数据（含网关请求），`-a` 的动画帧复用快照不额外打接口。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `SYSMON_GATEWAY` | 网关地址 | `http://localhost:8000` |
| `SYSMON_THEME` | 配色主题名或 JSON 路径 | tokyo-night |
| `SYSMON_CHART` | 负载图风格 area/wave/spectrum/heat | curses 版 wave，文本版 area |
| `NO_COLOR` | 置任意值 = 文本版去色 | — |
| `CAP_ROWS/CAP_COLS/CAP_FRAMES` | capture 画布与帧数 | 44/140/12 |

## 目录

```
sysmonitor/
├── sysmon.py            # curses TUI
├── sysmon_text.py       # 文本版
├── sysmon_capture.py    # 离线截图渲染
├── theme.py             # 主题加载器（两版共用）
└── themes/*.json        # 7 套配色
```

## 排障

- **主题键按了没反应**：部分终端把 Shift+Tab 映射成别的序列，`T` 正循环总可用。
- **波纹/动画静止**：按 `A` 确认未关动画；`--no-anim` 或数据暂停（`Space`）时相位也会冻结。
- **LLM 面板显示"--"或离线**：检查 `SYSMON_GATEWAY` 与网关状态 `curl <网关>/v1/gateway/status`；
  速率需至少 2 个采样窗口才有值。
- **颜色发灰/错乱**：终端未开 256 色，设 `TERM=xterm-256color`；纯 8 色终端自动回退到语义基础色。
- **字符显示成口框**：字体需支持 Block Elements（U+2580-259F）与 Braille（⠋ 等），
  JetBrains Mono / Sarasa / Nerd Font 均覆盖。
