# sysmonitor — 本地 AI 工作站实时监控

终端里的 nvtop 式监控，**只呈现当前的压力与负载**（不画时间曲线）：
**GPU 算力/显存/带宽 + CPU/内存/磁盘 + 网络收发 + 磁盘 I/O + 全系统进程 + LLM 模型吞吐**
（t/s 生成速率、p/s Prompt 处理速率、Prompt 进度动画、slots 占用/排队、累计 tokens）。
数据源：`nvidia-smi`、`psutil`、网关 `GET /v1/gateway/status`（无需额外服务）。

所有百分比都是一行式**当前值仪表**：`▰▱▱▱▱▱` ▰=已占用、▱=余量，即采即画。
配色语义：压力类（算力/带宽/CPU/交换/磁盘）按值分级着色 —— <span>绿 &lt;60%</span>、
<span>黄 60–85%</span>、<span>红 ≥85%</span>；容量类（显存/内存）固定青蓝 ——
模型常驻显存是"占"不是"忙"，避免误读为过载。

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
python3 sysmon.py                       # 默认：数据 1s 刷新、catppuccin-mocha 主题
python3 sysmon.py -i 0.5                # 数据采样间隔（动画独立 ~8fps，不受 -i 影响）
python3 sysmon.py --theme dracula       # 指定配色主题
python3 sysmon.py --gateway http://192.168.0.103:8000   # 远程网关
python3 sysmon.py --no-anim             # 静止模式（无动画）
python3 sysmon.py --list-themes         # 列出全部主题
```

### 快捷键

| 键 | 作用 |
|---|---|
| `Space` | 暂停/继续数据采样（数据与动效全冻结便于读数；标题墙钟继续走） |
| `A` / `a` | 开/关动画（脉冲、转圈、仪表微光、网速箭头、数值滑行；关闭即回退精确静态值） |
| `P` | 进程表排序循环：显存 → %CPU → %MEM |
| `T` | 下一个配色主题 |
| `Shift+Tab` | 上一个配色主题 |
| `q` / `Esc` | 退出 |

### 动效设计（参考 btop/htop 惯例，克制不扰读）

| 动效 | 表现 | 说明 |
|---|---|---|
| 平滑过渡 | 仪表条与数字在采样间隔内 ease-out 滑行（~0.5s 收敛） | 条不再每秒一跳；`A` 关/`Space` 暂停时直接显示真实值 |
| 微光扫掠 | 已填充段一个亮格自左向右漂移，其余略暗 | btop 式 shimmer；窄条（<10格）自动跳过 |
| 流向箭头 | 网速 `↓↑` 有流量时与 `⇣⇡` 交替 | 一眼区分"在流动"与"静止" |
| 进度分层 | Prompt 进度条实心段亮、余量段暗 | 与仪表同一视觉语言 |
| 阶段动画 | `⠋` Prompt 转圈 / `●○` 生成脉冲 / `◐` 加载 / 排队脉冲 | 原有保留 |

### 界面结构

```
 SYSMON  主机名  运行1d21h  时间                       ● Σ 317.3 t/s · 3912 p/s  ← 顶栏：吞吐+时长
[GPU 0] RTX PRO 6000 ...                │[GPU 1] ...          ← 每卡：温度/功耗/风扇
算力 100.0% ▰▰▰▰▰▰▰▰▰▰▰ (红)            │算力 ...              ← 三条当前值仪表
显存  93.9% ▰▰▰▰▰▰▰▰▱ (蓝) 89.8G/95.6G  │显存 ...              空闲量在明细里
带宽  52.0% ▰▰▰▰▰▱▱▱ (黄)               │带宽 ...
─────────────────────────────────────────
 LLM  Σ  317.3 t/s  3912 p/s  请求 5 ▲1排队              ● 网关   ← 聚合头（活动有脉冲）
  qwen3.8-flash-next vLLM ·GPU0,1  ⠼ Prompt ▰▰▰▰▱▱ 82% 320Ktok 12s  ← 第一行：阶段动画
  t/s   317.2  p/s   2200  ACT ▰▰▱▱ 2/2 ○+1排队  Σprompt 823M  Σgen 2.9M ← 第二行：当前值
─────────────────────────────────────────
CPU 42.3% ▰▰▰▰▱▱ load 3.9 3.0 1.8 3.96GHz │内存  70.2% ▰▰▰▰▰▰▱ 86.6G/123.3G 可用36.7G
每核 24: ▄▂█▁▆...（当前值方块）             │交换 / 磁盘 各一行仪表
 0:.. 1:.. 每核数字表（补足到右列同高）     │网络  ↓入 ↑出        ← 网卡实时速率
                                          │enp131s0 ↓14.8K/s ▱▱ ↑11.9K/s ▱▱ 192.168…
                                          │Σ累计 ↓10.0G ↑3.4G（开机以来）
                                          │磁盘 I/O  util=繁忙度
                                          │nvme0n1 12.3% ▰▰▱▱▱ 读23.1M/s 写4.5M/s
─────────────────────────────────────────
进程表（全系统，P 键切换排序 显存→%CPU→%MEM）：GPU/PID/USER/%CPU/%MEM/RSS/VRAM + 显存仪表 + 模型/命令行
```

### 显示区域

| 区域 | 内容 |
|---|---|
| GPU 面板 | 每卡：温度/功耗/风扇 + 算力、显存（已用/总量/空闲）、带宽 三条仪表 |
| LLM 面板 | 阶段动画、当前 t/s 与 p/s、ACT slot 占用仪表+排队、累计 Σ（见下表） |
| CPU 面板 | 总体仪表 + load/频率 + 每核方块图 + 每核数字表 |
| 系统仪表 | 内存（蓝，容量）、交换、磁盘（含读/写速率明细） |
| 网络 | 每活动网卡（排除 lo/veth/docker）当前 ↓入/↑出 速率 + mini 仪表 + IP + 累计流量；满格=链路速率（未知取本批最活跃口） |
| 磁盘 I/O | 物理盘 util（busy_time 差分，即盘繁忙度）+ 读/写吞吐 + IOPS，util 高=盘是瓶颈 |
| 进程表 | 全系统进程（内核线程除外）；`P` 键循环排序 显存→%CPU→%MEM；GPU 显存行带仪表与模型标签，busy 行分级高亮 |

### LLM 面板字段

| 字段 | 含义 |
|---|---|
| 阶段动画 | `⠋ Prompt ▰▱ %` 预填充中（转圈+进度条+已处理 tok+耗时）；`● 生成中` 呼吸脉冲；`◐ 加载` 转圈；`空闲 3m12s` 距上次活动 |
| `t/s` | 当前生成 tokens 速率（绿）；无活动显示 `--` |
| `p/s` | 当前 Prompt 处理 tokens 速率（琥珀）；与 t/s 独立，多路并发时可同时有值 |
| `ACT ▰▰▱▱ 2/2` | slot 占用仪表（进行中/slot 上限）；超限画满并显示 `○+N排队`（黄，脉冲） |
| `Σprompt` `Σgen` | 自模型加载以来累计（K=千 M=百万 tokens；换模型/重启清零） |
| `● 网关` | 网关连接状态；有吞吐时脉冲动画 |

### 颜色语义

| 颜色 | 含义 |
|---|---|
| 绿 ▰ | 压力 < 60% |
| 黄 ▰ | 压力 60–85%（忙） |
| 红 ▰ | 压力 ≥ 85%（满载/告警，温度同理） |
| 蓝 ▰ | 容量占用类：显存、内存（不表示压力） |
| 紫 | 模型名标签 |
| 主题强调底 | 面板标题/状态栏（深字浅底，随主题） |

## 主题（7 套）

`catppuccin-mocha`（默认）、`tokyo-night`、`nord`、`dracula`、`gruvbox-dark`、`one-dark`、`solarized-dark`

- 运行时：`T` / `Shift+Tab` 循环切换（curses 版）
- 启动指定：`--theme <名>` 或环境变量 `SYSMON_THEME=<名>`，也可直接给 JSON 文件路径
- 自定义：复制 `themes/catppuccin-mocha.json` 改色即可被自动发现。值为 256 色索引
  （0-255），`header` 可写 `{"fg":0,"bg":111}`；未写的角色沿用内置兜底（Tokyo Night 系）。
  语义角色：`header border dim accent ok warn crit mem cpu disk model` + `gpu`（面板色数组）

## sysmon_text.py（文本版）

```bash
python3 sysmon_text.py                  # 单次快照（适合脚本/巡检）
python3 sysmon_text.py -w -i 2          # 持续刷新（默认 2s）
python3 sysmon_text.py -w -a            # 动画模式：数据间隔内 ~8fps 重绘 spinner/脉冲（需 tty）
python3 sysmon_text.py --no-color       # 去色（重定向文件时用，或设 NO_COLOR=1）
python3 sysmon_text.py -w >> monitor.log  # 记录时间序列日志
```

文本版无交互按键；区块顺序 CPU→内存→磁盘→GPU→LLM→网络→磁盘 I/O→进程 Top；
所有占用同样用 ▰▱ 仪表条按上述颜色语义着色；
`-w` 下每轮重打数据（含网关请求），`-a` 的动画帧复用快照不额外打接口。

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `SYSMON_GATEWAY` | 网关地址 | `http://localhost:8000` |
| `SYSMON_THEME` | 配色主题名或 JSON 路径 | catppuccin-mocha |
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
- **动画静止**：按 `A` 确认未关动画；数据暂停（`Space`）时相位也会冻结。
- **LLM 面板显示"--"或离线**：检查 `SYSMON_GATEWAY` 与网关状态 `curl <网关>/v1/gateway/status`；
  速率需至少 2 个采样窗口才有值。
- **显存表是蓝的不是红的**：有意设计——显存/内存是容量占用（蓝），算力/CPU 才是压力（绿黄红）。
- **颜色发灰/错乱**：终端未开 256 色，设 `TERM=xterm-256color`；纯 8 色终端自动回退到语义基础色。
- **字符显示成口框**：字体需支持 Block Elements（U+2580-259F、▰▱）与 Braille（⠋ 等），
  JetBrains Mono / Sarasa / Nerd Font 均覆盖。

## 说明

早期版本提供过 60 点滚动负载图（面积/波柱/频谱/热力 4 风格 + `C` 键切换），
后按使用反馈移除：日常值守只关心**此刻**的压力与负载，当前值仪表更直观且省屏。
