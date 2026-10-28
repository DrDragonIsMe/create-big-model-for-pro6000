# 本地 AI 服务（LLM + 视频生成 + 对口型数字人）

Docker 部署，GPU 数量 1~2 张、单卡 48~96GB 自适应（摆放按本机 nvidia-smi 实算）。
本目录是全网同构服务器的源参考蓝本（家庭 `home-pro6000-dual`）；`tju-pro6000-single`（1×96GB）与 `workstation-4090-dual`（2×48GB）跑同一套代码。

## 日常使用

```bash
cd ~/AI
./scripts/start.sh          # 交互菜单：选 LLM（未下载自动先下载，已下载自动校准）→ 灰化不可用视频项 → 启动
./scripts/start.sh --switch video   # 快速切到视频服务（卸文本、起 ComfyUI）
./scripts/start.sh --switch llm     # 快速切回文本服务（懒加载：不预热，请求到了自动加载）
# 远程切换（局域网客户端无需 SSH，等价 --switch）：
curl -X POST http://<服务器IP>:8000/v1/gateway/switch -H 'Content-Type: application/json' -d '{"target":"video"}'
curl -X POST http://<服务器IP>:8000/v1/gateway/switch -H 'Content-Type: application/json' -d '{"target":"llm"}'
# 客户端 agent 首次连接先学习网关能力/协商协议（GPU 数量、按卡加载、状态轮询）：
curl http://<服务器IP>:8000/v1/gateway/capabilities    # 说明接口：能力、协商步骤、模型目录
curl http://<服务器IP>:8000/v1/gateway/gpus            # GPU 发现：数量、每卡占用、谁占了卡
# 把模型加载到指定 GPU（wait=false 异步返回 request_id，轮询 ops/{id} 拿加载状态）：
curl -X POST http://<服务器IP>:8000/v1/gateway/load -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b","gpus":[0],"wait":false}'
# 布局模式（2026-10-26）：查每个模型本机可用摆放（dual/tp/replica/pool/auto）+ ctx 估算 + 宿主持久化命令，mode 参数当场切：
curl http://<服务器IP>:8000/v1/gateway/modes?model=qwen3.8-27b
curl -X POST http://<服务器IP>:8000/v1/gateway/load -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b","mode":"tp","wait":false}'
./scripts/start.sh --down           # 停止全部服务（只停不删；--clean 才删容器）
./scripts/manage.sh list    # 查看模型清单与下载状态
./scripts/manage.sh download <名称>       # 下载模型主仓库（LLM 或视频）
./scripts/manage.sh download <名称> fp8   # 下载官方 FP8 变体到 <名称>-fp8（网关优先使用）
./scripts/manage.sh add     # 新增模型到清单（下次启动菜单可见）
./scripts/manage.sh check [--fix]  # 按模型文件校准清单（think/上下文/工具格式）
./scripts/manage.sh verify  # 通过网关逐模型实测：正文非空 + 工具调用解析
# GUI 管理后台（集群总览/下载/布局切换/任务/监控，等价上面这些命令的点点点）：浏览器开 http://<本机IP>:9001
python3 scripts/gen_video.py "提示词"   # MiniMax H3 文生视频（--turbo 加速，--first/--last 首/尾帧）
python3 sysmonitor/sysmon.py            # 实时监控：GPU/CPU/网速/磁盘IO/进程/模型吞吐，全当前值仪表（q退出 Space暂停 A动画 P进程排序 T切主题）
python3 sysmonitor/sysmon_web.py        # 实时监控 Web 版：浏览器看板 http://<本机IP>:8088（已随 compose 常驻 ai-sysmon-web；与 CLI 可同开）
```

### 集群统一管理 CLI

`./scripts/ai` 是四节点统一的远程管理入口，直接复用各节点现有 VPN/HTTP 网关，不需要 SSH，也不会修改节点信任关系。节点可用完整 `node_id` 或别名 `home` / `111` / `tju` / `dgx`；默认节点为 `home`。

```bash
./scripts/ai nodes                         # 节点、VPN 地址与协议能力
./scripts/ai status --all                  # 并发查询四节点；单点失败不影响其他节点
./scripts/ai models --all                  # 全部模型与加载状态
./scripts/ai --node 111 gpus               # GPU 容量、占用与在途请求
./scripts/ai --node tju modes --model qwen3.8-27b
./scripts/ai --node tju load qwen3.8-27b --async
./scripts/ai --node tju ops <request_id> --watch
./scripts/ai --node tju unload qwen3.8-27b
./scripts/ai --node home load qwen3.8-27b --mode dual
./scripts/ai --node home unload --all
./scripts/ai --node home switch llm
./scripts/ai --json status --all           # 机器可读输出，错误也为 JSON
```

`load` 默认同步等待就绪，`--async` 返回 `request_id`；`ops --watch` 默认最多等待 900 秒。`--gpus` / `--mode` / `--replicas` 只透传给网关，布局校验、显存调度和繁忙保护仍由网关执行。`--timeout` 控制单次 HTTP 超时，`ops` 另有 `--watch-timeout` 与 `--interval`。

DGX 是独立栈，只支持 `nodes` / `status` / `models` 只读查询；`load` / `unload` / `switch` / `modes` / `ops` 会明确报告不支持管理协议。任何节点不可达都会打印真实 HTTP/网络原因；`--all` 查询继续覆盖其他节点，全部完成后如有失败退出码为 1（用法错误为 2）。

非交互启动示例：
```bash
./scripts/start.sh --llm qwen3.8-27b --video wan2.2-s2v -y
./scripts/start.sh --llm qwen3.8-27b --tp 2 -y          # 跨卡 TP：整池显存换单请求 ctx（2×48GB：193K→895K），解码速度不涨
./scripts/start.sh --llm qwen3.8-27b --replicas 2 -y    # 每卡一份副本：聚合吞吐≈×2、每流 ctx 不变，视频让位
```

## 服务入口

- LLM API（OpenAI 兼容）: http://localhost:8000/v1/chat/completions
- ComfyUI 图形界面: http://localhost:8188（第二实例 8189）
- 监控看板（Web，常驻）: http://localhost:8088 —— sysmonitor GPU/吞吐/网速/磁盘/进程实时仪表，含温度与系统压力告警区；支持跟随系统 / 浅色 / 深色；终端版 `python3 sysmonitor/sysmon.py` 可同时开
- **管理后台（GUI，常驻）**: http://localhost:9001 —— 集群总览、温度/负载告警、DGX head/worker 统一内存状态、模型吞吐、内存/磁盘/网络与进程详情，以及模型下载、布局切换（gateway-layout-v1 目录）和任务日志；打开任意一台即可看到四台（浏览器直连各节点 agent:9001，DGX 经 relay 只读）。控制台支持跟随系统 / 浅色 / 深色。agent 只是遥控器：下载=manage.sh、持久布局=menu.py、运行态切换=网关，写配置的仍然只有 menu.py

文本模型 ↔ 视频的切换、`--switch` 快速切换（含远程 `gateway/switch` 接口）、显存腾挪、管理接口（`status` / `unload_all` / `switch`）、以及客户端 agent 的细粒度协商接口（`capabilities` 说明接口 / `gpus` GPU 发现 / `load` 指定 GPU 加载 / `ops` 状态轮询 / `unload` 按模型卸载）见 **`模型切换说明.md`**。

## 插拔显卡后

重跑 `./scripts/start.sh` 即可，脚本自动重新检测卡数并适配。

## 新机器一键初始化

完整部署流程（硬件要求 / 快速部署 / 手动分步 / 模型管理 / 踩坑）见 **`部署文档.md`**。

```bash
./scripts/init.sh                     # 交互模式（可选模型）
./scripts/init.sh --minimal           # 最小安装（27B + H3，~120GB）
./scripts/init.sh --full              # 全量安装（所有模型，~500GB）
./scripts/init.sh --no-start          # 只安装不启动
```

初始化脚本自动完成：系统依赖（Docker + NVIDIA Toolkit + locale）→ 模型下载 → 镜像构建 → 启动服务。

如需手动分步：

```bash
sudo ./scripts/install-prereqs.sh   # ① Docker + NVIDIA Container Toolkit（装完注销重登）
./scripts/manage.sh download qwen3.8-27b        # ② 下载 LLM 主仓库
./scripts/manage.sh download qwen3.8-27b fp8    #    可选：官方 FP8 变体（网关优先用 -fp8 目录）
./scripts/manage.sh download wan2.2-s2v         #    视频模型（自动附带共享组件）
./scripts/manage.sh check --fix                 # ③ 校准 models.json（下载后也会自动跑）
./scripts/start.sh                              # ④ 启动（自动检测 GPU、生成配置、拉起网关）
./scripts/manage.sh verify                      # ⑤ 端到端实测：正文非空 + 工具调用解析
```

注：`start.sh` 对选中的模型自动执行"缺则下载、有则校准"，②③ 可省略（手动跑可提前准备、断点续传更可控）。

`think`（是否思考模型）和 `max_ctx`（上下文上限）不用手填：校准脚本以模型自带的
`chat_template.jinja` / `config.json` 为准自动修正。网关据此决定是否给后端加
`--reasoning-parser qwen3`——非思考模型强开会导致正文为空（已踩过坑）。

## 上下文长度（YaRN 扩展）

各模型原生均为 256K；`models.json` 的 `gw.max_len` 是网关实际上限（钳制、调度都按它走）：

| 模型 | 实际上限 | 说明 |
|---|---|---|
| qwen3.8-flash-next | 1M | 官方 YaRN factor 4.0（混合注意力，KV 开销小） |
| qwen3.8-flash-next-iq3s | 1M | llama.cpp 引擎 + YaRN factor 4.0（`--rope-scaling yarn`），与 vLLM 旗舰同语义；实测单卡 96GB 显存 85GB 可跑，2×48GB 机自动跨卡分层（`--split-mode layer`）摊分 |
| qwen3.6-35b-a3b | 1.01M | 官方 YaRN factor 4.0；需 FP8 变体（bf16 下 KV 余量不够） |
| qwen3.8-27b | 900K | 官方 YaRN factor 4.0；vLLM 实测 KV 上限 907088，1M 差 ~6GB |
| qwen3-coder-30b-a3b | 512K | 稠密注意力 KV 开销大，factor 2.0；需 FP8 变体（bf16 下 512K 放不下） |
| qwen3-coder-next | 256K | 官方无 YaRN 方案，且 FP8 权重占满单卡，KV 无余量 |

YaRN 通过 `gw.extra_args`（`--hf-overrides`）+ `gw.env`（`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`）
按模型注入。注意静态 YaRN 在短文本上可能有轻微质量损失（官方注），且超长 prompt 的
首轮 prefill 很慢（开了 prefix caching，后续轮次只增量计算）。

## 模式说明

| 模式 | 触发 | 说明 |
|---|---|---|
| 双卡分工 | 2 张卡（默认） | LLM 默认占 GPU1，视频占 GPU0，真并行；视频未用时第二个 LLM 可放 GPU0 并存（按模型名路由） |
| 双卡 TP | 2 张卡 + `--tp 2` | vLLM 单引擎跨卡：整池 KV 换单请求 ctx（27b fp8 @2×48GB：193K→895K）；解码速度**不涨**（无 NVLink 卡 all-reduce 净亏，实测 bf16TP2 611 < fp8 单卡 658 tok/s）；视频/副本让位；远程等价 `gateway/load {"mode":"tp"}`（ctx 持久化仍需本命令落 gw.max_len） |
| 双卡副本 | 2 张卡 + `--replicas 2` | 每卡一份完整副本，网关按在途数分流，聚合吞吐≈×2、每流 ctx 不变；视频让位；远程等价 `gateway/load {"mode":"replica"}` |
| 双卡旗舰 | 2 张卡 + 选大模型 | qwen3.8-flash-next（125GB FP8）占双卡，准入按整机池 `gmem×张数×0.92`（2×48GB 机池容 88GB 会被拒，应选 iq3s）；start.sh 会自动停止运行中的 ComfyUI 腾显存（有任务在跑先提示确认）；下次选回视频时自动拉起 |
| 单卡共享 | 1 张卡 | LLM 限 ≤38GB（FP8），与视频分时共享显存 |
| 独占模式 | 旗舰量化模型 | qwen3.8-flash-next-iq3s（GGUF 84G，llama.cpp）：96GB 卡机吃满整卡；2×48GB 机自动跨卡按层摊分（`--split-mode layer` + 按显存比例的 `--tensor-split`）；视频自动灰化；`--switch video` 自动降回 27B 共存 |

## 已部署能力速查

以下所有模型文件均已下载，**无需额外下载**即可使用。`start.sh` 和 `manage.sh` 检测到文件存在时会自动跳过下载。

### 文本大模型（LLM，网关 vLLM / llama.cpp 双引擎）

| 模型 | 精度 | 大小 | 卡数 | 默认推荐 |
|---|---|---|---|---|
| qwen3.8-27b | FP8 | 29G | 1 卡（GPU1） | ✅ 默认在线 |
| qwen3.8-27b | BF16 | 54G | 1 卡 | 备用 |
| qwen3.6-35b-a3b | FP8 | 35G | 1 卡 | ✅ |
| qwen3.6-35b-a3b | BF16 | 67G | 1 卡 | 备用 |
| qwen3-coder-next | FP8 | 75G | 1 卡 | ✅ |
| qwen3-coder-30b-a3b | FP8 | 30G | 1 卡 | ✅ |
| qwen3-coder-30b-a3b | BF16 | 57G | 1 卡 | 备用 |
| qwen3.8-flash-next | FP8 | 173G | 2 卡 | ✅ 需独占双卡 |
| qwen3.8-flash-next-iq3s | GGUF IQ3_S + mmproj | 84G+0.9G | 1 卡（llama.cpp） | ✅ 单卡独占（parallel=2 自动规划，200B 级旗舰单卡版，支持图像/视频理解） |

### 视频/图像模型（ComfyUI）

| 能力 | 模型 | 状态 |
|---|---|---|
| 文生/图生视频 | MiniMax H3 33B（含 turbo LoRA、音频 VAE） | ✅ |
| 文生视频 | Wan2.2 A14B（高噪/低噪，FP16+FP8） | ✅ |
| 图生视频 | Wan2.2 A14B（高噪/低噪，FP16+FP8） | ✅ |
| 对口型（S2V） | Wan2.2 S2V 14B（BF16+FP8） | ✅ |
| 对口型（V2V） | InfiniteTalk（单人/双人唇形模型） | ✅ |
| 文生图 | Qwen-Image 2512 FP8 | ✅ |
| 数字人 | LongCat-Avatar 15B + LoRA | ✅ |

### 共享组件

已部署：Wan2.2 VAE、Wan2.1 VAE、UMT5-XXL 文本编码器（FP8+FP16）、CLIP ViT-H 视觉编码器、wav2vec2 中文语音编码器、所有 lightx2v / Wan 加速 LoRA。

---

## 目录

- `models.json` — 模型清单（显存/卡数/下载源），新增模型唯一入口
- `models/llm` — 文本模型 ｜ `models/comfy` — 视频模型（ComfyUI 目录结构）
- `scripts/` — ai（四节点统一网关管理 CLI）、init.sh（新机器一键初始化）、start.sh（启动）、manage.sh（模型）、menu.py（菜单与守卫逻辑）、comfyui_gpu1.sh（GPU1 第二 ComfyUI 实例，幂等）
- `sysmonitor/` — 实时监控（curses/文本/Web看板/截图；GPU/网速/磁盘IO/进程/模型吞吐 当前值仪表；Web 支持跟随系统、浅色与 7 套暗色配色、统一健康告警，并可分节点显示 DGX Spark head/worker 的统一内存与模型驻留；详见 `sysmonitor/README.md`）
- `console/` — GUI 管理后台 agent（9001；agent.py + web/ 静态前端 + console.json relay 白名单。读网关/sysmon、驱动 manage.sh/menu.py；bind-mount 代码，`docker compose restart console` 即生效）
- `模型切换说明.md` — 文本/视频切换与管理接口（客户端操作依据）
- `部署文档.md` — 通用部署指南（硬件要求 / 一键或手动部署 / 踩坑）
- `data/output` — 生成结果输出目录

## 注意

- 系统 Python 3.14 未被改动，一切运行在容器内
- 下载走 hf-mirror 镜像站；LTX-2 需先在 HuggingFace 网页接受协议
- 网关会钳制过大的 `max_tokens`（vLLM 对 prompt+输出超上下文一律 400，客户端会误判为"上下文溢出"提前 compact）；钳制设有保底预算（思考模型 16384 / 非思考 4096），低于保底说明上下文真满了，放开让 vLLM 原生 400 透传触发 compact，避免思考模型预算被钳光后只吐 thinking 报错
- 多模型放置策略：装得下就并存（每卡按已加载模型的 util 预算扣减，为其保留 KV cache 空间）；任何卡都装不下才卸最久未用的空闲模型；正忙的模型绝不驱逐，直接返回 503 让客户端稍后重试
- 客户端 agent 细粒度协商（2026-10-20 新增）：首次连接先 `GET /v1/gateway/capabilities` 学习能力与协议 → `GET /v1/gateway/gpus` 查 GPU 数量与占用 → `POST /v1/gateway/load {"model":...,"gpus":[0],"wait":false}` 指定 GPU 异步加载 → 轮询 `GET /v1/gateway/ops/{id}` 拿加载/切换状态（`transitions[]`）。推理请求也可带 `X-Gateway-Load-Async:1`（+可选 `X-Gateway-GPUs:0`）免预加载、202 异步协商。2026-10-26 起再加一步**布局协商**：`GET /v1/gateway/modes` 查每个模型的摆放模式（dual/tp/replica/pool/auto）、现行布局与 ctx 上限，`load` 带 `"mode":"…"` 或 `"replicas":N` 当场切摆放（可加推理头 `X-Gateway-Mode`）；网关对 models.json 只读，新 ctx 持久化按目录给的 `host_command` 在宿主执行。详见 `模型切换说明.md` 第 5.3 / 5.8 节。
- 自动卸载只由文本请求触发；视频请求直连 ComfyUI 不经过网关，跑大视频任务前需手动腾显存：`curl -X POST http://localhost:8000/v1/gateway/unload_all`（正忙/加载中的模型跳过并在 `busy` 字段列出，网关保持常驻、请求来了自动重载）
- 详细设计见方案文档（plans/miles-morales-siryn-batgirl.md）

**存储空间**：LLM 模型共 ~532G（9 个变体），ComfyUI 视频模型共 ~434G，HF 缓存 ~330K。总计约 **966GB**。所有文件均已下载完毕，`manage.sh download` 检测到已有文件时会自动跳过。
