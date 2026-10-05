# 本地 AI 服务（LLM + 视频生成 + 对口型数字人）

双 RTX PRO 6000 96GB，Docker 部署，GPU 数量 1~2 张自适应。

## 日常使用

```bash
cd ~/AI
./scripts/start.sh          # 交互菜单：选 LLM（未下载自动先下载，已下载自动校准）→ 灰化不可用视频项 → 启动
./scripts/start.sh --down   # 停止全部服务
./scripts/manage.sh list    # 查看模型清单与下载状态
./scripts/manage.sh download <名称>       # 下载模型主仓库（LLM 或视频）
./scripts/manage.sh download <名称> fp8   # 下载官方 FP8 变体到 <名称>-fp8（网关优先使用）
./scripts/manage.sh add     # 新增模型到清单（下次启动菜单可见）
./scripts/manage.sh check [--fix]  # 按模型文件校准清单（think/上下文/工具格式）
./scripts/manage.sh verify  # 通过网关逐模型实测：正文非空 + 工具调用解析
```

非交互启动示例：
```bash
./scripts/start.sh --llm qwen3.8-27b --video wan2.2-s2v -y
```

## 服务入口

- LLM API（OpenAI 兼容）: http://localhost:8000/v1/chat/completions
- ComfyUI 图形界面: http://localhost:8188

## 插拔显卡后

重跑 `./scripts/start.sh` 即可，脚本自动重新检测卡数并适配。

## 新机器重建

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

## 模式说明

| 模式 | 触发 | 说明 |
|---|---|---|
| 双卡分工 | 2 张卡（默认） | LLM 占 GPU1，视频占 GPU0，真并行 |
| 双卡旗舰 | 2 张卡 + 选大模型 | 480B/235B/Flash-Next 占双卡；480B-1.8bit 可与视频共存 |
| 单卡共享 | 1 张卡 | LLM 限 ≤38GB（FP8），与视频分时共享显存 |

## 目录

- `models.json` — 模型清单（显存/卡数/下载源），新增模型唯一入口
- `models/llm` — 文本模型 ｜ `models/comfy` — 视频模型（ComfyUI 目录结构）
- `scripts/` — start.sh（启动）、manage.sh（模型）、menu.py（菜单与守卫逻辑）
- `data/output` — 生成结果输出目录

## 注意

- 系统 Python 3.14 未被改动，一切运行在容器内
- 下载走 hf-mirror 镜像站；LTX-2 需先在 HuggingFace 网页接受协议
- 网关会钳制过大的 `max_tokens`（vLLM 对 prompt+输出超上下文一律 400，客户端会误判为"上下文溢出"提前 compact）
- 详细设计见方案文档（plans/miles-morales-siryn-batgirl.md）
