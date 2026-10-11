#!/usr/bin/env bash
# GPU1 上的第二个 ComfyUI 实例（ai-comfyui-gpu1，宿主端口 8189）——负载均衡用
# 与 ai-comfyui 同一镜像、同一模型目录；客户端按 /queue 深度择机排队。
#
# 它不在 compose 文件里（compose 的一个服务位只能绑一张卡），所以单独用本脚本管理：
#   ./scripts/comfyui_gpu1.sh            不存在则创建，已存在则启动（幂等）
#   ./scripts/comfyui_gpu1.sh --stop     停止（容器保留）
#   ./scripts/comfyui_gpu1.sh --rm       删除容器
#
# 显存提醒：GPU1 默认是文本模型（vLLM）的卡位，两者同时干活会抢显存。
# 跑大分辨率/长时长视频前先腾显存：
#   curl -X POST http://localhost:8000/v1/gateway/unload_all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NAME=ai-comfyui-gpu1
IMAGE=ai-comfyui:local
GPU=${COMFY_GPU1_INDEX:-1}     # 实例所在的卡
HOST_PORT=${COMFY_GPU1_PORT:-8189}
ROOT=${HOST_ROOT:-/home/pc/AI}

case "${1:-up}" in
  --stop) docker stop "$NAME" >/dev/null && echo "已停止 $NAME（容器保留，再跑本脚本即恢复）"; exit 0 ;;
  --rm)   docker rm -f "$NAME" >/dev/null && echo "已删除 $NAME"; exit 0 ;;
esac

if docker ps -q -f "name=^/${NAME}$" | grep -q .; then
  if [ "$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$NAME")" = "ai" ]; then
    echo "$NAME 带着 compose 项目标签（下次 compose up/down 会被删），重建中…"
    docker rm -f "$NAME" >/dev/null
  else
    echo "$NAME 已在运行"
    exit 0
  fi
elif docker ps -aq -f "name=^/${NAME}$" | grep -q .; then
  if [ "$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$NAME")" = "ai" ]; then
    echo "$NAME 带着 compose 项目标签（下次 compose up/down 会被删），重建中…"
    docker rm -f "$NAME" >/dev/null
  else
    docker start "$NAME" >/dev/null && echo "已启动现有容器 $NAME"
    exit 0
  fi
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "错误：缺少镜像 $IMAGE，先跑 ./scripts/start.sh 选一个视频任务把它构建出来"; exit 1; }

# compose 构建镜像时会把项目标签烧进 image config（project=ai / service=comfyui），
# docker run 出来的容器继承这些标签 → compose 每次 up/down 都把它当 comfyui 服务的
# 陈旧副本删掉（本实例已因此消失两次）。显式覆盖标签，让它脱离 compose 的项目视野。
LABELS=(
  -l com.docker.compose.project=ai-standalone
  -l com.docker.compose.service=comfyui-gpu1
  -l com.docker.compose.container-number=1
  -l com.docker.compose.config-hash=standalone
)

docker run -d --name "$NAME" --restart unless-stopped --gpus "device=${GPU}" \
  -p "0.0.0.0:${HOST_PORT}:8188" \
  -v "${ROOT}/models/comfy:/opt/ComfyUI/models:rw" \
  -v "${ROOT}/data:/data:rw" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  "${LABELS[@]}" \
  "$IMAGE" python3 main.py --listen 0.0.0.0 --port 8188 --highvram \
  --output-directory /data/output --input-directory /data/input >/dev/null

echo "已创建并启动 $NAME（GPU${GPU} → http://localhost:${HOST_PORT}）"
