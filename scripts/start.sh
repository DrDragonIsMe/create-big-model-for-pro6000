#!/usr/bin/env bash
# AI 服务统一启动入口：GPU 自适应 + 模型选择菜单 + 显存守卫
# 用法:
#   ./start.sh                  交互模式（菜单选模型）
#   ./start.sh --llm qwen3.8-27b --video wan2.2-s2v -y   非交互
#   ./start.sh --down           停止全部服务
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# docker 权限（装完 docker 需注销重登生效；未生效时给出提示）
if ! docker info >/dev/null 2>&1; then
  echo "错误：当前用户无 docker 权限。请注销重登（或执行: sudo usermod -aG docker $USER 后重登）" >&2
  exit 1
fi

if [[ "${1:-}" == "--down" ]]; then
  docker compose -f compose.yaml -f compose.selected.yaml down 2>/dev/null && echo "已停止" || echo "服务未在运行"
  exit 0
fi

# ① 生成 .env 与 compose.selected.yaml（含两级菜单与显存守卫）
python3 scripts/menu.py "$@"
echo ""

set -a; source ./.env; set +a

# ② 准备镜像
echo "==> 准备镜像"
# 网关镜像随 scripts/gateway.py 显式重建：compose up 的隐式构建在镜像已存在时会复用旧代码
docker build -q -f docker/gateway/Dockerfile -t ai-gateway:local . >/dev/null
if [[ "${LLM_ENGINE}" == "vllm" ]]; then
  docker pull vllm/vllm-openai:latest
else
  if ! docker pull ghcr.io/ggml-org/llama.cpp:server-cuda 2>/dev/null; then
    echo "预构建 llama.cpp 镜像拉取失败，改本地编译兜底镜像（约 10-20 分钟）…"
    docker build -t ghcr.io/ggml-org/llama.cpp:server-cuda ./docker/llamacpp
  fi
fi
if [[ "${COMFY_ENABLED}" == "1" && "${VIDEO_NAME}" != "none" ]]; then
  docker build -t ai-comfyui:local ./docker/comfyui
fi

# ③ 启动
echo "==> 启动服务"
docker compose -f compose.yaml -f compose.selected.yaml up -d

# ④ 健康检查（LLM 首次加载模型需 1-3 分钟）
echo "==> 等待服务就绪（最长 5 分钟）"
llm_ok=1; comfy_ok=1
[[ "${COMFY_ENABLED}" != "1" || "${VIDEO_NAME}" == "none" ]] && comfy_ok=0
for _ in $(seq 1 60); do
  if [[ $llm_ok != 0 ]] && { curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || curl -sf http://localhost:8000/health >/dev/null 2>&1; }; then
    llm_ok=0
  fi
  if [[ $comfy_ok != 0 ]] && curl -sf http://localhost:8188/system_stats >/dev/null 2>&1; then
    comfy_ok=0
  fi
  [[ $llm_ok == 0 && $comfy_ok == 0 ]] && break
  sleep 5
done

echo ""
echo "======== 启动结果 ========"
if [[ $llm_ok == 0 ]]; then
  echo "✓ LLM 网关就绪: http://localhost:8000  （暴露全部已下载模型，按需自动加载/空闲自动卸载，不动视频服务）"
  echo "  默认预加载: ${LLM_NAME}；切换模型只需在请求里改 model 名（冷启动需等权重加载）"
  echo "  测试: curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${LLM_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
else
  echo "✗ LLM 网关未就绪，查看日志: docker logs ai-gateway"
fi
if [[ $comfy_ok == 0 ]]; then
  echo "✓ ComfyUI 就绪: http://localhost:8188  (视频: ${VIDEO_NAME}, 上限 ${VIDEO_MAX_RES}p)"
else
  [[ "${COMFY_ENABLED}" == "1" && "${VIDEO_NAME}" != "none" ]] && echo "✗ ComfyUI 未就绪，查看日志: docker logs ai-comfyui"
fi

# 外网访问信息（局域网内任意设备可用）
LAN_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE "^(172\.(17|18|19)\.|127\.|::$)" | head -1)
if [[ -n "$LAN_IP" ]]; then
  echo ""
  echo "======== 外网访问（LAN: ${LAN_IP}）========"
  echo "  LLM API : http://${LAN_IP}:8000/v1/chat/completions"
  [[ $comfy_ok == 0 ]] && echo "  ComfyUI : http://${LAN_IP}:8188"
  echo "  Open WebUI: http://${LAN_IP}:3000"
  [[ -n "${LLM_API_KEY:-}" ]] && echo "  API Key : ${LLM_API_KEY}"
fi
