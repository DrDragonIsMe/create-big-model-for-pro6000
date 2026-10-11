#!/usr/bin/env bash
# AI 项目一键初始化（新机器）
#
# 用法:
#   ./scripts/init.sh                 交互模式（可选模型配置）
#   ./scripts/init.sh --minimal       最小安装：系统依赖 + qwen3.8-27b + minimax-h3
#   ./scripts/init.sh --full          全量安装：系统依赖 + 所有模型
#   ./scripts/init.sh --no-start      安装完不启动服务
#   ./scripts/init.sh --help          显示帮助
#
# 前置条件:
#   - Ubuntu 24.04（或兼容的 Debian）
#   - NVIDIA GPU + 驱动已安装（nvidia-smi 可运行）
#   - root/sudo 权限
#
# 流程:
#   1. 系统依赖（Docker + NVIDIA Container Toolkit + locale）
#   2. 下载/确认项目代码
#   3. 模型下载（按需）
#   4. 构建镜像 + 启动服务

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ──────────────────────────────────────────────
# 颜色与提示
# ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*" >&2; }

# 读取 sudo 密码（优先级：环境变量 > .env > 交互式）
read_sudo_pass() {
  if [[ -n "${SUDO_PASS:-}" ]]; then
    echo "$SUDO_PASS"
    return 0
  elif [[ -f .env ]]; then
    grep -oP '^SUDO_PASS=\K.*' .env 2>/dev/null && return 0
  fi
  # 交互式提示（仅在需要时）
  if [[ "$1" == "ask" ]]; then
    stty -echo
    read -rp "请输入 sudo 密码: " _pass
    echo
    stty echo
    echo "$_pass"
    return 0
  fi
  return 1
}

sudo_cmd() {
  local pass
  pass="$(read_sudo_pass)" || { err "无法获取 sudo 密码，请设置环境变量 SUDO_PASS"; exit 1; }
  echo "$pass" | sudo -S "$@" 2>/dev/null
}

# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="interactive"   # interactive | minimal | full
NO_START=0

for arg in "$@"; do
  case "$arg" in
    --minimal)  MODE="minimal";;
    --full)     MODE="full";;
    --no-start) NO_START=1;;
    --help|-h)
      cat <<'EOF'
用法:
  ./scripts/init.sh                交互模式（可选模型配置）
  ./scripts/init.sh --minimal      最小安装：系统依赖 + qwen3.8-27b + minimax-h3
  ./scripts/init.sh --full         全量安装：系统依赖 + 所有模型
  ./scripts/init.sh --no-start     安装完不启动服务
  ./scripts/init.sh --help         显示帮助

环境配置:
  SUDO_PASS=xxx  设置 sudo 密码（避免交互式提示）
  HF_TOKEN=xxx   HuggingFace Token（部分模型需要，如 LTX-2）

模式说明:
  --minimal  约 120GB 磁盘（27B LLM 54G + H3 视频 81G + 共享组件）
  --full     约 500GB 磁盘（所有 LLM + 所有视频 + 共享组件）

EOF
      exit 0;;
  esac
done

# ──────────────────────────────────────────────
# 检查 NVIDIA 驱动
# ──────────────────────────────────────────────
info "检查 NVIDIA 驱动…"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  err "未找到 nvidia-smi，请确认 NVIDIA 驱动已安装并重启。"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do
  echo "  GPU: $line"
done
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
echo "  GPU 数: $gpu_count"
if [[ "$gpu_count" -lt 1 ]]; then
  err "未检测到 NVIDIA GPU，初始化中止。"
  exit 1
fi

# ──────────────────────────────────────────────
# 第一步：系统级依赖
# ──────────────────────────────────────────────
echo ""
info "========================================"
info "  第一步：系统级依赖"
info "========================================"

bash scripts/install-prereqs.sh

# 加入 docker 组（如果不在）
if ! id -nG "$USER" 2>/dev/null | grep -qw docker; then
  info "将用户 $USER 加入 docker 组…"
  sudo_cmd usermod -aG docker "$USER" || true
  warn "docker 组变更需要注销重登生效。本次会话通过 sudo 绕过。"
fi

# Locale 修复
info "配置 locale…"
if ! locale -a 2>/dev/null | grep -q "en_US.utf8"; then
  if [[ -f /etc/locale.gen ]]; then
    sudo_cmd bash -c "sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen"
    sudo_cmd locale-gen en_US.UTF-8
    ok "en_US.UTF-8 locale 已生成"
  else
    warn "/etc/locale.gen 不存在，跳过 locale 配置"
  fi
else
  ok "en_US.UTF-8 locale 已存在"
fi

# ──────────────────────────────────────────────
# 第二步：项目代码
# ──────────────────────────────────────────────
echo ""
info "========================================"
info "  第二步：项目代码"
info "========================================"

if [[ ! -f models.json ]]; then
  err "当前目录没有 models.json，请确保在 AI 项目根目录下运行此脚本。"
  err "如果是克隆仓库，请先: git clone <repo-url> && cd AI"
  exit 1
fi
ok "项目代码已就位: $(pwd)"

# ──────────────────────────────────────────────
# 第三步：模型下载
# ──────────────────────────────────────────────
echo ""
info "========================================"
info "  第三步：模型下载"
info "========================================"

# 判断要下载的模型
download_llm=()
download_video=()
download_common=0

case "$MODE" in
  minimal)
    download_llm=("qwen3.8-27b")
    download_video=("minimax-h3")
    download_common=1
    ;;
  full)
    # 全部 LLM（不含 fp8，fp8 按需下载）
    for name in $(python3 -c "import json; man=json.load(open('models.json')); print(' '.join(man['llm'].keys()))"); do
      download_llm+=("$name")
    done
    # 全部视频
    for name in $(python3 -c "import json; man=json.load(open('models.json')); print(' '.join(man['video'].keys()))"); do
      download_video+=("$name")
    done
    download_common=1
    ;;
  interactive)
    # 交互选择
    echo ""
    echo "  可用 LLM 模型:"
    python3 <<'PYEOF'
import json
man = json.load(open("models.json"))
for k, v in man["llm"].items():
    vs = ", ".join(f"{vk} {vg}GB" for vk, vg in v["variants"].items())
    print(f"    {k:25s} {vs:30s} {v['desc'][:50]}")
PYEOF
    read -rp "  要下载的 LLM（逗号分隔，回车=默认 qwen3.8-27b）: " llm_input
    if [[ -z "$llm_input" ]]; then
      download_llm=("qwen3.8-27b")
    else
      IFS=',' read -ra _tmp <<< "$llm_input"
      for name in "${_tmp[@]}"; do
        name="$(echo "$name" | xargs)"
        [[ -n "$name" ]] && download_llm+=("$name")
      done
    fi

    echo ""
    echo "  可用视频模型:"
    python3 <<'PYEOF'
import json
man = json.load(open("models.json"))
for k, v in man["video"].items():
    print(f'    {k:25s} 480p:{v["vram_480p_fp8"]}GB 720p:{v["vram_720p_bf16"]}GB  {v["desc"][:50]}')
PYEOF
    read -rp "  要下载的视频（逗号分隔，回车=默认 minimax-h3）: " video_input
    if [[ -z "$video_input" ]]; then
      download_video=("minimax-h3")
    else
      IFS=',' read -ra _tmp <<< "$video_input"
      for name in "${_tmp[@]}"; do
        name="$(echo "$name" | xargs)"
        [[ -n "$name" ]] && download_video+=("$name")
      done
    fi

    read -rp "  是否下载共享组件（wav2vec2/VAE/文本编码器等）？[Y/n]: " common_input
    common_input="${common_input,,}"
    if [[ "$common_input" == "n" || "$common_input" == "no" ]]; then
      download_common=0
    else
      download_common=1
    fi
    ;;
esac

# 校验模型名
invalid=()
all_llm=$(python3 -c "import json; print(' '.join(json.load(open('models.json'))['llm'].keys()))")
all_video=$(python3 -c "import json; print(' '.join(json.load(open('models.json'))['video'].keys()))")

for name in "${download_llm[@]}"; do
  if ! echo "$all_llm" | grep -qw "$name"; then
    invalid+=("$name (LLM)"); warn "  未知 LLM 模型: $name，已跳过"
  fi
done
for name in "${download_video[@]}"; do
  if ! echo "$all_video" | grep -qw "$name"; then
    invalid+=("$name (视频)"); warn "  未知视频模型: $name，已跳过"
  fi
done

# 去重
download_llm=($(printf '%s\n' "${download_llm[@]}" | sort -u))
download_video=($(printf '%s\n' "${download_video[@]}" | sort -u))

# 预估磁盘
est_llm_gb=0
for name in "${download_llm[@]}"; do
  gb=$(python3 -c "import json; man=json.load(open('models.json')); print(man['llm']['$name']['variants']['bf16'])" 2>/dev/null || echo "?")
  est_llm_gb=$((est_llm_gb + gb))
done
est_video_gb=0
for name in "${download_video[@]}"; do
  gb=$(python3 -c "import json; man=json.load(open('models.json')); print(man['video']['$name']['vram_720p_bf16'])" 2>/dev/null || echo "?")
  est_video_gb=$((est_video_gb + gb))
done
est_total=$((est_llm_gb + est_video_gb + 30))

echo ""
info "准备下载的模型:"
for name in "${download_llm[@]}"; do echo "  LLM: $name (~${est_llm_gb}GB)"; done
for name in "${download_video[@]}"; do echo "  视频: $name (~${est_video_gb}GB)"; done
if [[ $download_common -eq 1 ]]; then
  echo "  共享组件 (wav2vec2/VAE/编码器… ~30GB)"
fi
echo ""
echo -e "  ${YELLOW}预计磁盘占用: ~${est_total}GB${NC}"

if [[ "$MODE" != "minimal" && "$MODE" != "full" ]]; then
  read -rp "  确认下载？[Y/n]: " confirm
  confirm="${confirm,,}"
  if [[ "$confirm" == "n" || "$confirm" == "no" ]]; then
    info "跳过模型下载"
  fi
fi

# 实际下载
echo ""
for name in "${download_llm[@]}"; do
  info "下载 LLM: $name"
  bash scripts/manage.sh download "$name" || warn "下载失败: $name"
done

for name in "${download_video[@]}"; do
  info "下载视频模型: $name"
  bash scripts/manage.sh download "$name" || warn "下载失败: $name"
done

if [[ $download_common -eq 1 ]]; then
  info "下载共享组件"
  bash scripts/manage.sh download common || warn "下载失败: common"
fi

# 自动校准
info "自动校准 models.json（think/max_ctx/工具格式）"
bash scripts/manage.sh check --fix || warn "校准跳过"

# ──────────────────────────────────────────────
# 第四步：启动服务
# ──────────────────────────────────────────────
echo ""
info "========================================"
info "  第四步：构建镜像 + 启动服务"
info "========================================"

if [[ $NO_START -eq 1 ]]; then
  info "已指定 --no-start，跳过服务启动。"
  info "启动命令: ./scripts/start.sh"
else
  # 非交互启动（使用默认配置）
  info "开始启动服务…"
  bash scripts/start.sh --llm "${download_llm[0]:-qwen3.8-27b}" --video "${download_video[0]:-minimax-h3}" -y
  echo ""
  ok "初始化完成！"
  echo ""
  echo "======== 常用命令 ========"
  echo "  切换服务: ./scripts/start.sh --switch video"
  echo "             ./scripts/start.sh --switch llm"
  echo "  停止服务: ./scripts/start.sh --down"
  echo "  查看日志: docker logs -f ai-gateway"
  echo "  卸载模型: curl -X POST http://localhost:8000/v1/gateway/unload_all"
  echo ""
  echo "======== 访问地址 ========"
  LAN_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE "^(172\.(17|18|19)\.|127\.|::$)" | head -1)
  if [[ -n "${LAN_IP:-}" ]]; then
    echo "  LLM API   : http://${LAN_IP}:8000"
    echo "  ComfyUI   : http://${LAN_IP}:8188"
    echo "  Open WebUI: http://${LAN_IP}:3000"
  else
    echo "  LLM API   : http://localhost:8000"
    echo "  ComfyUI   : http://localhost:8188"
    echo "  Open WebUI: http://localhost:3000"
  fi
fi
