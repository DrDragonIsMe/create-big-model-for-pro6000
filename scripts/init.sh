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
# 无参数调用＝只探测有无现成凭据，不提示；ask＝允许在无凭据时交互询问
read_sudo_pass() {
  local ask="${1:-}"
  if [[ -n "${SUDO_PASS:-}" ]]; then
    printf '%s\n' "$SUDO_PASS"
    return 0
  fi
  if [[ -f .env ]]; then
    local _p
    _p="$(grep -oP '^SUDO_PASS=\K.*' .env 2>/dev/null || true)"
    if [[ -n "$_p" ]]; then
      printf '%s\n' "$_p"
      return 0
    fi
  fi
  # 交互式提示（仅在需要、且确实有终端时）
  if [[ "$ask" == "ask" && -t 0 ]]; then
    local _pass
    stty -echo 2>/dev/null || true
    read -rp "请输入 sudo 密码: " _pass
    stty echo 2>/dev/null || true
    echo
    [[ -n "$_pass" ]] || return 1
    printf '%s\n' "$_pass"
    return 0
  fi
  return 1
}

# 能否拿到 root（已是 root / NOPASSWD / 有密码 / 可交互询问）
can_sudo() {
  [[ "$(id -u)" -eq 0 ]] && return 0
  sudo -n true 2>/dev/null && return 0
  read_sudo_pass >/dev/null 2>&1 && return 0
  [[ -t 0 ]] && return 0
  return 1
}

# 提权执行：root 直接跑，否则 sudo（NOPASSWD 或密码）。
# 拿不到权限时返回 126，由调用方决定中止还是告警——可选步骤不该带崩整个初始化。
sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return $?
  fi
  if sudo -n true 2>/dev/null; then
    sudo -n "$@"
    return $?
  fi
  local pass
  if pass="$(read_sudo_pass ask)"; then
    printf '%s\n' "$pass" | sudo -S -p '' "$@"
    return $?
  fi
  return 126
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
# count 每张卡输出一次，取首行；直接 tr 去空白会把 "2\n2" 粘成 "22"
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
[[ "$gpu_count" =~ ^[0-9]+$ ]] || gpu_count=0
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

# Docker + NVIDIA Container Toolkit 只有新机器才需要装。
# install-prereqs.sh 会整体覆盖 /etc/docker/daemon.json 并重启 docker，
# 在已就绪的机器上重复执行既慢（apt + 拉测试镜像）又会中断在跑的服务，所以先探测再决定。
prereqs_ready() {
  command -v docker >/dev/null 2>&1 || return 1
  command -v nvidia-ctk >/dev/null 2>&1 || return 1
  if docker info >/dev/null 2>&1; then
    docker info 2>/dev/null | grep -q nvidia || return 1
  else
    grep -q nvidia /etc/docker/daemon.json 2>/dev/null || return 1
  fi
  return 0
}

if prereqs_ready; then
  ok "Docker 与 NVIDIA Container Toolkit 已就绪，跳过系统依赖安装"
elif can_sudo; then
  info "安装系统依赖（Docker + NVIDIA Container Toolkit，需要 root）…"
  sudo_cmd bash scripts/install-prereqs.sh
else
  err "缺少 Docker/NVIDIA Container Toolkit，且当前拿不到 sudo 权限。"
  err "请设置 SUDO_PASS 后重跑，或先手动执行: sudo bash scripts/install-prereqs.sh"
  exit 1
fi

# 加入 docker 组（如果不在）；ssh 非登录 shell 下 $USER 可能为空，用 id -un
me="$(id -un)"
if id -nG "$me" 2>/dev/null | grep -qw docker; then
  ok "用户 $me 已在 docker 组"
else
  info "将用户 $me 加入 docker 组…"
  if sudo_cmd usermod -aG docker "$me"; then
    warn "docker 组变更需要注销重登生效。本次会话通过 sudo 绕过。"
  else
    warn "无 sudo 权限，请手动执行: sudo usermod -aG docker $me 后重新登录"
  fi
fi

# Locale 修复
info "配置 locale…"
if locale -a 2>/dev/null | grep -qiE '^en_US\.utf-?8$'; then
  ok "en_US.UTF-8 locale 已存在"
elif ! command -v locale-gen >/dev/null 2>&1; then
  warn "locale-gen 不可用（未安装 locales？），跳过"
elif sudo_cmd locale-gen en_US.UTF-8 && sudo_cmd update-locale LANG=en_US.UTF-8; then
  ok "en_US.UTF-8 locale 已生成"
else
  warn "生成 locale 需要 root，跳过（可手动: sudo locale-gen en_US.UTF-8）"
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

# 运行时目录：manage.sh 把 ./models bind-mount 进容器并以当前用户身份写入，
# 目录不存在时 docker 会以 root 建出来，后续下载直接 Permission denied
mkdir -p models/llm models/comfy data/input data/output data/abtest data/vllm-cache .jobs

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

# 校验模型名：清单里没有的直接剔除。
# 原来只是打个 warn 却把名字留在列表里，后面按名字取显存拿到 "?"，$(( )) 当场炸掉整个初始化
all_llm=$(python3 -c "import json; print(' '.join(json.load(open('models.json'))['llm'].keys()))")
all_video=$(python3 -c "import json; print(' '.join(json.load(open('models.json'))['video'].keys()))")

# $1=空格分隔的已知名单，其余参数为候选名；输出去重后的合法名
keep_known() {
  local known="$1"; shift
  local n
  for n in "$@"; do
    if grep -qw -- "$n" <<<"$known"; then echo "$n"
    else warn "  未知模型: $n，已跳过"
    fi
  done | sort -u
}

download_llm=($(keep_known "$all_llm" "${download_llm[@]}"))
download_video=($(keep_known "$all_video" "${download_video[@]}"))

# 预估磁盘（清单里的显存值可能是小数，统一向上取整再累加，避免 $(( )) 拿到小数或 "?"）
est_gb() {  # $1=取值的 python 表达式
  python3 -c "import json,math;man=json.load(open('models.json'));print(math.ceil(float($1)))" 2>/dev/null || echo 0
}
est_llm_gb=0
for name in "${download_llm[@]}"; do
  gb=$(est_gb "man['llm']['$name']['variants'].get('bf16',0)")
  est_llm_gb=$((est_llm_gb + gb))
done
est_video_gb=0
for name in "${download_video[@]}"; do
  gb=$(est_gb "man['video']['$name'].get('vram_720p_bf16',0)")
  est_video_gb=$((est_video_gb + gb))
done
est_total=$((est_llm_gb + est_video_gb + 30))

# 空间不够就提前收工，比下到一半 ENOSPC 好收拾
free_gb=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$free_gb" && "$free_gb" -lt "$est_total" ]]; then
  err "磁盘可用 ${free_gb}GB，放不下预计 ${est_total}GB 的模型。"
  err "先清理空间，或减少要下载的模型后重跑。"
  exit 1
fi

echo ""
info "准备下载的模型:"
for name in "${download_llm[@]}"; do echo "  LLM: $name (~${est_llm_gb}GB)"; done
for name in "${download_video[@]}"; do echo "  视频: $name (~${est_video_gb}GB)"; done
if [[ $download_common -eq 1 ]]; then
  echo "  共享组件 (wav2vec2/VAE/编码器… ~30GB)"
fi
echo ""
echo -e "  ${YELLOW}预计磁盘占用: ~${est_total}GB${NC}"

SKIP_DOWNLOAD=0
if [[ "$MODE" != "minimal" && "$MODE" != "full" ]]; then
  read -rp "  确认下载？[Y/n]: " confirm
  confirm="${confirm,,}"
  if [[ "$confirm" == "n" || "$confirm" == "no" ]]; then
    SKIP_DOWNLOAD=1
    info "跳过模型下载"
  fi
fi

# 实际下载
echo ""
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
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
