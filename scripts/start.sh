#!/usr/bin/env bash
# AI 服务统一启动入口：GPU 自适应 + 模型选择菜单 + 显存守卫
#
# 用法:
#   ./start.sh                  交互模式（菜单选模型）
#   ./start.sh --llm qwen3.8-27b --video wan2.2-s2v -y   非交互
#   ./start.sh --switch video   切到视频服务（起 ComfyUI + 卸载文本模型腾显存）
#   ./start.sh --switch llm     切到文本服务（懒加载，不预热）
#   ./start.sh --down           停止全部服务（只停不删）
#   ./start.sh --clean          停止并删除容器
#   ./start.sh --help, -h       显示帮助信息
#
# 可用模型:
#   --llm   : qwen3.8-27b | qwen3.6-35b-a3b | qwen3-coder-next | qwen3.8-flash-next | qwen3.8-flash-next-iq3s | qwen3-coder-30b-a3b | none
#   --video : wan2.2-s2v | wan2.2-s2v-720p | infinitetalk | minimax-h3 | wan2.2-t2v | wan2.2-i2v | ltx-2 | none
#
# 说明:
#   - 默认同时启动 LLM + ComfyUI（双卡各占一张）
#   - 单卡时 LLM 与 ComfyUI 互斥（先 LLM 后 ComfyUI，或只选其一）
#   - --video none 时只启动文本模型；--llm none 时只启动 ComfyUI
#   - 选"不跑视频"（含独占双卡旗舰模型）会自动停止正在运行的 ComfyUI（含第二实例 gpu1）
#     释放显存；重新选择视频启动时自动拉起。有任务在跑时会先提示确认（-y 跳过）
#   - ComfyUI 模型加载需冷启动 1-3 分钟，首次请求挂起等待加载
#   - vLLM 模型空闲后自动卸载显存（不占用时内存 < 1GB）
#   - LLM API 默认绑定所有模型（切换模型只改请求中的 model 字段）
set -euo pipefail

# ──────────────────────────────────────────────
# 帮助信息
# ──────────────────────────────────────────────
show_help() {
    cat << 'HELPEOF'
用法:
  ./start.sh                    交互模式（菜单选模型，自动检测 GPU 数）
  ./start.sh --llm <模型> --video <视频|none> [-y]   非交互一键启动（模型名见下方清单）
  ./start.sh --switch video|llm [模型名]             快速切换文本/视频（网关常驻，无需重启）
  ./start.sh --down / --clean                        停止全部 / 停止并删除容器
  ./start.sh --help, -h                              显示本帮助

快速切换（记不住长命令，用这些）:
  ./scripts/start.sh --switch video                  # 视频优先：起 ComfyUI，空闲文本后端自动让显存
  ./scripts/start.sh --switch llm                    # 文本优先：懒加载，不预热不占显存
  ./scripts/start.sh --switch llm qwen3.8-flash-next-iq3s
  # 局域网远程切换（客户端无需 SSH，网关常驻）:
  curl -X POST http://<服务器IP>:8000/v1/gateway/switch -H 'Content-Type: application/json' -d '{"target":"video"}'
  curl -X POST http://<服务器IP>:8000/v1/gateway/switch -H 'Content-Type: application/json' -d '{"target":"llm","model":"qwen3.8-27b"}'

一键启动命令（全部可启动模型；缺文件会自动先下载；★=已下载 ○=未下载）:
HELPEOF
    # 模型清单从 models.json 动态生成，新增模型无需再改本帮助
    python3 - "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" <<'PY'
import json, os, sys
root = sys.argv[1]
man = json.load(open(os.path.join(root, "models.json")))
def has_llm(n):
    for cand in (n + "-fp8", n):
        p = os.path.join(root, "models/llm", cand)
        if os.path.isdir(p) and os.listdir(p):
            return True
    return False
print("  文本模型（--llm）:")
for n, d in man["llm"].items():
    vs = " / ".join(f"{v}:{int(gb)}G" for v, gb in d["variants"].items())
    mark = "★" if has_llm(n) else "○"
    gpu = f"需{d['min_gpus']}卡独占" if d["min_gpus"] > 1 else "1卡"
    brief = (d.get("menu_brief") or d.get("desc") or "")[:42]
    print(f"    {mark} ./scripts/start.sh --llm {n} --video none -y   # {vs} {d['engine']} {gpu} {brief}")
print("  视频模型（--video，文本默认 qwen3.8-27b 共存）:")
for n, d in man["video"].items():
    print(f"    ./scripts/start.sh --llm qwen3.8-27b --video {n} -y   # {(d.get('desc') or '')[:46]}")
print("  只跑文本: --video none ｜ 只跑视频: --llm none ｜ 默认: 27B + wan2.2-s2v")
PY
    cat << 'HELPEOF'

选项:
  --llm <模型>      文本模型名（上面清单任选；默认 qwen3.8-27b；none=不启动文本）
  --video <模型>    视频模型名（上面清单任选；默认 wan2.2-s2v；none=不启动视频）
  -y, --yes         跳过确认提示（非交互模式）
  --switch <目标>   切换服务对象（网关常驻，只决定显存归谁；也可写 -switch）
                     video [名称]   起 ComfyUI（缺失则按当前 .env 重建）+ 卸载空闲文本后端腾显存
                     llm [模型名]   文本模式就绪提示（懒加载：模型等请求自动加载，不预热不占显存）
  --down            停止所有服务（只停不删，容器保留，再跑 start.sh 即恢复）
  --clean           停止并删除容器（改 GPU 数/换 compose 结构后用；会连带删除手工创建的
                    ai-comfyui-gpu1，需要时用 scripts/comfyui_gpu1.sh 重建）

示例:
  # 默认启动（27B + wan2.2-s2v）
  ./start.sh
  # 单卡旗舰（记不住就抄这条）
  ./start.sh --llm qwen3.8-flash-next-iq3s --video none -y
  # 双卡旗舰（自动停 ComfyUI 腾显存）
  ./start.sh --llm qwen3.8-flash-next --video none -y
  # 只启动文本 / 只跑视频
  ./start.sh --llm qwen3.6-35b-a3b --video none -y
  ./start.sh --llm none --video infinitetalk -y

硬件配置:
  - 双 NVIDIA RTX PRO 6000 Blackwell 96GB，GPU 数 1~2 自适应
  - 2 卡: LLM(GPU1) + ComfyUI(GPU0) 并行；1 卡: LLM 与视频互斥分时
  - 双卡旗舰 qwen3.8-flash-next 独占双卡；单卡旗舰 qwen3.8-flash-next-iq3s 独占单卡
    （旗舰模式下运行中的 ComfyUI 会被自动停止释放显存）

服务端口:
  - LLM API : http://localhost:8000       (网关，暴露全部已下载模型，按需加载/空闲卸载)
  - ComfyUI : http://localhost:8188      (图生视频/唇形同步)
  - Open WebUI: http://localhost:3000   (LLM Web 界面)

服务管理:
  - 启动:     ./start.sh
  - 停止:     ./start.sh --down   （只停不删；--clean 才删容器）
  - 查看日志: docker logs ai-gateway / ai-comfyui / ai-llamacpp-<模型>
  - 重启:     ./start.sh -y   （无需先 --down）
HELPEOF
}

# ──────────────────────────────────────────────
# 解析 --help/-h 参数
# ──────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# docker 权限（装完 docker 需注销重登生效；未生效时给出提示）
if ! docker info >/dev/null 2>&1; then
  echo "错误：当前用户无 docker 权限。请注销重登（或执行: sudo usermod -aG docker $USER 后重登）" >&2
  exit 1
fi

if [[ "${1:-}" == "--down" || "${1:-}" == "--clean" ]]; then
  clean=0; [[ "${1:-}" == "--clean" ]] && clean=1
  if [[ $clean == 1 ]]; then
    docker compose -f compose.yaml -f compose.selected.yaml down --timeout 20 2>/dev/null \
      && echo "已停止并删除容器（--clean：compose 会连带删除带本项目标签、但未在 compose 文件里声明的容器，含 ai-comfyui-gpu1）" \
      || echo "服务未在运行"
  else
    # 默认只停不删：compose down 是"删除"语义，会把第二 ComfyUI 实例这类
    # 带项目标签但未声明的容器一并删掉，重建只能靠手工命令，故不作为默认行为
    docker compose -f compose.yaml -f compose.selected.yaml stop 2>/dev/null \
      && echo "已停止（容器保留，再跑 start.sh 即恢复）" || echo "服务未在运行"
    if docker ps -q -f name='^/ai-comfyui-gpu1$' | grep -q .; then
      docker stop ai-comfyui-gpu1 >/dev/null && echo "已停止 ai-comfyui-gpu1"
    fi
  fi
  # vLLM / llama.cpp 后端容器由网关按需拉起、用完即弃：删除它们才释放显存，下次请求自动重建
  orphans=$(docker ps -aq --filter "name=^/ai-vllm-" --filter "name=^/ai-llamacpp-")
  if [[ -n "$orphans" ]]; then
    docker rm -f $orphans >/dev/null && echo "已清理 vLLM/llama.cpp 后端容器（释放显存）"
  fi
  exit 0
fi

# ──────────────────────────────────────────────
# 快速切换服务对象: --switch video | --switch llm [模型名]
#   video  让 ComfyUI 接管显存：配置缺失/容器缺失由 compose 补齐 + 起第二实例，
#          卸载空闲文本后端等显存调度统一委托网关 /v1/gateway/switch（远程客户端同款入口）
#   llm    文本模式就绪提示：全部懒加载，模型等请求到达自动加载；显存不足时网关自动卸载空闲模型（含视频）
# 网关与 Open WebUI 常驻不动，切换只决定"显存归谁"，不需要 --down
# ──────────────────────────────────────────────
if [[ "${1:-}" == "-switch" || "${1:-}" == "--switch" ]]; then
  target="${2:-}"; who="${3:-}"
  set -a; [[ -f .env ]] && source ./.env; set +a

  gw()      { curl -s -m "${2:-8}" "http://localhost:8000$1" 2>/dev/null || true; }
  gw_post() { # $1=path $2=JSON body（POST）
    curl -s -m 120 -X POST -H 'Content-Type: application/json' -d "$2" "http://localhost:8000$1" 2>/dev/null || true
  }
  running() { docker ps -q -f "name=^/$1$" | grep -q .; }
  exists()  { docker ps -aq -f "name=^/$1$" | grep -q .; }
  show_status() {
    local j; j="$(gw /v1/gateway/status)"
    [[ -n "$j" ]] || return 0
    GW_JSON="$j" python3 - <<'PY' 2>/dev/null || true
import os, json
try:
    d = json.loads(os.environ.get("GW_JSON") or "{}")
except Exception:
    raise SystemExit
b = d.get("backends") or {}
print("  文本后端:", ", ".join(k + "(" + v["status"] + ")" for k, v in b.items()) or "无（按需自动加载）")
print("  显存空闲:", ", ".join("GPU%s %dGB/%dGB" % (g, v["free"] // 1024, v["total"] // 1024)
                               for g, v in (d.get("gpu_free") or {}).items()))
PY
  }

  case "$target" in
  video)
    # compose.selected.yaml 由上次启动的选择决定；不含 comfyui 时按 .env 重新生成
    # （写 .env / compose 配置是宿主机职责，网关不碰；网关只负责"拉起容器 + 卸载文本"）
    regen=0
    if ! docker compose -f compose.yaml -f compose.selected.yaml config --services 2>/dev/null | grep -qx comfyui; then
      vname="$who"; [[ -z "$vname" || "$vname" == "none" ]] && vname="${VIDEO_NAME:-}"
      [[ -z "$vname" || "$vname" == "none" ]] && vname=minimax-h3
      llm="${LLM_NAME:-qwen3.8-27b}"
      # 双卡旗舰与视频物理互斥：切视频时 LLM 降到单卡可并存的那档
      [[ "${AI_MODE:-}" == "flagship" || "${LLM_TP_SIZE:-1}" != "1" ]] && llm=qwen3.8-27b
      echo "==> 当前配置不含视频，重新生成：视频 ${vname} + 文本 ${llm}（双卡各一）"
      python3 scripts/menu.py --llm "$llm" --video "$vname" -y >/dev/null
      set -a; source ./.env; set +a
      regen=1
    fi
    # 容器创建走 compose（唯一创建源，compose.selected.yaml 缺失或容器被删时重建）
    if [[ $regen == 1 ]] || ! exists ai-comfyui; then
      docker compose -f compose.yaml -f compose.selected.yaml up -d comfyui >/dev/null 2>&1 \
        && echo "✓ ai-comfyui 已启动 → http://localhost:8188"
    fi
    # 拉起（含 gpu1）与卸载空闲文本后端统一委托网关 switch 接口——远程客户端同款入口
    j="$(gw_post /v1/gateway/switch '{"target":"video"}')"
    if [[ -n "$j" ]]; then
      GW_JSON="$j" python3 - <<'PY' 2>/dev/null || true
import os, json
try:
    d = json.loads(os.environ.get("GW_JSON") or "{}")
except Exception:
    raise SystemExit
for c in d.get("missing") or []:
    print(f"  ⚠ {c} 容器缺失（需先在宿主机初始化视频配置）")
for c in d.get("started") or []:
    print(f"  ✓ {c} 已启动")
for c in d.get("already_running") or []:
    print(f"  ✓ {c} 已在运行")
for s in d.get("skipped") or []:
    print(f"  ⚠ {s.get('container')} 启动失败: {s.get('error')}")
print("  已卸载空闲文本后端:", ", ".join(d.get("unloaded") or []) or "无")
busy = d.get("busy") or []
if busy: print("  仍在加载（有在途请求/正在启动，未动）:", ", ".join(x["model"] for x in busy))
PY
    else
      # 网关不可达：退回本地操作（仍能起 ComfyUI，只是无法卸载文本后端）
      echo "⚠ 网关不可达，退回本地操作"
      docker compose -f compose.yaml -f compose.selected.yaml up -d comfyui >/dev/null 2>&1 \
        && echo "✓ ai-comfyui 已启动 → http://localhost:8188"
      if exists ai-comfyui-gpu1; then
        running ai-comfyui-gpu1 || { docker start ai-comfyui-gpu1 >/dev/null && echo "✓ ai-comfyui-gpu1 已启动 → http://localhost:8189"; }
      else
        echo "- 第二实例不存在（不擅自创建，它会和 GPU1 上的 LLM 抢显存）：需要时 ./scripts/comfyui_gpu1.sh"
      fi
      curl -s -m 120 -X POST http://localhost:8000/v1/gateway/unload_all >/dev/null 2>&1 || true
    fi
    echo "==> 视频服务状态"
    for pair in ai-comfyui:8188 ai-comfyui-gpu1:8189; do
      c="${pair%%:*}"; p="${pair##*:}"
      running "$c" || continue
      curl -sf -m 8 "http://localhost:$p/system_stats" >/dev/null 2>&1 \
        && echo "  ✓ $c (:$p) 就绪" || echo "  ⚠ $c 运行中但未响应（首次冷启动 1-3 分钟）"
    done
    echo "  提交任务: python3 scripts/gen_video.py \"提示词\"   （--server http://localhost:8189 指定第二实例）"
    show_status
    exit 0
    ;;

  llm|text)
    [[ -z "$(gw /v1/models)" ]] && { echo "错误：网关未运行，先跑 ./scripts/start.sh"; exit 1; }
    name="$who"; [[ -z "$name" ]] && name="${LLM_NAME:-qwen3.8-27b}"
    # 显式指定了模型名就校验，防手滑（清单里没有直接报错，而不是静默忽略）
    if [[ -n "$who" ]] && ! python3 -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open('models.json'))['llm'] else 1)" "$who"; then
      echo "错误：清单里没有 LLM '$who'（可用 ./scripts/manage.sh list 查看）" >&2
      exit 1
    fi
    echo "==> 文本模式已就绪（全部懒加载：不预热、不占显存）"
    echo "  默认模型 ${name}：请求到达时自动加载（冷启动 1-5 分钟），无需服务端操作"
    echo "  显存不足时网关自动卸载空闲模型让位（文本按 LRU，视频模型空闲也平等对待）"
    echo "  视频侧：ComfyUI 容器保留，要跑视频直接 ./scripts/start.sh --switch video"
    echo "  远程切换（局域网内任意设备）: curl -X POST http://<服务器IP>:8000/v1/gateway/switch -d '{\"target\":\"llm\"}'"
    show_status
    exit 0
    ;;

  *)
    echo "用法: $0 --switch video [视频模型名] | $0 --switch llm [文本模型名]" >&2
    exit 1
    ;;
  esac
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

# ③ 协调 ComfyUI 容器（compose 只管 ai-comfyui；ai-comfyui-gpu1 是手动创建的第二实例）
YES=0; for a in "$@"; do [[ "$a" == "-y" || "$a" == "--yes" ]] && YES=1; done
comfy_busy() {  # $1=端口；有任务在跑/排队返回 0
  curl -s -m 3 "http://localhost:$1/queue" 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); sys.exit(0 if (d['queue_running'] or d['queue_pending']) else 1)" 2>/dev/null
}
if [[ "${COMFY_ENABLED}" == "1" && "${VIDEO_NAME}" != "none" ]]; then
  # 选了视频：第二实例存在但 stopped 时一并拉起（compose 不认识它）
  if docker ps -aq -f name='^/ai-comfyui-gpu1$' | grep -q . \
     && ! docker ps -q -f name='^/ai-comfyui-gpu1$' | grep -q .; then
    echo "==> 拉起第二 ComfyUI 实例 ai-comfyui-gpu1（GPU1 负载均衡）"
    docker start ai-comfyui-gpu1 >/dev/null
  fi
  docker ps -aq -f name='^/ai-comfyui-gpu1$' | grep -q . \
    || echo "提示：第二实例 ai-comfyui-gpu1 不存在，需要时用 ./scripts/comfyui_gpu1.sh 创建（GPU1 会与 LLM 共用）"
else
  # 没选视频（含独占双卡旗舰）：停掉仍在运行的 ComfyUI，否则显存不释放、LLM 加载会 503
  for pair in ai-comfyui:8188 ai-comfyui-gpu1:8189; do
    c="${pair%%:*}"; port="${pair##*:}"
    docker ps -q -f name="^/${c}$" | grep -q . || continue
    if comfy_busy "$port"; then
      echo "⚠ ${c} 有视频任务在运行/排队"
      if [[ $YES == 0 ]]; then
        read -rp "仍要停止 ${c} 释放显存吗？[y/N] > " ans
        [[ "$ans" =~ ^[yY] ]] || { echo "已取消，未停止 ${c}"; continue; }
      fi
    fi
    echo "==> 停止 ${c}（当前配置不含视频，释放显存给 LLM）"
    docker stop "$c" >/dev/null
  done
fi

# ④ 启动
echo "==> 启动服务"
docker compose -f compose.yaml -f compose.selected.yaml up -d

# ⑤ 健康检查（LLM 首次加载模型需 1-3 分钟）
echo "==> 等待服务就绪（最长 5 分钟）"
llm_ok=1; comfy_ok=1
comfy_wanted=0
[[ "${COMFY_ENABLED}" == "1" && "${VIDEO_NAME}" != "none" ]] && comfy_wanted=1
[[ $comfy_wanted == 0 ]] && comfy_ok=2   # 2 = 本配置不含视频，不检查也不报"就绪"
for _ in $(seq 1 60); do
  if [[ $llm_ok != 0 ]] && { curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 || curl -sf http://localhost:8000/health >/dev/null 2>&1; }; then
    llm_ok=0
  fi
  if [[ $comfy_ok == 1 ]] && curl -sf http://localhost:8188/system_stats >/dev/null 2>&1; then
    comfy_ok=0
  fi
  [[ $llm_ok == 0 && $comfy_ok != 1 ]] && break
  sleep 5
done

echo ""
echo "======== 启动结果 ========"
if [[ $llm_ok == 0 ]]; then
  echo "✓ LLM 网关就绪: http://localhost:8000  （暴露全部已下载模型，按需自动加载/空闲自动卸载，不动视频服务）"
  echo "  默认模型: ${LLM_NAME}（懒加载：请求到达自动加载，冷启动需等权重加载）"
  echo "  测试: curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${LLM_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
else
  echo "✗ LLM 网关未就绪，查看日志: docker logs ai-gateway"
fi
if [[ $comfy_wanted == 1 ]]; then
  if [[ $comfy_ok == 0 ]]; then
    echo "✓ ComfyUI 就绪: http://localhost:8188  (视频: ${VIDEO_NAME}, 上限 ${VIDEO_MAX_RES}p)"
  else
    echo "✗ ComfyUI 未就绪，查看日志: docker logs ai-comfyui"
  fi
else
  echo "- ComfyUI 未启用（视频: none）"
fi

# 外网访问信息（局域网内任意设备可用）
LAN_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE "^(172\.(17|18|19)\.|127\.|::$)" | head -1)
if [[ -n "$LAN_IP" ]]; then
  echo ""
  echo "======== 外网访问（LAN: ${LAN_IP}）========"
  echo "  LLM API : http://${LAN_IP}:8000/v1/chat/completions"
  [[ $comfy_wanted == 1 && $comfy_ok == 0 ]] && echo "  ComfyUI : http://${LAN_IP}:8188"
  echo "  Open WebUI: http://${LAN_IP}:3000"
  [[ -n "${LLM_API_KEY:-}" ]] && echo "  API Key : ${LLM_API_KEY}"
fi
