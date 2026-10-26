#!/usr/bin/env bash
# 模型管理：list / download / add / check / verify
# 用法:
#   ./manage.sh list                        列出清单与下载状态
#   ./manage.sh download qwen3.8-27b        下载 LLM 主仓库（下完自动校准 think/max_ctx）
#   ./manage.sh download qwen3.8-27b fp8    下载官方 FP8 变体到 <名称>-fp8（网关优先使用）
#   ./manage.sh download wan2.2-s2v         下载视频模型（自动附带 common 组件）
#   ./manage.sh download qwen-image         下载生图模型（image 段，按 dest 文件表逐个归位）
#   ./manage.sh download common             只下载共享组件
#   ./manage.sh add                         交互式新增模型到 models.json
#   ./manage.sh check [--fix]               按已下载模型文件校准 models.json（think/上下文/工具格式）
#   ./manage.sh verify [模型...]            通过网关逐模型实测：正文非空 + 工具调用解析
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! docker info >/dev/null 2>&1; then
  echo "错误：无 docker 权限，请注销重登后使用" >&2; exit 1
fi

cmd="${1:-list}"; shift || true

case "$cmd" in
list)
  python3 - <<'EOF'
import fnmatch, json, os
man = json.load(open("models.json"))
def has(d): return os.path.isdir(d) and bool(os.listdir(d))
def vhas(d):
    # 口径与 menu.py video_downloaded / console agent comfy_downloaded 三处同源：
    # 模式取 basename（common 段 includes 带 repo 路径）+ 大小写不敏感（官方文件名有大小写笔误）
    p = f"models/comfy/{d['dest']}"
    if not os.path.isdir(p): return False
    files = [f.lower() for f in os.listdir(p)]
    pats = [os.path.basename(pat).lower() for pat in (d.get("includes") or ["*"])]
    return all(any(fnmatch.fnmatch(f, pat) for f in files) for pat in pats)
print("== LLM ==")
for n, d in man["llm"].items():
    vs = "/".join(f"{v}:{g}GB" for v, g in d["variants"].items())
    print(f"  {'★' if has(f'models/llm/{n}') else ' '} {n:<24} {vs:<18} {d['engine']}, min_gpus={d['min_gpus']}  {d['desc']}")
print("== 视频 ==")
for n, d in man["video"].items():
    print(f"  {'★' if vhas(d) else ' '} {n:<24} 480p:{d['vram_480p_fp8']}GB 720p:{d['vram_720p_bf16']}GB  {d['desc']}")
print("== 生图 ==")
for n, d in man.get("image", {}).items():
    dest = d.get("dest", {})
    ok = bool(dest) and all(os.path.isfile(f"models/comfy/{dd}/{ff}") for dd, ff in dest.items())
    print(f"  {'★' if ok else ' '} {n:<24} ~{d.get('size_gb','?')}GB  {d.get('desc', n)}  → {d.get('cli', '')}")
print("== 共享组件 ==")
for n, d in man["common"].items():
    print(f"  {'★' if has('models/comfy/' + d['dest']) else ' '} {n:<24} -> comfy/{d['dest']}")
EOF
  ;;

download)
  name="${1:?用法: manage.sh download <名称> [fp8]}"
  variant="${2:-}"
  JOB="$(mktemp /tmp/hf-job-XXXX.json)"
  # 未显式提供 HF_TOKEN 时读取仓库根的 .hf_token（LTX-2 等受限仓库必需）
  if [[ -z "${HF_TOKEN:-}" && -f .hf_token ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < .hf_token)"
  fi
  python3 - "$name" "$variant" > "$JOB" <<'EOF'
import json, sys
man = json.load(open("models.json"))
name, variant = sys.argv[1], sys.argv[2]
tasks = []
if name == "common":
    for n, d in man["common"].items():
        tasks.append({"mode": "comfy", "name": n, **d})
elif name in man["video"]:
    d = dict(man["video"][name]); d.pop("extra", None)
    tasks.append({"mode": "comfy", "name": name, **d})
    for en, ed in man["video"][name].get("extra", {}).items():
        tasks.append({"mode": "comfy", "name": f"{name}/{en}", **ed})
    for cn, cd in man["common"].items():
        tasks.append({"mode": "comfy", "name": f"common/{cn}", **cd})
elif name in man.get("image", {}):
    d = man["image"][name]
    for dest_dir, fname in d["dest"].items():
        tasks.append({"mode": "comfy", "name": f"image/{name}/{dest_dir}", "repo": d["repo"],
                      "includes": [fname], "dest": dest_dir})
    for cn, cd in man["common"].items():
        tasks.append({"mode": "comfy", "name": f"common/{cn}", **cd})
elif name in man["llm"]:
    d = man["llm"][name]
    if variant == "fp8":
        if not d.get("repo_fp8"):
            sys.exit(f"错误：{name} 没有官方 FP8 仓库（repo_fp8 未配置）")
        tasks.append({"mode": "llm", "name": f"{name}-fp8", "repo": d["repo_fp8"]})
    elif variant:
        sys.exit(f"错误：不支持的变体 '{variant}'（仅支持 fp8；默认下载主仓库）")
    else:
        tasks.append({"mode": "llm", "name": name, "repo": d["repo"],
                      "includes": d.get("gguf_include") and [d["gguf_include"]] or None})
        # GGUF 模型的多模态投影器（mmproj），与主分片同仓库，进同一目录
        if d.get("mmproj_include"):
            tasks.append({"mode": "llm", "name": name, "repo": d["repo"],
                          "includes": [d["mmproj_include"]]})
else:
    sys.exit(f"错误：清单里没有 '{name}'，可用 manage.sh list 查看")
print(json.dumps({"tasks": tasks}, ensure_ascii=False))
EOF
  echo "==> 下载任务: $(python3 -c "import json;print(len(json.load(open('$JOB'))['tasks']))") 项"
  docker pull python:3.12-slim >/dev/null 2>&1 || true
  docker run --rm \
    # MANAGE_RUN_AS：console agent 容器内是 root，直接 id -u 会把宿主 models/ 文件下成 root-owned；
    # agent 按 models 目录真实属主注入该变量。宿主手跑时未设置，行为与从前完全一致。
    --user "${MANAGE_RUN_AS:-$(id -u):$(id -g)}" \
    -v "$PWD/models:/models" -v "$JOB:/job.json:ro" -v "$PWD/scripts:/opt:ro" \
    -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_HOME=/models/.hf-cache -e HOME=/tmp -e PIP_CACHE_DIR=/tmp/pip-cache \
    -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    python:3.12-slim bash -c "pip install -q huggingface_hub hf_transfer hf_xet && python3 /opt/hf_fetch.py /job.json"
  rm -f "$JOB"
  echo "==> 完成: $name"
  # LLM 下完后以模型文件为准自动校准 models.json（think / max_ctx / 工具格式）
  if python3 -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open('models.json'))['llm'] else 1)" "$name"; then
    python3 scripts/check_llm_config.py "$name" --fix
  fi
  ;;

check)
  python3 scripts/check_llm_config.py "$@"
  ;;

verify)
  python3 scripts/verify_llm.py "$@"
  ;;

add)
  python3 - <<'EOF'
import json
man = json.load(open("models.json"))
print("新增模型到 models.json（Ctrl-C 取消）")
kind = input("类型 [llm/video/common] (llm): ").strip() or "llm"
name = input("唯一名称，如 qwen3.8-coder-27b: ").strip()
if not name or name in man.get(kind, {}):
    raise SystemExit("名称为空或已存在")
repo = input("HuggingFace 仓库，如 Qwen/Qwen3.8-27B: ").strip()
if kind == "llm":
    engine = input("引擎 [vllm/llamacpp] (vllm): ").strip() or "vllm"
    min_gpus = int(input("最少 GPU 数 (1): ").strip() or 1)
    vs = {}
    for v in input("变体列表，逗号分隔 (bf16,fp8): ").strip().split(","):
        v = v.strip()
        if v:
            vs[v] = float(input(f"  {v} 显存 GB: ").strip())
    entry = {"repo": repo, "engine": engine, "min_gpus": min_gpus, "variants": vs,
             "ctx_default": int(input("默认上下文 (65536): ").strip() or 65536),
             "desc": input("描述: ").strip() or name}
    repo_fp8 = input("官方 FP8 仓库（没有则留空）: ").strip()
    if repo_fp8:
        entry["repo_fp8"] = repo_fp8
    if engine == "llamacpp":
        entry["gguf_include"] = input("GGUF 文件名匹配模式，如 *UD-1.8bit*.gguf: ").strip()
        entry["coexist"] = (input("支持视频共存切分? [y/N]: ").strip().lower() == "y")
    man["llm"][name] = entry
else:
    dest = input("ComfyUI 目标子目录，如 diffusion_models: ").strip() or "diffusion_models"
    inc = [p.strip() for p in input("文件名匹配模式，逗号分隔: ").strip().split(",") if p.strip()]
    entry = {"repo": repo, "includes": inc or ["*"], "dest": dest}
    if kind == "video":
        entry["vram_480p_fp8"] = float(input("480p fp8 显存 GB: ").strip())
        entry["vram_720p_bf16"] = float(input("720p bf16 显存 GB: ").strip())
    entry["desc"] = input("描述: ").strip() or name
    man.setdefault(kind, {})[name] = entry
json.dump(man, open("models.json", "w"), ensure_ascii=False, indent=2)
print(f"已写入 models.json，下次启动菜单可见: {name}")
print("提示: think/max_ctx 无需手填——下载后会自动校准，也可随时跑 scripts/check_llm_config.py --fix")
EOF
  ;;

*)
  echo "未知命令: $cmd（可用: list / download / add / check / verify）" >&2; exit 1;;
esac
