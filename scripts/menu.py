#!/usr/bin/env python3
"""启动入口核心：检测 GPU → 选 LLM → 按余量灰化视频选项 → 生成 .env + compose.selected.yaml"""
import argparse, json, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def load_manifest():
    with open("models.json", encoding="utf-8") as f:
        return json.load(f)

def detect_gpus():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("错误：检测不到 NVIDIA GPU（nvidia-smi 失败）")
    names = [l for l in out.stdout.strip().splitlines() if l.strip()]
    mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()
    return len(names), [int(m) for m in mem]  # MiB

def llm_downloaded(name):
    d = os.path.join("models/llm", name)
    return os.path.isdir(d) and bool(os.listdir(d))

def video_downloaded(name, man):
    d = man["video"][name]["dest"]
    p = os.path.join("models/comfy", d)
    return os.path.isdir(p) and bool(os.listdir(p))

def _nonempty_dir(p):
    return os.path.isdir(p) and bool(os.listdir(p))

def ensure_assets(llm_name, variant, d, video_name, man):
    """新机器兜底：选中的模型没下载就先下载（manage.sh 下载后自带校准），
    已下载的跑一次 check --fix 校准（think/max_ctx 以模型文件为准，幂等）。"""
    def sh(cmd):
        print(f"\n==> 模型未下载，自动下载: {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            sys.exit("错误：模型下载失败，可重试或手动执行 " + " ".join(cmd))

    main_dir = os.path.join("models/llm", llm_name)
    fp8_dir = main_dir + "-fp8"
    # 选了 fp8 且有官方 FP8 仓库：只缺哪个下哪个（网关优先用 -fp8 目录）
    if variant == "fp8" and d.get("repo_fp8") and not _nonempty_dir(fp8_dir):
        sh(["bash", "scripts/manage.sh", "download", llm_name, "fp8"])
    if not _nonempty_dir(main_dir) and not _nonempty_dir(fp8_dir):
        sh(["bash", "scripts/manage.sh", "download", llm_name])
    if video_name != "none" and not video_downloaded(video_name, man):
        sh(["bash", "scripts/manage.sh", "download", video_name])
    subprocess.run([sys.executable, "scripts/check_llm_config.py", llm_name, "--fix"],
                   check=False)

def env_quote(v):
    """.env 同时被 start.sh source 和 docker compose 解析：
    含空格/通配符等字符的值必须加引号，否则 source 时会把后缀当命令执行"""
    s = str(v)
    if s and not re.fullmatch(r"[A-Za-z0-9_.,:/=@%+-]+", s):
        s = '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s

def pick_variant(llm, budget_gb):
    """在显存预算内选最大可用变体；预算内没有则返回 None"""
    best = None
    for v, gb in llm["variants"].items():
        if gb <= budget_gb and (best is None or gb > llm["variants"][best]):
            best = v
    return best

def video_tier(vdef, free_gb, slack):
    """返回 (标记, 说明, max_res)：'720' / '480' / 'risky480' / 'no'"""
    if free_gb >= vdef["vram_720p_bf16"]:
        return "720", "", 720
    if free_gb >= vdef["vram_480p_fp8"]:
        return "480", "仅 480p", 480
    if free_gb >= vdef["vram_480p_fp8"] - slack:
        return "risky480", "480p+offload(慢)", 480
    return "no", "", None

def compute(args, man, gpu_count, gpu_mem):
    meta = man["meta"]
    gmem = round(gpu_mem[0] / 1024)  # GB，同型号卡取首卡
    llms = man["llm"]

    # ---------- 第 1 级：LLM ----------
    llm_name = args.llm
    if llm_name is not None and llm_name not in llms:
        sys.exit(f"错误：清单里没有 LLM '{llm_name}'（可用 manage.sh list 查看，或 manage.sh add 新增）")
    if gpu_count == 1:
        budget = meta["single_llm_budget_gb"]
    else:
        budget = round(gmem * meta["dual_llm_util"])
    if llm_name is None:
        print(f"\n检测到 {gpu_count} 张 GPU（单卡 {gmem}GB）")
        print("第 1 步 · 选文本模型（★=已下载，✗=已弃用/跑不动；选中未下载的会自动先下载）：")
        avail = []
        for i, (n, d) in enumerate(llms.items(), 1):
            if d["min_gpus"] > gpu_count:
                continue
            if d.get("deprecated") and not llm_downloaded(n):
                continue  # 明确不用且未下载：不显示
            star = "★" if llm_downloaded(n) else " "
            vs = " / ".join(f"{v}:{gb}GB" for v, gb in d["variants"].items())
            fit = pick_variant(d, budget)
            full_cap = gmem - meta["safety_gap_gb"] - 2
            full_fit = pick_variant(d, full_cap) if (gpu_count == 1 and not fit) else None
            selectable = bool(fit) or bool(full_fit)
            mark = "" if fit else ("  ⚠ 独占单卡（视频不可用）" if full_fit else "  ✗ 单卡放不下")
            risk = " ⚠" if d.get("arch_risk") else ""
            dep = " ✗已弃用" if d.get("deprecated") else ""
            brief = d.get("menu_brief", d.get("desc", ""))
            print(f"  {i}) {star} {n:<24} {vs:<20} {brief}{risk}{dep}{mark}")
            if selectable and not d.get("deprecated"):
                avail.append(n)
        while True:
            c = input("输入编号或名称 > ").strip()
            hit = None
            if c.isdigit() and 1 <= int(c) <= len(llms):
                hit = list(llms.keys())[int(c) - 1]
            elif c in llms:
                hit = c
            if hit and hit in avail:
                llm_name = hit
                break
            print("  无效或不可用，请重选")
    d = llms[llm_name]
    if d["min_gpus"] > gpu_count:
        sys.exit(f"错误：{llm_name} 需要 {d['min_gpus']} 张卡，当前只有 {gpu_count} 张")
    if d.get("arch_risk") and not args.yes:
        print(f"\n⚠ 提醒：{d['arch_risk']}")
        input("按回车继续，Ctrl-C 取消 > ")

    if args.variant:
        variant = args.variant
        if variant not in d["variants"]:
            sys.exit(f"错误：{llm_name} 没有变体 '{variant}'，可选: {list(d['variants'])}")
    else:
        if gpu_count == 1:
            vbudget = meta["single_llm_budget_gb"]
        elif d["min_gpus"] == 2:
            vbudget = 10 ** 6  # 旗舰模型：直接取最大变体
        else:
            vbudget = round(gmem * meta["dual_llm_util"])
        variant = pick_variant(d, vbudget)
        if variant is None and gpu_count == 1:
            # 单卡独占：single 预算内装不下，退而求其次吃满整卡（视频不可共存）
            variant = pick_variant(d, gmem - meta["safety_gap_gb"] - 2)
        if variant is None:
            sys.exit(f"错误：{llm_name} 的所有变体都超过当前显存预算，请换模型")
    vram = d["variants"][variant]
    engine = d["engine"]
    # 单卡/双卡分工模式下，非旗舰模型的变体必须单卡放得下
    if gpu_count == 1 and vram > gmem - meta["safety_gap_gb"] - 2:
        sys.exit(f"错误：单卡可用显存约 {gmem - meta['safety_gap_gb'] - 2}GB，"
                 f"{llm_name}:{variant} 为 {vram}GB，请换更小的量化版或其他模型")
    if gpu_count >= 2 and d["min_gpus"] == 1 and vram > round(gmem * meta["dual_llm_util"]):
        sys.exit(f"错误：{llm_name}:{variant}（{vram}GB）单卡放不下，请选双卡旗舰模型")

    # ---------- 摆放 ----------
    comfy_enabled = 1
    comfy_devices, llm_devices, tp = "0", "0", 1
    mem_util, tensor_split = None, None
    if gpu_count == 1:
        if vram <= meta["single_llm_budget_gb"]:
            mode = "single"
            llm_devices = "0"
            mem_util = round((vram + meta["safety_gap_gb"]) / gmem, 2)
            free_video = gmem - vram - meta["safety_gap_gb"] - 2
        elif vram <= gmem - meta["safety_gap_gb"] - 2:
            # 单卡独占旗舰：模型吃满整卡，视频不可共存（与双卡旗舰同语义）
            mode = "flagship"
            llm_devices = "0"
            mem_util = round((vram + meta["safety_gap_gb"]) / gmem, 2)
            comfy_enabled = 0
            free_video = 0
        else:
            sys.exit(f"错误：单卡可用显存约 {gmem - meta['safety_gap_gb'] - 2}GB，"
                     f"{llm_name}:{variant} 为 {vram}GB，装不下")
    elif vram <= round(gmem * meta["dual_llm_util"]):
        mode = "dual"
        llm_devices = "1"
        mem_util = 0.92  # 独占一卡，尽量多留 KV 上下文
        free_video = gmem - meta["video_reserve_gb"]
    else:
        mode = "flagship"
        llm_devices = "0,1"
        if engine == "vllm":
            if vram > gmem * gpu_count * 0.92:
                sys.exit(f"错误：{llm_name}:{variant}（{vram}GB）超过 {gpu_count} 卡总显存"
                         f"（{int(gmem * gpu_count * 0.92)}GB 可用），无法运行；"
                         f"请改选 GGUF 量化版或其他模型")
            tp = 2
            mem_util = 0.92
            per_card_free = gmem - vram / 2 - 10
            free_video = per_card_free  # 两卡对称，视频在哪张卡都一样
        else:  # llamacpp
            if d.get("coexist"):
                reserve = meta["coexist_video_reserve_gb"]
                g0 = gmem - reserve
                g1 = gmem - meta["video_reserve_gb"]
                tensor_split = f"{g0 / gmem:.3f},{g1 / gmem:.3f}"
                free_video = reserve
            else:
                comfy_enabled = 0
                free_video = 0

    # ---------- 上下文长度：按显卡剩余显存尽量开大 ----------
    ctx_cap = d.get("max_ctx", d["ctx_default"])
    if engine == "vllm":
        total_budget = gmem * mem_util * (tp if tp > 1 else 1)
        kv_gb = max(0.0, total_budget - vram - 4)  # 权重之外可用于 KV 的显存
        est_tokens = int(kv_gb * 15300)  # 本机实测 ~15.2K tokens/GB（27B@96GB 混合架构）
        max_len = max(32768, min(ctx_cap, est_tokens))
    else:
        # llamacpp：上下文即 --ctx-size，直接按 models.json 的 gw.max_len（已按 KV 余量配好）
        max_len = int(d.get("gw", {}).get("max_len") or d["ctx_default"])

    # ---------- 第 2 级：视频（依赖 LLM 余量，✗ 灰化） ----------
    videos = man["video"]
    slack = 12  # offload 余量
    menu = []
    for n, vd in videos.items():
        tier, note, res = video_tier(vd, free_video, slack) if comfy_enabled else ("no", "", None)
        star = "★" if video_downloaded(n, man) else " "
        menu.append((n, tier, note, res, star))
    print("\n第 2 步 · 选视频任务（按上一步剩余显存自动判定）：") if llm_name == args.llm or args.llm is None else None
    for i, (n, tier, note, res, star) in enumerate(menu, 1):
        vd = videos[n]
        if tier == "no":
            tag = "✗ LLM 独占或显存不足"
        elif tier == "720":
            tag = f"✓ 余{int(free_video)}GB, 可 720p"
        elif tier == "480":
            tag = f"✓ 仅 480p (余{int(free_video)}GB)"
        else:
            tag = f"⚠ 480p+offload (余{int(free_video)}GB)"
        print(f"  {i}) {n} {star} {tag}  {vd['desc']}")

    video_name = args.video
    ok = {n for n, t, _, _, _ in menu if t != "no"}
    if video_name is None:
        if not ok:
            print("  （无可用视频选项，自动选择'不跑视频'）")
            video_name = "none"
        elif not args.yes:
            while True:
                c = input("输入编号/名称，或 n=不跑视频 > ").strip().lower()
                if c in ("n", "none", ""):
                    video_name = "none" if c else None
                    if video_name is None:
                        continue
                    break
                hit = None
                if c.isdigit() and 1 <= int(c) <= len(menu):
                    hit = menu[int(c) - 1][0]
                elif c in videos:
                    hit = c
                if hit and hit in ok:
                    video_name = hit
                    break
                print("  该项不可用（灰化），请重选")
        else:
            video_name = "none"
    if video_name != "none":
        if video_name not in ok:
            sys.exit(f"错误：视频选项 '{video_name}' 在当前 LLM/卡数下不可用（显存不足或被独占）")
        tier = next(t for n, t, _, _, _ in menu if n == video_name)
        max_res = 720 if tier == "720" else 480
    else:
        max_res = None
        comfy_enabled = 0 if mode == "flagship" else comfy_enabled
    if video_name == "none" and not args.yes and args.video is None:
        comfy_enabled = 0

    # ---------- 模型就位：缺则下载，有则校准（新机器一键起） ----------
    ensure_assets(llm_name, variant, d, video_name, man)

    # ---------- 输出 ----------
    api_key = os.environ.get("LLM_API_KEY", "")
    env = {
        "AI_GPU_COUNT": gpu_count, "AI_MODE": mode,
        "LLM_NAME": llm_name, "LLM_ENGINE": engine, "LLM_REPO": d.get("repo_fp8") if (variant == "fp8" and d.get("repo_fp8")) else d["repo"],
        "LLM_VARIANT": variant, "LLM_VRAM_GB": vram,
        # 官方 FP8 已下载则用独立目录，否则从 bf16 动态量化
        "LLM_MODEL_PATH": (f"/models/{llm_name}-fp8"
            if (variant == "fp8" and d.get("repo_fp8")
                and os.path.isdir(f"models/llm/{llm_name}-fp8") and os.listdir(f"models/llm/{llm_name}-fp8"))
            else f"/models/{llm_name}"),
        "LLM_TP_SIZE": tp, "LLM_MEM_UTIL": mem_util if mem_util else "",
        "LLM_MAX_LEN": max_len,
        "LLM_DEVICES": llm_devices, "LLM_QUANT": ("--quantization fp8" if variant == "fp8" and not (d.get("repo_fp8") and os.path.isdir(f"models/llm/{llm_name}-fp8") and os.listdir(f"models/llm/{llm_name}-fp8")) else ""),
        "LLAMA_TENSOR_SPLIT": tensor_split or "", "LLAMA_INCLUDE": d.get("gguf_include", ""),
        "COMFY_ENABLED": comfy_enabled, "COMFY_DEVICES": comfy_devices,
        "VIDEO_NAME": video_name, "VIDEO_MAX_RES": max_res or "",
        "LLM_API_KEY": api_key,
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HOST_ROOT": ROOT,
    }
    with open(".env", "w") as f:
        for k, v in env.items():
            f.write(f"{k}={env_quote(v)}\n")

    write_compose(env, d)
    print(f"\n已生成 .env（模式 {mode}）与 compose.selected.yaml：")
    print(f"  LLM : {llm_name}:{variant} ({vram}GB, GPU [{llm_devices}]"
          + (f", TP={tp}" if tp > 1 else "")
          + (f", tensor-split={tensor_split}" if tensor_split else "") + ")")
    print(f"  视频: {video_name}" + (f"（上限 {max_res}p, GPU [{comfy_devices}]）" if video_name != "none" else "（不启动 ComfyUI）"))
    return env

def yq_devices(ids):  # ["0","1"] -> YAML list
    return "[" + ", ".join(f'"{i}"' for i in ids) + "]"

def write_compose(env, llm_def):
    # LLM 不再静态定义：ai-gateway（compose.yaml）独占 8000，按请求动态拉起/卸载后端
    lines = ["services:"]
    common = """    restart: unless-stopped
    environment:
      - HF_ENDPOINT=${HF_ENDPOINT}"""
    if env["COMFY_ENABLED"] and env["VIDEO_NAME"] != "none":
        lines.append(f"""  comfyui:
    build: ./docker/comfyui
    image: ai-comfyui:local
    container_name: ai-comfyui
{common}
    ports: ["0.0.0.0:8188:8188"]
    volumes:
      - {ROOT}/models/comfy:/opt/ComfyUI/models
      - {ROOT}/data:/data
    command: >-
      python3 main.py --listen 0.0.0.0 --port 8188 --highvram
      --output-directory /data/output --input-directory /data/input
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: {yq_devices(env['COMFY_DEVICES'].split(','))}
              capabilities: [gpu]""")
    # 无视频时 LLM 后端全由网关动态拉起，selected 里没有服务；
    # 空映射写成 services: {}——裸 "services:" 会被 compose 以 "services must be a mapping" 拒绝
    if len(lines) == 1:
        lines[0] = "services: {}"
    with open("compose.selected.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser(description="AI 服务启动配置")
    ap.add_argument("--llm", help="LLM 名称（非交互）")
    ap.add_argument("--video", help="视频任务名称或 none（非交互）")
    ap.add_argument("--variant", help="强制变体，如 fp8/bf16/ud1.8")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()
    man = load_manifest()
    gpu_count, gpu_mem = detect_gpus()
    compute(args, man, gpu_count, gpu_mem)

if __name__ == "__main__":
    main()
