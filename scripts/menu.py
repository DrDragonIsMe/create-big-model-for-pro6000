#!/usr/bin/env python3
"""启动入口核心：检测 GPU → 选 LLM → 按余量灰化视频选项 → 生成 .env + compose.selected.yaml"""
import argparse, json, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# YaRN 扩展上限复用 check_llm_config 的同一份实现：rope 缩放规则要在两处各写一遍的话，
# 迟早算出两个不同的天花板（那边校验放行 900000、这边却按原生 262144 把配置砍下来）。
# 该模块级只有 import 和一个纯路径常量，导入无副作用。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_llm_config import yarn_override_limit

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
        # 单卡放不下、但整池放得下：放行，由下面的 flagship 分支按 gpu_count 张卡做张量并行。
        # 原来这里无条件 exit，把本文件已实现的 TP 路径堵死了——
        # 于是 2×48GB 机器永远用不上 54GB bf16，只能退到 fp8 单卡。
        pool_gb = int(gmem * gpu_count * 0.92)
        if vram > pool_gb:
            sys.exit(f"错误：{llm_name}:{variant}（{vram}GB）单卡放不下，"
                     f"{gpu_count} 卡总可用 {pool_gb}GB 也放不下，请换更小的量化版")

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
        llm_devices = ",".join(str(i) for i in range(gpu_count))
        if engine == "vllm":
            if vram > gmem * gpu_count * 0.92:
                sys.exit(f"错误：{llm_name}:{variant}（{vram}GB）超过 {gpu_count} 卡总显存"
                         f"（{int(gmem * gpu_count * 0.92)}GB 可用），无法运行；"
                         f"请改选 GGUF 量化版或其他模型")
            tp = gpu_count
            mem_util = 0.92
            per_card_free = gmem - vram / gpu_count - 10
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

    # ---------- 多副本：每卡一个后端，网关按在途请求数分流 ----------
    # 本机实测（2×RTX4090 48GB，qwen3.8-27b）：bf16 TP=2 双卡 611 tok/s，
    # 反而低于 fp8 单卡 658 tok/s 还把两张卡都占满——解码是显存带宽瓶颈，
    # TP 切权重和 fp8 减字节两条路效果相互抵消，剩下 PCIe 无 NVLink 的 all-reduce 净亏。
    # 所以吞吐要靠"每卡一份 fp8 副本"，不是靠张量并行。
    replicas = max(1, int(args.replicas or 1))
    if replicas > 1:
        if engine != "vllm":
            sys.exit(f"错误：--replicas 只支持 vLLM 后端（{llm_name} 用的是 {engine}）")
        if replicas > gpu_count:
            sys.exit(f"错误：--replicas {replicas} 超过本机 GPU 数 {gpu_count}，"
                     f"每副本至少要一整个卡")
        if vram > round(gmem * meta["dual_llm_util"]):
            sys.exit(f"错误：{llm_name}:{variant}（{vram}GB）单卡放不下，副本模式跑不起来；"
                     f"要跨卡张量并行请去掉 --replicas，改用 --variant bf16")
        mode = "replica"
        llm_devices = ",".join(str(i) for i in range(replicas))
        tp = 1                          # 每副本一卡，KV 预算按单卡算
        if replicas < gpu_count:
            comfy_devices = str(gpu_count - 1)   # 剩下最高位那张卡留给视频
            free_video = gmem - meta["video_reserve_gb"]
        else:
            comfy_enabled = 0
            free_video = 0

    # ---------- 上下文长度：按显卡剩余显存尽量开大 ----------
    # 天花板取"原生上限"与"YaRN 扩展上限"的较大者：配了 YaRN 的模型本就是为跑超长而配，
    # 只按 max_ctx 封顶会把刻意扩出来的上下文当越界砍掉（27B：262144 → 1048576）。
    ctx_cap = max(int(d.get("max_ctx") or d["ctx_default"]), yarn_override_limit(d))
    # KV 预算要按"网关实际加载的那份权重"算。model_dir 在没有显式指定精度时优先用
    # <name>-fp8 目录，而 variants 里的 vram 是 menu 选中的变体——两者不一致时
    # （如 96GB 机上 menu 自动选 bf16:54GB、网关却加载 fp8:28GB）会低估可用 KV，
    # 把人家调好的长上下文当成"装不下"给砍掉。
    vram_eff = vram
    if engine == "vllm" and not args.variant and d.get("repo_fp8") \
       and _nonempty_dir(f"models/llm/{llm_name}-fp8") and "fp8" in d["variants"]:
        vram_eff = d["variants"]["fp8"]
    if engine == "vllm":
        total_budget = gmem * mem_util * (tp if tp > 1 else 1)
        kv_gb = max(0.0, total_budget - vram_eff - 4)  # 权重之外可用于 KV 的显存
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

    # ---------- 把本机算出的布局写回 models.json 的 gw ----------
    # 网关是容器，只 bind-mount 了 /opt/models.json，读不到宿主 .env：上面 .env 里的
    # LLM_TP_SIZE / LLM_MAX_LEN / LLM_MEM_UTIL 对它毫无作用（全仓库只有 start.sh 读过一次
    # LLM_TP_SIZE）。不写回 gw，本机布局就只是装饰，而沿用别台机器带来的 gw
    # （本机现在是 gpus=1、max_len=900000）会让 48GB 卡在首个请求加载时直接起不来。
    with open("models.json", encoding="utf-8") as f:
        man_disk = json.load(f)       # ensure_assets 刚跑过 check --fix，必须从盘上重读
    gw_old = man_disk["llm"][llm_name].setdefault("gw", {})
    gw_new = {"gpus": llm_devices}
    if mem_util is not None:          # llamacpp 旗舰走 tensor_split，没有 util 概念
        gw_new["util"] = mem_util
    if engine == "vllm":
        # 精度只在用户显式 --variant 时钉住；没指定就删键、回到 model_dir 的默认口径
        # （-fp8 优先）。否则上一次 --variant bf16 会一直赖在配置里，变成
        # "自动选了 fp8、网关却还在加载 54GB bf16" 的单卡装不下错配。
        # 另外只对 vLLM 动这个键：GGUF 的 variant 是量化名（如 ud1.8）不是目录后缀，
        # 写进去会让 model_dir 找不到 <name>-ud1.8 而把模型判成"未下载"。
        gw_new["variant"] = variant if args.variant else None
        # 副本数只在 >1 时落盘，单副本删键：既不改变现有机器（参考机）的 models.json，
        # 也让"没配 replicas"永远等价于今天的单容器行为
        gw_new["replicas"] = replicas if replicas > 1 else None
        # max_len 每次按本机实算，绝不与旧值取 min：旧值可能是别台机器（96GB→900000）
        # 或本机别种布局（fp8 单卡→186048）留下的快照，取 min 等于把快照一路继承下来——
        # 真机验证过：切到 bf16 双卡 TP=2 时本可给到 ~46 万，却被上一个布局的 186048 钉死。
        # 该防的是"算错"，靠的是按实际加载的权重估 KV + YaRN 天花板两头，不是靠不敢抬高。
        gw_new["max_len"] = int(max_len)
    changed = {k: (gw_old.get(k), v) for k, v in gw_new.items() if gw_old.get(k) != v}
    if changed:
        for k, (_prev, v) in changed.items():
            if v is None:
                gw_old.pop(k, None)
            else:
                gw_old[k] = v
        # 必须就地截断写，不能 tmp+rename：compose.yaml 是以"单文件"bind-mount
        # ./models.json → /opt/models.json，rename 会换 inode，容器会一直挂在旧 inode 上，
        # 宿主改完网关却永远读不到（实测宿主 186048 / 容器 900000）。
        # 仓库存 models.json 的两处写者（check_llm_config.py:151、manage.sh add）都是
        # open(out,"w") 就地截断，正是为了保住这个 inode；顺带也不补结尾换行，
        # 免得 git 把整份文件记成改动。
        json.dump(man_disk, open("models.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"==> 已按本机显存改写 models.json 的 gw[{llm_name!r}]（网关下个请求即生效）：")
        for k, (old, new) in changed.items():
            print(f"      {k}: {old} → {'（清除，回到默认口径）' if new is None else new}")

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
        "LLM_REPLICAS": replicas,
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
    ap.add_argument("--replicas", type=int, default=1,
                    help="同一 LLM 起几个后端副本（每副本一卡，网关按在途请求数分流）。"
                         "如 2×48GB 双卡各跑一份 fp8 把吞吐翻倍")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()
    man = load_manifest()
    gpu_count, gpu_mem = detect_gpus()
    compute(args, man, gpu_count, gpu_mem)

if __name__ == "__main__":
    main()
