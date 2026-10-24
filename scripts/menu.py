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
import hwprofile  # 探测 + 派生预算：meta 里那批绝对 GB 的唯一替代口径

def load_manifest():
    with open("models.json", encoding="utf-8") as f:
        return json.load(f)

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
    """返回 (标记, 说明, max_res)：'720' / '480' / 'risky480' / 'no'

    优先用 calib_vram.py video 实测的单卡峰值（vram_*_peak_gb），没有实测值才退回清单
    声明值。声明值是流水线各阶段显存之和，而 ComfyUI 在阶段之间会打
    "Unloaded partially: N MB freed, M MB remains loaded"——阶段并不要求同时在卡上，
    所以之和系统性偏大：minimax-h3 声明 55GB，2×4090 上实测单卡峰值只有 44.8GB 且出片成功。
    继续拿 55 做准入，就是把每台 48GB 机器都判成"跑不了"。

    slack（offload 余量）只打折声明值：它那 12GB 本就是为了吃掉"阶段之和"的高估；
    实测峰值里没有这部分水分，再打一次折等于把同一个宽容计两遍，会把真装不下的
    机器判成"能跑"（48GB 机 free=40 vs 实测 45.7 必须还是 no）。
    """
    meas_720 = vdef.get("vram_720p_bf16_peak_gb")
    meas_480 = vdef.get("vram_480p_fp8_peak_gb")
    need_720 = meas_720 or vdef["vram_720p_bf16"]
    need_480 = meas_480 or vdef["vram_480p_fp8"]
    if free_gb >= need_720:
        return "720", "", 720
    if free_gb >= need_480:
        return "480", "仅 480p", 480
    if not meas_480 and free_gb >= need_480 - slack:
        return "risky480", "480p+offload(慢)", 480
    return "no", "", None

def compute(args, man, gpu_count, gpu_mem):
    meta = man["meta"]
    gmem = round(gpu_mem[0] / 1024)  # GB，同型号卡取首卡
    # 各角色预算从探测结果派生，meta 只在人类显式写过且量纲对得上时当覆盖项
    # （48GB 机上 meta 的 video_reserve_gb=8 是 96GB 参考机遗留值，会被退回 margin）
    bud = hwprofile.budgets(meta, gmem)
    llms = man["llm"]
    # 派生出来的预算一定要打出来：meta 的绝对数被忽略掉是"安全垫按本机重算"，
    # 悄悄重算就等同于清单在骗人
    print(f"本机预算（从 {gpu_count}×{gmem}GB 派生，margin {bud['margin']}GB）："
          f"LLM 单卡 {bud['llm_card']}GB / 视频单卡 {bud['video_card']}GB / "
          f"单卡共存给 LLM {bud['single_llm']}GB")
    if bud["overridden"]:
        print(f"  · 取自 models.json meta 的覆盖项：{'、'.join(bud['overridden'])}")
    if bud["ignored"]:
        print(f"  · meta 里这些键是别台大显存机器的绝对值，本机按 margin 重算："
              f"{'、'.join(bud['ignored'])}"
              f"（video_reserve_gb={meta.get('video_reserve_gb')}GB 相当于单卡 "
              f"{round(100.0 * meta.get('video_reserve_gb', 0) / gmem)}%，"
              f"派生值 {bud['margin']}GB）")

    # ---------- 第 1 级：LLM ----------
    llm_name = args.llm
    if llm_name is not None and llm_name not in llms:
        sys.exit(f"错误：清单里没有 LLM '{llm_name}'（可用 manage.sh list 查看，或 manage.sh add 新增）")
    if gpu_count == 1:
        budget = bud["single_llm"]
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
            vbudget = bud["single_llm"]
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
        if vram <= bud["single_llm"]:
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
        free_video = bud["video_card"]
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
                # coexist_video_reserve_gb=46 也是参考机（96GB 卡）的数，量纲是"半张卡"。
                # 卡 0 上留给视频的显存最多半卡，超过就说明这数是从大卡机继承来的：
                # 照搬到 48GB 机会只剩 2GB 给 LLM，tensor_split 直接算成废值。
                reserve = min(meta["coexist_video_reserve_gb"], gmem / 2)
                g0 = gmem - reserve
                g1 = bud["video_card"]
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
            free_video = bud["video_card"]
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
        # 密度两个来源：kv_tokens_per_gb 是 calib_vram.py text 从 vLLM 自己日志里量的
        # （qwen3.8-27b@48GB：KV 11.99GiB / 190,622 tokens = 15898 tokens/GB）；
        # 15300 是更早那台 96GB 机器上估出来的旧值，留着当"没量过"的兜底。
        # 这数是 per-model 的属性、不是机器的：27B 是 Qwen3_5ForConditionalGeneration
        # 混合 Mamba cache，KV 才这么密；换成纯 attention 模型会低一个量级。
        density = int(d.get("kv_tokens_per_gb") or 15300)
        est_tokens = int(kv_gb * density)
        max_len = max(32768, min(ctx_cap, est_tokens))
        if d.get("kv_tokens_per_gb"):
            # 有实测值就再拿它反查一遍并发度：max_len 超过 kv_gb×密度就意味着一个完整请求
            # 都装不满（并发 <1.0），vLLM 起来也只是摆设。防的是陈旧实测值——
            # 密度是上一个布局/上一次量化留下的，而这里的 kv_gb 已经按当前布局重算过了。
            kv_cap = int(kv_gb * density)
            if max_len > kv_cap:
                print(f"  ⚠ 实测密度 {density} tokens/GB × KV {kv_gb:.1f}GB 只装得下 "
                      f"{kv_cap} tokens（并发<1.0），max_len {int(max_len)} → {kv_cap}")
                max_len = max(4096, kv_cap)  # 4096 兜底：再短就不是能用的上下文了
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
        # 就地截断写，不能 tmp+rename：compose.yaml 以"单文件"bind-mount ./models.json，
        # rename 换 inode 会让运行中的容器一直挂在旧 inode 上（宿主 186048 / 容器 900000）。
        # 规则与 check_llm_config.py:151、manage.sh add 同源，收在 hwprofile 里一份。
        hwprofile.write_manifest_inplace("models.json", man_disk)
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

def report_occupancy(entries, strict):
    """摆放前把"本机上不归本系统管的显存持有者/容器"打出来。

    为什么要专门打：下面所有预算都是"整张卡 - margin"算出来的，卡上却要有别的进程先咬走
    一截，算出来的 max_len/util 就是空中楼阁——参考那次误判就是别人的进程占着显存，
    菜单却判成"这个模型放不下"。默认只警告继续跑（显存空着也是空着，占的人多半是自己人）；
    --strict 或 AI_STRICT=1 直接退出，供部署脚本卡住不放行。
    """
    bad = [e for e in entries if not e.get("owned", True)]
    for e in entries:
        if e.get("kind") == "notice":
            print(f"（审计降级）{e['msg']}")
    if not bad:
        return
    print("\n" + "!" * 62)
    print("⚠ 显存被本系统之外的东西占着（ai-/bench- 容器之外）：")
    for e in bad:
        if e["kind"] == "gpu_process":
            print(f"    PID {e['pid']:<8} 用户={e['user']:<10} 已跑={e['etime']:<12} "
                  f"{e['mem_mib'] / 1024:.1f}GiB  {e['args'][:90]}")
        else:
            print(f"    容器 {e['container']:<28} 镜像={e['image']:<28} {e['status']}")
    print("  后果：实际可用显存比下面算出来的小，模型可能加载失败或被误判成"
          "「显存放不下」（这台机器上真发生过一次）。")
    print("  处理：确认这些进程可以让位后停掉它们，或重跑 "
          "`python3 scripts/menu.py --video none -y` 只起文本。")
    print("!" * 62)
    if strict:
        sys.exit("错误：--strict/AI_STRICT=1 下不做带外显存的摆放。"
                 "清掉上面列出的占用者后重跑，或去掉 --strict 接受告警继续。")


def sync_meta(man, prof):
    """把探测到的显存写回 models.json 的 meta：那上面的 "96" 是参考机的数，
    在 48GB 机器上一眼看上去就是错的（gpu_mem_gb 全仓库没有消费者，纯粹给人看的，
    所以才更要把本机真值写上去）。只在值真的变了时才动盘，幂等。"""
    meta = man.setdefault("meta", {})
    gib = int(round(prof["gmem_mib"] / 1024))
    new = {"gpu_mem_gb": gib, "gpu_count": prof["gpu_count"]}
    changed = {k: (meta.get(k), v) for k, v in new.items() if meta.get(k) != v}
    if not changed:
        return
    meta.update({k: v for k, (_o, v) in changed.items()})
    hwprofile.write_manifest_inplace("models.json", man)
    print(f"==> models.json 的 meta 已按本机改写（原值是别台机器的显存）：")
    for k, (old, val) in changed.items():
        print(f"      meta.{k}: {old} → {val}")
    if prof["mixed"]:
        print(f"      注意：本机卡型号不一致 {prof['names']}，按最小的那张算")

def main():
    ap = argparse.ArgumentParser(description="AI 服务启动配置")
    ap.add_argument("--llm", help="LLM 名称（非交互）")
    ap.add_argument("--video", help="视频任务名称或 none（非交互）")
    ap.add_argument("--variant", help="强制变体，如 fp8/bf16/ud1.8")
    ap.add_argument("--replicas", type=int, default=1,
                    help="同一 LLM 起几个后端副本（每副本一卡，网关按在途请求数分流）。"
                         "如 2×48GB 双卡各跑一份 fp8 把吞吐翻倍")
    ap.add_argument("--strict", action="store_true",
                    help="发现本系统之外的显存占用者时直接退出（等价 AI_STRICT=1）")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    args = ap.parse_args()
    man = load_manifest()
    try:
        prof = hwprofile.detect()
    except RuntimeError as e:
        sys.exit(f"错误：{e}")
    # 先审计再摆放：占用者在别人手里，算出来的预算全是乐观值
    report_occupancy(hwprofile.foreign_occupancy(),
                     args.strict or os.environ.get("AI_STRICT") == "1")
    sync_meta(man, prof)
    compute(args, man, prof["gpu_count"], [prof["gmem_mib"]])

if __name__ == "__main__":
    main()
