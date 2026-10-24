#!/usr/bin/env python3
"""实测显存：把 models.json 里"猜出来的 GB"换成"量出来的 GB"。

  calib_vram.py text  <llm>            从已加载后端的日志量 KV tokens/GB → llm.<llm>.kv_tokens_per_gb
  calib_vram.py video <name> [--res]   真跑一次工作流，采样单卡峰值 → video.<name>.vram_*_peak_gb

为什么要它：meta 里的绝对 GB 和 video 的 vram_* 都是别台机器/另一套口径留下的数
（vram_* 是流水线各阶段显存之和，ComfyUI 阶段间会打 "Unloaded partially: N MB freed,
M MB remains loaded"，说明这些阶段本来就不要求同时在卡上）。用它们做准入判定，
48GB 机器上就把实测 44.8GB 峰值能跑的 minimax-h3 判成"放不下"。

两个子命令都只读容器、只写 models.json（就地截断写，见 hwprofile.write_manifest_inplace），
绝不自己拉起/停止任何容器：加载是网关的职责，脚本抢过去就会和网关打架。
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hwprofile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MANIFEST = "models.json"

# 测试提示词：短的、任何模型都能出画的，只为把显存推上去而不是评质量
TEST_PROMPT = "一只橘猫在窗台上打盹，阳光照进来，镜头缓慢推进"


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def save(man, changed_lines):
    hwprofile.write_manifest_inplace(MANIFEST, man)
    print("==> 实测值已写入 models.json（就地写，网关下个请求即生效）：")
    for line in changed_lines:
        print(f"      {line}")


def report(where, old, new, unit=""):
    if old == new:
        print(f"    {where}: {new}{unit}（与盘上相同，未改写）")
        return False
    print(f"    {where}: {old if old is not None else '（无）'} → {new}{unit}")
    return True


def running_backends(llm_name):
    """当前在跑的该模型后端容器名（含多副本 -r0/-r1 后缀）。"""
    out = hwprofile.run(["docker", "ps", "--format", "{{.Names}}"])
    if out is None:
        return None  # docker 不可用，和"没有在跑"是两回事，调用方要分开报
    pref = (f"ai-vllm-{llm_name}", f"ai-llamacpp-{llm_name}")
    return sorted(n for n in out.split() if any(n == p or n.startswith(p + "-") for p in pref))


def cmd_text(args):
    man = load_manifest()
    d = man["llm"].get(args.llm)
    if d is None:
        sys.exit(f"错误：清单里没有 LLM '{args.llm}'（可用 ./scripts/manage.sh list 查看）")
    if d.get("engine", "vllm") != "vllm":
        # llama.cpp 没有 KV cache 预留这一步，日志里不出那两行，量不出这个口径
        print(f"提示：{args.llm} 用 {d.get('engine')} 后端，"
              f"kv_tokens_per_gb 只有 vLLM 的日志里有依据，跳过")
        return 0
    cands = running_backends(args.llm)
    if cands is None:
        sys.exit("错误：docker CLI 不可用（或未加入 docker 组）。"
                 "请用 docker ps 能正常输出的身份运行，或手动执行 "
                 "docker logs ai-vllm-%s 找那两行 KV 日志" % args.llm)
    if not cands:
        # 网关是懒加载的：没请求过就没有后端容器，日志自然还不存在。
        # 这里绝不替用户把模型拉起来——那会和在跑的网关抢显存、还可能挤掉别的模型。
        print(f"未标定：{args.llm} 的后端容器没在跑（网关按需加载）。")
        print("  先发一次请求让它加载，再重跑本命令：")
        print("    curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\\n"
              f"      -d '{{\"model\":\"{args.llm}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'")
        return 1
    dens = None
    for cname in cands:
        dens = hwprofile.kv_density_from_log(cname)
        if dens:
            break
    if dens is None:
        print(f"未标定：{'、'.join(cands)} 的日志里还没有 KV cache 那两行"
              f"（模型仍在加载，或首请求未完成）。等加载完再重跑。")
        return 1
    new = int(dens["tokens_per_gb"])
    old = d.get("kv_tokens_per_gb")
    where = f"llm.{args.llm}.kv_tokens_per_gb"
    print(f"  实测来源 {args.llm}（{'、'.join(cands)}）："
          f"KV {dens['kv_gib']}GiB / {dens['kv_tokens']} tokens"
          + (f" / max_len {dens.get('max_model_len')}"
             f" 并发 {dens.get('concurrency')}x" if dens.get("concurrency") else ""))
    if not report(where, old, new, " tokens/GB"):
        return 0
    d["kv_tokens_per_gb"] = new
    save(man, [f"{where}: {old} → {new}"])
    return 0


# ────────────────────────── video ──────────────────────────

class Sampler(threading.Thread):
    """每秒采一次每卡已用显存。用线程而不是子进程轮询，是为了不和工作流的 HTTP 轮询互相拖时间。"""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = {}
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            out = hwprofile.run(["nvidia-smi", "--query-gpu=index,memory.used",
                                 "--format=csv,noheader,nounits"])
            if out:
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        idx, used = int(parts[0]), int(float(parts[1]))
                    except ValueError:
                        continue
                    self.samples.setdefault(idx, []).append(used)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def comfy_get(port, path, timeout=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def workflow_path(vdef, res):
    """workflow 字段可能是单个路径，也可能是按分辨率分开的字典；两种清单写法都认。"""
    wf = vdef.get("workflow")
    if isinstance(wf, dict):
        return wf.get(str(res)) or wf.get(f"{res}p") or wf.get("default")
    return wf


def tune(graph, steps):
    """就地改测试工作流：换提示词、（可选）降步数。其余参数原样保留——
    帧数/分辨率决定了激活显存，改了就不是"这台机器跑这个模型要多少显存"的答案了。"""
    touched = 0
    nodes = graph.get("prompt", graph) if isinstance(graph, dict) else {}
    for node in nodes.values():
        inp = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inp, dict):
            continue
        for k, v in list(inp.items()):
            if isinstance(v, str) and k in ("prompt", "text", "positive") and v.strip():
                inp[k] = TEST_PROMPT
                touched += 1
            elif steps and k == "steps" and isinstance(v, (int, float)):
                inp[k] = int(steps)
                touched += 1
            elif k == "noise_seed" and isinstance(v, (int, float)):
                # 必须换种子：工作流图哈希相同的话 ComfyUI 会走 execution_cached，
                # 几乎不重新采样，采到的"峰值"其实是上一次残留，会系统性低报。
                inp[k] = int(time.time()) % 10**8
                touched += 1
    return touched


def cmd_video(args):
    man = load_manifest()
    vdef = man["video"].get(args.video)
    if vdef is None:
        sys.exit(f"错误：清单里没有视频模型 '{args.video}'"
                 f"（可用 ./scripts/manage.sh list 查看）")
    res = args.res or int(vdef.get("res_default") or vdef.get("default_res") or 480)
    if res not in (480, 720):
        sys.exit(f"错误：--res 只支持 480/720（当前 {res}）")
    key = "vram_480p_fp8_peak_gb" if res == 480 else "vram_720p_bf16_peak_gb"
    if args.peak_gb is not None:
        # 人工录入实测峰值，用来解开一个自锁死循环：自动测量要求 ComfyUI 在跑，
        # ComfyUI 只在准入通过时才起，而准入又要求实测峰值——全新机器上第一圈过不去。
        # minimax-h3 就是实例：清单声明 55GB 是流水线阶段之和，2×4090 实测单卡峰值
        # 45.7GB 且出片成功，却因 55 > 46 被永久判成"放不下"，也就永远测不了。
        # 操作者手工量到后从这里记一次；等 ComfyUI 能起来了，不带 --peak-gb 重跑，
        # 由自动测量覆盖这个值。
        if report(f"video.{args.video}.{key}", vdef.get(key), args.peak_gb, "GB（人工录入）"):
            vdef[key] = args.peak_gb
            save(man, [f"{args.video} {key} = {args.peak_gb}（人工录入，待自动复测覆盖）"])
        return
    rel = workflow_path(vdef, res)
    if not rel:
        sys.exit(f"错误：models.json 的 video.{args.video} 没有 workflow 路径，无法自测")
    path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        sys.exit(f"错误：工作流文件不存在：{path}（先跑 ./scripts/manage.sh download common）")
    with open(path, encoding="utf-8") as f:
        graph = json.load(f)

    try:
        comfy_get(args.port, "/system_stats", timeout=5)
    except (urllib.error.URLError, OSError, ValueError) as e:
        # 同样不替用户起容器：ComfyUI 起不起、占哪张卡由 start.sh / 网关决定
        print(f"未标定：ComfyUI 没有在 127.0.0.1:{args.port} 应答（{e}）")
        print("  先把视频服务起起来，再重跑本命令：")
        print("    ./scripts/start.sh --switch video")
        print(f"    python3 scripts/calib_vram.py video {args.video} --res {res}")
        return 1

    if not isinstance(graph, dict):
        sys.exit(f"错误：{path} 不是 ComfyUI API 格式的工作流（顶层应是节点字典）")
    touched = tune(graph, args.steps)
    print(f"==> 提交实测工作流 {rel}（{res}p，改动 {touched} 处测试参数）到 :{args.port}")
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/prompt",
                                data=json.dumps({"prompt": graph}).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        sys.exit(f"错误：ComfyUI 拒绝该工作流（HTTP {e.code}）：{e.read().decode(errors='replace')[:500]}")
    pid = resp.get("prompt_id")
    if not pid:
        sys.exit(f"错误：ComfyUI 未受理（{resp}）")

    samp = Sampler()
    samp.start()
    deadline = time.time() + args.timeout
    hist = {}
    try:
        while time.time() < deadline:
            try:
                hist = comfy_get(args.port, f"/history/{pid}", timeout=10) or {}
            except (urllib.error.URLError, OSError, ValueError):
                pass  # ComfyUI 忙到短暂不响应查询属正常，继续等
            if hist.get(pid):
                break
            time.sleep(2)
        else:
            print(f"未标定：{args.timeout}s 内没等到 {pid} 出结果，停止等待（已采到的样本丢弃）")
            return 1
        # 出结果后再多采 3 秒：解码/存盘收尾的显存尖峰常落在 history 落盘前后，
        # 立刻停采样会系统性低报峰值（实测差 1.7GB），而低报会放进真装不下的模型。
        time.sleep(3)
    finally:
        samp.stop()
        samp.join(timeout=5)

    outs = hist.get(pid) or {}
    status = (outs.get("status") or {})
    if status and status.get("status_str") not in (None, "success"):
        print(f"未标定：ComfyUI 报 {status.get('status_str')}（不写实测值，避免把失败的峰值当基准）")
        return 1
    prof = hwprofile.video_peak_from_samples(samp.samples)
    if not prof["peak_mib"]:
        print("未标定：没采到任何显存样本（nvidia-smi 不可用？）")
        return 1
    # 取"所有卡里最高的那张"的峰值：网关把 ComfyUI 放到哪张卡是运行时决定的，
    # 准入判定必须保证换成最差那张卡也放得下，否则 models.json 里的 peak 就是假基准。
    worst = max(prof["peak_mib"].items(), key=lambda kv: kv[1])
    peak_gb = math.ceil(worst[1] / 1024.0 * 10) / 10
    old = vdef.get(key)
    print(f"  采样 {prof['seconds']}s，单卡峰值 GPU{worst[0]} = {worst[1]}MiB"
          f"（各卡 {prof['peak_mib']}）")
    if not report(f"video.{args.video}.{key}", old, peak_gb, " GB"):
        return 0
    vdef[key] = peak_gb
    save(man, [f"video.{args.video}.{key}: {old} → {peak_gb}"])
    print(f"  说明：清单声明的阶段之和是 "
          f"{vdef.get('vram_480p_fp8') if res == 480 else vdef.get('vram_720p_bf16')}GB，"
          f"菜单准入会改用实测 {peak_gb}GB")
    return 0


def main():
    ap = argparse.ArgumentParser(description="实测显存并写回 models.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text", help="从已加载的 vLLM 日志量 KV tokens/GB")
    t.add_argument("llm", help="LLM 名称，如 qwen3.8-27b")
    t.set_defaults(fn=cmd_text)
    v = sub.add_parser("video", help="跑一次工作流量单卡峰值")
    v.add_argument("video", help="视频模型名，如 minimax-h3")
    v.add_argument("--res", type=int, choices=[480, 720], help="测试分辨率，默认取清单值")
    v.add_argument("--steps", type=int, help="临时降步数（只为省时间，不影响峰值口径）")
    v.add_argument("--port", type=int, default=8188, help="ComfyUI 宿主端口，默认 8188")
    v.add_argument("--timeout", type=int, default=1800, help="等工作流结束的最长秒数")
    v.add_argument("--peak-gb", dest="peak_gb", type=float,
                   help="跳过自动测量，直接录入已人工实测的单卡峰值 GB。只用于破首次自锁："
                        "没实测值→准入按阶段之和拒绝→ComfyUI 起不来→永远测不到")
    v.set_defaults(fn=cmd_video)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
