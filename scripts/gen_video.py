#!/usr/bin/env python3
"""MiniMax H3 文生/图生视频 CLI 客户端（走 ComfyUI REST API）

用法:
    python3 scripts/gen_video.py "一只柴犬在月球上喝茶，电影感"
    python3 scripts/gen_video.py "海浪拍打礁石" --seconds 10 --width 1280 --height 736
    python3 scripts/gen_video.py "女孩转身微笑" --first 首帧.png --turbo
输出: 打印生成的文件名（ComfyUI output 目录，即本机 data/output/）
首/尾帧图片需先放入 data/input/（ComfyUI 输入目录）
"""
import argparse, json, os, sys, time
import urllib.request

# 本机 ComfyUI 直连：系统代理的 no_proxy 用 CIDR 写法，urllib 不认识，会误走代理导致 502
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

API = os.environ.get("COMFY_API", "http://127.0.0.1:8188")
WF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data/workflows/minimax_h3_t2v_api.json")
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def frames_for(seconds):
    """按时长算帧数，对齐模型的 17k+5 网格（124 帧 ≈ 5 秒）"""
    m = max(5, round(seconds * 24))
    return m + (5 - m % 17) % 17


def submit(wf):
    data = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{API}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(OPENER.open(req).read()).get("prompt_id")


def wait(pid, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(OPENER.open(f"{API}/history/{pid}").read())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    print("ERROR:", json.dumps(st, ensure_ascii=False)[:800])
                    return None
                if st.get("completed") or h[pid].get("outputs"):
                    return h[pid]
        except Exception:
            pass
        time.sleep(5)
    print("TIMEOUT")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=5, help="时长秒（最长 15）")
    ap.add_argument("--steps", type=int, default=0, help="采样步数（默认 20，turbo 8）")
    ap.add_argument("--turbo", action="store_true", help="挂 8 步 turbo LoRA 加速")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--first", default="", help="首帧图片名（data/input/ 内）")
    ap.add_argument("--last", default="", help="尾帧图片名（data/input/ 内）")
    args = ap.parse_args()

    steps = args.steps or (8 if args.turbo else 20)
    wf = json.load(open(WF))
    wf["5"]["inputs"]["prompt"] = args.prompt
    wf["5"]["inputs"]["width"] = args.width
    wf["5"]["inputs"]["height"] = args.height
    wf["5"]["inputs"]["length"] = frames_for(args.seconds)
    wf["6"]["inputs"]["noise_seed"] = args.seed if args.seed >= 0 else int(time.time()) % (2**63)
    wf["9"]["inputs"]["steps"] = steps

    model_src = ["1", 0]
    if args.turbo:
        wf["15"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA,
                               "strength_model": 1.0}}
        model_src = ["15", 0]
    wf["7"]["inputs"]["model"] = model_src
    wf["9"]["inputs"]["model"] = model_src

    for key, img in (("first_frame", args.first), ("last_frame", args.last)):
        if img:
            nid = f"2{len(wf)}"
            wf[nid] = {"class_type": "LoadImage", "inputs": {"image": img}}
            wf["5"]["inputs"][key] = [nid, 0]

    pid = submit(wf)
    print(f"submitted: {pid}  ({args.width}x{args.height}, {args.seconds}s, {steps}步"
          f"{'+turbo' if args.turbo else ''})")
    r = wait(pid)
    if not r:
        sys.exit(1)
    for out in r.get("outputs", {}).values():
        for v in out.get("videos", []) or out.get("images", []):
            sub = v.get("subfolder")
            rel = f"{sub}/{v.get('filename')}" if sub else v.get("filename")
            print(f"output: data/output/{rel}  ({API}/view?filename={v.get('filename')}"
                  f"{'&subfolder=' + sub if sub else ''}&type=output)")


if __name__ == "__main__":
    main()
