#!/usr/bin/env python3
"""Qwen-Image 文生图 CLI 客户端（走 ComfyUI REST API）

用法:
    python3 scripts/gen_image.py "一只在月球上喝茶的柴犬，电影感"
    python3 scripts/gen_image.py "海报：2026 新年" --width 1080 --height 1920 --steps 25
输出: 打印生成的文件名（ComfyUI output 目录，即本机 data/output/）
"""
import argparse, json, os, sys, time
import urllib.request

# 本机 ComfyUI 直连：系统代理的 no_proxy 用 CIDR 写法，urllib 不认识，会误走代理导致 502
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

API = os.environ.get("COMFY_API", "http://127.0.0.1:8188")
WF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data/workflows/qwen_image_t2i_api.json")


def submit(wf):
    data = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{API}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(OPENER.open(req).read()).get("prompt_id")


def wait(pid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(OPENER.open(f"{API}/history/{pid}").read())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    print("ERROR:", json.dumps(st, ensure_ascii=False)[:500])
                    return None
                if st.get("completed") or h[pid].get("outputs"):
                    return h[pid]
        except Exception:
            pass
        time.sleep(3)
    print("TIMEOUT")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--negative", default="低质量，模糊，变形，文字错误，水印")
    ap.add_argument("--width", type=int, default=1328)
    ap.add_argument("--height", type=int, default=1328)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    wf = json.load(open(WF))
    wf["5"]["inputs"]["text"] = args.prompt
    wf["6"]["inputs"]["text"] = args.negative
    wf["7"]["inputs"]["width"] = args.width
    wf["7"]["inputs"]["height"] = args.height
    wf["8"]["inputs"]["steps"] = args.steps
    wf["8"]["inputs"]["cfg"] = args.cfg
    wf["8"]["inputs"]["seed"] = args.seed if args.seed >= 0 else int(time.time()) % (2**32)

    pid = submit(wf)
    print(f"submitted: {pid}")
    r = wait(pid)
    if not r:
        sys.exit(1)
    for out in r.get("outputs", {}).values():
        for img in out.get("images", []):
            print(f"output: data/output/{img.get('filename')}  "
                  f"({API}/view?filename={img.get('filename')}&type=output)")


if __name__ == "__main__":
    main()
