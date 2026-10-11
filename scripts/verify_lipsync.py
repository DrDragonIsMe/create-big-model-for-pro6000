#!/usr/bin/env python3
"""InfiniteTalk 对口型端到端验证：
适配 workflow_api.json → 检查模型文件 → 提交 ComfyUI API → 轮询 → 报告输出视频"""
import json, os, sys, time, urllib.request, wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = f"{ROOT}/data/workflows/workflow_api.json"
COMFY = "http://localhost:8188"
M = f"{ROOT}/models/comfy"

# 模型文件映射（按实际下载到的文件名）
PATCH = {
    "1": {"model_name": "wan_2.1_vae.safetensors"},
    "2": {"model_name": "umt5_xxl_fp16.safetensors"},
    "3": {"clip_name": "clip_vision_h.safetensors"},
    "4": {"model": "wav2vec2-chinese-base_fp32.safetensors", "base_precision": "fp32"},
    "5": {"model": "Wan2_1-InfiniTetalk-Single_fp16.safetensors"},
    "6": {"audio": "1.wav"},
    "7": {"image": "ref_image.png"},
    "17": {"image": "ref_image.png"},
    "11": {"lora": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
    "13": {"model": "wan2.1_i2v_480p_14B_bf16.safetensors", "attention_mode": "sdpa"},
}

SUBDIR = {
    "model_name": ["vae", "text_encoders"],
    "clip_name": ["clip_vision"],
    "model": ["diffusion_models", "wav2vec2"],
    "lora": ["loras"],
}

def find_file(name, keys):
    cands = []
    for k in keys:
        cands += [d for d in SUBDIR.get(k, [])]
    for d in cands + ["", "diffusion_models", "text_encoders", "vae", "clip_vision", "loras", "wav2vec2"]:
        p = os.path.join(M, d, name) if d else os.path.join(M, name)
        if os.path.isfile(p):
            return p
    return None

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(OPENER.open(req, timeout=60))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:800]
        sys.exit(f"HTTP {e.code}: {body}")

def main():
    w = json.load(open(WF))
    # 1) 适配
    for nid, patch in PATCH.items():
        if nid not in w:
            sys.exit(f"工作流缺少节点 {nid}")
        w[nid]["inputs"].update(patch)
    # 2) 文件检查（带重试：下载器可能正在搬运文件）
    missing = []
    for attempt in range(4):
        missing = []
        for nid, patch in PATCH.items():
            for k, v in patch.items():
                if k in ("base_precision", "attention_mode"):
                    continue
                if k in ("audio", "image"):
                    ok = os.path.isfile(f"{ROOT}/data/input/{v}")
                else:
                    ok = find_file(v, [k]) is not None
                if not ok:
                    missing.append(f"节点{nid} {k}={v}")
        if not missing:
            break
        time.sleep(10)
    if missing:
        sys.exit("缺少模型文件:\n  " + "\n  ".join(missing))
    # 3) 音频时长 → 帧数（25fps）；窗口对齐：9 + N*72（81窗口-9运动帧），避免末窗音频饿死
    with wave.open(f"{ROOT}/data/input/1.wav") as f:
        dur = f.getnframes() / f.getframerate()
    nfr = max(250, min(1000, int(dur * 25)))
    nfr = 9 + ((nfr - 9) // 72) * 72
    w["8"]["inputs"]["num_frames"] = nfr
    w["10"]["inputs"]["width"], w["10"]["inputs"]["height"] = 832, 464
    w["16"]["inputs"]["filename_prefix"] = "IT_verify"
    print(f"音频 {dur:.1f}s → {nfr} 帧 @25fps, 分辨率 832x464")
    # 4) 提交
    r = post(f"{COMFY}/prompt", {"prompt": w})
    pid = r["prompt_id"]
    print(f"已提交 prompt_id={pid}")
    # 5) 轮询
    t0 = time.time()
    while time.time() - t0 < 2400:
        h = json.load(OPENER.open(f"{COMFY}/history/{pid}", timeout=30))
        if pid in h:
            outs = h[pid]["outputs"]
            for n, o in outs.items():
                if "gifs" in o or "videos" in o:
                    for v in o.get("gifs", []) + o.get("videos", []):
                        print(f"✓ 输出: {v.get('filename')} (节点{n})")
            print("节点执行:", {n: ("✓" if o else "?") for n, o in outs.items()})
            sys.exit(0)
        time.sleep(20)
    sys.exit("超时：40 分钟未完成")

if __name__ == "__main__":
    main()
