#!/usr/bin/env python3
"""在 python:3.12-slim 容器内运行。按任务清单下载模型并归置目录。
任务 JSON: {"tasks":[{"mode":"llm|comfy","name":..., "repo":..., "fallback_repo":...,
           "includes":[...], "dest":...}]}
挂载: -v $PWD/models:/models -v <job.json>:/job.json:ro
环境: HF_ENDPOINT(镜像站), HF_TOKEN(可选), HF_HUB_ENABLE_HF_TRANSFER=1
"""
import fnmatch, json, os, shutil, sys

MODELS = "/models"

def log(*a):
    print("[fetch]", *a, flush=True)

def snapshot(repo, patterns, local_dir):
    from huggingface_hub import snapshot_download
    kwargs = dict(repo_id=repo, local_dir=local_dir, max_workers=8)
    if patterns:
        kwargs["allow_patterns"] = patterns
    token = os.environ.get("HF_TOKEN") or None
    if token:
        kwargs["token"] = token
    return snapshot_download(**kwargs)

def fetch_comfy(t):
    repo = t["repo"]
    try:
        tmp = os.path.join(MODELS, "_tmp", repo.replace("/", "_"))
        snapshot(repo, t.get("includes"), tmp)
    except Exception as e:
        fb = t.get("fallback_repo")
        if not fb:
            raise
        log(f"主仓库 {repo} 不可用({e})，回退 {fb}")
        repo = fb
        tmp = os.path.join(MODELS, "_tmp", repo.replace("/", "_"))
        snapshot(repo, t.get("includes"), tmp)
    dest = os.path.join(MODELS, "comfy", t["dest"])
    os.makedirs(dest, exist_ok=True)
    patterns = t.get("includes") or ["*"]
    moved = 0
    for root, _, files in os.walk(tmp):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), tmp)
            if rel.startswith(".cache") or f.endswith((".incomplete", ".metadata")):
                continue
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(f, p) for p in patterns):
                target = os.path.join(dest, f)
                if not os.path.exists(target):
                    shutil.move(os.path.join(root, f), target)
                moved += 1
    log(f"{t['name']}: {moved} 个文件 -> comfy/{t['dest']}")
    shutil.rmtree(os.path.join(MODELS, "_tmp", repo.replace("/", "_")), ignore_errors=True)

def fetch_llm(t):
    dest = os.path.join(MODELS, "llm", t["name"])
    os.makedirs(dest, exist_ok=True)
    snapshot(t["repo"], t.get("includes"), dest)
    log(f"{t['name']} -> llm/{t['name']}")

def main():
    job = json.load(open(sys.argv[1]))
    for t in job["tasks"]:
        for attempt in range(1, 1000):
            try:
                log(f"开始: {t['name']}  ({t['repo']})  第{attempt}次")
                (fetch_llm if t["mode"] == "llm" else fetch_comfy)(t)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (401, 403, 404):
                    log(f"致命错误(HTTP {code})，重试无意义，直接退出: {e}")
                    sys.exit(1)
                log(f"失败(第{attempt}次): {e}")
                import time
                time.sleep(min(60, 10 * attempt))
        else:
            sys.exit(1)
    shutil.rmtree(os.path.join(MODELS, "_tmp"), ignore_errors=True)
    log("全部完成")

if __name__ == "__main__":
    main()
