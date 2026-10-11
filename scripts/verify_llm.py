#!/usr/bin/env python3
"""通过网关对 LLM 做端到端验证（新机器重建后跑一遍）。

每个模型三项检查：
1. 普通问答：content 必须非空（回归"只返回 thinking/正文为空"事故）；
   非思考模型不应返回 reasoning 字段；思考模型思考应被分离到 reasoning 字段。
2. 工具调用：模型应通过 qwen3_xml parser 产出结构化 tool_calls。
3. max_tokens 钳制：max_tokens 拉满上下文上限也必须放行（回归"67K/256K
   就被误触发 compact"事故——vLLM 对 prompt+max_tokens 超限一律 400）。

用法：
  python3 scripts/verify_llm.py                # 验证所有已下载 LLM
  python3 scripts/verify_llm.py qwen3.8-27b    # 只验证指定模型
  环境变量 GW 可覆盖网关地址（默认 http://localhost:8000）

注意：思考模型默认思考开销大，max_tokens 给足（2048），否则思考被截断
会得到 finish_reason=length + 空正文，那是预算问题而非配置错误。
"""
import json, os, sys, urllib.request

GW = os.environ.get("GW", "http://localhost:8000")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def post(path, payload, timeout=1800):
    req = urllib.request.Request(GW + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())

def reasoning_of(m):
    return (m.get("reasoning_content") or m.get("reasoning") or "").strip()

def check(name, d):
    think = bool(d.get("think"))
    gw = d.get("gw", {})
    max_len = int(gw.get("max_len", d.get("max_ctx", d.get("ctx_default", 32768))))
    ok = True
    r = post("/v1/chat/completions", {
        "model": name,
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 2048 if think else 800})
    c = r["choices"][0]
    m, fr = c["message"], c.get("finish_reason")
    content = (m.get("content") or "").strip()
    reasoning = reasoning_of(m)
    if not content:
        print(f"[{name}] 普通问答: 正文为空!! finish={fr} "
              f"reasoning={'有' if reasoning else '无'}", flush=True)
        print(f"  响应: {json.dumps(m, ensure_ascii=False)[:400]}", flush=True)
        ok = False
    else:
        print(f"[{name}] 普通问答: OK (finish={fr}, "
              f"reasoning={'有' if reasoning else '无'})", flush=True)
    if not think and reasoning:
        print(f"[{name}] 警告: 非思考模型返回了 reasoning，"
              f"检查 models.json 的 think 与网关 --reasoning-parser", flush=True)
        ok = False
    if think and fr == "stop" and not reasoning:
        print(f"[{name}] 提示: 思考模型本次未产出 reasoning（可能思考极短），不判失败", flush=True)

    r = post("/v1/chat/completions", {
        "model": name,
        "messages": [{"role": "user", "content": "北京现在天气怎么样？请用工具查询"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}],
        "max_tokens": 2048 if think else 800})
    m = r["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    parsed = False
    if tc:
        try:
            args = json.loads(tc[0]["function"]["arguments"])
            parsed = tc[0]["function"]["name"] == "get_weather" and "city" in args
        except Exception:
            pass
    if parsed:
        print(f"[{name}] 工具调用: OK", flush=True)
    else:
        print(f"[{name}] 工具调用: 未正确解析!! "
              f"raw={json.dumps(tc or m.get('content'), ensure_ascii=False)[:300]}", flush=True)
        ok = False

    # max_tokens 钳制回归：max_tokens 拉满上下文上限，网关应钳制放行而不是 400
    # （vLLM 对 prompt+max_tokens 超限一律 400，客户端会误判为上下文溢出提前 compact）
    try:
        r = post("/v1/chat/completions", {
            "model": name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_len})
        if "error" in r:
            print(f"[{name}] max_tokens 钳制: 仍被拒!! {str(r['error'])[:200]}", flush=True)
            ok = False
        else:
            print(f"[{name}] max_tokens 钳制: OK", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[{name}] max_tokens 钳制: HTTP {e.code}!! {e.read()[:200]}", flush=True)
        ok = False
    return ok

def main():
    man = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))["llm"]
    names = sys.argv[1:] or [
        n for n in man
        if os.path.isdir(os.path.join(ROOT, "models/llm", n))
        or os.path.isdir(os.path.join(ROOT, "models/llm", n + "-fp8"))]
    fails = []
    for n in names:
        if n not in man:
            print(f"[{n}] 不在 models.json 的 llm 清单中", flush=True)
            fails.append(n)
            continue
        try:
            if not check(n, man[n]):
                fails.append(n)
        except Exception as e:
            print(f"[{n}] 请求异常: {e}", flush=True)
            fails.append(n)
    print("== 总结:", "全部通过" if not fails else f"失败: {fails}", flush=True)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
