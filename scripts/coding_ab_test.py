#!/usr/bin/env python3
"""编程质量对照测试：同一组题测当前服务端，输出供人工对比"""
import json, os, urllib.request

BASE = os.environ.get("AB_BASE", "http://localhost:8000")

TASKS = [
    ("t1_lru", "实现一个 O(1) 的 LRUCache（get/put），要求并发安全，给出完整可运行代码并简要说明设计", 1500),
    ("t2_algo", "LeetCode 困难难度：给定整数数组，找出和为 K 的连续子数组个数。要求 O(n) 解法，写完整 Python 代码并解释正确性", 1500),
    ("t3_edge", "写一个 Python 函数，安全地解析一个可能嵌套任意深度的 JSON 字符串，所有叶子数值统一转为 Decimal（避免浮点误差），循环引用要报错。边界情况考虑周全", 1800),
]

def gen(prompt, max_tokens):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "system", "content": "你是顶级编程专家，直接输出代码和关键说明"},
                          {"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    m = d["choices"][0]["message"]
    return (m.get("content") or ("[空] reasoning=%d字符" % len(m.get("reasoning_content") or ""))).strip(), d["usage"]["completion_tokens"]

import sys
out = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
for tid, prompt, mt in TASKS:
    text, ntok = gen(prompt, mt)
    open(f"{out}/{tid}.md", "w").write(text)
    print(f"{tid}: {ntok} tokens -> {out}/{tid}.md")
