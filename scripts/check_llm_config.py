#!/usr/bin/env python3
"""校准 models.json 的 LLM 配置，以已下载模型文件为准（地面真相）。

检查项：
1. think        — chat_template.jinja 是否含 <think>/enable_thinking 标记。
                  网关据此决定是否加 --reasoning-parser qwen3；
                  非思考模型强开会把正文吞进 reasoning（正文为空的已知事故）。
2. 工具调用格式  — 模板是否输出 <function= XML 格式（gateway 固定 qwen3_xml parser）。
                  非 XML 格式（如 hermes JSON）会报警告，需改 gateway 的 parser。
3. max_ctx      — 与 config.json 的 max_position_embeddings（含 YaRN 换算）对齐。
4. gw.max_len   — 不得超过模型实际上限；gw.extra_args 里配置了 YaRN
                  --hf-overrides 时，上限按扩展后（original × factor）计算。

用法：
  python3 scripts/check_llm_config.py          # 只报告
  python3 scripts/check_llm_config.py --fix    # 自动修正 models.json
  python3 scripts/check_llm_config.py qwen3.8-27b --fix   # 只处理指定模型
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log(*a):
    print(*a, flush=True)

def native_max_ctx(cfg):
    mpe = cfg.get("max_position_embeddings", 0)
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    if rs.get("rope_type") == "yarn" and rs.get("factor"):
        return int(rs.get("original_max_position_embeddings", mpe) * rs["factor"])
    return mpe

def yarn_override_limit(d):
    """gw.extra_args 的 --hf-overrides 里配了 YaRN 时，返回扩展后上限，否则 0。"""
    args = d.get("gw", {}).get("extra_args") or []
    for i, a in enumerate(args):
        if a == "--hf-overrides" and i + 1 < len(args):
            try:
                ov = json.loads(args[i + 1])
            except Exception:
                return 0
            rs = (ov.get("text_config", {}).get("rope_parameters")
                  or ov.get("rope_scaling") or {})
            if rs.get("rope_type") == "yarn" and rs.get("factor"):
                return int(rs.get("original_max_position_embeddings", 0) * rs["factor"])
    return 0

def check_model(name, d, fix, man):
    """返回问题数"""
    problems = 0
    base = os.path.join(ROOT, "models/llm", name)
    if not (os.path.isdir(base) and os.listdir(base)):
        cand = base + "-fp8"
        if os.path.isdir(cand) and os.listdir(cand):
            base = cand
        else:
            log(f"[{name}] 跳过（未下载）")
            return 0

    tpl_path = os.path.join(base, "chat_template.jinja")
    tpl = ""
    if os.path.exists(tpl_path):
        tpl = open(tpl_path, encoding="utf-8").read()
    else:
        tc = os.path.join(base, "tokenizer_config.json")
        if os.path.exists(tc):
            tpl = json.load(open(tc, encoding="utf-8")).get("chat_template") or ""
    if not tpl:
        log(f"[{name}] 警告: 找不到 chat_template，无法检测 think/工具格式")
        problems += 1
    else:
        # --- think ---
        actual_think = ("enable_thinking" in tpl) or ("<think>" in tpl)
        if bool(d.get("think")) != actual_think:
            log(f"[{name}] think 应为 {actual_think}（当前 {bool(d.get('think'))}）"
                + (" → 已修正" if fix else ""))
            if fix:
                d["think"] = actual_think
            problems += 1
        # --- 工具格式 ---
        if "<function=" not in tpl:
            log(f"[{name}] 警告: 模板不是 <function= XML 格式，"
                f"gateway 固定的 qwen3_xml parser 可能不适用，需人工确认")
            problems += 1

    cfg_path = os.path.join(base, "config.json")
    if os.path.exists(cfg_path):
        limit = native_max_ctx(json.load(open(cfg_path, encoding="utf-8")))
        if limit:
            if d.get("max_ctx") != limit:
                log(f"[{name}] max_ctx 应为 {limit}（当前 {d.get('max_ctx')}）"
                    + (" → 已修正" if fix else ""))
                if fix:
                    d["max_ctx"] = limit
                problems += 1
            gw_len = int(d.get("gw", {}).get("max_len", 0))
            eff_limit = max(limit, yarn_override_limit(d))
            if gw_len > eff_limit:
                log(f"[{name}] gw.max_len {gw_len} 超过模型上限 {eff_limit}"
                    + (" → 已修正" if fix else ""))
                if fix:
                    d.setdefault("gw", {})["max_len"] = eff_limit
                problems += 1
    if "gw" not in d:
        log(f"[{name}] 提示: 缺少 gw 段（gpus/util/max_len），网关将用默认值")
    if not problems:
        log(f"[{name}] OK")
    return problems

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    fix = "--fix" in sys.argv
    man = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
    names = args or list(man["llm"].keys())
    total = 0
    for n in names:
        if n not in man["llm"]:
            log(f"[{n}] 不在 models.json 的 llm 清单中")
            total += 1
            continue
        total += check_model(n, man["llm"][n], fix, man)
    if fix and total:
        out = os.path.join(ROOT, "models.json")
        json.dump(man, open(out, "w"), ensure_ascii=False, indent=2)
        log("models.json 已更新")
    log(f"== 共 {total} 处问题" + ("（已修正可修复项）" if fix and total else ""))
    sys.exit(1 if total and not fix else 0)

if __name__ == "__main__":
    main()
