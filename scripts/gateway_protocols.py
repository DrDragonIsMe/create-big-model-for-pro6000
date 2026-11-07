import json
import time
import uuid


class ProtocolError(ValueError):
    pass


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _text_parts(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return [str(content)]
    out = []
    for block in content:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict) and block.get("text") is not None:
            out.append(str(block["text"]))
    return out


def _image_url(block):
    value = block.get("image_url", block.get("image"))
    if isinstance(value, dict):
        value = value.get("url")
    if not value and block.get("source"):
        source = block["source"]
        if source.get("type") == "base64":
            value = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
        elif source.get("type") == "url":
            value = source.get("url")
    if not value:
        raise ProtocolError("图片块缺少 image_url 或 source")
    result = {"url": value}
    if block.get("detail"):
        result["detail"] = block["detail"]
    return {"type": "image_url", "image_url": result}


def _openai_content(content, role="user"):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    blocks = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            raise ProtocolError("content 块必须是对象")
        kind = block.get("type", "text")
        if kind in ("text", "input_text", "output_text"):
            blocks.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind in ("input_image", "image_url", "image"):
            blocks.append(_image_url(block))
        elif kind in ("thinking", "redacted_thinking", "reasoning"):
            continue
        elif kind in ("input_file", "document", "file"):
            raise ProtocolError("当前本地后端不支持文件/PDF输入，请先提取文本后发送")
        elif kind in ("input_audio", "audio"):
            raise ProtocolError("当前模型接口不支持音频输入")
        else:
            raise ProtocolError(f"不支持的 content 类型: {kind}")
    if not blocks:
        return ""
    if all(block["type"] == "text" for block in blocks):
        return "\n".join(block["text"] for block in blocks)
    if role == "assistant" and any(block["type"] != "text" for block in blocks):
        raise ProtocolError("assistant 历史消息不能包含图片块")
    return blocks


def _responses_tools(tools):
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise ProtocolError("tools 元素必须是对象")
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = dict(tool["function"])
        elif tool.get("type") in (None, "function", "custom") or tool.get("name"):
            fn = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", tool.get("input_schema")) or {"type": "object"},
            }
        else:
            raise ProtocolError(f"本地网关不支持托管工具: {tool.get('type')}")
        if not fn.get("name"):
            raise ProtocolError("function 工具缺少 name")
        fn.setdefault("parameters", {"type": "object"})
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice(choice, anthropic=False):
    if choice is None:
        return None
    if isinstance(choice, str):
        values = {"any": "required", "auto": "auto", "required": "required", "none": "none"}
        if choice not in values:
            raise ProtocolError(f"不支持的 tool_choice: {choice}")
        return values[choice]
    if not isinstance(choice, dict):
        raise ProtocolError("tool_choice 必须是字符串或对象")
    kind = choice.get("type")
    if kind in ("auto", "required", "none", "any"):
        return _tool_choice(kind)
    if kind in ("function", "tool"):
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        if not name:
            raise ProtocolError("指定 tool_choice 时缺少工具名")
        return {"type": "function", "function": {"name": name}}
    if kind == "allowed_tools":
        return "required"
    raise ProtocolError(f"不支持的 tool_choice 类型: {kind}")


def _apply_common_chat_options(body, out, responses=False):
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"),
                           ("top_k", "top_k"), ("parallel_tool_calls", "parallel_tool_calls")):
        if body.get(source) is not None:
            out[target] = body[source]
    if responses:
        if body.get("max_output_tokens") is not None:
            out["max_tokens"] = body["max_output_tokens"]
        elif body.get("max_tokens") is not None:
            out["max_tokens"] = body["max_tokens"]
        tools = _responses_tools(body.get("tools"))
        choice = _tool_choice(body.get("tool_choice"))
        text_format = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
        schema = body.get("output_schema")
        if text_format:
            kind = text_format.get("type", "text")
            if kind == "json_schema":
                schema = text_format
            elif kind == "json_object":
                out["response_format"] = {"type": "json_object"}
            elif kind != "text":
                raise ProtocolError(f"不支持的 text.format: {kind}")
        if schema:
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("name", "structured_output"),
                    "schema": schema.get("schema", schema.get("json_schema", schema)),
                    "strict": schema.get("strict", False),
                },
            }
        reasoning = body.get("reasoning") or {}
        if reasoning.get("effort"):
            out["reasoning_effort"] = reasoning["effort"]
            out["chat_template_kwargs"] = {"enable_thinking": reasoning["effort"] != "none"}
    else:
        out["max_tokens"] = body.get("max_tokens", 4096)
        if body.get("stop_sequences") is not None:
            out["stop"] = body["stop_sequences"]
        tools = []
        for tool in body.get("tools") or []:
            if not tool.get("name"):
                raise ProtocolError("Anthropic 工具缺少 name")
            tools.append({"type": "function", "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object"},
            }})
        choice = _tool_choice(body.get("tool_choice"), anthropic=True)
        thinking = body.get("thinking") or {}
        if thinking.get("type") == "enabled":
            out["chat_template_kwargs"] = {"enable_thinking": True}
            budget = thinking.get("budget_tokens")
            if budget:
                out["reasoning_effort"] = "high" if budget >= 12000 else "medium" if budget >= 4000 else "low"
        elif thinking.get("type") == "disabled":
            out["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        out["tools"] = tools
    if choice is not None:
        out["tool_choice"] = choice


def _merge_leading_system(messages):
    # 后端 chat 模板只接受一条置顶 system（Codex 会把基础指令和权限说明发成
    # 两条 developer 消息，合并成一条才不会 400 "System message must be at the beginning."）
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        i += 1
    if i < 2:
        return messages
    merged = "\n\n".join(text for msg in messages[:i] for text in _text_parts(msg.get("content")))
    return [{"role": "system", "content": merged}, *messages[i:]]


def responses_to_chat(body):
    if body.get("previous_response_id"):
        raise ProtocolError("previous_response_id 需要服务端会话存储；请改为发送完整 input 历史")
    inp = body.get("input")
    if isinstance(inp, str):
        items = [{"type": "message", "role": "user", "content": inp}]
    elif isinstance(inp, list):
        items = inp
    else:
        raise ProtocolError("input 必须是字符串或数组")
    messages = []
    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ProtocolError("input 元素必须是对象")
        kind = item.get("type", "message" if item.get("role") else "")
        if kind in ("message", "input_message"):
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                raise ProtocolError(f"不支持的消息角色: {role}")
            messages.append({"role": role, "content": _openai_content(item.get("content", ""), role)})
        elif kind in ("function_call", "tool_use"):
            call_id = item.get("call_id", item.get("id") or _id("call"))
            arguments = item.get("arguments")
            if arguments is None:
                arguments = json.dumps(item.get("input", {}), ensure_ascii=False)
            call = {"id": call_id, "type": "function", "function": {
                "name": item.get("name", ""), "arguments": arguments}}
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind in ("function_call_output", "tool_result"):
            output = item.get("output", item.get("content", ""))
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", item.get("tool_use_id", "")),
                             "content": _openai_content(output)})
        elif kind == "reasoning":
            continue
        else:
            raise ProtocolError(f"不支持的 input 类型: {kind or '<missing>'}")
    instructions = body.get("instructions")
    if instructions:
        messages.insert(0, {"role": "system", "content": _openai_content(instructions)})
    messages = _merge_leading_system(messages)
    if not messages:
        raise ProtocolError("input 不能为空")
    out = {"model": body.get("model", ""), "messages": messages,
           "stream": bool(body.get("stream", False))}
    _apply_common_chat_options(body, out, responses=True)
    return out


def anthropic_to_chat(body):
    source = body.get("messages")
    if not isinstance(source, list) or not source:
        raise ProtocolError("messages 必须是非空数组")
    messages = []
    inline_system = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": "\n".join(_text_parts(system))})
    for msg in source:
        if not isinstance(msg, dict):
            raise ProtocolError("messages 元素必须是对象")
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            blocks = content if isinstance(content, list) else []
            calls = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls.append({"id": block.get("id") or _id("call"), "type": "function", "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}})
            text = "\n".join(_text_parts(content))
            message = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        elif role == "user":
            if not isinstance(content, list):
                messages.append({"role": "user", "content": content})
                continue
            pending = []
            def flush_pending():
                if pending:
                    messages.append({"role": "user", "content": _openai_content(pending)})
                    pending.clear()
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    flush_pending()
                    messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""),
                                     "content": _openai_content(block.get("content", ""))})
                else:
                    pending.append(block)
            flush_pending()
        elif role == "system":
            # Claude Code 会在 messages 里内联 system 条目；后端只认置顶那一条，
            # 所以提取文本并入前导 system，而不是当不支持的角色拒掉。
            inline_system.extend(_text_parts(content))
        else:
            raise ProtocolError(f"Anthropic 不支持角色: {role}")
    if inline_system:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = "\n\n".join([messages[0]["content"], *inline_system])
        else:
            messages.insert(0, {"role": "system", "content": "\n\n".join(inline_system)})
    messages = _merge_leading_system(messages)
    out = {"model": body.get("model", ""), "messages": messages,
           "stream": bool(body.get("stream", False))}
    _apply_common_chat_options(body, out)
    return out


def _split_prompt_tokens(usage):
    """vLLM usage → Anthropic 语义：prompt 全量拆成 未命中缓存的 input + cache_read。

    cached_tokens 仅在命中自动前缀缓存时出现；cache_creation vLLM 不报，恒 0。
    Claude Code 的上下文占用 = input + cache_read + cache_creation = prompt_tokens。
    """
    prompt = usage.get("prompt_tokens", 0) or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    cached = min(cached, prompt)
    return prompt - cached, cached


def _usage(usage, anthropic=False):
    usage = usage or {}
    if anthropic:
        input_tokens, cache_read = _split_prompt_tokens(usage)
        return {"input_tokens": input_tokens,
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cache_read}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
        "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
        "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
    }


def _response_item_ids(response_id):
    suffix = response_id.removeprefix("resp_")
    return f"msg_{suffix}", f"rs_{suffix}"


def chat_to_response(data, response_id=None):
    if not data.get("choices"):
        raise ProtocolError("后端响应缺少 choices")
    response_id = response_id or _id("resp")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    message_id, reasoning_id = _response_item_ids(response_id)
    output = []
    if msg.get("reasoning_content"):
        output.append({"id": reasoning_id, "type": "reasoning", "summary": [{
            "type": "summary_text", "text": msg["reasoning_content"]}]})
    if msg.get("content") or not msg.get("tool_calls"):
        output.append({"id": message_id, "type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": msg.get("content") or "", "annotations": []}]})
    for index, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function") or {}
        call_id = call.get("id") or f"call_{index}"
        output.append({"id": f"fc_{call_id}", "type": "function_call", "status": "completed",
                       "call_id": call_id, "name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
    finish = choice.get("finish_reason") or "stop"
    incomplete = finish == "length"
    return {
        "id": response_id, "object": "response", "created_at": data.get("created", int(time.time())),
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "model": data.get("model", ""), "output": output, "error": None,
        "parallel_tool_calls": True, "tools": [], "tool_choice": "auto",
        "usage": _usage(data.get("usage")),
    }


def chat_to_anthropic(data):
    if not data.get("choices"):
        raise ProtocolError("后端响应缺少 choices")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    content = []
    if msg.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": msg["reasoning_content"], "signature": ""})
    if msg.get("content") or not msg.get("tool_calls"):
        content.append({"type": "text", "text": msg.get("content") or ""})
    for index, call in enumerate(msg.get("tool_calls") or []):
        fn = call.get("function") or {}
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            arguments = {"_raw": fn.get("arguments", "")}
        content.append({"type": "tool_use", "id": call.get("id") or f"call_{index}",
                        "name": fn.get("name", ""), "input": arguments})
    if not content:
        content.append({"type": "text", "text": ""})
    stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(
        choice.get("finish_reason") or "stop", "end_turn")
    return {"id": _id("msg"), "type": "message", "role": "assistant", "content": content,
            "model": data.get("model", ""), "stop_reason": stop, "stop_sequence": None,
            "usage": _usage(data.get("usage"), anthropic=True)}


async def chat_events(lines):
    data_lines = []
    async for line in lines:
        if line == "":
            if data_lines:
                raw = "\n".join(data_lines)
                data_lines.clear()
                if raw == "[DONE]":
                    return
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines and data_lines != ["[DONE]"]:
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return


def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_responses(lines, model_name, response_id=None):
    response_id = response_id or _id("resp")
    created = int(time.time())
    message_id, reasoning_id = _response_item_ids(response_id)
    base = {"id": response_id, "object": "response", "created_at": created,
            "status": "in_progress", "model": model_name, "output": [], "error": None}
    yield _sse("response.created", {"type": "response.created", "response": base})
    text, reasoning = [], []
    tools = {}
    usage = {}
    finish = None
    current = None
    next_output_index = 0

    def close_current():
        nonlocal current
        if current is None:
            return []
        kind, index = current["kind"], current["output_index"]
        events = []
        if kind == "text":
            value = "".join(text)
            events.append(_sse("response.output_text.done", {"type": "response.output_text.done",
                "response_id": response_id, "item_id": message_id, "output_index": index,
                "content_index": 0, "text": value}))
            events.append(_sse("response.content_part.done", {"type": "response.content_part.done",
                "response_id": response_id, "item_id": message_id, "output_index": index, "content_index": 0,
                "part": {"type": "output_text", "text": value, "annotations": []}}))
            events.append(_sse("response.output_item.done", {"type": "response.output_item.done",
                "response_id": response_id, "output_index": index,
                "item": {"id": message_id, "type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": value, "annotations": []}]}}))
        elif kind == "reasoning":
            value = "".join(reasoning)
            events.append(_sse("response.reasoning_summary_text.done", {"type": "response.reasoning_summary_text.done",
                "response_id": response_id, "item_id": reasoning_id, "output_index": index,
                "summary_index": 0, "text": value}))
            events.append(_sse("response.reasoning_summary_part.done", {"type": "response.reasoning_summary_part.done",
                "response_id": response_id, "item_id": reasoning_id, "output_index": index, "summary_index": 0,
                "part": {"type": "summary_text", "text": value}}))
            events.append(_sse("response.output_item.done", {"type": "response.output_item.done",
                "response_id": response_id, "output_index": index,
                "item": {"id": reasoning_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": value}]}}))
        current = None
        return events

    def start(kind):
        nonlocal current, next_output_index
        events = close_current()
        index = next_output_index
        next_output_index += 1
        current = {"kind": kind, "output_index": index}
        if kind == "text":
            item = {"id": message_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
            part = {"type": "output_text", "text": "", "annotations": []}
            events.append(_sse("response.output_item.added", {"type": "response.output_item.added",
                "response_id": response_id, "output_index": index, "item": item}))
            events.append(_sse("response.content_part.added", {"type": "response.content_part.added",
                "response_id": response_id, "item_id": message_id, "output_index": index,
                "content_index": 0, "part": part}))
        else:
            item = {"id": reasoning_id, "type": "reasoning", "summary": []}
            events.append(_sse("response.output_item.added", {"type": "response.output_item.added",
                "response_id": response_id, "output_index": index, "item": item}))
            events.append(_sse("response.reasoning_summary_part.added", {"type": "response.reasoning_summary_part.added",
                "response_id": response_id, "item_id": reasoning_id, "output_index": index, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}}))
        return events

    async for event in chat_events(lines):
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason") or finish
            value = delta.get("reasoning_content")
            if value:
                if not current or current["kind"] != "reasoning":
                    for output in start("reasoning"):
                        yield output
                reasoning.append(value)
                yield _sse("response.reasoning_summary_text.delta", {"type": "response.reasoning_summary_text.delta",
                    "response_id": response_id, "item_id": reasoning_id, "output_index": current["output_index"],
                    "summary_index": 0, "delta": value})
            value = delta.get("content")
            if value:
                if not current or current["kind"] != "text":
                    for output in start("text"):
                        yield output
                text.append(value)
                yield _sse("response.output_text.delta", {"type": "response.output_text.delta",
                    "response_id": response_id, "item_id": message_id, "output_index": current["output_index"],
                    "content_index": 0, "delta": value})
            for call in delta.get("tool_calls") or []:
                tool_index = call.get("index", 0)
                if tool_index not in tools:
                    for output in close_current():
                        yield output
                    output_index = next_output_index
                    next_output_index += 1
                    call_id = call.get("id") or f"call_{tool_index}"
                    state = {"output_index": output_index, "call_id": call_id,
                             "name": "", "arguments": []}
                    tools[tool_index] = state
                    item = {"id": f"fc_{call_id}", "type": "function_call", "status": "in_progress",
                            "call_id": call_id, "name": "", "arguments": ""}
                    yield _sse("response.output_item.added", {"type": "response.output_item.added",
                        "response_id": response_id, "output_index": output_index, "item": item})
                state = tools[tool_index]
                if call.get("id"):
                    state["call_id"] = call["id"]
                fn = call.get("function") or {}
                state["name"] += fn.get("name") or ""
                args = fn.get("arguments") or ""
                if args:
                    state["arguments"].append(args)
                    yield _sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta", "response_id": response_id,
                        "item_id": f"fc_{state['call_id']}", "output_index": state["output_index"],
                        "delta": args})
    for output in close_current():
        yield output
    for state in sorted(tools.values(), key=lambda item: item["output_index"]):
        arguments = "".join(state["arguments"])
        yield _sse("response.function_call_arguments.done", {"type": "response.function_call_arguments.done",
            "response_id": response_id, "item_id": f"fc_{state['call_id']}", "output_index": state["output_index"],
            "arguments": arguments})
        yield _sse("response.output_item.done", {"type": "response.output_item.done",
            "response_id": response_id, "output_index": state["output_index"],
            "item": {"id": f"fc_{state['call_id']}", "type": "function_call", "status": "completed",
                     "call_id": state["call_id"], "name": state["name"], "arguments": arguments}})
    data = {"id": response_id, "created": created, "model": model_name, "choices": [{"message": {
        "content": "".join(text), "reasoning_content": "".join(reasoning),
        "tool_calls": [{"id": state["call_id"], "type": "function", "function": {
            "name": state["name"], "arguments": "".join(state["arguments"])}}
            for state in sorted(tools.values(), key=lambda item: item["output_index"])]},
        "finish_reason": finish or ("tool_calls" if tools else "stop")}], "usage": usage}
    response = chat_to_response(data, response_id=response_id)
    event_type = "response.incomplete" if response["status"] == "incomplete" else "response.completed"
    yield _sse(event_type, {"type": event_type, "response": response})


async def stream_anthropic(lines, model_name):
    message = {"id": _id("msg"), "type": "message", "role": "assistant", "content": [],
               "model": model_name, "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 0, "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": message})
    current = None
    block_count = 0
    tools = {}
    finish = None
    usage = {}

    def close_current():
        nonlocal current
        if current is None:
            return None
        index = current["index"]
        current = None
        return _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    def start(kind, call=None):
        nonlocal current, block_count
        previous = close_current()
        index = block_count
        block_count += 1
        if kind == "text":
            block = {"type": "text", "text": ""}
        elif kind == "thinking":
            block = {"type": "thinking", "thinking": "", "signature": ""}
        else:
            block = {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}}
        current = {"kind": kind, "index": index}
        event = _sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})
        return previous, event

    async for event in chat_events(lines):
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason") or finish
            for kind, field, delta_type in (("thinking", "reasoning_content", "thinking_delta"),
                                            ("text", "content", "text_delta")):
                value = delta.get(field)
                if not value:
                    continue
                if not current or current["kind"] != kind:
                    previous, started = start(kind)
                    if previous:
                        yield previous
                    yield started
                yield _sse("content_block_delta", {"type": "content_block_delta", "index": current["index"],
                    "delta": {"type": delta_type, "thinking" if kind == "thinking" else "text": value}})
            for call in delta.get("tool_calls") or []:
                tool_index = call.get("index", 0)
                fn = call.get("function") or {}
                if tool_index not in tools:
                    call_id = call.get("id") or f"call_{tool_index}"
                    previous, started = start("tool_use", {"id": call_id, "name": fn.get("name", "")})
                    if previous:
                        yield previous
                    yield started
                    tools[tool_index] = {"index": current["index"]}
                state = tools[tool_index]
                args = fn.get("arguments") or ""
                if args:
                    yield _sse("content_block_delta", {"type": "content_block_delta", "index": state["index"],
                        "delta": {"type": "input_json_delta", "partial_json": args}})
    stopped = close_current()
    if stopped:
        yield stopped
    stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(
        finish or ("tool_calls" if tools else "stop"), "end_turn")
    input_tokens, cache_read = _split_prompt_tokens(usage or {})
    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                 "usage": {"input_tokens": input_tokens,
                                           "cache_read_input_tokens": cache_read,
                                           "output_tokens": (usage or {}).get("completion_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})
