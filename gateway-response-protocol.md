# AI Gateway 模型与响应协议说明

本文供客户端升级和协议适配使用。网关地址默认为 `http://<服务器>:8000`。

## 1. 模型发现：`GET /v1/models`

返回当前已下载、可调度的模型。未加载的模型也会返回，客户端不需要先判断 `loaded` 再决定是否调用。

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.8-flash-next",
      "object": "model",
      "created": 0,
      "owned_by": "ai-gateway",
      "loaded": true,
      "context_length": 262144,
      "native_context_length": 262144,
      "capabilities": {
        "input_modalities": ["text", "image", "video"],
        "output_modalities": ["text"],
        "streaming": true,
        "tool_calls": true,
        "structured_outputs": true,
        "reasoning": true,
        "vision": true,
        "video_input": true,
        "api_endpoints": [
          "/v1/chat/completions",
          "/v1/responses",
          "/v1/messages"
        ]
      }
    }
  ]
}
```

### 字段语义

| 字段 | 类型 | 客户端用法 |
|---|---|---|
| `id` | string | 所有推理请求的 `model` 值 |
| `loaded` | boolean | 是否已确认 `ready`；`false` 不代表不可用，首次请求会自动加载 |
| `context_length` | integer | 当前实际生效窗口；计算 prompt、输出预算和 compact 阈值时使用此值 |
| `native_context_length` | integer | 模型原生窗口，仅用于展示或判断 YaRN 扩展，不用于当前请求预算 |
| `capabilities.input_modalities` | string[] | 可包含 `text`、`image`、`video` |
| `capabilities.output_modalities` | string[] | 当前固定为 `["text"]` |
| `capabilities.streaming` | boolean | 是否支持 SSE |
| `capabilities.tool_calls` | boolean | 是否支持函数工具调用 |
| `capabilities.structured_outputs` | boolean | 是否可请求 JSON/JSON Schema 输出 |
| `capabilities.reasoning` | boolean | 是否为推理模型 |
| `capabilities.vision` | boolean | 是否支持图片输入 |
| `capabilities.video_input` | boolean | 是否支持视频输入 |
| `capabilities.api_endpoints` | string[] | 该模型可使用的推理端点 |

客户端应对未知字段保持向前兼容，不要因为出现新增 capability 而拒绝解析。

## 2. OpenAI Responses：`POST /v1/responses`

### 2.1 简单文本请求

```json
{
  "model": "qwen3.8-flash-next",
  "input": "只回复 OK",
  "max_output_tokens": 128,
  "stream": false
}
```

也支持指令、消息数组和图片输入：

```json
{
  "model": "qwen3.8-flash-next",
  "instructions": "你是编程助手，回答保持简洁",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "描述这张图片"
        },
        {
          "type": "input_image",
          "image_url": "data:image/png;base64,..."
        }
      ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_output_tokens": 1024
}
```

`role` 支持 `system`、`developer`、`user`、`assistant`；`developer` 会转换为系统指令。

后端模型的 chat template 只接受一条置顶系统指令，因此开头连续的 `system`/`developer` 消息（连同 `instructions` 字段带来的那条）会按原顺序合并成一条。OpenAI Codex CLI 每个请求都发两条前置 `developer` 消息（基础指令 + 权限说明），不合并会被后端以 `System message must be at the beginning.` 拒绝。

### 2.2 工具调用请求

```json
{
  "model": "qwen3.8-flash-next",
  "input": "北京天气如何？",
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string"
          }
        },
        "required": ["city"]
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "max_output_tokens": 512
}
```

`tool_choice` 支持以下形式：

```json
"auto"
```

```json
"required"
```

```json
"none"
```

```json
{
  "type": "function",
  "name": "get_weather"
}
```

### 2.3 工具结果回传

```json
{
  "model": "qwen3.8-flash-next",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "北京天气如何？"
    },
    {
      "type": "function_call",
      "call_id": "call_001",
      "name": "get_weather",
      "arguments": "{\"city\":\"北京\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_001",
      "output": "{\"temperature\":18,\"condition\":\"晴\"}"
    }
  ]
}
```

### 2.4 结构化输出

推荐使用 OpenAI Responses 的 `text.format`：

```json
{
  "model": "qwen3.8-flash-next",
  "input": "提取信息：张三，30岁",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  }
}
```

也支持 JSON 对象模式：

```json
{
  "text": {
    "format": {
      "type": "json_object"
    }
  }
}
```

### 2.5 Responses 非流式返回

```json
{
  "id": "resp_xxx",
  "object": "response",
  "created_at": 1794018018,
  "status": "completed",
  "incomplete_details": null,
  "model": "qwen3.8-flash-next",
  "output": [
    {
      "id": "rs_xxx",
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "推理内容"
        }
      ]
    },
    {
      "id": "msg_xxx",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "OK",
          "annotations": []
        }
      ]
    }
  ],
  "error": null,
  "usage": {
    "input_tokens": 55,
    "output_tokens": 26,
    "total_tokens": 81,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens_details": {
      "reasoning_tokens": 22
    }
  }
}
```

工具调用时，`output` 中包含：

```json
{
  "id": "fc_call_001",
  "type": "function_call",
  "status": "completed",
  "call_id": "call_001",
  "name": "get_weather",
  "arguments": "{\"city\":\"北京\"}"
}
```

状态约定：

- `completed`：正常完成，包括需要执行工具的情况。
- `incomplete`：输出被长度限制截断。
- `incomplete_details.reason`：当前为 `max_output_tokens`。

客户端应从 `output` 中按 `type` 读取，不要假设第一项一定是文本，也不要把工具调用状态写成 `tool_calls`。

## 3. Anthropic Messages：`POST /v1/messages`

### 3.1 请求

```json
{
  "model": "qwen3.8-flash-next",
  "max_tokens": 1024,
  "system": "你是编程助手",
  "messages": [
    {
      "role": "user",
      "content": "只回复 OK"
    }
  ],
  "stream": false
}
```

`system` 支持字符串或文本块数组（`cache_control` 会被忽略）。Claude Code 还会在 `messages` 里内联 `role: "system"` 条目（会话提醒等），这类条目会提取文本并入置顶系统指令，而不是按不支持的角色返回 400 `Anthropic 不支持角色: system`。

### 3.2 返回

```json
{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "model": "qwen3.8-flash-next",
  "content": [
    {
      "type": "thinking",
      "thinking": "推理内容",
      "signature": ""
    },
    {
      "type": "text",
      "text": "OK"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 55,
    "output_tokens": 30
  }
}
```

工具调用内容块：

```json
{
  "type": "tool_use",
  "id": "call_001",
  "name": "get_weather",
  "input": {
    "city": "北京"
  }
}
```

工具结果请求块：

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_001",
      "content": "{\"temperature\":18}"
    }
  ]
}
```

`stop_reason` 映射：

| 含义 | 值 |
|---|---|
| 正常结束 | `end_turn` |
| 调用工具 | `tool_use` |
| 达到输出上限 | `max_tokens` |

### 3.3 Token 估算

```http
POST /v1/messages/count_tokens
```

返回：

```json
{
  "input_tokens": 123
}
```

该接口是本地估算值，不应当作后端 tokenizer 的精确结果。

## 4. Anthropic 风格 Responses

端点仍然是：

```http
POST /v1/responses
```

可通过以下任一方式选择：

```http
anthropic-version: 2023-06-01
```

```http
x-api-format: anthropic
```

```text
/v1/responses?format=anthropic
```

也支持请求体特征：

```json
{
  "anthropic_version": "2023-06-01",
  "model": "qwen3.8-flash-next",
  "input": "你好"
}
```

或者工具使用 `input_schema`、输入项使用 `input_message`。

注意：该入口使用 Anthropic 风格请求识别和错误结构，但成功响应仍然是 `object: "response"` 的 Responses 结构，不是 `/v1/messages` 的 `type: "message"` 结构。

## 5. SSE 流式读取

请求增加：

```json
{
  "stream": true
}
```

每个事件同时包含 `event` 和 `data`：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"你","output_index":0}
```

### 5.1 OpenAI Responses 主要事件

推荐按 `event` 或 `data.type` 分发：

- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.reasoning_summary_text.delta`
- `response.output_text.delta`
- `response.function_call_arguments.delta`
- `response.output_text.done`
- `response.function_call_arguments.done`
- `response.output_item.done`
- `response.completed`
- `response.incomplete`

结束依据为 `response.completed` 或 `response.incomplete`。Responses 流不依赖 `data: [DONE]`。

工具参数需要按 `output_index` 累积 `response.function_call_arguments.delta`，直到收到 `response.function_call_arguments.done`，再对完整 `arguments` 执行 `JSON.parse`。

### 5.2 Anthropic Messages 主要事件

- `message_start`
- `content_block_start`
- `content_block_delta`
- `content_block_stop`
- `message_delta`
- `message_stop`

`content_block_delta.delta.type` 包括：

- `text_delta`
- `thinking_delta`
- `input_json_delta`

结束依据是 `message_stop`。最终输出 token 数从 `message_delta.usage.output_tokens` 读取。

## 6. 错误格式

OpenAI 风格：

```json
{
  "error": {
    "message": "错误说明",
    "type": "invalid_request_error"
  }
}
```

Anthropic 风格：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "错误说明"
  }
}
```

常见 HTTP 状态：

| 状态 | 含义 |
|---|---|
| 400 | 请求格式、字段或输入类型不支持 |
| 404 | 模型、响应 ID 不存在 |
| 502 | 模型加载或上游调用失败 |
| 503 | GPU 资源不足或繁忙模型无法驱逐 |

## 7. 客户端升级建议

1. 启动时读取 `/v1/models`，以 `id` 建立模型目录。
2. 上下文预算只使用 `context_length`。
3. 根据 `capabilities` 禁用客户端不支持的图片、视频、工具或结构化输出选项。
4. 不要因为 `loaded=false` 阻止调用，网关会自动加载。
5. Responses 按 `output[].type` 处理 `reasoning`、`message`、`function_call`。
6. 流式工具参数必须累积完整后再解析 JSON。
7. usage 以非流式最终响应或流式结束事件为准。
8. 对新增字段和新增 SSE 事件采取忽略策略，保证向前兼容。

## 8. 当前边界

- 系统指令统一归一化为唯一一条置顶消息：Responses 开头连续的 `system`/`developer` 按序合并，Anthropic `messages` 内联的 `system` 并入置顶。中途 system 的位置语义不保留——后端 chat template 只认一条置顶系统指令。
- 不支持托管 `web_search`、文件/PDF、音频输入，相关请求会明确返回 400。
- 不支持 `previous_response_id` 会话续接，客户端应发送完整 `input` 历史。
- 非流式 Responses 的 GET/DELETE 仅使用进程内缓存，客户端不应把它当作持久会话存储。
