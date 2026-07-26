import {
  calculateCost,
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type StopReason,
  type Tool
} from "@earendil-works/pi-ai";

type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  tool_calls?: Array<{
    function: { name: string; arguments: Record<string, unknown> };
  }>;
};

type OllamaResponse = {
  message?: {
    content?: string;
    thinking?: string;
    tool_calls?: Array<{
      function?: { name?: string; arguments?: Record<string, unknown> | string };
    }>;
  };
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
  error?: string;
};

type OllamaToolCall = NonNullable<NonNullable<OllamaResponse["message"]>["tool_calls"]>[number];

/**
 * Calls Ollama's native /api/chat endpoint and adapts its completed response
 * to Pi's standard event stream.  Ollama has no API-key requirement by
 * default, but callers may still use a protected reverse proxy.
 */
export function streamOllamaChat(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const output: AssistantMessage = {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
    },
    stopReason: "stop",
    timestamp: Date.now()
  };

  void (async () => {
    try {
      stream.push({ type: "start", partial: output });
      const request = {
        model: model.id,
        messages: toOllamaMessages(context),
        tools: context.tools?.map(toOllamaTool),
        stream: false,
        think: Boolean(model.reasoning && options?.reasoning),
        options: { num_predict: options?.maxTokens ?? model.maxTokens }
      };
      const payload = await options?.onPayload?.(request, model) ?? request;
      const apiKey = options?.apiKey?.trim();
      const response = await fetch(`${normalizeOllamaBaseUrl(model.baseUrl)}/api/chat`, {
        method: "POST",
        signal: options?.signal,
        headers: {
          "content-type": "application/json",
          ...(apiKey && apiKey !== "ollama" ? { authorization: `Bearer ${apiKey}` } : {})
        },
        body: JSON.stringify(payload)
      });
      await options?.onResponse?.({
        status: response.status,
        headers: Object.fromEntries(response.headers.entries())
      }, model);
      const body = await parseOllamaResponse(response);
      if (!response.ok || body.error) {
        throw new Error(body.error || `Ollama returned HTTP ${response.status}`);
      }
      if (!body.message) {
        throw new Error("Ollama response did not include a message");
      }

      appendThinking(stream, output, body.message.thinking);
      appendText(stream, output, body.message.content);
      appendToolCalls(stream, output, body.message.tool_calls);
      output.usage.input = body.prompt_eval_count ?? 0;
      output.usage.output = body.eval_count ?? 0;
      output.usage.totalTokens = output.usage.input + output.usage.output;
      calculateCost(model, output.usage);
      output.stopReason = ollamaStopReason(body.done_reason, Boolean(body.message.tool_calls?.length));
      stream.push({
        type: "done",
        reason: output.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
        message: output
      });
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
    } finally {
      stream.end();
    }
  })();

  return stream;
}

export function normalizeOllamaBaseUrl(rawBaseUrl: string): string {
  return rawBaseUrl.replace(/\/+$/, "").replace(/\/api\/chat$/i, "");
}

function toOllamaMessages(context: Context): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  if (context.systemPrompt?.trim()) {
    messages.push({ role: "system", content: context.systemPrompt });
  }
  for (const message of context.messages) {
    const converted = toOllamaMessage(message);
    if (converted) {
      messages.push(converted);
    }
  }
  return messages;
}

function toOllamaMessage(message: Message): OllamaMessage | undefined {
  if (message.role === "user") {
    return { role: "user", content: messageContent(message.content) };
  }
  if (message.role === "toolResult") {
    return { role: "tool", content: messageContent(message.content) };
  }
  const text = message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n");
  const thinking = message.content
    .filter((part) => part.type === "thinking")
    .map((part) => part.thinking)
    .join("\n");
  const toolCalls = message.content
    .filter((part) => part.type === "toolCall")
    .map((part) => ({ function: { name: part.name, arguments: part.arguments } }));
  if (!text && !thinking && toolCalls.length === 0) {
    return undefined;
  }
  return {
    role: "assistant",
    content: text,
    ...(thinking ? { thinking } : {}),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {})
  };
}

function messageContent(content: Message["content"]): string {
  if (typeof content === "string") {
    return content;
  }
  return content.map((part) => part.type === "text" ? part.text : "[image omitted]").join("\n");
}

function toOllamaTool(tool: Tool): Record<string, unknown> {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters
    }
  };
}

async function parseOllamaResponse(response: Response): Promise<OllamaResponse> {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text) as OllamaResponse;
  } catch {
    throw new Error(`Ollama returned invalid JSON: ${text.slice(0, 500)}`);
  }
}

function appendThinking(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  thinking: string | undefined
): void {
  if (!thinking) {
    return;
  }
  const contentIndex = output.content.push({ type: "thinking", thinking }) - 1;
  stream.push({ type: "thinking_start", contentIndex, partial: output });
  stream.push({ type: "thinking_delta", contentIndex, delta: thinking, partial: output });
  stream.push({ type: "thinking_end", contentIndex, content: thinking, partial: output });
}

function appendText(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  text: string | undefined
): void {
  if (!text) {
    return;
  }
  const contentIndex = output.content.push({ type: "text", text }) - 1;
  stream.push({ type: "text_start", contentIndex, partial: output });
  stream.push({ type: "text_delta", contentIndex, delta: text, partial: output });
  stream.push({ type: "text_end", contentIndex, content: text, partial: output });
}

function appendToolCalls(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  toolCalls: OllamaToolCall[] | undefined
): void {
  for (const call of toolCalls ?? []) {
    const name = call.function?.name;
    if (!name) {
      continue;
    }
    const argumentsValue = call.function?.arguments;
    const argumentsObject = typeof argumentsValue === "string"
      ? safeJsonObject(argumentsValue)
      : argumentsValue ?? {};
    const toolCall = {
      type: "toolCall" as const,
      id: crypto.randomUUID(),
      name,
      arguments: argumentsObject
    };
    const contentIndex = output.content.push(toolCall) - 1;
    stream.push({ type: "toolcall_start", contentIndex, partial: output });
    stream.push({ type: "toolcall_delta", contentIndex, delta: JSON.stringify(argumentsObject), partial: output });
    stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
  }
}

function safeJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function ollamaStopReason(doneReason: string | undefined, hasToolCalls: boolean): Extract<StopReason, "stop" | "length" | "toolUse"> {
  if (hasToolCalls || doneReason === "tool_calls") {
    return "toolUse";
  }
  return doneReason === "length" ? "length" : "stop";
}
