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
  done?: boolean;
  error?: string;
};

type OllamaToolCall = NonNullable<NonNullable<OllamaResponse["message"]>["tool_calls"]>[number];

/**
 * Calls Ollama's native /api/chat endpoint and adapts its NDJSON response
 * to Pi's standard event stream. Forwarding token deltas is important: Pi
 * uses them as an idle-progress signal while a local model is generating.
 * Ollama has no API-key requirement by
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
        stream: true,
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
      const deltaState = { thinkingIndex: undefined as number | undefined, textIndex: undefined as number | undefined };
      let completed: OllamaResponse | undefined;
      let responseError: string | undefined;
      let toolCalls: OllamaToolCall[] | undefined;
      await readOllamaResponses(response, (chunk) => {
        if (chunk.error) {
          responseError = chunk.error;
          return;
        }
        appendThinkingDelta(stream, output, deltaState, chunk.message?.thinking);
        appendTextDelta(stream, output, deltaState, chunk.message?.content);
        if (chunk.message?.tool_calls?.length) {
          toolCalls = chunk.message.tool_calls;
        }
        if (chunk.done) {
          completed = chunk;
        }
      });
      if (!response.ok || responseError) {
        throw new Error(responseError || `Ollama returned HTTP ${response.status}`);
      }
      if (!completed) {
        throw new Error("Ollama stream ended before its final response");
      }

      finishTextDelta(stream, output, deltaState);
      finishThinkingDelta(stream, output, deltaState);
      appendToolCalls(stream, output, toolCalls);
      output.usage.input = completed.prompt_eval_count ?? 0;
      output.usage.output = completed.eval_count ?? 0;
      output.usage.totalTokens = output.usage.input + output.usage.output;
      calculateCost(model, output.usage);
      output.stopReason = ollamaStopReason(completed.done_reason, Boolean(toolCalls?.length));
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

async function readOllamaResponses(
  response: Response,
  onResponse: (chunk: OllamaResponse) => void
): Promise<void> {
  if (!response.body) {
    throw new Error("Ollama response did not include a body");
  }
  const decoder = new TextDecoder();
  let buffered = "";
  const consumeLine = (line: string): void => {
    const value = line.trim();
    if (!value) {
      return;
    }
    try {
      onResponse(JSON.parse(value) as OllamaResponse);
    } catch {
      throw new Error(`Ollama returned invalid JSON: ${value.slice(0, 500)}`);
    }
  };
  for await (const bytes of response.body) {
    buffered += decoder.decode(bytes, { stream: true });
    const lines = buffered.split(/\r?\n/);
    buffered = lines.pop() ?? "";
    for (const line of lines) {
      consumeLine(line);
    }
  }
  buffered += decoder.decode();
  consumeLine(buffered);
}

type DeltaState = { thinkingIndex: number | undefined; textIndex: number | undefined };

function appendThinkingDelta(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  state: DeltaState,
  thinking: string | undefined
): void {
  if (!thinking) {
    return;
  }
  if (state.thinkingIndex === undefined) {
    state.thinkingIndex = output.content.push({ type: "thinking", thinking: "" }) - 1;
    stream.push({ type: "thinking_start", contentIndex: state.thinkingIndex, partial: output });
  }
  const block = output.content[state.thinkingIndex];
  if (block?.type !== "thinking") {
    throw new Error("Ollama thinking stream content index is invalid");
  }
  block.thinking += thinking;
  stream.push({ type: "thinking_delta", contentIndex: state.thinkingIndex, delta: thinking, partial: output });
}

function appendTextDelta(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  state: DeltaState,
  text: string | undefined
): void {
  if (!text) {
    return;
  }
  if (state.textIndex === undefined) {
    state.textIndex = output.content.push({ type: "text", text: "" }) - 1;
    stream.push({ type: "text_start", contentIndex: state.textIndex, partial: output });
  }
  const block = output.content[state.textIndex];
  if (block?.type !== "text") {
    throw new Error("Ollama text stream content index is invalid");
  }
  block.text += text;
  stream.push({ type: "text_delta", contentIndex: state.textIndex, delta: text, partial: output });
}

function finishThinkingDelta(stream: AssistantMessageEventStream, output: AssistantMessage, state: DeltaState): void {
  if (state.thinkingIndex === undefined) {
    return;
  }
  const block = output.content[state.thinkingIndex];
  if (block?.type === "thinking") {
    stream.push({ type: "thinking_end", contentIndex: state.thinkingIndex, content: block.thinking, partial: output });
  }
}

function finishTextDelta(stream: AssistantMessageEventStream, output: AssistantMessage, state: DeltaState): void {
  if (state.textIndex === undefined) {
    return;
  }
  const block = output.content[state.textIndex];
  if (block?.type === "text") {
    stream.push({ type: "text_end", contentIndex: state.textIndex, content: block.text, partial: output });
  }
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
