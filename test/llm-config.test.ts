import assert from "node:assert/strict";
import test from "node:test";
import {
  createLlmRuntime,
  loadLlmRuntimeConfig,
  normalizeAnthropicMessagesBaseUrl,
  normalizeOpenAIBaseUrl,
  normalizeOpenAICompletionsBaseUrl
} from "../src/llm-config.js";
import { normalizeOllamaBaseUrl, streamOllamaChat } from "../src/ollama.js";

test("normalizes full chat completions endpoint to OpenAI-compatible base URL", () => {
  assert.equal(
    normalizeOpenAICompletionsBaseUrl("https://example.test/api/openai/chat/completions"),
    "https://example.test/api/openai"
  );
});

test("registers LLM runtime from LLM_* environment", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai/chat/completions",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "feature/deepseek"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(runtime.model.provider, "baizhi-openai");
  assert.equal(runtime.model.id, "feature/deepseek");
  assert.equal(runtime.model.baseUrl, "https://example.test/api/openai");
  assert.equal(runtime.model.api, "openai-completions");
  assert.deepEqual(runtime.metadata.costPerMillionTokens, {
    input: 3,
    output: 6,
    cacheRead: 0.025,
    cacheWrite: 0
  });
  assert.equal(runtime.metadata.costCurrency, "CNY");
  assert.equal("apiKey" in runtime.metadata, false);
});

test("registers OpenAI Responses runtime when LLM_API_TYPE requests it", () => {
  assert.equal(
    normalizeOpenAIBaseUrl("https://example.test/api/openai/responses", "openai-responses"),
    "https://example.test/api/openai"
  );
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "sec/gpt-5.5",
    LLM_API_TYPE: "openai-responses"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(config.apiType, "openai-responses");
  assert.equal(runtime.model.api, "openai-responses");
  assert.equal(runtime.model.baseUrl, "https://example.test/api/openai");
});

test("registers native Ollama without requiring an API key", () => {
  assert.equal(
    normalizeOllamaBaseUrl("http://localhost:11434/api/chat"),
    "http://localhost:11434"
  );
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "http://localhost:11434/api/chat",
    LLM_DEFAULT_MODEL: "local-model",
    LLM_API_TYPE: "ollama"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(config.apiKey, "ollama");
  assert.equal(runtime.model.api, "ollama");
  assert.equal(runtime.model.baseUrl, "http://localhost:11434");
});

test("adapts a native Ollama chat response into Pi events", async (t) => {
  const originalFetch = globalThis.fetch;
  let requestUrl = "";
  let requestBody: Record<string, unknown> | undefined;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    requestUrl = input.toString();
    requestBody = JSON.parse(String(init?.body));
    return new Response([
      JSON.stringify({ message: { thinking: "inspect the target" }, done: false }),
      JSON.stringify({ message: { content: "I will run a command." }, done: false }),
      JSON.stringify({
        message: { tool_calls: [{ function: { name: "bash", arguments: { command: "id" } } }] },
        done: true,
        done_reason: "tool_calls",
        prompt_eval_count: 11,
        eval_count: 7
      })
    ].join("\n") + "\n", { status: 200, headers: { "content-type": "application/x-ndjson" } });
  }) as typeof fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  const runtime = createLlmRuntime(loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "http://localhost:11434",
    LLM_DEFAULT_MODEL: "local-model",
    LLM_API_TYPE: "ollama"
  }));
  const events = [];
  for await (const event of streamOllamaChat(runtime.model, {
    systemPrompt: "system instruction",
    messages: [{ role: "user", content: "solve this", timestamp: Date.now() }]
  }, { maxTokens: 99, reasoning: "low" })) {
    events.push(event);
  }

  assert.equal(requestUrl, "http://localhost:11434/api/chat");
  assert.deepEqual(requestBody, {
    model: "local-model",
    messages: [
      { role: "system", content: "system instruction" },
      { role: "user", content: "solve this" }
    ],
    stream: true,
    think: true,
    options: { num_predict: 99 }
  });
  assert.equal(events.at(-1)?.type, "done");
  assert.equal(events.some((event) => event.type === "thinking_delta"), true);
  assert.equal(events.some((event) => event.type === "text_delta"), true);
  const done = events.at(-1);
  assert.equal(done?.type === "done" && done.reason, "toolUse");
  if (done?.type === "done") {
    assert.equal(done.message.usage.input, 11);
    assert.equal(done.message.usage.output, 7);
    assert.deepEqual(done.message.content, [
      { type: "thinking", thinking: "inspect the target" },
      { type: "text", text: "I will run a command." },
      { type: "toolCall", id: done.message.content[2]?.type === "toolCall" ? done.message.content[2].id : "", name: "bash", arguments: { command: "id" } }
    ]);
  }
});

test("registers an Anthropic Messages planner alongside OpenAI executor roles", () => {
  assert.equal(
    normalizeAnthropicMessagesBaseUrl("https://planner.example.test/v1/messages"),
    "https://planner.example.test"
  );
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://executor.example.test/v1",
    LLM_API_KEY: "executor-key",
    LLM_DEFAULT_MODEL: "qwen-executor",
    LLM_PLANNER_API_TYPE: "anthropic-messages",
    LLM_PLANNER_BASE_URL: "https://planner.example.test/v1",
    LLM_PLANNER_API_KEY: "planner-key",
    LLM_PLANNER_MODEL: "minimax-m2.5"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(config.roles.planner.apiType, "anthropic-messages");
  assert.equal(runtime.models.planner.api, "anthropic-messages");
  assert.equal(runtime.models.planner.baseUrl, "https://planner.example.test");
  assert.equal(runtime.models.planner.provider, "baizhi-openai-planner");
  assert.equal(runtime.models.executor.api, "openai-completions");
  assert.equal(runtime.models.executor.baseUrl, "https://executor.example.test/v1");
  assert.equal(runtime.metadata.models.planner.apiType, "anthropic-messages");
  assert.equal("planner-key" in runtime.metadata.models.planner, false);
});

test("defaults to Chat Completions when LLM_API_TYPE is omitted", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai/responses",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "sec/gpt-5.5"
  });
  assert.equal(config.apiType, "openai-completions");
});

test("defaults all roles to the shared model with a 32k completion budget", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2"
  });
  assert.equal(config.defaultMaxTokens, 32_768);
  assert.equal(config.thinkingFormat, "zai");
  for (const role of ["planner", "executor", "supervisor", "projector"] as const) {
    assert.equal(config.roles[role].modelId, "glm-5.2");
    assert.equal(config.roles[role].maxTokens, 32_768);
    assert.equal(config.roles[role].thinkingLevel, "off");
    assert.equal(config.roles[role].apiType, "openai-completions");
  }
  const runtime = createLlmRuntime(config);
  for (const role of ["planner", "executor", "supervisor", "projector"] as const) {
    assert.equal(runtime.models[role].provider, "baizhi-openai");
    assert.equal(runtime.models[role].id, "glm-5.2");
    assert.equal(runtime.models[role].maxTokens, 32_768);
  }
  assert.equal(runtime.model, runtime.models.planner);
});

test("registers per-role models, budgets and thinking levels from LLM_<ROLE>_* overrides", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_MAX_TOKENS: "16384",
    LLM_EXECUTOR_MODEL: "deepseek-v4-pro-202606",
    LLM_PLANNER_MAX_TOKENS: "65536",
    LLM_PLANNER_THINKING: "low",
    LLM_SUPERVISOR_MODEL: "glm-5.2"
  });
  assert.equal(config.roles.executor.modelId, "deepseek-v4-pro-202606");
  assert.equal(config.roles.executor.maxTokens, 16_384);
  assert.equal(config.roles.planner.maxTokens, 65_536);
  assert.equal(config.roles.planner.thinkingLevel, "low");
  assert.equal(config.roles.projector.modelId, "glm-5.2");
  const runtime = createLlmRuntime(config);
  assert.equal(runtime.models.executor.id, "deepseek-v4-pro-202606");
  assert.equal(runtime.models.planner.maxTokens, 65_536);
  assert.equal(runtime.models.supervisor.maxTokens, 16_384);
  // planner/supervisor/projector share the default model id in one provider;
  // the executor variant gets its own registration.
  assert.equal(runtime.models.planner.provider, "baizhi-openai");
  assert.equal(runtime.models.executor.provider, "baizhi-openai");
  assert.equal(runtime.metadata.models.executor.modelId, "deepseek-v4-pro-202606");
});

test("registers a dedicated provider for roles with their own base URL or API key", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_EXECUTOR_BASE_URL: "https://backup.test/v1/chat/completions",
    LLM_EXECUTOR_API_KEY: "backup-key",
    LLM_EXECUTOR_MODEL: "glm-5.2"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(runtime.models.executor.provider, "baizhi-openai-executor");
  assert.equal(runtime.models.executor.baseUrl, "https://backup.test/v1");
  assert.equal(runtime.models.planner.provider, "baizhi-openai");
  assert.equal(runtime.models.planner.baseUrl, "https://example.test/api/openai");
  assert.equal("backup-key" in runtime.metadata.models.executor, false);
});

test("keeps per-role budgets distinct when roles share a model id", () => {
  const config = loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_PLANNER_MAX_TOKENS: "65536"
  });
  const runtime = createLlmRuntime(config);
  assert.equal(runtime.models.planner.maxTokens, 65_536);
  assert.equal(runtime.models.executor.maxTokens, 32_768);
  assert.notEqual(runtime.models.planner.id, runtime.models.executor.id);
});

test("rejects unsupported thinking level and format values", () => {
  assert.throws(() => loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_PLANNER_THINKING: "ultra"
  }), /Unsupported thinking level/);
  assert.throws(() => loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_THINKING_FORMAT: "xml"
  }), /Unsupported LLM_THINKING_FORMAT/);
  assert.throws(() => loadLlmRuntimeConfig({
    LLM_API_BASE_URL: "https://example.test/api/openai",
    LLM_API_KEY: "test-key",
    LLM_DEFAULT_MODEL: "glm-5.2",
    LLM_PLANNER_API_TYPE: "unknown-protocol"
  }), /Unsupported LLM_API_TYPE/);
});
