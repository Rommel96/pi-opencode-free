// SPDX-License-Identifier: GPL-3.0-or-later
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
// pi's extension loader aliases the pi-ai root to the /compat entrypoint and
// only whitelists exact subpaths (/compat, /oauth, /providers/all), so deeper
// subpath imports like .../api/openai-completions fail to resolve at load time.
import { openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { discoverModels, type OpenCodeModelInfo } from "./discovery.js";

const nativeOpenAICompletionsStream = openAICompletionsApi().streamSimple;
const nativeOpenAIResponsesStream = openAIResponsesApi().streamSimple;

const PROVIDER_ID = "opencode-free";
const BASE_URL = "https://opencode.ai/zen/v1";

// Describes Zen's upstream quirks to Pi's native openai-completions engine
// (max_tokens field name, reasoning_content replay) — no custom streaming code.
const OPENCODE_COMPAT = {
  supportsStore: false,
  supportsDeveloperRole: false,
  // Zen often ends streams without finish_reason; let pi infer stop/toolUse
  // instead of throwing "Stream ended without finish_reason".
  supportsFinishReason: false,
  maxTokensField: "max_tokens" as const,
  requiresReasoningContentOnAssistantMessages: true,
};

/** Maps a discovered model to the provider config shape. */
function toProviderModel(m: OpenCodeModelInfo) {
  return {
    id: m.id.replace(/^opencode\//, ""),
    name: m.name,
    // Per-model API override: models routed via @ai-sdk/openai speak the
    // Responses API (e.g. muse-spark), everything else uses chat completions.
    api: m.api ?? "openai-completions",
    reasoning: m.reasoning ?? false,
    thinkingLevelMap: m.thinkingLevelMap,
    input: (m.input?.includes("image") ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    contextWindow: m.contextWindow ?? 128_000,
    maxTokens: m.maxTokens ?? 16_384,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: OPENCODE_COMPAT,
  };
}

function toStoredModel(config: ReturnType<typeof toProviderModel>) {
  return { ...config, provider: PROVIDER_ID, baseUrl: BASE_URL };
}

function fromStoredModel<M extends { provider?: unknown; baseUrl?: unknown }>(m: M) {
  const { provider: _provider, baseUrl: _baseUrl, ...config } = m;
  return config as unknown as ReturnType<typeof toProviderModel>;
}

const OPENCODE_UA = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14";
const HEX = "0123456789abcdef";
const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const pick = (chars: string): string => chars[Math.floor(Math.random() * chars.length)] as string;
const repeat = (n: number, chars: string): string =>
  Array.from({ length: n }, () => pick(chars)).join("");

/**
 * A session id the free tier accepts: `ses_` + 12 lowercase hex + 14 base62.
 * The last three hex digits must land in 0xFC0..0xFFF (decrementing-timestamp layout).
 */
export const newZenSessionId = (): string =>
  `ses_${repeat(9, HEX)}f${pick("cdef")}${pick(HEX)}${repeat(14, BASE62)}`;

const GATE_TOOLS = [
  {
    type: "function",
    function: {
      name: "bash",
      description: "Placeholder required by the OpenCode free-tier request shape.",
      parameters: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read",
      description: "Placeholder required by the OpenCode free-tier request shape.",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
    },
  },
];

/** Extract active tools from transcript system messages (toolsAdded/toolsRemoved). */
export function getCurrentTools(messages: unknown): unknown[] {
  if (!Array.isArray(messages)) return [];
  const tools = new Map<string, unknown>();
  for (const message of messages) {
    if (!message || typeof message !== "object" || (message as { role?: string }).role !== "system") {
      continue;
    }
    const removed = (message as { toolsRemoved?: Array<{ name?: string }> }).toolsRemoved;
    if (Array.isArray(removed)) {
      for (const t of removed) {
        if (t?.name) tools.delete(t.name);
      }
    }
    const added = (message as { toolsAdded?: Array<{ name?: string }> }).toolsAdded;
    if (Array.isArray(added)) {
      for (const t of added) {
        if (t?.name) tools.set(t.name, t);
      }
    }
  }
  return [...tools.values()];
}

export function shapeZenPayload(payload: unknown): unknown {
  if (!payload || typeof payload !== "object") return payload;
  const p = { ...(payload as Record<string, unknown>) };
  p.stream = true;
  const tools = Array.isArray(p.tools) ? [...p.tools] : [];
  const names = new Set(
    tools.map((t) => (t as { function?: { name?: string } })?.function?.name).filter(Boolean),
  );
  let injected = false;
  for (const gate of GATE_TOOLS) {
    if (!names.has(gate.function.name)) {
      tools.push(gate);
      injected = true;
    }
  }
  p.tools = tools;
  if (injected && p.tool_choice === undefined) p.tool_choice = "auto";
  return p;
}

export default function opencodeDirectExtension(pi: ExtensionAPI): void {
  // Registered once with an empty list; models come exclusively through
  // refreshModels (snapshot restore on session init, live discovery on
  // explicit refresh) — pi's native model lifecycle.
  pi.registerProvider(PROVIDER_ID, {
    name: "OpenCode Direct (Free)",
    baseUrl: BASE_URL,
    // Placeholder: Pi's auth composition requires a key; real requests go
    // keyless via the streamSimple shim below.
    apiKey: "none",
    api: "openai-completions",
    headers: {
      "x-opencode-client": "cli",
      "x-opencode-project": "global",
      "User-Agent": OPENCODE_UA,
    },
    models: [],
    async refreshModels(ctx) {
      if (!ctx.allowNetwork) {
        return ctx.stored?.models
          .filter(m => m.provider === PROVIDER_ID && (m.api === "openai-completions" || m.api === "openai-responses"))
          .map(fromStoredModel) ?? [];
      }
      if (ctx.signal.aborted) return [];
      const discovered = await discoverModels({ signal: ctx.signal });
      if (discovered.length === 0) return []; // keep previous snapshot; retry on next refresh
      const configs = discovered.map(toProviderModel);
      await ctx.publish({ persist: { models: configs.map(toStoredModel), checkedAt: Date.now() } });
      return configs;
    },
    // Keyless shim over pi's native engines, dispatched by model API:
    // Zen's free tier rejects every Bearer token with 401, and
    // `Authorization: null` is the OpenAI SDK's supported omission.
    // Streaming, tools, reasoning, and cost handling stay fully native.
    streamSimple: (model, context, options) => {
      const api = (model as { api?: string }).api;
      const native =
        api === "openai-responses"
          ? (nativeOpenAIResponsesStream as typeof nativeOpenAICompletionsStream)
          : nativeOpenAICompletionsStream;
      const session = newZenSessionId();
      const currentTools =
        (context as { tools?: unknown[] })?.tools && (context as { tools?: unknown[] }).tools!.length > 0
          ? (context as { tools?: unknown[] }).tools
          : getCurrentTools((context as { messages?: unknown[] })?.messages);
      const effectiveContext =
        currentTools && currentTools.length > 0
          ? { ...context, tools: currentTools }
          : context;
      return native(
        model as Parameters<typeof nativeOpenAICompletionsStream>[0],
        effectiveContext as Parameters<typeof nativeOpenAICompletionsStream>[1],
        {
        ...options,
        headers: {
          ...options?.headers,
          Authorization: null,
          "User-Agent": OPENCODE_UA,
          "x-opencode-session": session,
          "x-opencode-session-id": session,
          "x-opencode-client": "cli",
          "x-opencode-project": "global",
        },
        onPayload: async (payload, hookModel) => {
          const modified = (await options?.onPayload?.(payload, hookModel)) ?? payload;
          const shaped = shapeZenPayload(modified);
          return shaped;
        },
      });
    },
  });
}

