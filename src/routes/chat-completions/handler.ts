import type { Context } from "hono"

import consola from "consola"
import { streamSSE, type SSEMessage } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { getTokenCount } from "~/lib/tokenizer"
import { isNullish } from "~/lib/utils"
import {
  createChatCompletions,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
} from "~/services/copilot/create-chat-completions"

export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  let payload = await c.req.json<ChatCompletionsPayload>()
  consola.debug("Request payload:", JSON.stringify(payload).slice(-400))

  // Find the selected model
  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )

  // Calculate and display token count
  try {
    if (selectedModel) {
      const tokenCount = await getTokenCount(payload, selectedModel)
      consola.info("Current token count:", tokenCount)
    } else {
      consola.warn("No model selected, skipping token count calculation")
    }
  } catch (error) {
    consola.warn("Failed to calculate token count:", error)
  }

  if (state.manualApprove) await awaitApproval()

  // ── Sanitize messages to avoid 400 errors ──────────────────
  payload = sanitizePayload(payload)

  if (isNullish(payload.max_tokens)) {
    payload = {
      ...payload,
      max_tokens: selectedModel?.capabilities.limits.max_output_tokens,
    }
    consola.debug("Set max_tokens to:", JSON.stringify(payload.max_tokens))
  }

  const response = await createChatCompletions(payload)

  if (isNonStreaming(response)) {
    consola.debug("Non-streaming response:", JSON.stringify(response))
    return c.json(response)
  }

  consola.debug("Streaming response")
  return streamSSE(c, async (stream) => {
    for await (const chunk of response) {
      // Skip [DONE] sentinel — it's not valid JSON and causes parse errors
      if (chunk.data === "[DONE]") {
        consola.debug("Received [DONE], ending stream")
        break
      }
      consola.debug("Streaming chunk:", JSON.stringify(chunk))
      await stream.writeSSE(chunk as SSEMessage)
    }
  })
}

// ── Sanitize payload to prevent common 400 errors ──────────────
function sanitizePayload(payload: ChatCompletionsPayload): ChatCompletionsPayload {
  let messages = [...payload.messages]

  // Fix 1: Remove orphaned tool_result messages whose tool_use_id
  // doesn't exist in the preceding assistant message's tool_calls.
  // This happens when OpenClaw compacts conversation history and
  // drops an assistant message with tool_calls but keeps the tool results.
  const validToolCallIds = new Set<string>()
  const cleaned: typeof messages = []

  for (const msg of messages) {
    // Track tool_call IDs from assistant messages
    if (msg.role === "assistant" && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        validToolCallIds.add(tc.id)
      }
    }

    // Filter out tool messages with orphaned IDs
    if (msg.role === "tool" && msg.tool_call_id) {
      if (!validToolCallIds.has(msg.tool_call_id)) {
        consola.warn(`Dropping orphaned tool result: ${msg.tool_call_id}`)
        continue
      }
    }

    // Also check content arrays for tool_result blocks (Anthropic format)
    if (Array.isArray(msg.content)) {
      const filteredContent = (msg.content as Array<{ type: string; tool_use_id?: string }>).filter((part) => {
        if (part.type === "tool_result" && part.tool_use_id) {
          if (!validToolCallIds.has(part.tool_use_id)) {
            consola.warn(`Dropping orphaned tool_result block: ${part.tool_use_id}`)
            return false
          }
        }
        return true
      })
      if (filteredContent.length === 0) {
        consola.warn("Dropping message with no remaining content after tool_result cleanup")
        continue
      }
      msg.content = filteredContent
    }

    cleaned.push(msg)
  }

  // Fix 2: Rough token budget — trim older messages if payload is too large.
  // Keep system message(s) + last N messages to stay under ~120k chars (~100k tokens).
  const MAX_CHARS = 480_000 // ~120k tokens rough estimate
  let totalChars = JSON.stringify(cleaned).length

  if (totalChars > MAX_CHARS) {
    consola.warn(`Payload too large (${totalChars} chars), trimming older messages`)

    // Separate system messages from the rest
    const systemMsgs = cleaned.filter(m => m.role === "system" || m.role === "developer")
    const nonSystem = cleaned.filter(m => m.role !== "system" && m.role !== "developer")

    // Keep removing from the front (oldest) until under budget
    while (nonSystem.length > 2 && JSON.stringify([...systemMsgs, ...nonSystem]).length > MAX_CHARS) {
      const removed = nonSystem.shift()
      consola.debug(`Trimmed message role=${removed?.role}`)
    }

    messages = [...systemMsgs, ...nonSystem]
    consola.info(`Trimmed to ${messages.length} messages (${JSON.stringify(messages).length} chars)`)
  } else {
    messages = cleaned
  }

  return { ...payload, messages }
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
