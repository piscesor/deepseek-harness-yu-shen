/**
 * Fixed-density heuristic token pricing shared by the meter service and the
 * pure context-breakdown projection, so both surfaces price identical content
 * to identical numbers.
 *
 * @module @deepseek-ai/dsh-token-meter/estimate
 */

import type { ContentBlock, Message } from '@deepseek-ai/dsh-llm'
import type { EpochHeader } from '@deepseek-ai/dsh-session'

/** Fixed text-density estimate for narrow (ASCII-range) text until exact tokenization is needed. */
const CHARS_PER_TOKEN = 4

/**
 * Characters per token for wide-script text. CJK, kana, and hangul tokenizers
 * (Qwen, Llama, most multilingual BPEs) emit roughly one token per 1.2-1.6
 * characters, so pricing them at the ASCII density of four characters per
 * token undercounts a Chinese conversation by two to three times and lets the
 * request reach the provider's hard context limit before compaction fires.
 */
const WIDE_CHARS_PER_TOKEN = 1.4

/** Whether one code point belongs to a script that tokenizes near one token per character. */
function isWideCodePoint(codePoint: number): boolean {
  return (codePoint >= 0x2e80 && codePoint <= 0x9fff) // CJK radicals through CJK Unified Ideographs
    || (codePoint >= 0xa960 && codePoint <= 0xa97f) // Hangul Jamo Extended-A
    || (codePoint >= 0xac00 && codePoint <= 0xd7ff) // Hangul syllables and Jamo Extended-B
    || (codePoint >= 0xf900 && codePoint <= 0xfaff) // CJK compatibility ideographs
    || (codePoint >= 0xfe30 && codePoint <= 0xfe4f) // CJK compatibility forms
    || (codePoint >= 0xff00 && codePoint <= 0xffef) // Halfwidth and fullwidth forms
    || (codePoint >= 0x20000 && codePoint <= 0x3ffff) // CJK Unified Ideographs Extension B and beyond
}

/** Whether one code point is an emoji, which most tokenizers spend one to two tokens on. */
function isEmojiCodePoint(codePoint: number): boolean {
  return (codePoint >= 0x1f000 && codePoint <= 0x1faff)
    || (codePoint >= 0x2600 && codePoint <= 0x27bf)
}

/**
 * Price one text run under the script-aware fixed-density heuristic: ASCII-range
 * text keeps the classic four-characters-per-token density, wide scripts are
 * priced at their real (much denser) token cost, and emoji cost one token each.
 * ASCII-only input prices exactly as `Math.ceil(text.length / 4)`, so every
 * caller that previously relied on that density keeps its numbers.
 * @param text - raw text to price without mutation.
 * @returns heuristic tokens for the text.
 */
export function estimateTextTokens(text: string): number {
  let narrow = 0
  let wide = 0
  let emoji = 0
  for (const character of text) {
    const codePoint = character.codePointAt(0) ?? 0
    if (isEmojiCodePoint(codePoint)) emoji += 1
    else if (isWideCodePoint(codePoint)) wide += 1
    else narrow += 1
  }
  return Math.ceil(narrow / CHARS_PER_TOKEN) + Math.ceil(wide / WIDE_CHARS_PER_TOKEN) + emoji
}

/** Per-block structural overhead for JSON framing and type tags. */
const BLOCK_OVERHEAD = 4

/** Role-field framing overhead added to every priced message. */
export const ROLE_OVERHEAD = 4

/**
 * Structural JSON price of one block outside the typed pricing arms: the
 * fixed heuristic for merge-extended blocks and for image references, whose
 * request price is route-owned rather than fixed.
 * @param block - block to price without mutation.
 * @returns heuristic tokens for the block's JSON structure.
 */
export function estimateStructuralBlock(block: ContentBlock): number {
  return BLOCK_OVERHEAD + estimateTextTokens(JSON.stringify(block))
}

/**
 * Price content blocks recursively under the fixed density heuristic.
 * @param blocks - content blocks to price without mutation.
 * @returns heuristic tokens including per-block structural overhead.
 */
export function estimateContent(blocks: readonly ContentBlock[]): number {
  let tokens = 0
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
      case 'reasoning':
        tokens += estimateTextTokens(block.text) + BLOCK_OVERHEAD
        break
      case 'tool-call':
        tokens += estimateTextTokens(block.name)
          + estimateTextTokens(block.arguments)
          + BLOCK_OVERHEAD
        break
      case 'tool-result':
        tokens += estimateContent(block.content) + BLOCK_OVERHEAD
        break
      default:
        // ContentBlockMap is merge-extensible; unknown blocks (and image
        // references, whose request price is route-owned) retain a
        // conservative structural JSON price under the fixed heuristic.
        tokens += estimateStructuralBlock(block)
    }
  }
  return tokens
}

/**
 * Price the rendered system prompt: the `system/message` surface node's text.
 * Adapters serialize the prompt as a plain string — a system-role message or
 * the request's system field — not as a typed content block, so the price is
 * text density plus role framing with no per-block overhead.
 * @param message - system-role message to price without mutation.
 * @returns heuristic system-prompt tokens; 0 for empty content ("no system prompt").
 */
export function estimateSystemMessage(message: Message): number {
  if (message.content.length === 0) return 0
  let tokens = 0
  for (const block of message.content) {
    tokens += block.type === 'text' ? estimateTextTokens(block.text) : estimateTextTokens(JSON.stringify(block))
  }
  return tokens + ROLE_OVERHEAD
}

/**
 * Heuristically price one model-visible message.
 * @param message - message to price without mutation.
 * @returns content and role-framing tokens under the fixed heuristic; a
 *   system-role message prices as {@link estimateSystemMessage}.
 */
export function estimateMessage(message: Message): number {
  if (message.role === 'system') return estimateSystemMessage(message)
  return estimateContent(message.content) + ROLE_OVERHEAD
}

/**
 * Price the tool-schema part of a canonical request envelope — the envelope's
 * only priced field, since the system prompt is a surface node.
 * @param header - canonical envelope, or undefined before any request.
 * @returns heuristic tool-schema tokens; 0 when absent or empty.
 */
export function estimateToolsTokens(header: EpochHeader | undefined): number {
  if (header?.tools === undefined || header.tools.length === 0) return 0
  return estimateTextTokens(JSON.stringify(header.tools)) + BLOCK_OVERHEAD
}
