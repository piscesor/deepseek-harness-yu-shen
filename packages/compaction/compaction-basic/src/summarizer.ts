/**
 * Default one-shot summarization and durable checkpoint framing.
 *
 * @module @deepseek-ai/dsh-compaction-basic/summarizer
 */

import type { Context } from '@deepseek-ai/cordis'
import {
  contentHasImage,
  CONTEXT_WINDOW_EXCEEDED_CODE,
  BlockAssembler,
  LlmError,
} from '@deepseek-ai/dsh-llm'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'
import type {
  ContentBlock, FinishReason, GenerateOptions, Message, RequestMessage, TokenUsage, ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { Agent } from '@deepseek-ai/dsh-agent'

interface SummaryConfig {
  readonly summarizationProvider: string
  readonly summarizationModel: string
  readonly maxTokens: number
}

/** Exact provider/model pair that receives one summarization request. */
export type SummaryRoute = Pick<GenerateOptions, 'provider' | 'model'>

/** Tags wrapping the structured summary inside the landed checkpoint node. */
const SUMMARY_OPEN_TAG = '<compacted-summary>'
const SUMMARY_CLOSE_TAG = '</compacted-summary>'

/**
 * The summarization directive, delivered as the FINAL user message after the
 * replayed conversation rather than as a distinct summarizer system prompt.
 * Keeping the conversation's own system prompt, tools, and message prefix in
 * front of it makes the auxiliary call a genuine prefix of the last routed
 * request, so the provider's KV cache is reused instead of invalidated.
 */
const COMPACTION_INSTRUCTION = [
  'You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.',
  '',
  'Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.',
  '',
  '## Primary Request and Intent',
  "- [the user's original and evolving goals; quote verbatim where the exact wording matters]",
  '',
  '## Key Technical Concepts',
  '- [technologies, frameworks, patterns, and conventions in play]',
  '',
  '## Files and Code',
  '- [exact path: why it matters, key changes or snippets]',
  '',
  '## Errors and Fixes',
  '- [error: how it was resolved, plus any related user feedback]',
  '',
  '## Pending Jobs',
  '- [explicitly requested work not yet completed]',
  '',
  '## Current Work',
  '- [precisely what was in progress at this checkpoint]',
  '',
  '## Next Step',
  '- [the single next action, directly in line with the most recent request, or "(none)"]',
  '',
  '## Critical Context',
  '- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]',
  '',
  'Rules:',
  '- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.',
  '- Capture user feedback and explicit instructions faithfully, especially corrections.',
  '- Do NOT mention this summarization request or that the context was compacted.',
  '- Output only the checkpoint text: do not call any tool or take any other action.',
  `- If the conversation already contains a ${SUMMARY_OPEN_TAG} block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure.`,
].join('\n')

/** Framing that makes the replacement user message established context. */
const CHECKPOINT_PREAMBLE =
  'This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.'

/**
 * The replayed conversation surface the summarizer condenses. Reproducing the
 * last routed request's system prompt, tools, and leading messages verbatim
 * lets the auxiliary call reuse the provider's warm prefix cache; the trailing
 * compaction instruction is then the only novel input.
 */
export interface SummarizationInput {
  /** The conversation's tool schemas, reused for prefix-cache alignment; absent when the request carried none. */
  readonly tools?: readonly ToolSchema[]
  /** The derived system head, when present, followed by the shadowed region in surface order. */
  readonly messages: readonly Message[]
}

/** Safe summary content plus the exact auxiliary call envelope recorded with it. */
export type SummaryResult = {
  summary: ContentBlock[]
  provider: string
  model: string
  maxTokens?: number
  /** Provider-reported usage for this summarization request. */
  usage?: TokenUsage
} & (
  | {
    /** Complete provider output before the text-only summary projection. */
    rawOutput: ContentBlock[]
    /** Identifies exactly one call through this context's `ctx.llm.stream()`. */
    llmStreamCall: true
  }
  | {
    /** Optional complete output from an unmarked template, remote, or other summarizer. */
    rawOutput?: ContentBlock[]
    /** An unmarked result does not identify a call through this context's LLM seam. */
    llmStreamCall?: never
  }
)

/**
 * Run the default cache-reusing `ctx.llm.stream()` summarization call: replay
 * the conversation prefix, then append the compaction instruction as the final
 * user message so the provider's warm prefix cache is reused.
 * @param ctx - context providing the LLM service.
 * @param config - resolved backend configuration.
 * @param input - replayed conversation prefix (system, tools, and leading messages) to condense.
 * @param agent - supplies tool history and session id.
 * @param conversation - route used when `config` names no summary target; `undefined` when the conversation has none.
 * @param signal - optional cancellation forwarded to the adapter.
 * @returns safe text-only summary blocks and the exact call envelope and output.
 */
export async function summarizeWithLlm(
  ctx: Context,
  config: SummaryConfig,
  input: SummarizationInput,
  agent: Agent,
  conversation: SummaryRoute | undefined,
  signal?: AbortSignal,
): Promise<SummaryResult> {
  const configured = config.summarizationProvider.length === 0
    ? undefined
    : { provider: config.summarizationProvider, model: config.summarizationModel }
  const target = configured ?? conversation
  if (target === undefined) {
    throw new Error(
      'no provider/model available for summarization: set both BasicCompactionConfig summarization fields, route one request, or set both AgentOptions fields',
    )
  }

  const assembler = new BlockAssembler()
  const messages: RequestMessage[] = [
    ...input.messages,
    deepFreeze({
      role: 'user',
      content: [{ type: 'text', text: COMPACTION_INSTRUCTION }],
    }),
  ]
  const options: GenerateOptions = {
    provider: target.provider,
    model: target.model,
    messages,
    toolHistory: agent.session.toolHistory(),
    ...input.tools === undefined ? {} : { tools: [...input.tools] },
    maxTokens: config.maxTokens,
    sessionId: agent.session.id,
    purpose: 'compaction',
    ...signal === undefined ? {} : { signal },
  }
  for await (const chunk of ctx.llm.stream(options)) assembler.push(chunk)
  const error = finishError(assembler.finish)
  if (error !== undefined) throw error

  const rawOutput = assembler.blocks()
  const summary = summaryText(rawOutput)
  if (!summary.some(block => block.text.trim().length > 0)) {
    throw new Error('summarization produced no text summary content')
  }
  return {
    summary,
    rawOutput,
    llmStreamCall: true,
    provider: options.provider,
    model: options.model,
    maxTokens: config.maxTokens,
    ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
  }
}

/** Failure classes a smaller summarization input can resolve. */
const RETRYABLE_SUMMARY_CODES = new Set<string>([CONTEXT_WINDOW_EXCEEDED_CODE, 'MAX_TOKENS'])

/** Upper bound on summarization attempts; every retry sends strictly less input. */
const MAX_SUMMARY_ATTEMPTS = 4

/** Smallest slice retained when an attempt shrinks; below this a checkpoint carries no signal. */
const MIN_RETRY_SLICE_TOKENS = 2048

/** Route-derived facts the retry loop needs in order to bound a retried input. */
export interface SummaryRetryBudget {
  /** Adapter-reported capacity of the summarization target, when the adapter knows it. */
  contextWindow(): Promise<number | undefined>
  /** Fixed-heuristic price of one message, matching the conversation meter. */
  estimate(message: Message): number
}

/** The failure code carried by a summarization finish, when it has one. */
function summaryFailureCode(error: unknown): string | undefined {
  if (typeof error !== 'object' || error === null) return undefined
  const code = (error as { code?: unknown }).code
  return typeof code === 'string' ? code : undefined
}

/**
 * Keep the system head plus the newest messages that fit one retry budget.
 *
 * The system message carries the conversation's durable framing and keeps the
 * auxiliary call a prefix of the routed request, so it is never trimmed; the
 * conversation is cut from its oldest end, which is exactly what a compaction
 * checkpoint is supposed to absorb anyway.
 * @param input - the summarization input that just failed.
 * @param contextWindow - adapter capacity of the summarization target, when known.
 * @param maxTokens - output cap reserved for the summary itself.
 * @param estimate - fixed-heuristic message price, matching the conversation meter.
 * @returns a strictly smaller input, or `undefined` when nothing can be removed.
 */
export function trimRetryInput(
  input: SummarizationInput,
  contextWindow: number | undefined,
  maxTokens: number,
  estimate: (message: Message) => number,
): SummarizationInput | undefined {
  const messages = [...input.messages]
  const head = messages[0]?.role === 'system' ? messages.shift() : undefined
  if (messages.length === 0) return undefined
  const priced = messages.map(message => estimate(message))
  const total = priced.reduce((sum, tokens) => sum + tokens, 0)
  const headTokens = head === undefined ? 0 : estimate(head)
  // Half the current input always fits the "strictly smaller" contract; the
  // route capacity further bounds it so the retried request lands under the
  // window instead of merely under the previous attempt.
  const half = Math.floor(total / 2)
  const capacity = contextWindow === undefined
    ? half
    : contextWindow - maxTokens - headTokens - MIN_RETRY_SLICE_TOKENS
  const target = Math.max(Math.min(half, capacity), Math.min(half, MIN_RETRY_SLICE_TOKENS))
  let kept = 0
  let accumulated = 0
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    accumulated += priced[index] ?? 0
    if (kept > 0 && accumulated > target) break
    kept += 1
  }
  if (kept >= messages.length) return undefined
  const tail = messages.slice(messages.length - kept)
  return { ...input, messages: head === undefined ? tail : [head, ...tail] }
}

/**
 * Summarize the replayed prefix, retrying with a smaller prefix when the
 * provider rejects the auxiliary call for size or stops it at the output cap.
 *
 * Overflow recovery runs exactly when the conversation no longer fits, so a
 * full-prefix summarization request can be rejected by the very window it is
 * meant to relieve. Shrinking the input keeps the checkpoint — and therefore
 * the turn — alive; the alternative is surfacing the original context-window
 * failure and making the user start a new session.
 * @param ctx - context providing the LLM service.
 * @param config - resolved backend configuration.
 * @param input - replayed conversation prefix (system, tools, and leading messages) to condense.
 * @param agent - supplies routed-model history, fallback model, and session id.
 * @param conversation - route used when `config` names no summary target; `undefined` when the conversation has none.
 * @param budget - capacity and pricing for retry sizing.
 * @param signal - optional cancellation forwarded to the adapter.
 * @returns safe text-only summary blocks and the exact call envelope and output.
 */
export async function summarizeWithOverflowFallback(
  ctx: Context,
  config: SummaryConfig,
  input: SummarizationInput,
  agent: Agent,
  conversation: SummaryRoute | undefined,
  budget: SummaryRetryBudget,
  signal?: AbortSignal,
): Promise<SummaryResult> {
  let current = input
  let firstError: unknown
  for (let attempt = 0; attempt < MAX_SUMMARY_ATTEMPTS; attempt += 1) {
    try {
      return await summarizeWithLlm(ctx, config, current, agent, conversation, signal)
    } catch (error: unknown) {
      firstError ??= error
      const code = summaryFailureCode(error)
      if (code === undefined || !RETRYABLE_SUMMARY_CODES.has(code) || signal?.aborted === true) {
        throw error
      }
      const next = trimRetryInput(
        current,
        await budget.contextWindow(),
        config.maxTokens,
        message => budget.estimate(message),
      )
      if (next === undefined) throw firstError
      current = next
    }
  }
  throw firstError
}

/**
 * Wrap raw summary blocks in the durable checkpoint framing.
 * @param summary - safe text-only model output.
 * @returns content for the synthesized replacement user message.
 */
export function frameSummary(summary: readonly ContentBlock[]): ContentBlock[] {
  return [
    { type: 'text', text: `${CHECKPOINT_PREAMBLE}\n\n${SUMMARY_OPEN_TAG}` },
    ...summary,
    { type: 'text', text: SUMMARY_CLOSE_TAG },
  ]
}

/** Map a terminal summarization finish to its fail-closed error. */
function finishError(finish: FinishReason): Error | undefined {
  switch (finish.kind) {
    case 'error':
    case 'aborted': {
      return LlmError.fromFailure(finish.failure)
    }
    case 'max-tokens': {
      const error = new Error('summarization truncated at the token cap (incomplete checkpoint)') as Error & { code?: string }
      error.code = 'MAX_TOKENS'
      return error
    }
    default:
      return undefined
  }
}

/** Reject visual output and keep only text before synthesizing a user message. */
function summaryText(
  blocks: readonly ContentBlock[],
): Array<Extract<ContentBlock, { type: 'text' }>> {
  if (contentHasImage(blocks)) {
    throw new LlmError('compaction summary cannot contain image output', 'UNSUPPORTED_CONTENT')
  }
  return blocks.filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
}
