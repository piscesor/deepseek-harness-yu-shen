import { describe, expect, it } from 'vitest'
import { createSystemMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  estimateContent,
  estimateMessage,
  estimateSystemMessage,
  estimateTextTokens,
} from '../src/estimate.ts'

function textInput(text: string) {
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  })
}

describe('script-aware text density', () => {
  it('keeps the ASCII density bit-for-bit', () => {
    expect(estimateTextTokens('')).toBe(0)
    expect(estimateTextTokens('abcd')).toBe(1)
    expect(estimateTextTokens('abcdefgh')).toBe(2)
    expect(estimateContent([{ type: 'text', text: 'abcdefgh' }])).toBe(2 + 4)
  })

  it('prices wide-script text at its real token cost', () => {
    // CJK prose runs near one token per 1.2-1.6 characters; the ASCII density
    // of four characters per token would price these six characters as two.
    expect(estimateTextTokens('中文中文中文')).toBe(5)
    expect(estimateTextTokens('ひらがな')).toBe(3)
    expect(estimateTextTokens('한국어')).toBe(3)
    expect(estimateContent([{ type: 'text', text: '中文中文中文' }])).toBe(5 + 4)
  })

  it('prices mixed text by summing the narrow, wide, and emoji budgets', () => {
    expect(estimateTextTokens('abcd中文')).toBe(3)
    expect(estimateTextTokens('😀')).toBe(1)
    expect(estimateTextTokens('😀😀😀😀😀')).toBe(5)
  })

  it('prices a Chinese conversation far above its character-count/4 figure', () => {
    const chinese = '继续实现这个功能。'.repeat(50)
    const asciiDensity = Math.ceil(chinese.length / 4)
    expect(estimateTextTokens(chinese)).toBeGreaterThan(asciiDensity * 2)
  })

  it('routes system and user messages through the same density', () => {
    const chinese = '继续实现这个功能。'
    expect(estimateSystemMessage(createSystemMessage(chinese))).toBe(estimateTextTokens(chinese) + 4)
    expect(estimateMessage(textInput(chinese))).toBe(estimateTextTokens(chinese) + 4 + 4)
  })
})
