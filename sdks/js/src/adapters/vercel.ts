/**
 * Vercel AI SDK adapter for EvalForge.
 *
 * Usage:
 *   import { generateText, streamText } from 'ai';
 *   import { fromVercel } from 'evalforge';
 *   import { run } from 'evalforge';
 *
 *   const result = await generateText({
 *     model: openai('gpt-4o'),
 *     prompt: 'What is the capital of France?',
 *     tools: { webSearch: ... }
 *   });
 *
 *   const tracePath = fromVercel(result, {
 *     question: 'What is the capital of France?',
 *     model: 'gpt-4o'
 *   });
 *   const evalResult = run(tracePath, { metrics: ['faithfulness'] });
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Trace } from '../types';

export interface VercelAdapterOptions {
  agentName?: string;
  model?: string;
  question?: string;
  expectedTools?: string[];
}

export function fromVercel(
  result: any,
  options: VercelAdapterOptions = {}
): string {
  /**
   * Convert Vercel AI SDK generateText() result to EvalForge trace JSON.
   *
   * Supports:
   * - result.text — final answer
   * - result.toolCalls — array of tool invocations
   * - result.toolResults — array of tool results
   * - result.steps — array of reasoning steps
   * - result.usage — token usage stats
   * - result.finishReason — why generation stopped
   */

  const {
    agentName = 'vercel-agent',
    model = 'unknown',
    question = '',
    expectedTools = [],
  } = options;

  const steps: Trace['steps'] = [];
  let stepId = 1;
  let finalAnswer = '';
  let totalTokens = 0;

  // Extract final answer
  if (result?.text) {
    finalAnswer = result.text;
  } else if (typeof result === 'string') {
    finalAnswer = result;
  }

  // Extract token usage
  if (result?.usage) {
    totalTokens = (result.usage.promptTokens ?? 0) +
                  (result.usage.completionTokens ?? 0);
  }

  // Extract tool calls and results
  const toolCalls = result?.toolCalls ?? [];
  const toolResults = result?.toolResults ?? [];

  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i];
    const tr = toolResults[i];

    steps.push({
      step_id: stepId++,
      type: 'tool_call',
      tool: tc.toolName ?? tc.name ?? 'unknown_tool',
      input: tc.args ?? tc.input ?? {},
      output: tr ? { result: String(tr.result ?? '') } : {},
      duration_ms: 0,
    });
  }

  // Extract steps if present (multi-step generations)
  if (Array.isArray(result?.steps)) {
    for (const step of result.steps) {
      if (step.text && step.text !== finalAnswer) {
        steps.push({
          step_id: stepId++,
          type: 'thought',
          content: step.text,
        });
      }
      // Tool calls inside steps
      for (const tc of step.toolCalls ?? []) {
        const tr = (step.toolResults ?? []).find(
          (r: any) => r.toolCallId === tc.toolCallId
        );
        steps.push({
          step_id: stepId++,
          type: 'tool_call',
          tool: tc.toolName ?? 'unknown_tool',
          input: tc.args ?? {},
          output: tr ? { result: String(tr.result ?? '') } : {},
          duration_ms: 0,
        });
      }
    }
  }

  const trace: Trace = {
    evalforge_version: '0.1',
    trace_id: `vercel-${Date.now()}`,
    timestamp: new Date().toISOString(),
    metadata: {
      framework: 'vercel-ai',
      model,
      agent_name: agentName,
      duration_ms: 0,
      total_tokens: totalTokens,
    },
    input: {
      user: question,
      system: '',
    },
    steps,
    output: {
      answer: finalAnswer,
      finish_reason: result?.finishReason,
    },
    eval_hints: {
      expected_tools: expectedTools,
      expected_answer: null,
      context_documents: [],
    },
  };

  const tmp = path.join(os.tmpdir(), `evalforge_vercel_${Date.now()}.json`);
  fs.writeFileSync(tmp, JSON.stringify(trace, null, 2));
  return tmp;
}
