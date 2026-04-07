/**
 * Mastra adapter for EvalForge.
 *
 * Usage:
 *   import { Agent } from '@mastra/core';
 *   import { fromMastra } from 'evalforge/adapters/mastra';
 *   import { run } from 'evalforge';
 *
 *   const agent = new Agent({ name: 'my-agent', ... });
 *   const result = await agent.generate('What is the capital of France?');
 *
 *   const tracePath = fromMastra(result, { agentName: 'my-agent', model: 'gpt-4o' });
 *   const evalResult = run(tracePath, { metrics: ['faithfulness'] });
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { Trace } from '../types';

export interface MastraAdapterOptions {
  agentName?: string;
  model?: string;
  question?: string;
  expectedTools?: string[];
}

export function fromMastra(
  result: any,
  options: MastraAdapterOptions = {}
): string {
  /**
   * Convert Mastra agent.generate() result to EvalForge trace JSON file.
   *
   * Supports:
   * - result.text — final answer string
   * - result.steps — array of reasoning steps
   * - result.toolCalls — array of tool invocations
   * - result.usage — token usage stats
   */

  const {
    agentName = 'mastra-agent',
    model = 'unknown',
    question = '',
    expectedTools = [],
  } = options;

  const steps: Trace['steps'] = [];
  let stepId = 1;
  let finalAnswer = '';
  let totalTokens = 0;

  // Extract final answer
  if (typeof result === 'string') {
    finalAnswer = result;
  } else if (result?.text) {
    finalAnswer = result.text;
  } else if (result?.content) {
    finalAnswer = String(result.content);
  }

  // Extract token usage
  if (result?.usage) {
    totalTokens = (result.usage.promptTokens ?? 0) +
                  (result.usage.completionTokens ?? 0);
  }

  // Extract steps if present
  if (Array.isArray(result?.steps)) {
    for (const step of result.steps) {
      if (step.text) {
        steps.push({
          step_id: stepId++,
          type: 'thought',
          content: step.text,
        });
      }
    }
  }

  // Extract tool calls if present
  if (Array.isArray(result?.toolCalls)) {
    for (const tc of result.toolCalls) {
      steps.push({
        step_id: stepId++,
        type: 'tool_call',
        tool: tc.toolName ?? tc.name ?? 'unknown_tool',
        input: tc.args ?? tc.input ?? {},
        output: { result: String(tc.result ?? '') },
        duration_ms: 0,
      });
    }
  }

  // Extract tool results from steps if toolCalls not separate
  if (Array.isArray(result?.toolResults)) {
    for (const tr of result.toolResults) {
      steps.push({
        step_id: stepId++,
        type: 'tool_call',
        tool: tr.toolName ?? 'unknown_tool',
        input: tr.args ?? {},
        output: { result: String(tr.result ?? '') },
        duration_ms: 0,
      });
    }
  }

  const trace: Trace = {
    evalforge_version: '0.1',
    trace_id: `mastra-${Date.now()}`,
    timestamp: new Date().toISOString(),
    metadata: {
      framework: 'mastra',
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
    },
    eval_hints: {
      expected_tools: expectedTools,
      expected_answer: null,
      context_documents: [],
    },
  };

  const tmp = path.join(os.tmpdir(), `evalforge_mastra_${Date.now()}.json`);
  fs.writeFileSync(tmp, JSON.stringify(trace, null, 2));
  return tmp;
}
