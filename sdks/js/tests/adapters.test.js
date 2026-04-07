const { test } = require('node:test');
const assert = require('node:assert');
const path = require('path');
const fs = require('fs');

const { fromMastra, fromVercel } = require('../dist/adapters/index.js');

test('fromMastra with text result', () => {
  const result = { text: 'The capital of France is Paris.' };
  const tracePath = fromMastra(result, {
    agentName: 'test-agent',
    model: 'gpt-4o',
    question: 'What is the capital of France?'
  });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.metadata.framework, 'mastra');
  assert.strictEqual(trace.output.answer, 'The capital of France is Paris.');
  assert.strictEqual(trace.input.user, 'What is the capital of France?');
});

test('fromMastra with tool calls', () => {
  const result = {
    text: 'Canberra is the capital.',
    toolCalls: [{
      toolName: 'web_search',
      args: { query: 'capital of Australia' },
      result: 'Canberra is the capital of Australia.'
    }]
  };
  const tracePath = fromMastra(result, {
    question: 'What is the capital of Australia?',
    expectedTools: ['web_search']
  });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.steps.length, 1);
  assert.strictEqual(trace.steps[0].type, 'tool_call');
  assert.strictEqual(trace.steps[0].tool, 'web_search');
  assert.deepStrictEqual(trace.eval_hints.expected_tools, ['web_search']);
});

test('fromMastra with string result', () => {
  const tracePath = fromMastra('Direct answer.', { agentName: 'simple' });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.output.answer, 'Direct answer.');
});

test('fromMastra with token usage', () => {
  const result = {
    text: 'Answer.',
    usage: { promptTokens: 100, completionTokens: 50 }
  };
  const tracePath = fromMastra(result);
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.metadata.total_tokens, 150);
});

test('fromVercel with text result', () => {
  const result = {
    text: 'The capital of France is Paris.',
    usage: { promptTokens: 50, completionTokens: 20 },
    finishReason: 'stop'
  };
  const tracePath = fromVercel(result, {
    question: 'What is the capital of France?',
    model: 'gpt-4o'
  });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.metadata.framework, 'vercel-ai');
  assert.strictEqual(trace.output.answer, 'The capital of France is Paris.');
  assert.strictEqual(trace.metadata.total_tokens, 70);
  assert.strictEqual(trace.output.finish_reason, 'stop');
});

test('fromVercel with tool calls', () => {
  const result = {
    text: 'Canberra is the capital.',
    toolCalls: [{
      toolName: 'web_search',
      args: { query: 'capital of Australia' }
    }],
    toolResults: [{
      result: 'Canberra is the capital of Australia.'
    }]
  };
  const tracePath = fromVercel(result, {
    question: 'Capital of Australia?',
    expectedTools: ['web_search']
  });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.strictEqual(trace.steps.length, 1);
  assert.strictEqual(trace.steps[0].tool, 'web_search');
  assert.deepStrictEqual(trace.eval_hints.expected_tools, ['web_search']);
});

test('fromVercel with multi-step result', () => {
  const result = {
    text: 'Final answer.',
    steps: [
      { text: 'Thinking...', toolCalls: [], toolResults: [] },
      {
        text: '',
        toolCalls: [{ toolName: 'search', toolCallId: '1', args: { q: 'test' } }],
        toolResults: [{ toolCallId: '1', result: 'search result' }]
      }
    ]
  };
  const tracePath = fromVercel(result, { question: 'Test?' });
  const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
  assert.ok(trace.steps.length >= 1);
});
