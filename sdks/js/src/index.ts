import { spawnSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { findBinary } from './binary';
import { EvalResult, RunOptions, Trace } from './types';

export { EvalResult, MetricResult, RunOptions, Trace } from './types';
export * from './adapters';

function parseOutput(output: string): EvalResult {
  const metrics = [];
  const lines = output.split('\n');

  let traceId = '';
  let framework = '';
  let overallPassed = true;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (line.startsWith('Trace ID:')) traceId = line.split(':')[1].trim();
    if (line.startsWith('Framework:')) framework = line.split(':')[1].trim();
    if (line === 'Overall: FAIL') overallPassed = false;

    const metricMatch = line.match(/^(\w+)\s+([\d.]+)\s+(PASS|FAIL)$/);
    if (metricMatch) {
      const reason = lines[i + 1]?.trim().startsWith('Reason:')
        ? lines[i + 1].replace('Reason:', '').trim()
        : '';
      metrics.push({
        metric: metricMatch[1],
        score: parseFloat(metricMatch[2]),
        passed: metricMatch[3] === 'PASS',
        reason,
      });
    }
  }

  return { traceId, framework, metrics, passed: overallPassed };
}

export function run(tracePath: string, options: RunOptions): EvalResult {
  const binary = findBinary();
  const args = [
    'run',
    '--trace', tracePath,
    '--metrics', options.metrics.join(','),
    '--threshold', String(options.threshold ?? 0.7),
  ];

  if (options.mock) args.push('--mock');
  if (options.rubric) { args.push('--rubric'); args.push(options.rubric); }

  const env = { ...process.env };
  if (options.apiKey) env.ANTHROPIC_API_KEY = options.apiKey;

  const result = spawnSync(binary, args, { encoding: 'utf8', env });
  if (result.error) throw result.error;

  return parseOutput(result.stdout);
}

export function demo(): EvalResult {
  const trace: Trace = {
    evalforge_version: '0.1',
    trace_id: 'demo-js-001',
    timestamp: new Date().toISOString(),
    metadata: {
      framework: 'openai-agents',
      model: 'gpt-4o',
      agent_name: 'demo-agent',
      duration_ms: 1200,
      total_tokens: 450,
    },
    input: {
      user: 'What is the capital of Australia?',
      system: 'You are a helpful assistant.',
    },
    steps: [
      { step_id: 1, type: 'thought', content: 'I know this — Canberra.' },
      {
        step_id: 2,
        type: 'tool_call',
        tool: 'web_search',
        input: { query: 'capital of Australia' },
        output: { result: 'Canberra is the capital of Australia.' },
        duration_ms: 400,
      },
    ],
    output: { answer: 'The capital of Australia is Canberra.' },
    eval_hints: {
      expected_tools: ['web_search'],
      expected_answer: 'Canberra',
      context_documents: [],
    },
  };

  const tmp = path.join(os.tmpdir(), `evalforge_demo_${Date.now()}.json`);
  fs.writeFileSync(tmp, JSON.stringify(trace));

  return run(tmp, { metrics: ['faithfulness'], mock: true });
}
