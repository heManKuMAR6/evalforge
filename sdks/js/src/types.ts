export interface MetricResult {
  metric: string;
  score: number;
  passed: boolean;
  reason: string;
}

export interface EvalResult {
  traceId: string;
  framework: string;
  metrics: MetricResult[];
  passed: boolean;
}

export interface RunOptions {
  metrics: string[];
  threshold?: number;
  mock?: boolean;
  apiKey?: string;
  rubric?: string;
}

export interface Trace {
  evalforge_version: string;
  trace_id: string;
  timestamp: string;
  metadata: {
    framework: string;
    model: string;
    agent_name: string;
    duration_ms: number;
    total_tokens: number;
  };
  input: {
    user: string;
    system: string;
  };
  steps: Array<{
    step_id: number;
    type: string;
    content?: string;
    tool?: string;
    input?: Record<string, unknown>;
    output?: Record<string, unknown>;
    duration_ms?: number;
  }>;
  output: {
    answer: string;
    finish_reason?: string;
  };
  eval_hints: {
    expected_tools: string[];
    expected_answer: string | null;
    context_documents: string[];
  };
}
