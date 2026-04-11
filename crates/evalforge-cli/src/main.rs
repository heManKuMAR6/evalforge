use chrono::Utc;
use clap::{Parser, Subcommand};
use evalforge_core::metrics::answer_relevance::{
    extract_answer_relevance_input, score_answer_relevance,
};
use evalforge_core::metrics::faithfulness::{extract_faithfulness_input, score_faithfulness};
use evalforge_core::metrics::context_precision::{
    extract_context_precision_input, score_context_precision,
};
use evalforge_core::metrics::g_eval::{extract_g_eval_input, score_g_eval};
use evalforge_core::metrics::goal_completion::{
    extract_goal_completion_input, score_goal_completion,
};
use evalforge_core::metrics::hallucination::{extract_hallucination_input, score_hallucination};
use evalforge_core::metrics::tool_accuracy::{
    extract_tool_accuracy_input, score_tool_accuracy, ToolAccuracyResult,
};
use evalforge_core::metrics::code_correctness::{
    extract_code_correctness_input, score_code_correctness,
};
use evalforge_core::metrics::code_quality::{extract_code_quality_input, score_code_quality};
use evalforge_core::metrics::code_security::{extract_code_security_input, score_code_security};
use evalforge_core::trace::load_trace;

#[derive(Parser)]
#[command(name = "evalforge", version = "0.9.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load and score a trace file
    Run {
        /// Path to the trace JSON file
        #[arg(long)]
        trace: String,

        /// Comma-separated list of metrics to run (e.g. faithfulness)
        #[arg(long)]
        metrics: Option<String>,

        /// Pass/fail threshold for all metrics (default: 0.7)
        #[arg(long, default_value_t = 0.7)]
        threshold: f64,

        /// Return a fake score without calling the API
        #[arg(long, default_value_t = false)]
        mock: bool,

        /// Custom evaluation rubric for g_eval metric
        #[arg(long)]
        rubric: Option<String>,

        /// Save results to a JSON file
        #[arg(long)]
        output: Option<String>,
    },

    /// Evaluate all trace files in a directory
    Batch {
        /// Path to directory containing trace JSON files
        #[arg(long)]
        traces: String,

        /// Comma-separated list of metrics to run
        #[arg(long)]
        metrics: String,

        /// Pass/fail threshold for all metrics (default: 0.7)
        #[arg(long, default_value_t = 0.7)]
        threshold: f64,

        /// Return fake scores without calling the API
        #[arg(long, default_value_t = false)]
        mock: bool,

        /// Directory to save individual result JSON files
        #[arg(long)]
        output: Option<String>,

        /// Custom evaluation rubric for g_eval metric
        #[arg(long)]
        rubric: Option<String>,
    },

    /// Compare judge scores against human labels to calibrate thresholds
    Calibrate {
        /// Path to directory containing trace JSON files
        #[arg(long)]
        traces: String,

        /// Path to labels JSON file
        #[arg(long)]
        labels: String,

        /// Comma-separated list of metrics to calibrate
        #[arg(long)]
        metrics: String,

        /// Skip real API calls and use mock scores
        #[arg(long, default_value_t = false)]
        mock: bool,
    },

    /// Compare before/after eval result directories to detect regressions or improvements
    Compare {
        /// Path to directory containing before JSON result files
        #[arg(long)]
        before: String,

        /// Path to directory containing after JSON result files
        #[arg(long)]
        after: String,

        /// Comma-separated metrics to compare (default: all found)
        #[arg(long)]
        metrics: Option<String>,
    },

    /// Analyze score trends across sequential eval run outputs
    Trend {
        /// Path to directory containing JSON run output files
        #[arg(long)]
        history: String,

        /// Comma-separated list of metrics to analyze
        #[arg(long)]
        metrics: String,

        /// Number of most recent runs to include in the window
        #[arg(long, default_value_t = 10)]
        window: u32,

        /// Exit with code 1 if regression is detected
        #[arg(long, default_value_t = false)]
        exit_on_regression: bool,
    },

    /// Generate a self-contained HTML report from batch result JSON files
    Report {
        /// Directory containing result JSON files from batch --output
        #[arg(long)]
        results: String,

        /// Path to save the HTML file (e.g. report.html)
        #[arg(long)]
        output: String,

        /// Title for the report
        #[arg(long, default_value = "EvalForge Report")]
        title: String,
    },

    /// Compare multiple models on the same set of traces
    Models {
        /// Path to directory containing trace JSON files
        #[arg(long)]
        traces: String,

        /// Comma-separated list of metrics to evaluate
        #[arg(long)]
        metrics: String,

        /// Comma-separated list of model names to compare
        #[arg(long)]
        models: String,

        /// Use mock scores without calling the API
        #[arg(long, default_value_t = false)]
        mock: bool,
    },

    /// Test whether an agent reliably uses a specific skill/tool
    Skills {
        #[command(subcommand)]
        command: SkillsCommands,
    },
}

#[derive(Subcommand)]
enum SkillsCommands {
    /// Test skill invocation accuracy, argument correctness, and result utilization
    Test {
        /// Name of the skill/tool to test (e.g. web_search)
        #[arg(long)]
        skill: String,

        /// Path to directory containing trace JSON files
        #[arg(long)]
        traces: String,

        /// Use mock mode (reads real trace data)
        #[arg(long, default_value_t = false)]
        mock: bool,
    },
}

struct MetricScore {
    score: f64,
    pass: bool,
    reason: String,
    rubric: Option<String>,
    method: &'static str,
    judge_model: &'static str,
}

impl From<ToolAccuracyResult> for MetricScore {
    fn from(r: ToolAccuracyResult) -> Self {
        MetricScore {
            score: r.score,
            pass: r.pass,
            reason: r.reason,
            rubric: None,
            method: "deterministic",
            judge_model: "none",
        }
    }
}

/// Compute slope via ordinary least squares on xs=[0,1,...,n-1].
/// Returns None if fewer than 2 points or denominator is zero.
fn linear_slope(ys: &[f64]) -> Option<f64> {
    let n = ys.len();
    if n < 2 {
        return None;
    }
    let n_f = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = ys.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();
    let denom = n_f * sum_x2 - sum_x.powi(2);
    if denom == 0.0 {
        return Some(0.0);
    }
    Some((n_f * sum_xy - sum_x * sum_y) / denom)
}

#[derive(Debug, PartialEq)]
enum Agreement {
    Agree,
    TooGenerous,
    TooHarsh,
}

fn calibrate_agreement(judge_score: f64, human_score: f64) -> Agreement {
    if judge_score > human_score + 0.1 {
        Agreement::TooGenerous
    } else if judge_score < human_score - 0.1 {
        Agreement::TooHarsh
    } else {
        Agreement::Agree
    }
}

/// Score a single metric on a trace. Returns None for unknown metrics.
fn score_metric(
    t: &evalforge_core::trace::Trace,
    name: &str,
    mock: bool,
    api_key: &str,
    threshold: f64,
    rubric: Option<&str>,
) -> Option<MetricScore> {
    match name {
        "faithfulness" => {
            let input = extract_faithfulness_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.91,
                    pass: true,
                    reason: "Mock score — skipping live API call".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_faithfulness(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring faithfulness: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "tool_accuracy" => {
            let input = extract_tool_accuracy_input(t);
            Some(if mock {
                MetricScore {
                    score: 1.0,
                    pass: true,
                    reason: "Mock score — all expected tools used".to_string(),
                    rubric: None,
                    method: "deterministic",
                    judge_model: "none",
                }
            } else {
                score_tool_accuracy(&input, threshold).into()
            })
        }
        "goal_completion" => {
            let input = extract_goal_completion_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.85,
                    pass: true,
                    reason: "Mock score — goal appears completed".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_goal_completion(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring goal_completion: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "hallucination" => {
            let input = extract_hallucination_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.95,
                    pass: true,
                    reason: "Mock score — no hallucinations detected".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_hallucination(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring hallucination: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "g_eval" => {
            let rubric_str = match rubric {
                Some(r) => r,
                None => {
                    eprintln!("Error: --rubric is required when using g_eval metric");
                    std::process::exit(1);
                }
            };
            Some(if mock {
                MetricScore {
                    score: 0.88,
                    pass: true,
                    reason: "Mock score — response meets rubric criteria".to_string(),
                    rubric: Some(rubric_str.to_string()),
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                let input = extract_g_eval_input(t, rubric_str);
                match score_g_eval(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: Some(r.rubric),
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring g_eval: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "context_precision" => {
            let input = extract_context_precision_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.80,
                    pass: true,
                    reason: "Mock score — all retrieved context was relevant".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_context_precision(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring context_precision: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "answer_relevance" => {
            let input = extract_answer_relevance_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.95,
                    pass: true,
                    reason: "Mock score — answer directly addresses the question".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_answer_relevance(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring answer_relevance: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "code_correctness" => {
            let input = extract_code_correctness_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.85,
                    pass: true,
                    reason: "Mock — code appears correct".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_code_correctness(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring code_correctness: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "code_quality" => {
            let input = extract_code_quality_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.80,
                    pass: true,
                    reason: "Mock — code quality is good".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_code_quality(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring code_quality: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        "code_security" => {
            let input = extract_code_security_input(t);
            Some(if mock {
                MetricScore {
                    score: 0.95,
                    pass: true,
                    reason: "Mock — no security issues found".to_string(),
                    rubric: None,
                    method: "llm_judge",
                    judge_model: "claude-haiku-4-5-20251001",
                }
            } else {
                match score_code_security(&input, api_key, threshold) {
                    Ok(r) => MetricScore {
                        score: r.score,
                        pass: r.pass,
                        reason: r.reason,
                        rubric: None,
                        method: "llm_judge",
                        judge_model: "claude-haiku-4-5-20251001",
                    },
                    Err(e) => {
                        eprintln!("Error scoring code_security: {}", e);
                        std::process::exit(1);
                    }
                }
            })
        }
        other => {
            eprintln!("Warning: unknown metric '{}', skipping.", other);
            None
        }
    }
}

/// Returns (passed, total, pass_pct) for a slice of per-trace overall pass flags.
fn batch_outcome(pass_flags: &[bool]) -> (usize, usize, f64) {
    let total = pass_flags.len();
    let passed = pass_flags.iter().filter(|&&p| p).count();
    let rate = if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64 * 100.0
    };
    (passed, total, rate)
}

/// Compute pass rate (0–100) from a slice of overall-pass booleans.
fn report_pass_rate(flags: &[bool]) -> f64 {
    if flags.is_empty() {
        return 0.0;
    }
    let passed = flags.iter().filter(|&&p| p).count();
    passed as f64 / flags.len() as f64 * 100.0
}

/// Compute the arithmetic mean of a slice of scores.
fn report_metric_average(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Returns estimated cost per run for a named model.
fn model_cost(model: &str) -> f64 {
    match model {
        "gpt-4o" => 0.023,
        "gpt-4o-mini" => 0.001,
        "claude-sonnet-4-6" | "claude-sonnet" => 0.019,
        "claude-haiku-4-5-20251001" | "claude-haiku" => 0.0008,
        _ => 0.005,
    }
}

/// Returns the mock base score for a given metric name.
fn mock_base_score(metric: &str) -> f64 {
    match metric {
        "faithfulness" => 0.91,
        "tool_accuracy" => 1.0,
        "goal_completion" => 0.85,
        "hallucination" => 0.95,
        "g_eval" => 0.88,
        "context_precision" => 0.80,
        "answer_relevance" => 0.95,
        "code_correctness" => 0.85,
        "code_quality" => 0.80,
        "code_security" => 0.95,
        _ => 0.80,
    }
}

/// Applies a per-model score offset to a base score, clamped to [0, 1].
fn mock_score_for_model(base_score: f64, model: &str) -> f64 {
    let adjusted = match model {
        "gpt-4o-mini" => base_score - 0.19,
        "gpt-4o" => base_score,
        "claude-haiku-4-5-20251001" | "claude-haiku" => base_score - 0.03,
        "claude-sonnet-4-6" | "claude-sonnet" => base_score + 0.02,
        _ => base_score - 0.10,
    };
    adjusted.clamp(0.0, 1.0)
}

/// Returns the index of the model with the highest average score.
fn best_quality_idx(avg_scores: &[f64]) -> usize {
    avg_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Returns the index of the model with the best score-to-cost ratio.
fn best_value_idx(avg_scores: &[f64], costs: &[f64]) -> usize {
    avg_scores
        .iter()
        .zip(costs.iter())
        .enumerate()
        .max_by(|(_, (s1, c1)), (_, (s2, c2))| {
            let r1 = if **c1 == 0.0 { 0.0 } else { *s1 / *c1 };
            let r2 = if **c2 == 0.0 { 0.0 } else { *s2 / *c2 };
            r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Returns true if `skill` was invoked in any step of the trace.
fn skill_invoked(trace: &evalforge_core::trace::Trace, skill: &str) -> bool {
    trace
        .steps
        .iter()
        .any(|s| s.step_type == "tool_call" && s.tool.as_deref() == Some(skill))
}

/// Returns true if every tool_call for `skill` has non-empty input args.
fn skill_args_correct(trace: &evalforge_core::trace::Trace, skill: &str) -> bool {
    let calls: Vec<_> = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call" && s.tool.as_deref() == Some(skill))
        .collect();
    if calls.is_empty() {
        return false;
    }
    calls.iter().all(|s| {
        s.input
            .as_ref()
            .map(|v| !v.is_null() && v != &serde_json::Value::Object(Default::default()))
            .unwrap_or(false)
    })
}

/// Returns true if the final answer references words found in any tool output for `skill`.
fn skill_result_utilized(trace: &evalforge_core::trace::Trace, skill: &str) -> bool {
    let answer_lower = trace.output.answer.to_lowercase();
    trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call" && s.tool.as_deref() == Some(skill))
        .filter_map(|s| s.output.as_ref())
        .any(|output| {
            let output_str = output.to_string().to_lowercase();
            output_str
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 5)
                .any(|word| answer_lower.contains(word))
        })
}

/// Overall skill score = average of the three component rates.
fn skill_score(invocation: f64, args: f64, utilization: f64) -> f64 {
    (invocation + args + utilization) / 3.0
}

fn delta_symbol(delta: f64) -> &'static str {
    if delta > 0.05 {
        "✅"
    } else if delta < -0.05 {
        "⚠"
    } else {
        "→"
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            trace,
            metrics,
            threshold,
            mock,
            rubric,
            output,
        } => {
            let t = match load_trace(&trace) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: failed to load trace from '{}': {}", trace, e);
                    std::process::exit(1);
                }
            };

            println!("EvalForge v0.9.0");
            println!("─────────────────────────────");
            println!("Trace ID:   {}", t.trace_id);
            println!("Framework:  {}", t.metadata.framework);
            println!("Model:      {}", t.metadata.model);
            println!("Agent:      {}", t.metadata.agent_name);
            println!("Steps:      {}", t.steps.len());
            println!("Duration:   {}ms", t.metadata.duration_ms);
            println!("Tokens:     {}", t.metadata.total_tokens);
            println!("─────────────────────────────");
            println!("Status: Trace loaded successfully. Ready to score.");

            let Some(metrics_str) = metrics else {
                return;
            };

            let metric_names: Vec<&str> = metrics_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if metric_names.is_empty() {
                return;
            }

            // Resolve API key once (not needed for mock)
            let api_key = if !mock {
                match std::env::var("ANTHROPIC_API_KEY") {
                    Ok(k) => k,
                    Err(_) => {
                        eprintln!(
                            "Error: ANTHROPIC_API_KEY not set. Use --mock to test without an API key."
                        );
                        std::process::exit(1);
                    }
                }
            } else {
                String::new()
            };

            let mut results: Vec<(&str, MetricScore)> = Vec::new();

            for name in &metric_names {
                match *name {
                    "faithfulness" => {
                        let input = extract_faithfulness_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.91,
                                pass: true,
                                reason: "Mock score — skipping live API call".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_faithfulness(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring faithfulness: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "tool_accuracy" => {
                        let input = extract_tool_accuracy_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 1.0,
                                pass: true,
                                reason: "Mock score — all expected tools used".to_string(),
                                rubric: None,
                                method: "deterministic",
                                judge_model: "none",
                            }
                        } else {
                            score_tool_accuracy(&input, threshold).into()
                        };
                        results.push((name, scored));
                    }
                    "goal_completion" => {
                        let input = extract_goal_completion_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.85,
                                pass: true,
                                reason: "Mock score — goal appears completed".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_goal_completion(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring goal_completion: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "hallucination" => {
                        let input = extract_hallucination_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.95,
                                pass: true,
                                reason: "Mock score — no hallucinations detected".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_hallucination(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring hallucination: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "g_eval" => {
                        let rubric_str = match &rubric {
                            Some(r) => r.as_str(),
                            None => {
                                eprintln!(
                                    "Error: --rubric is required when using g_eval metric"
                                );
                                std::process::exit(1);
                            }
                        };
                        let scored = if mock {
                            MetricScore {
                                score: 0.88,
                                pass: true,
                                reason: "Mock score — response meets rubric criteria".to_string(),
                                rubric: Some(rubric_str.to_string()),
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            let input = extract_g_eval_input(&t, rubric_str);
                            match score_g_eval(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: Some(r.rubric),
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring g_eval: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "context_precision" => {
                        let input = extract_context_precision_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.80,
                                pass: true,
                                reason: "Mock score — all retrieved context was relevant"
                                    .to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_context_precision(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring context_precision: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "answer_relevance" => {
                        let input = extract_answer_relevance_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.95,
                                pass: true,
                                reason: "Mock score — answer directly addresses the question"
                                    .to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_answer_relevance(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring answer_relevance: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "code_correctness" => {
                        let input = extract_code_correctness_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.85,
                                pass: true,
                                reason: "Mock — code appears correct".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_code_correctness(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring code_correctness: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "code_quality" => {
                        let input = extract_code_quality_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.80,
                                pass: true,
                                reason: "Mock — code quality is good".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_code_quality(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring code_quality: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    "code_security" => {
                        let input = extract_code_security_input(&t);
                        let scored = if mock {
                            MetricScore {
                                score: 0.95,
                                pass: true,
                                reason: "Mock — no security issues found".to_string(),
                                rubric: None,
                                method: "llm_judge",
                                judge_model: "claude-haiku-4-5-20251001",
                            }
                        } else {
                            match score_code_security(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
                                    method: "llm_judge",
                                    judge_model: "claude-haiku-4-5-20251001",
                                },
                                Err(e) => {
                                    eprintln!("Error scoring code_security: {}", e);
                                    std::process::exit(1);
                                }
                            }
                        };
                        results.push((name, scored));
                    }
                    other => {
                        eprintln!("Warning: unknown metric '{}', skipping.", other);
                    }
                }
            }

            if results.is_empty() {
                return;
            }

            println!("─────────────────────────────");
            println!("Scoring Results");
            println!("─────────────────────────────");
            let all_pass = results.iter().all(|(_, r)| r.pass);
            for (name, r) in &results {
                let status = if r.pass { "PASS" } else { "FAIL" };
                println!("{:<16} {:.2}   {}", name, r.score, status);
                println!("Reason: {}", r.reason);
                if let Some(rb) = &r.rubric {
                    println!("Rubric: \"{}\"", rb);
                }
            }
            println!("─────────────────────────────");
            if all_pass {
                println!("Overall: PASS");
            } else {
                println!("Overall: FAIL");
            }

            if let Some(output_path) = &output {
                let run_timestamp = Utc::now().to_rfc3339();
                let metrics_json: Vec<serde_json::Value> = results
                    .iter()
                    .map(|(name, r)| {
                        serde_json::json!({
                            "metric": name,
                            "score": r.score,
                            "passed": r.pass,
                            "reason": r.reason,
                            "method": r.method,
                            "judge_model": r.judge_model,
                            "threshold": threshold,
                            "timestamp": run_timestamp,
                        })
                    })
                    .collect();

                let output_json = serde_json::json!({
                    "evalforge_version": "0.6.0",
                    "trace_id": t.trace_id,
                    "framework": t.metadata.framework,
                    "timestamp": run_timestamp,
                    "overall_passed": all_pass,
                    "metrics": metrics_json,
                });

                let path = std::path::Path::new(output_path);
                if let Some(parent) = path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        eprintln!("Error: failed to create output directory: {}", e);
                        std::process::exit(1);
                    }
                }

                let pretty = serde_json::to_string_pretty(&output_json)
                    .expect("failed to serialize results");
                if let Err(e) = std::fs::write(path, pretty) {
                    eprintln!("Error: failed to write output file: {}", e);
                    std::process::exit(1);
                }

                println!("Results saved to: {}", output_path);
            }

            if !all_pass {
                std::process::exit(1);
            }
        }

        Commands::Batch {
            traces,
            metrics,
            threshold,
            mock,
            output,
            rubric,
        } => {
            let dir = std::path::Path::new(&traces);
            let mut trace_paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect(),
                Err(e) => {
                    eprintln!("Error: cannot read traces directory '{}': {}", traces, e);
                    std::process::exit(1);
                }
            };
            trace_paths.sort();

            let metric_names: Vec<&str> = metrics
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let api_key = if !mock {
                match std::env::var("ANTHROPIC_API_KEY") {
                    Ok(k) => k,
                    Err(_) => {
                        eprintln!(
                            "Error: ANTHROPIC_API_KEY not set. Use --mock to test without an API key."
                        );
                        std::process::exit(1);
                    }
                }
            } else {
                String::new()
            };

            println!("EvalForge — Batch Evaluation");
            println!("─────────────────────────────");
            println!("Traces:  {} found", trace_paths.len());
            println!("Metrics: {}", metric_names.join(", "));
            println!("─────────────────────────────");

            // Print header row
            let name_col = 24usize;
            let metric_col = 14usize;
            print!("{:<width$}", "Trace", width = name_col);
            for m in &metric_names {
                print!("{:<width$}", m, width = metric_col);
            }
            println!("Overall");

            // Score each trace; collect overall pass/fail per trace
            let mut rows: Vec<bool> = Vec::new();

            for path in &trace_paths {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let t = match load_trace(path.to_str().unwrap_or("")) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Warning: skipping '{}': {}", filename, e);
                        continue;
                    }
                };

                let rubric_ref = rubric.as_deref();
                let mut metric_scores: Vec<MetricScore> = Vec::new();
                for name in &metric_names {
                    if let Some(s) = score_metric(&t, name, mock, &api_key, threshold, rubric_ref) {
                        metric_scores.push(s);
                    }
                }

                let overall = metric_scores.iter().all(|s| s.pass);

                // Print result row
                print!("{:<width$}", filename, width = name_col);
                for s in &metric_scores {
                    let cell = format!("{:.2} {}", s.score, if s.pass { "PASS" } else { "FAIL" });
                    print!("{:<width$}", cell, width = metric_col);
                }
                println!("{}", if overall { "PASS" } else { "FAIL" });

                // Save to output dir if requested
                if let Some(ref out_dir) = output {
                    let run_timestamp = Utc::now().to_rfc3339();
                    let metrics_json: Vec<serde_json::Value> = metric_scores
                        .iter()
                        .zip(metric_names.iter())
                        .map(|(r, name)| {
                            serde_json::json!({
                                "metric": name,
                                "score": r.score,
                                "passed": r.pass,
                                "reason": r.reason,
                                "method": r.method,
                                "judge_model": r.judge_model,
                                "threshold": threshold,
                                "timestamp": run_timestamp,
                            })
                        })
                        .collect();

                    let out_json = serde_json::json!({
                        "evalforge_version": "0.8.0",
                        "trace_id": t.trace_id,
                        "framework": t.metadata.framework,
                        "timestamp": run_timestamp,
                        "overall_passed": overall,
                        "metrics": metrics_json,
                    });

                    let out_path = std::path::Path::new(out_dir).join(format!("{}.json", t.trace_id));
                    if let Some(parent) = out_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let pretty = serde_json::to_string_pretty(&out_json)
                        .expect("failed to serialize results");
                    if let Err(e) = std::fs::write(&out_path, pretty) {
                        eprintln!("Warning: failed to write '{}': {}", out_path.display(), e);
                    }
                }

                rows.push(overall);
            }

            let pass_flags: Vec<bool> = rows;
            let (passed, total, rate) = batch_outcome(&pass_flags);
            let failed = total - passed;
            let fail_rate = 100.0 - rate;

            println!("─────────────────────────────");
            println!("Summary");
            println!("─────────────────────────────");
            println!("Total:   {} traces", total);
            println!("Passed:  {} ({:.0}%)", passed, rate);
            println!("Failed:  {} ({:.0}%)", failed, fail_rate);
            println!("─────────────────────────────");

            let batch_pass = passed == total;
            if batch_pass {
                println!("Batch result: PASS");
            } else {
                println!("Batch result: FAIL");
                std::process::exit(1);
            }
        }

        Commands::Calibrate {
            traces,
            labels,
            metrics,
            mock,
        } => {
            // Load labels file
            let labels_text = match std::fs::read_to_string(&labels) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: cannot read labels file '{}': {}", labels, e);
                    std::process::exit(1);
                }
            };
            let labels_json: serde_json::Value = match serde_json::from_str(&labels_text) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: invalid labels JSON: {}", e);
                    std::process::exit(1);
                }
            };
            let label_entries = match labels_json["labels"].as_array() {
                Some(a) => a.clone(),
                None => {
                    eprintln!("Error: labels JSON must have a top-level 'labels' array.");
                    std::process::exit(1);
                }
            };

            // Load all trace files from the traces directory
            let dir = std::path::Path::new(&traces);
            let mut trace_paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect(),
                Err(e) => {
                    eprintln!("Error: cannot read traces directory '{}': {}", traces, e);
                    std::process::exit(1);
                }
            };
            trace_paths.sort();

            // Build a map of trace_id -> trace for quick lookup
            let mut trace_map: std::collections::HashMap<String, evalforge_core::trace::Trace> =
                std::collections::HashMap::new();
            for p in &trace_paths {
                if let Ok(t) = load_trace(p.to_str().unwrap_or("")) {
                    trace_map.insert(t.trace_id.clone(), t);
                }
            }

            let metric_names: Vec<&str> = metrics
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            for metric_name in &metric_names {
                // Collect labels for this metric
                let relevant: Vec<&serde_json::Value> = label_entries
                    .iter()
                    .filter(|e| e["metric"].as_str() == Some(metric_name))
                    .collect();

                if relevant.is_empty() {
                    eprintln!(
                        "Warning: no labels found for metric '{}', skipping.",
                        metric_name
                    );
                    continue;
                }

                let mut agree_count: usize = 0;
                let mut generous_count: usize = 0;
                let mut harsh_count: usize = 0;
                let mut human_scores: Vec<f64> = Vec::new();
                let mut judge_scores: Vec<f64> = Vec::new();

                for entry in &relevant {
                    let trace_id = entry["trace_id"].as_str().unwrap_or("");
                    let human_score = match entry["human_score"].as_f64() {
                        Some(s) => s,
                        None => continue,
                    };
                    human_scores.push(human_score);

                    // Get judge score (mock or real)
                    let judge_score = if mock {
                        // Use fixed mock score per metric
                        match *metric_name {
                            "faithfulness" => 0.91,
                            "tool_accuracy" => 1.0,
                            "goal_completion" => 0.85,
                            "hallucination" => 0.95,
                            "g_eval" => 0.88,
                            "context_precision" => 0.80,
                            "answer_relevance" => 0.95,
                            _ => 0.80,
                        }
                    } else {
                        match trace_map.get(trace_id) {
                            Some(t) => match *metric_name {
                                "faithfulness" => {
                                    let input = extract_faithfulness_input(t);
                                    let api_key =
                                        std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                                    score_faithfulness(&input, &api_key, 0.7)
                                        .map(|r| r.score)
                                        .unwrap_or(0.0)
                                }
                                _ => 0.0,
                            },
                            None => {
                                eprintln!(
                                    "Warning: trace '{}' not found in traces directory.",
                                    trace_id
                                );
                                0.0
                            }
                        }
                    };
                    judge_scores.push(judge_score);

                    match calibrate_agreement(judge_score, human_score) {
                        Agreement::Agree => agree_count += 1,
                        Agreement::TooGenerous => generous_count += 1,
                        Agreement::TooHarsh => harsh_count += 1,
                    }
                }

                let total = relevant.len();
                let avg_human =
                    human_scores.iter().sum::<f64>() / human_scores.len().max(1) as f64;
                let avg_judge =
                    judge_scores.iter().sum::<f64>() / judge_scores.len().max(1) as f64;
                let delta = avg_judge - avg_human;
                let recommended_threshold = avg_human;

                let agree_pct = (agree_count as f64 / total as f64 * 100.0).round() as u32;
                let generous_pct = (generous_count as f64 / total as f64 * 100.0).round() as u32;
                let harsh_pct = (harsh_count as f64 / total as f64 * 100.0).round() as u32;

                println!("EvalForge — Calibration Report");
                println!("─────────────────────────────");
                println!("Metric:           {}", metric_name);
                println!("Traces evaluated: {}", total);
                println!("─────────────────────────────");
                println!(
                    "Agreement:        {}/{} ({}%)",
                    agree_count, total, agree_pct
                );
                println!(
                    "Too generous:     {}/{} ({}%)",
                    generous_count, total, generous_pct
                );
                println!(
                    "Too harsh:        {}/{} ({}%)",
                    harsh_count, total, harsh_pct
                );
                println!("─────────────────────────────");
                println!("Avg human score:  {:.2}", avg_human);
                println!("Avg judge score:  {:.2}", avg_judge);
                let sign = if delta >= 0.0 { "+" } else { "" };
                println!("Score delta:      {}{:.2}", sign, delta);
                println!("─────────────────────────────");
                println!("Recommended threshold: {:.2}", recommended_threshold);
                println!("─────────────────────────────");
            }
        }

        Commands::Compare {
            before,
            after,
            metrics,
        } => {
            // Helper to load all JSON files from a directory and return (trace_id, metric_name -> score) maps
            fn load_results_dir(
                dir_path: &str,
            ) -> Result<
                std::collections::HashMap<String, std::collections::HashMap<String, f64>>,
                String,
            > {
                let dir = std::path::Path::new(dir_path);
                let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                    Ok(entries) => entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                        .collect(),
                    Err(e) => return Err(format!("cannot read directory '{}': {}", dir_path, e)),
                };
                paths.sort();

                let mut map: std::collections::HashMap<
                    String,
                    std::collections::HashMap<String, f64>,
                > = std::collections::HashMap::new();
                for path in &paths {
                    let text = match std::fs::read_to_string(path) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    let data: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let trace_id = match data["trace_id"].as_str() {
                        Some(id) => id.to_string(),
                        None => continue,
                    };
                    let mut metric_map: std::collections::HashMap<String, f64> =
                        std::collections::HashMap::new();
                    if let Some(arr) = data["metrics"].as_array() {
                        for entry in arr {
                            if let (Some(name), Some(score)) =
                                (entry["metric"].as_str(), entry["score"].as_f64())
                            {
                                metric_map.insert(name.to_string(), score);
                            }
                        }
                    }
                    map.insert(trace_id, metric_map);
                }
                Ok(map)
            }

            let before_map = match load_results_dir(&before) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };
            let after_map = match load_results_dir(&after) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            };

            // Find matched trace IDs
            let mut matched_ids: Vec<String> = before_map
                .keys()
                .filter(|id| after_map.contains_key(*id))
                .cloned()
                .collect();
            matched_ids.sort();

            // Gather all metric names across matched traces (or use --metrics filter)
            let all_metric_names: Vec<String> = {
                let mut names: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for id in &matched_ids {
                    if let Some(bm) = before_map.get(id) {
                        names.extend(bm.keys().cloned());
                    }
                    if let Some(am) = after_map.get(id) {
                        names.extend(am.keys().cloned());
                    }
                }
                let mut sorted: Vec<String> = names.into_iter().collect();
                sorted.sort();
                sorted
            };

            let metric_names: Vec<String> = match &metrics {
                Some(s) => s
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect(),
                None => all_metric_names,
            };

            println!("EvalForge — Comparison Report");
            println!("─────────────────────────────");
            println!("Before:  {}  ({} traces)", before, before_map.len());
            println!("After:   {}  ({} traces)", after, after_map.len());
            println!("Matched: {} traces", matched_ids.len());
            println!("─────────────────────────────");
            println!("{:<18} {:<9} {:<9} {}", "Metric", "Before", "After", "Delta");

            let mut improved = 0usize;
            let mut unchanged = 0usize;
            let mut regressed = 0usize;

            for metric in &metric_names {
                // Average before/after scores across all matched traces
                let mut before_scores: Vec<f64> = Vec::new();
                let mut after_scores: Vec<f64> = Vec::new();
                for id in &matched_ids {
                    if let Some(s) = before_map.get(id).and_then(|m| m.get(metric)) {
                        before_scores.push(*s);
                    }
                    if let Some(s) = after_map.get(id).and_then(|m| m.get(metric)) {
                        after_scores.push(*s);
                    }
                }
                if before_scores.is_empty() && after_scores.is_empty() {
                    continue;
                }
                let avg_before = before_scores.iter().sum::<f64>()
                    / before_scores.len().max(1) as f64;
                let avg_after =
                    after_scores.iter().sum::<f64>() / after_scores.len().max(1) as f64;
                let delta = avg_after - avg_before;
                let symbol = delta_symbol(delta);
                let sign = if delta >= 0.0 { "+" } else { "" };
                println!(
                    "{:<18} {:<9.2} {:<9.2} {}{:.2} {}",
                    metric, avg_before, avg_after, sign, delta, symbol
                );

                match symbol {
                    "✅" => improved += 1,
                    "⚠" => regressed += 1,
                    _ => unchanged += 1,
                }
            }

            println!("─────────────────────────────");
            println!("Summary");
            println!("─────────────────────────────");
            println!("Improved metrics:  {}", improved);
            println!("Unchanged metrics: {}", unchanged);
            println!("Regressed metrics: {}", regressed);
            println!("─────────────────────────────");
            if regressed > 0 {
                println!("Overall: REGRESSION ⚠");
                std::process::exit(1);
            } else if improved > 0 {
                println!("Overall: IMPROVEMENT ✅");
            } else {
                println!("Overall: UNCHANGED →");
            }
        }

        Commands::Trend {
            history,
            metrics,
            window,
            exit_on_regression,
        } => {
            // Collect and sort JSON files in the history directory
            let dir = std::path::Path::new(&history);
            let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect(),
                Err(e) => {
                    eprintln!("Error: cannot read history directory '{}': {}", history, e);
                    std::process::exit(1);
                }
            };
            paths.sort();

            let total_files = paths.len();
            if total_files < 2 {
                eprintln!(
                    "Error: need at least 2 run files in '{}', found {}.",
                    history, total_files
                );
                std::process::exit(1);
            }

            // Apply window
            let window_usize = window as usize;
            if paths.len() > window_usize {
                paths = paths[paths.len() - window_usize..].to_vec();
            }

            let metric_names: Vec<&str> = metrics
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            // Build per-metric score lists
            let mut metric_scores: std::collections::HashMap<&str, Vec<f64>> =
                metric_names.iter().map(|&m| (m, Vec::new())).collect();

            for path in &paths {
                let text = match std::fs::read_to_string(path) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let data: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(arr) = data["metrics"].as_array() {
                    for entry in arr {
                        let name = entry["metric"].as_str().unwrap_or("");
                        if let Some(scores) = metric_scores.get_mut(name) {
                            if let Some(s) = entry["score"].as_f64() {
                                scores.push(s);
                            }
                        }
                    }
                }
            }

            println!("EvalForge — Trend Analysis");
            println!("─────────────────────────────");
            println!("History:  {}", history);
            println!("Window:   {} runs", window);
            println!("Files:    {} found", total_files);
            println!("─────────────────────────────");
            println!(
                "{:<20} {:<10} {:<12} {}",
                "Metric", "Slope", "Direction", "Regression"
            );

            // Regression threshold matches Python SDK default
            const REGRESSION_THRESHOLD: f64 = -0.02;
            let mut any_regression = false;

            for name in &metric_names {
                let scores = metric_scores.get(name).map(|v| v.as_slice()).unwrap_or(&[]);
                match linear_slope(scores) {
                    None => {
                        println!("{:<20} {:<10} {:<12} {}", name, "n/a", "n/a", "n/a");
                    }
                    Some(slope) => {
                        let direction = if slope > 0.01 {
                            "improving"
                        } else if slope < -0.01 {
                            "degrading"
                        } else {
                            "stable"
                        };
                        let regression = slope < REGRESSION_THRESHOLD;
                        if regression {
                            any_regression = true;
                        }
                        let reg_label = if regression { "YES ⚠" } else { "no" };
                        println!(
                            "{:<20} {:+.4}    {:<12} {}",
                            name, slope, direction, reg_label
                        );
                    }
                }
            }

            println!("─────────────────────────────");
            if any_regression {
                println!("Overall: REGRESSION DETECTED");
                if exit_on_regression {
                    std::process::exit(1);
                }
            } else {
                println!("Overall: STABLE");
            }
        }

        Commands::Report {
            results,
            output,
            title,
        } => {
            // Read all JSON files from the results directory
            let dir = std::path::Path::new(&results);
            let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect(),
                Err(e) => {
                    eprintln!(
                        "Error: cannot read results directory '{}': {}",
                        results, e
                    );
                    std::process::exit(1);
                }
            };
            paths.sort();

            if paths.is_empty() {
                eprintln!("Error: no JSON files found in '{}'", results);
                std::process::exit(1);
            }

            // Parse each result file into (trace_id, overall_passed, Vec<(metric, score, passed)>)
            let mut trace_data: Vec<(String, bool, Vec<(String, f64, bool)>)> = Vec::new();
            for path in &paths {
                let text = match std::fs::read_to_string(path) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let data: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let trace_id = data["trace_id"].as_str().unwrap_or("unknown").to_string();
                let overall_passed = data["overall_passed"].as_bool().unwrap_or(false);
                let mut metrics: Vec<(String, f64, bool)> = Vec::new();
                if let Some(arr) = data["metrics"].as_array() {
                    for m in arr {
                        if let (Some(name), Some(score)) =
                            (m["metric"].as_str(), m["score"].as_f64())
                        {
                            metrics.push((
                                name.to_string(),
                                score,
                                m["passed"].as_bool().unwrap_or(false),
                            ));
                        }
                    }
                }
                trace_data.push((trace_id, overall_passed, metrics));
            }

            // Summary stats
            let total = trace_data.len();
            let pass_flags: Vec<bool> = trace_data.iter().map(|(_, p, _)| *p).collect();
            let passed = pass_flags.iter().filter(|&&p| p).count();
            let failed = total - passed;
            let pass_rate = report_pass_rate(&pass_flags);

            // Per-metric average scores
            let mut metric_acc: std::collections::HashMap<String, Vec<f64>> =
                std::collections::HashMap::new();
            for (_, _, metrics) in &trace_data {
                for (name, score, _) in metrics {
                    metric_acc.entry(name.clone()).or_default().push(*score);
                }
            }
            let mut all_metrics: Vec<String> = metric_acc.keys().cloned().collect();
            all_metrics.sort();

            let timestamp = Utc::now().to_rfc3339();
            let pass_rate_str = format!("{:.1}%", pass_rate);

            // Metrics summary table rows
            let metric_rows: String = all_metrics
                .iter()
                .map(|name| {
                    let avg = report_metric_average(
                        metric_acc.get(name).map(|v| v.as_slice()).unwrap_or(&[]),
                    );
                    format!("      <tr><td>{}</td><td>{:.3}</td></tr>", name, avg)
                })
                .collect::<Vec<_>>()
                .join("\n");

            // Per-trace table
            let metric_headers: String = all_metrics
                .iter()
                .map(|name| format!("<th>{}</th>", name))
                .collect::<Vec<_>>()
                .join("");

            let trace_rows: String = trace_data
                .iter()
                .map(|(trace_id, overall_passed, metrics)| {
                    let overall_cls = if *overall_passed { "pass-badge" } else { "fail-badge" };
                    let overall_lbl = if *overall_passed { "PASS" } else { "FAIL" };
                    let cells: String = all_metrics
                        .iter()
                        .map(|name| match metrics.iter().find(|(n, _, _)| n == name) {
                            Some((_, score, passed)) => {
                                let cls = if *passed { "pass-badge" } else { "fail-badge" };
                                let lbl = if *passed { "PASS" } else { "FAIL" };
                                format!(
                                    "<td>{:.2} <span class=\"{}\">{}</span></td>",
                                    score, cls, lbl
                                )
                            }
                            None => "<td>—</td>".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    format!(
                        "      <tr><td>{}</td>{}<td><span class=\"{}\">{}</span></td></tr>",
                        trace_id, cells, overall_cls, overall_lbl
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            let html = format!(
                r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 32px; }}
h1 {{ color: #7c3aed; font-size: 2rem; margin-bottom: 8px; }}
.subtitle {{ color: #8b949e; margin-bottom: 32px; font-size: 0.9rem; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; text-align: center; }}
.card-value {{ font-size: 2rem; font-weight: 700; color: #c9d1d9; }}
.card-label {{ color: #8b949e; font-size: 0.85rem; margin-top: 4px; }}
.card.pass .card-value {{ color: #3fb950; }}
.card.fail .card-value {{ color: #f85149; }}
.card.rate .card-value {{ color: #7c3aed; }}
section {{ margin-bottom: 32px; }}
h2 {{ color: #c9d1d9; font-size: 1.1rem; margin-bottom: 12px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ background: #161b22; color: #8b949e; font-size: 0.8rem; text-transform: uppercase; padding: 10px 14px; text-align: left; }}
td {{ padding: 10px 14px; border-bottom: 1px solid #21262d; }}
tr:hover td {{ background: #161b22; }}
.pass-badge {{ color: #3fb950; font-weight: 600; }}
.fail-badge {{ color: #f85149; font-weight: 600; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Generated: {timestamp}</p>
<div class="cards">
  <div class="card"><div class="card-value">{total}</div><div class="card-label">Total Traces</div></div>
  <div class="card pass"><div class="card-value">{passed}</div><div class="card-label">Passed</div></div>
  <div class="card fail"><div class="card-value">{failed}</div><div class="card-label">Failed</div></div>
  <div class="card rate"><div class="card-value">{pass_rate_str}</div><div class="card-label">Pass Rate</div></div>
</div>
<section>
  <h2>Metrics Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Avg Score</th></tr></thead>
    <tbody>
{metric_rows}
    </tbody>
  </table>
</section>
<section>
  <h2>Per-Trace Results</h2>
  <table>
    <thead><tr><th>Trace ID</th>{metric_headers}<th>Overall</th></tr></thead>
    <tbody>
{trace_rows}
    </tbody>
  </table>
</section>
</body>
</html>"#,
                title = title,
                timestamp = timestamp,
                total = total,
                passed = passed,
                failed = failed,
                pass_rate_str = pass_rate_str,
                metric_rows = metric_rows,
                metric_headers = metric_headers,
                trace_rows = trace_rows,
            );

            if let Err(e) = std::fs::write(&output, &html) {
                eprintln!("Error: failed to write report to '{}': {}", output, e);
                std::process::exit(1);
            }

            println!("Report saved to: {}", output);
        }

        Commands::Models {
            traces,
            metrics,
            models: models_arg,
            mock,
        } => {
            let dir = std::path::Path::new(&traces);
            let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                    .collect(),
                Err(e) => {
                    eprintln!(
                        "Error: cannot read traces directory '{}': {}",
                        traces, e
                    );
                    std::process::exit(1);
                }
            };
            paths.sort();

            let metric_names: Vec<&str> = metrics
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let model_names: Vec<&str> = models_arg
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            println!("EvalForge — Model Comparison");
            println!("─────────────────────────────");
            println!("Traces:  {} found", paths.len());
            println!("Metrics: {}", metric_names.join(", "));
            println!("─────────────────────────────");

            // Column widths
            let model_col = 20usize;
            let metric_col = 16usize;
            let avg_col = 10usize;

            // Header row
            print!("{:<width$}", "Model", width = model_col);
            for m in &metric_names {
                print!("{:<width$}", m, width = metric_col);
            }
            print!("{:<width$}", "Avg Score", width = avg_col);
            println!("Cost/Run");

            // Gather per-model results
            let mut all_avg_scores: Vec<f64> = Vec::new();
            let mut all_costs: Vec<f64> = Vec::new();

            for model in &model_names {
                let cost = model_cost(model);

                let per_metric_avgs: Vec<f64> = if mock {
                    // Mock: apply model-specific offset to base scores
                    metric_names
                        .iter()
                        .map(|m| mock_score_for_model(mock_base_score(m), model))
                        .collect()
                } else {
                    // Real: score each trace, average per metric
                    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
                        Ok(k) => k,
                        Err(_) => {
                            eprintln!(
                                "Error: ANTHROPIC_API_KEY not set. Use --mock to test without an API key."
                            );
                            std::process::exit(1);
                        }
                    };
                    let mut metric_acc: Vec<Vec<f64>> =
                        metric_names.iter().map(|_| Vec::new()).collect();

                    for path in &paths {
                        let t = match load_trace(path.to_str().unwrap_or("")) {
                            Ok(t) => t,
                            Err(_) => continue,
                        };
                        for (mi, mname) in metric_names.iter().enumerate() {
                            if let Some(s) =
                                score_metric(&t, mname, false, &api_key, 0.7, None)
                            {
                                metric_acc[mi].push(s.score);
                            }
                        }
                    }

                    metric_acc
                        .iter()
                        .map(|scores| {
                            if scores.is_empty() {
                                0.0
                            } else {
                                scores.iter().sum::<f64>() / scores.len() as f64
                            }
                        })
                        .collect()
                };

                let avg = if per_metric_avgs.is_empty() {
                    0.0
                } else {
                    per_metric_avgs.iter().sum::<f64>() / per_metric_avgs.len() as f64
                };

                // Print data row
                print!("{:<width$}", model, width = model_col);
                for &s in &per_metric_avgs {
                    print!("{:<width$.2}", s, width = metric_col);
                }
                print!("{:<width$.2}", avg, width = avg_col);
                println!("${:.4}", cost);

                all_avg_scores.push(avg);
                all_costs.push(cost);
            }

            println!("─────────────────────────────");

            if model_names.is_empty() {
                return;
            }

            let best_q = best_quality_idx(&all_avg_scores);
            let best_v = best_value_idx(&all_avg_scores, &all_costs);
            let most_exp = all_costs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            println!(
                "Best quality:   {:<16} ({:.2})",
                model_names[best_q], all_avg_scores[best_q]
            );
            println!(
                "Best value:     {:<16} (${:.4}/run, {:.2} avg)",
                model_names[best_v], all_costs[best_v], all_avg_scores[best_v]
            );
            println!(
                "Most expensive: {:<16} (${:.4}/run)",
                model_names[most_exp], all_costs[most_exp]
            );
            println!("─────────────────────────────");
        }

        Commands::Skills { command } => match command {
            SkillsCommands::Test {
                skill,
                traces,
                mock: _mock,
            } => {
                let dir = std::path::Path::new(&traces);
                let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
                    Ok(entries) => entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
                        .collect(),
                    Err(e) => {
                        eprintln!(
                            "Error: cannot read traces directory '{}': {}",
                            traces, e
                        );
                        std::process::exit(1);
                    }
                };
                paths.sort();

                // Per-trace results: (filename, invoked, args_ok, utilized)
                let mut rows: Vec<(String, bool, bool, bool)> = Vec::new();

                for path in &paths {
                    let filename = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let t = match load_trace(path.to_str().unwrap_or("")) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Warning: skipping '{}': {}", filename, e);
                            continue;
                        }
                    };
                    let invoked = skill_invoked(&t, &skill);
                    let args_ok = if invoked { skill_args_correct(&t, &skill) } else { false };
                    let utilized = if invoked { skill_result_utilized(&t, &skill) } else { false };
                    rows.push((filename, invoked, args_ok, utilized));
                }

                let total = rows.len();

                // Aggregate counters
                let invoked_count = rows.iter().filter(|(_, inv, _, _)| *inv).count();
                let args_correct_count = rows.iter().filter(|(_, inv, ok, _)| *inv && *ok).count();
                let utilized_count = rows.iter().filter(|(_, inv, _, ut)| *inv && *ut).count();

                let invocation_rate = if total == 0 {
                    0.0
                } else {
                    invoked_count as f64 / total as f64
                };
                let args_rate = if invoked_count == 0 {
                    1.0
                } else {
                    args_correct_count as f64 / invoked_count as f64
                };
                let util_rate = if invoked_count == 0 {
                    1.0
                } else {
                    utilized_count as f64 / invoked_count as f64
                };
                let score = skill_score(invocation_rate, args_rate, util_rate);
                let pass = score >= 0.7;

                println!("EvalForge — Agent Skills Test");
                println!("─────────────────────────────");
                println!("Skill:     {}", skill);
                println!("Traces:    {} evaluated", total);
                println!("─────────────────────────────");
                println!(
                    "Invocation accuracy:    {}/{}   ({:.0}%)",
                    invoked_count,
                    total,
                    invocation_rate * 100.0
                );
                println!(
                    "Argument correctness:   {}/{}   ({:.0}%)",
                    args_correct_count,
                    invoked_count,
                    args_rate * 100.0
                );
                println!(
                    "Result utilization:     {}/{}   ({:.0}%)",
                    utilized_count,
                    invoked_count,
                    util_rate * 100.0
                );
                println!("─────────────────────────────");
                println!(
                    "Skill score:  {:.2}   {}",
                    score,
                    if pass { "PASS" } else { "FAIL" }
                );
                println!("─────────────────────────────");
                println!("Details:");
                for (filename, invoked, args_ok, utilized) in &rows {
                    if *invoked {
                        let args_sym = if *args_ok { "✓" } else { "✗" };
                        let util_sym = if *utilized { "✓" } else { "✗" };
                        let row_pass = *args_ok && *utilized;
                        println!(
                            "{:<24} invoked ✓   args {}   utilized {}   {}",
                            filename,
                            args_sym,
                            util_sym,
                            if row_pass { "PASS" } else { "FAIL" }
                        );
                    } else {
                        println!("{:<24} not invoked ✗                      SKIP", filename);
                    }
                }
                println!("─────────────────────────────");
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        batch_outcome, best_quality_idx, best_value_idx, calibrate_agreement, delta_symbol,
        linear_slope, model_cost, report_metric_average, report_pass_rate, skill_args_correct,
        skill_invoked, skill_result_utilized, skill_score, Agreement,
    };

    #[test]
    fn test_batch_all_pass() {
        let (passed, total, rate) = batch_outcome(&[true, true]);
        assert_eq!(passed, 2);
        assert_eq!(total, 2);
        assert!((rate - 100.0).abs() < 1e-9);
        // batch passes when all pass
        assert_eq!(passed, total);
    }

    #[test]
    fn test_batch_one_fail() {
        let (passed, total, rate) = batch_outcome(&[true, false]);
        assert_eq!(passed, 1);
        assert_eq!(total, 2);
        assert!((rate - 50.0).abs() < 1e-9);
        // batch fails when any trace fails
        assert!(passed < total);
    }

    #[test]
    fn test_batch_pass_rate() {
        let (passed, total, rate) = batch_outcome(&[true, true, true, false]);
        assert_eq!(passed, 3);
        assert_eq!(total, 4);
        assert!((rate - 75.0).abs() < 1e-9);
    }

    #[test]
    fn test_calibration_agreement() {
        // judge=0.85, human=0.80 → within 0.1 → Agree
        assert_eq!(calibrate_agreement(0.85, 0.80), Agreement::Agree);
    }

    #[test]
    fn test_calibration_too_generous() {
        // judge=0.95, human=0.80 → judge > human + 0.1 → TooGenerous
        assert_eq!(calibrate_agreement(0.95, 0.80), Agreement::TooGenerous);
    }

    #[test]
    fn test_calibration_too_harsh() {
        // judge=0.65, human=0.80 → judge < human - 0.1 → TooHarsh
        assert_eq!(calibrate_agreement(0.65, 0.80), Agreement::TooHarsh);
    }

    #[test]
    fn test_recommended_threshold() {
        // average of [0.8, 1.0] = 0.90
        let scores = vec![0.8_f64, 1.0_f64];
        let avg = scores.iter().sum::<f64>() / scores.len() as f64;
        assert!((avg - 0.90).abs() < 1e-9);
    }

    #[test]
    fn test_slope_flat() {
        let ys = vec![0.9, 0.9, 0.9, 0.9, 0.9];
        let slope = linear_slope(&ys).unwrap();
        assert!((slope - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_slope_perfect_increase() {
        // y = 0.1 * x  →  slope = 0.1
        let ys: Vec<f64> = (0..5).map(|i| i as f64 * 0.1).collect();
        let slope = linear_slope(&ys).unwrap();
        assert!((slope - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_slope_degrading() {
        // 0.91 → 0.67 over 5 steps: slope = -0.06
        let ys = vec![0.91, 0.85, 0.79, 0.73, 0.67];
        let slope = linear_slope(&ys).unwrap();
        assert!(slope < -0.02, "expected regression slope, got {}", slope);
        assert!((slope - (-0.06)).abs() < 1e-9);
    }

    #[test]
    fn test_slope_requires_two_points() {
        assert!(linear_slope(&[]).is_none());
        assert!(linear_slope(&[0.9]).is_none());
        assert!(linear_slope(&[0.9, 0.8]).is_some());
    }

    #[test]
    fn test_slope_two_points() {
        // y = [0.0, 1.0]  →  slope = 1.0
        let slope = linear_slope(&[0.0, 1.0]).unwrap();
        assert!((slope - 1.0).abs() < 1e-9);
    }

    // --- compare tests ---

    #[test]
    fn test_compare_improvement() {
        let before = 0.72_f64;
        let after = 0.91_f64;
        let delta = after - before;
        assert!((delta - 0.19).abs() < 1e-9, "delta should be +0.19, got {}", delta);
        assert_eq!(delta_symbol(delta), "✅");
    }

    #[test]
    fn test_compare_regression() {
        let before = 0.91_f64;
        let after = 0.65_f64;
        let delta = after - before;
        assert!((delta - (-0.26)).abs() < 1e-9, "delta should be -0.26, got {}", delta);
        assert_eq!(delta_symbol(delta), "⚠");
    }

    #[test]
    fn test_compare_unchanged() {
        let before = 0.85_f64;
        let after = 0.87_f64;
        let delta = after - before;
        assert!((delta - 0.02).abs() < 1e-9, "delta should be +0.02, got {}", delta);
        assert_eq!(delta_symbol(delta), "→");
    }

    #[test]
    fn test_compare_delta_symbol() {
        // improvement: delta > +0.05
        assert_eq!(delta_symbol(0.10), "✅");
        assert_eq!(delta_symbol(0.06), "✅");
        // unchanged: -0.05 <= delta <= +0.05
        assert_eq!(delta_symbol(0.05), "→");
        assert_eq!(delta_symbol(0.0), "→");
        assert_eq!(delta_symbol(-0.05), "→");
        // regression: delta < -0.05
        assert_eq!(delta_symbol(-0.06), "⚠");
        assert_eq!(delta_symbol(-0.30), "⚠");
    }

    // --- report tests ---

    #[test]
    fn test_report_pass_rate() {
        // 3 pass, 1 fail → 75%
        let flags = vec![true, true, true, false];
        let rate = report_pass_rate(&flags);
        assert!((rate - 75.0).abs() < 1e-9, "expected 75.0, got {}", rate);
    }

    #[test]
    fn test_report_metric_average() {
        // faithfulness scores 0.8 and 0.9 → average 0.85
        let scores = vec![0.8_f64, 0.9_f64];
        let avg = report_metric_average(&scores);
        assert!((avg - 0.85).abs() < 1e-9, "expected 0.85, got {}", avg);
    }

    // --- models tests ---

    #[test]
    fn test_model_cost_gpt4o() {
        assert!(
            (model_cost("gpt-4o") - 0.023).abs() < 1e-9,
            "gpt-4o cost should be 0.023"
        );
    }

    #[test]
    fn test_model_cost_haiku() {
        assert!(
            (model_cost("claude-haiku") - 0.0008).abs() < 1e-9,
            "claude-haiku cost should be 0.0008"
        );
        assert!(
            (model_cost("claude-haiku-4-5-20251001") - 0.0008).abs() < 1e-9,
            "claude-haiku-4-5-20251001 cost should be 0.0008"
        );
    }

    #[test]
    fn test_model_best_quality() {
        // claude-sonnet (idx 3) has highest avg score
        let scores = vec![0.69_f64, 0.90, 0.87, 0.92];
        assert_eq!(
            best_quality_idx(&scores),
            3,
            "claude-sonnet should be best quality"
        );
    }

    #[test]
    fn test_model_best_value() {
        // haiku (idx 2): ratio = 0.87/0.0008 = 1087.5 — beats all others
        let scores = vec![0.69_f64, 0.90, 0.87, 0.92];
        let costs = vec![0.001_f64, 0.023, 0.0008, 0.019];
        assert_eq!(
            best_value_idx(&scores, &costs),
            2,
            "claude-haiku should be best value"
        );
    }

    // --- skills tests ---

    fn make_trace(
        tool: &str,
        args: Option<serde_json::Value>,
        output: Option<serde_json::Value>,
        answer: &str,
    ) -> evalforge_core::trace::Trace {
        evalforge_core::trace::Trace {
            evalforge_version: "0.1".to_string(),
            trace_id: "test-001".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            metadata: evalforge_core::trace::Metadata {
                framework: "test".to_string(),
                model: "test".to_string(),
                agent_name: "test-agent".to_string(),
                duration_ms: 0,
                total_tokens: 0,
            },
            input: evalforge_core::trace::Input {
                user: "test question".to_string(),
                system: "".to_string(),
            },
            steps: vec![evalforge_core::trace::Step {
                step_id: 1,
                step_type: "tool_call".to_string(),
                content: None,
                tool: Some(tool.to_string()),
                input: args,
                output,
                duration_ms: Some(100),
            }],
            output: evalforge_core::trace::Output {
                answer: answer.to_string(),
                finish_reason: None,
            },
            eval_hints: evalforge_core::trace::EvalHints {
                expected_tools: vec![],
                expected_answer: None,
                context_documents: vec![],
            },
        }
    }

    #[test]
    fn test_skill_invocation_found() {
        let trace = make_trace(
            "web_search",
            Some(serde_json::json!({"query": "test"})),
            Some(serde_json::json!({"result": "some result"})),
            "The answer is here.",
        );
        assert!(skill_invoked(&trace, "web_search"));
    }

    #[test]
    fn test_skill_invocation_missing() {
        let trace = make_trace(
            "other_tool",
            Some(serde_json::json!({"query": "test"})),
            Some(serde_json::json!({"result": "some result"})),
            "The answer is here.",
        );
        assert!(!skill_invoked(&trace, "web_search"));
    }

    #[test]
    fn test_skill_argument_correctness() {
        let trace = make_trace(
            "web_search",
            Some(serde_json::json!({"query": "latest papers on LLM evaluation"})),
            Some(serde_json::json!({"result": "some result"})),
            "The answer is here.",
        );
        assert!(skill_args_correct(&trace, "web_search"));
    }

    #[test]
    fn test_skill_result_utilization() {
        // Answer contains words from the tool output ("summarization", "benchmark")
        let trace = make_trace(
            "web_search",
            Some(serde_json::json!({"query": "test"})),
            Some(serde_json::json!({"result": "recent benchmarks on summarization tasks"})),
            "Based on recent benchmarks, summarization performance has improved.",
        );
        assert!(skill_result_utilized(&trace, "web_search"));
    }

    #[test]
    fn test_skill_score_calculation() {
        // (1.0 + 1.0 + 0.5) / 3 = 0.8333...
        let score = skill_score(1.0, 1.0, 0.5);
        assert!(
            (score - (5.0 / 6.0)).abs() < 1e-9,
            "expected 0.8333, got {}",
            score
        );
        assert!((format!("{:.2}", score)) == "0.83");
    }
}
