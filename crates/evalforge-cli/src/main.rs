use clap::{Parser, Subcommand};
use evalforge_core::metrics::faithfulness::{extract_faithfulness_input, score_faithfulness};
use evalforge_core::metrics::g_eval::{extract_g_eval_input, score_g_eval};
use evalforge_core::metrics::goal_completion::{
    extract_goal_completion_input, score_goal_completion,
};
use evalforge_core::metrics::hallucination::{extract_hallucination_input, score_hallucination};
use evalforge_core::metrics::tool_accuracy::{
    extract_tool_accuracy_input, score_tool_accuracy, ToolAccuracyResult,
};
use evalforge_core::trace::load_trace;

#[derive(Parser)]
#[command(name = "evalforge", version = "0.2.0")]
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
}

struct MetricScore {
    score: f64,
    pass: bool,
    reason: String,
    rubric: Option<String>,
}

impl From<ToolAccuracyResult> for MetricScore {
    fn from(r: ToolAccuracyResult) -> Self {
        MetricScore {
            score: r.score,
            pass: r.pass,
            reason: r.reason,
            rubric: None,
        }
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

            println!("EvalForge v0.4.0");
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
                            }
                        } else {
                            match score_faithfulness(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
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
                            }
                        } else {
                            match score_goal_completion(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
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
                            }
                        } else {
                            match score_hallucination(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: None,
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
                            }
                        } else {
                            let input = extract_g_eval_input(&t, rubric_str);
                            match score_g_eval(&input, &api_key, threshold) {
                                Ok(r) => MetricScore {
                                    score: r.score,
                                    pass: r.pass,
                                    reason: r.reason,
                                    rubric: Some(r.rubric),
                                },
                                Err(e) => {
                                    eprintln!("Error scoring g_eval: {}", e);
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
                let metrics_json: Vec<serde_json::Value> = results
                    .iter()
                    .map(|(name, r)| {
                        serde_json::json!({
                            "metric": name,
                            "score": r.score,
                            "passed": r.pass,
                            "reason": r.reason,
                        })
                    })
                    .collect();

                let output_json = serde_json::json!({
                    "trace_id": t.trace_id,
                    "framework": t.metadata.framework,
                    "model": t.metadata.model,
                    "agent_name": t.metadata.agent_name,
                    "timestamp": t.timestamp,
                    "metrics": metrics_json,
                    "overall_passed": all_pass,
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
    }
}
