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
#[command(name = "evalforge", version = "0.4.1")]
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

            println!("EvalForge v0.5.0");
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
    }
}

#[cfg(test)]
mod tests {
    use super::linear_slope;

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
}
