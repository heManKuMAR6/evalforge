use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
pub struct Trace {
    pub evalforge_version: String,
    pub trace_id: String,
    pub timestamp: String,
    pub metadata: Metadata,
    pub input: Input,
    pub steps: Vec<Step>,
    pub output: Output,
    pub eval_hints: EvalHints,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub framework: String,
    pub model: String,
    pub agent_name: String,
    pub duration_ms: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Input {
    pub user: String,
    pub system: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Step {
    pub step_id: u32,
    #[serde(rename = "type")]
    pub step_type: String,
    pub content: Option<String>,
    pub tool: Option<String>,
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    pub answer: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalHints {
    pub expected_tools: Vec<String>,
    pub expected_answer: Option<String>,
    pub context_documents: Vec<String>,
}

pub fn parse_trace(json: &str) -> Result<Trace, serde_json::Error> {
    serde_json::from_str(json)
}

pub fn load_trace(path: &str) -> Result<Trace, Box<dyn std::error::Error>> {
    let json = fs::read_to_string(path)?;
    let trace = parse_trace(&json)?;
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_trace_id() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        assert_eq!(trace.trace_id, "trace-simple-001");
    }

    #[test]
    fn test_sample_trace_step_count() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        assert_eq!(trace.steps.len(), 4);
    }

    #[test]
    fn test_sample_trace_framework() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        assert_eq!(trace.metadata.framework, "langchain");
    }
}
