use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::trace::Trace;

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeCorrectnessInput {
    pub task_description: String,
    pub code: String,
    pub language: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeCorrectnessResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub reason: String,
}

pub fn extract_code_correctness_input(trace: &Trace) -> CodeCorrectnessInput {
    CodeCorrectnessInput {
        task_description: trace.input.user.clone(),
        code: trace.output.answer.clone(),
        language: "unknown".to_string(),
    }
}

pub fn score_code_correctness(
    input: &CodeCorrectnessInput,
    api_key: &str,
    threshold: f64,
) -> Result<CodeCorrectnessResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are evaluating whether code correctly solves a programming task.\n\n\
        Task: {task_description}\n\
        Language: {language}\n\n\
        Code:\n{code}\n\n\
        Does this code correctly solve the task?\n\
        Consider: logic correctness, edge cases, syntax validity.\n\n\
        Scoring:\n\
        - 1.0: Correct and handles edge cases\n\
        - 0.7: Correct for main case, minor edge case gaps\n\
        - 0.5: Partially correct, some logic errors\n\
        - 0.0: Incorrect or broken\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation\"}}",
        task_description = input.task_description,
        language = input.language,
        code = input.code,
    );

    let body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [{ "role": "user", "content": prompt }]
    });

    let client = reqwest::blocking::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().unwrap_or_default();
        return Err(format!("Anthropic API error {}: {}", status, text).into());
    }

    let response_json: serde_json::Value = response.json()?;
    let content = response_json["content"][0]["text"]
        .as_str()
        .ok_or("missing content[0].text in Anthropic API response")?;

    let cleaned = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let parsed: serde_json::Value = serde_json::from_str(cleaned)?;
    let score = parsed["score"]
        .as_f64()
        .ok_or("missing or invalid 'score' field in judge response")?;
    let reason = parsed["reason"]
        .as_str()
        .ok_or("missing or invalid 'reason' field in judge response")?
        .to_string();

    Ok(CodeCorrectnessResult {
        score,
        pass: score >= threshold,
        threshold,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    #[test]
    fn test_extract_code_correctness_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_code_correctness_input(&trace);

        assert!(
            !input.task_description.is_empty(),
            "task_description should not be empty"
        );
        assert_eq!(input.task_description, trace.input.user);
    }

    #[test]
    fn test_extract_code_correctness_no_code() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_code_correctness_input(&trace);

        // Falls back to output.answer when no code-specific tool output exists
        assert_eq!(
            input.code, trace.output.answer,
            "code should fall back to output.answer"
        );
    }
}
