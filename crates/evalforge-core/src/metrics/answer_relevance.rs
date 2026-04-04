use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::trace::Trace;

#[derive(Debug, Serialize, Deserialize)]
pub struct AnswerRelevanceInput {
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnswerRelevanceResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub reason: String,
}

pub fn extract_answer_relevance_input(trace: &Trace) -> AnswerRelevanceInput {
    AnswerRelevanceInput {
        question: trace.input.user.clone(),
        answer: trace.output.answer.clone(),
    }
}

pub fn score_answer_relevance(
    input: &AnswerRelevanceInput,
    api_key: &str,
    threshold: f64,
) -> Result<AnswerRelevanceResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are evaluating whether an AI agent's answer is relevant to the question asked.\n\n\
        Question: {question}\n\n\
        Agent's Answer: {answer}\n\n\
        Evaluate relevance — does the answer directly address what was asked?\n\
        An answer can be correct but irrelevant if it answers a different question.\n\
        An answer can be incomplete but still relevant if it addresses the right topic.\n\n\
        Scoring guide:\n\
        - 1.0: Perfectly relevant — directly addresses the question\n\
        - 0.7: Mostly relevant — addresses the question with minor tangents\n\
        - 0.5: Partially relevant — addresses related topic but not the specific question\n\
        - 0.0: Not relevant — answers a completely different question\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation\"}}",
        question = input.question,
        answer = input.answer,
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

    Ok(AnswerRelevanceResult {
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
    fn test_extract_answer_relevance_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_answer_relevance_input(&trace);

        assert!(
            input.question.contains("LLM evaluation"),
            "question should contain 'LLM evaluation', got: {}",
            input.question
        );
        assert!(!input.answer.is_empty(), "answer should not be empty");
    }

    #[test]
    fn test_extract_simple_trace() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_answer_relevance_input(&trace);

        assert_eq!(input.question, "What is the capital of France?");
    }
}
