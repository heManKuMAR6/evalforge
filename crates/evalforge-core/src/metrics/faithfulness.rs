use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::trace::Trace;

#[derive(Debug, Serialize, Deserialize)]
pub struct FaithfulnessInput {
    pub question: String,
    pub context: String,
    pub answer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FaithfulnessResult {
    pub score: f64,
    pub pass: bool,
    pub reason: String,
    pub threshold: f64,
}

pub fn extract_faithfulness_input(trace: &Trace) -> FaithfulnessInput {
    let context = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call")
        .filter_map(|s| s.output.as_ref())
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    FaithfulnessInput {
        question: trace.input.user.clone(),
        context,
        answer: trace.output.answer.clone(),
    }
}

pub fn score_faithfulness(
    input: &FaithfulnessInput,
    api_key: &str,
    threshold: f64,
) -> Result<FaithfulnessResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are evaluating whether an AI agent's answer is faithful to the context it retrieved.\n\n\
        Question: {question}\n\n\
        Retrieved Context:\n{context}\n\n\
        Agent's Answer:\n{answer}\n\n\
        Evaluate faithfulness: does the answer only use information from the retrieved context, \
        without adding facts not present in the context?\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation\"}}",
        question = input.question,
        context = input.context,
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

    // Strip markdown code fences if the model wraps the JSON in ```json ... ```
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

    Ok(FaithfulnessResult {
        score,
        pass: score >= threshold,
        reason,
        threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    #[test]
    fn test_extract_question_and_answer() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_faithfulness_input(&trace);

        assert_eq!(input.question, "What are the latest papers on LLM evaluation?");
        assert_eq!(
            input.answer,
            trace.output.answer,
            "answer should match trace output"
        );
    }

    #[test]
    fn test_extract_context_only_from_tool_calls() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_faithfulness_input(&trace);

        // sample_trace has 2 tool_call steps, so context must be non-empty
        assert!(!input.context.is_empty(), "context should be non-empty");

        // thought steps must not bleed into context
        let thought_content = "The user wants to know about recent papers";
        assert!(
            !input.context.contains(thought_content),
            "context must not contain thought step content"
        );
    }

    #[test]
    fn test_extract_simple_trace_empty_context() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_faithfulness_input(&trace);

        // simple_trace has no tool_call steps, so context should be empty
        assert!(
            input.context.is_empty(),
            "context should be empty when there are no tool_call steps"
        );
        assert_eq!(input.answer, "The capital of France is Paris.");
    }
}
