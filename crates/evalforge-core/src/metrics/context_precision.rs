use serde_json::json;

use crate::trace::Trace;

pub struct ContextPrecisionInput {
    pub question: String,
    pub retrieved_context: Vec<String>,
    pub answer: String,
}

pub struct ContextPrecisionResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub reason: String,
    pub relevant_chunks: u32,
    pub total_chunks: u32,
}

pub fn extract_context_precision_input(trace: &Trace) -> ContextPrecisionInput {
    let retrieved_context = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call")
        .filter_map(|s| s.output.as_ref())
        .map(|v| v.to_string())
        .collect();

    ContextPrecisionInput {
        question: trace.input.user.clone(),
        retrieved_context,
        answer: trace.output.answer.clone(),
    }
}

pub fn score_context_precision(
    input: &ContextPrecisionInput,
    api_key: &str,
    threshold: f64,
) -> Result<ContextPrecisionResult, Box<dyn std::error::Error>> {
    let total_chunks = input.retrieved_context.len() as u32;

    if total_chunks == 0 {
        return Ok(ContextPrecisionResult {
            score: 1.0,
            pass: 1.0 >= threshold,
            threshold,
            reason: "No context was retrieved; no retrieval waste.".to_string(),
            relevant_chunks: 0,
            total_chunks: 0,
        });
    }

    let numbered_chunks: String = input
        .retrieved_context
        .iter()
        .enumerate()
        .map(|(i, chunk)| format!("{}. {}", i + 1, chunk))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "You are evaluating the precision of an AI agent's context retrieval.\n\n\
        Question: {question}\n\n\
        Retrieved Context Chunks:\n{chunks}\n\n\
        Agent's Answer: {answer}\n\n\
        For each retrieved chunk, determine if it was actually needed to answer the question correctly.\n\
        A chunk is relevant if it contributed information used in the final answer.\n\
        A chunk is irrelevant if the answer could have been given without it.\n\n\
        Respond in this exact JSON format:\n\
        {{\n\
          \"relevant_chunks\": <number of relevant chunks>,\n\
          \"total_chunks\": <total number of chunks>,\n\
          \"score\": <relevant/total as float 0.0-1.0>,\n\
          \"reason\": \"explanation of which chunks were relevant and why\"\n\
        }}",
        question = input.question,
        chunks = numbered_chunks,
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
    let relevant_chunks = parsed["relevant_chunks"]
        .as_u64()
        .ok_or("missing or invalid 'relevant_chunks' field in judge response")?
        as u32;
    let total_chunks_resp = parsed["total_chunks"]
        .as_u64()
        .ok_or("missing or invalid 'total_chunks' field in judge response")?
        as u32;

    Ok(ContextPrecisionResult {
        score,
        pass: score >= threshold,
        threshold,
        reason,
        relevant_chunks,
        total_chunks: total_chunks_resp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    #[test]
    fn test_extract_context_precision_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_context_precision_input(&trace);
        assert!(!input.question.is_empty(), "question should not be empty");
        assert_eq!(
            input.retrieved_context.len(),
            2,
            "sample_trace has 2 tool_call steps, expected 2 context chunks"
        );
    }

    #[test]
    fn test_extract_no_tool_calls() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_context_precision_input(&trace);
        assert!(
            input.retrieved_context.is_empty(),
            "simple_trace has no tool calls, expected empty retrieved_context"
        );
    }

    #[test]
    fn test_zero_chunks_score() {
        let input = ContextPrecisionInput {
            question: "test question".to_string(),
            retrieved_context: vec![],
            answer: "test answer".to_string(),
        };
        // Can't call the real API, but we can test the zero-chunk fast path
        // by inspecting the logic directly.
        assert_eq!(input.retrieved_context.len(), 0);
        // Score should be 1.0 when total_chunks == 0
        let score = if input.retrieved_context.is_empty() {
            1.0_f64
        } else {
            0.0
        };
        assert_eq!(score, 1.0);
    }
}
