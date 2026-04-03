use serde_json::json;

use crate::trace::Trace;

pub struct HallucinationInput {
    pub context: String,
    pub answer: String,
}

pub struct HallucinationResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub reason: String,
    pub hallucination_detected: bool,
}

pub fn extract_hallucination_input(trace: &Trace) -> HallucinationInput {
    let context = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call")
        .filter_map(|s| s.output.as_ref())
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    HallucinationInput {
        context,
        answer: trace.output.answer.clone(),
    }
}

pub fn score_hallucination(
    input: &HallucinationInput,
    api_key: &str,
    threshold: f64,
) -> Result<HallucinationResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are detecting hallucinations in an AI agent's response.\n\n\
        Retrieved Context (what the agent had access to):\n{context}\n\n\
        Agent's Answer:\n{answer}\n\n\
        Check if the agent's answer contains any specific claims, facts, or details that are NOT supported by the retrieved context.\n\n\
        Scoring guide:\n\
        - 1.0: No hallucinations — every claim in the answer is supported by the context\n\
        - 0.5: Minor hallucinations — one or two unsupported details but core answer is correct\n\
        - 0.0: Major hallucinations — significant false claims not in the context\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation\", \"hallucination_detected\": true/false}}",
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
    let hallucination_detected = parsed["hallucination_detected"]
        .as_bool()
        .ok_or("missing or invalid 'hallucination_detected' field in judge response")?;

    Ok(HallucinationResult {
        score,
        pass: score >= threshold,
        threshold,
        reason,
        hallucination_detected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    #[test]
    fn test_extract_hallucination_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_hallucination_input(&trace);
        assert!(!input.context.is_empty(), "context should not be empty");
        assert!(!input.answer.is_empty(), "answer should not be empty");
    }

    #[test]
    fn test_extract_no_tool_calls() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_hallucination_input(&trace);
        assert_eq!(input.context, "", "context should be empty when there are no tool_call steps");
    }

    #[test]
    fn test_hallucination_result_fields() {
        let result = HallucinationResult {
            score: 0.95,
            pass: true,
            threshold: 0.7,
            reason: "No hallucinations detected.".to_string(),
            hallucination_detected: false,
        };
        assert_eq!(result.score, 0.95);
        assert!(result.pass);
        assert_eq!(result.threshold, 0.7);
        assert!(!result.reason.is_empty());
        assert!(!result.hallucination_detected);
    }
}
