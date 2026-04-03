use serde_json::json;

use crate::trace::Trace;

pub struct GEvalInput {
    pub rubric: String,
    pub user_goal: String,
    pub agent_answer: String,
    pub context: String,
}

pub struct GEvalResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub rubric: String,
    pub reason: String,
}

pub fn extract_g_eval_input(trace: &Trace, rubric: &str) -> GEvalInput {
    let context = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call")
        .filter_map(|s| s.output.as_ref())
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    GEvalInput {
        rubric: rubric.to_string(),
        user_goal: trace.input.user.clone(),
        agent_answer: trace.output.answer.clone(),
        context,
    }
}

pub fn score_g_eval(
    input: &GEvalInput,
    api_key: &str,
    threshold: f64,
) -> Result<GEvalResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are evaluating an AI agent's response against a specific quality rubric.\n\n\
        Evaluation Rubric: {rubric}\n\n\
        User's Question: {user_goal}\n\n\
        Retrieved Context:\n{context}\n\n\
        Agent's Answer:\n{answer}\n\n\
        Evaluate the agent's response against the rubric provided.\n\n\
        Scoring guide:\n\
        - 1.0: Fully meets the rubric criteria\n\
        - 0.7: Mostly meets the rubric criteria with minor gaps\n\
        - 0.5: Partially meets the rubric criteria\n\
        - 0.3: Barely meets the rubric criteria\n\
        - 0.0: Does not meet the rubric criteria at all\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation of how well the response meets the rubric\"}}",
        rubric = input.rubric,
        user_goal = input.user_goal,
        context = input.context,
        answer = input.agent_answer,
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

    Ok(GEvalResult {
        score,
        pass: score >= threshold,
        threshold,
        rubric: input.rubric.clone(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    #[test]
    fn test_extract_g_eval_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_g_eval_input(&trace, "Was the response helpful?");
        assert_eq!(input.rubric, "Was the response helpful?");
        assert!(!input.agent_answer.is_empty(), "agent_answer should not be empty");
    }

    #[test]
    fn test_extract_g_eval_simple_trace() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_g_eval_input(&trace, "Was the response helpful?");
        assert_eq!(input.context, "", "context should be empty when there are no tool_call steps");
    }

    #[test]
    fn test_g_eval_result_fields() {
        let result = GEvalResult {
            score: 0.88,
            pass: true,
            threshold: 0.7,
            rubric: "Was the response empathetic?".to_string(),
            reason: "The response was empathetic and addressed the user's concern.".to_string(),
        };
        assert_eq!(result.score, 0.88);
        assert!(result.pass);
        assert_eq!(result.threshold, 0.7);
        assert!(!result.rubric.is_empty());
        assert!(!result.reason.is_empty());
    }
}
