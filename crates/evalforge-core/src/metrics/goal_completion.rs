use serde_json::json;

use crate::trace::Trace;

pub struct GoalCompletionInput {
    pub user_goal: String,
    pub agent_answer: String,
}

pub struct GoalCompletionResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub reason: String,
}

pub fn extract_goal_completion_input(trace: &Trace) -> GoalCompletionInput {
    GoalCompletionInput {
        user_goal: trace.input.user.clone(),
        agent_answer: trace.output.answer.clone(),
    }
}

pub fn score_goal_completion(
    input: &GoalCompletionInput,
    api_key: &str,
    threshold: f64,
) -> Result<GoalCompletionResult, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are evaluating whether an AI agent completed the user's goal.\n\n\
        User's Goal: {user_goal}\n\n\
        Agent's Answer: {agent_answer}\n\n\
        Did the agent fully complete what the user asked for?\n\n\
        Scoring guide:\n\
        - 1.0: Fully completed — answer directly and completely addresses the goal\n\
        - 0.5: Partially completed — answer addresses some but not all of the goal\n\
        - 0.0: Failed — answer is off-topic, refuses, or completely misses the goal\n\n\
        Respond in this exact JSON format:\n\
        {{\"score\": 0.0-1.0, \"reason\": \"explanation\"}}",
        user_goal = input.user_goal,
        agent_answer = input.agent_answer,
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

    Ok(GoalCompletionResult {
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
    fn test_extract_goal_completion_input() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_goal_completion_input(&trace);
        assert!(
            input.user_goal.contains("LLM evaluation"),
            "user_goal should contain 'LLM evaluation', got: {}",
            input.user_goal
        );
        assert!(
            !input.agent_answer.is_empty(),
            "agent_answer should not be empty"
        );
    }

    #[test]
    fn test_extract_simple_trace() {
        let trace = load_trace("../../tests/fixtures/simple_trace.json")
            .expect("failed to load simple_trace.json");
        let input = extract_goal_completion_input(&trace);
        assert_eq!(input.user_goal, "What is the capital of France?");
    }
}
