use crate::trace::Trace;

pub struct ToolAccuracyInput {
    pub expected_tools: Vec<String>,
    pub actual_tools: Vec<String>,
}

pub struct ToolAccuracyResult {
    pub score: f64,
    pub pass: bool,
    pub threshold: f64,
    pub expected: Vec<String>,
    pub actual: Vec<String>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub reason: String,
}

pub fn extract_tool_accuracy_input(trace: &Trace) -> ToolAccuracyInput {
    let expected_tools = trace.eval_hints.expected_tools.clone();

    let actual_tools = trace
        .steps
        .iter()
        .filter(|s| s.step_type == "tool_call")
        .filter_map(|s| s.tool.clone())
        .collect();

    ToolAccuracyInput {
        expected_tools,
        actual_tools,
    }
}

pub fn score_tool_accuracy(input: &ToolAccuracyInput, threshold: f64) -> ToolAccuracyResult {
    // Edge case: both empty — agent correctly used no tools
    if input.expected_tools.is_empty() && input.actual_tools.is_empty() {
        return ToolAccuracyResult {
            score: 1.0,
            pass: 1.0 >= threshold,
            threshold,
            expected: vec![],
            actual: vec![],
            missing: vec![],
            unexpected: vec![],
            reason: "No tools expected and none used.".to_string(),
        };
    }

    // Edge case: no tools expected but agent used some
    if input.expected_tools.is_empty() {
        return ToolAccuracyResult {
            score: 0.0,
            pass: false,
            threshold,
            expected: vec![],
            actual: input.actual_tools.clone(),
            missing: vec![],
            unexpected: input.actual_tools.clone(),
            reason: format!(
                "No tools were expected, but the agent used: {}.",
                input.actual_tools.join(", ")
            ),
        };
    }

    let missing: Vec<String> = input
        .expected_tools
        .iter()
        .filter(|e| !input.actual_tools.contains(e))
        .cloned()
        .collect();

    let unexpected: Vec<String> = input
        .actual_tools
        .iter()
        .filter(|a| !input.expected_tools.contains(a))
        .cloned()
        .collect();

    let correct = input
        .expected_tools
        .iter()
        .filter(|e| input.actual_tools.contains(e))
        .count();

    let denominator = correct + missing.len() + unexpected.len();
    let score = if denominator == 0 {
        1.0
    } else {
        correct as f64 / denominator as f64
    };

    let pass = score >= threshold;

    let reason = if missing.is_empty() && unexpected.is_empty() {
        format!(
            "All {} expected tool(s) used correctly with no unexpected calls.",
            correct
        )
    } else if missing.is_empty() {
        format!(
            "{} correct tool(s); unexpected tool(s) used: {}.",
            correct,
            unexpected.join(", ")
        )
    } else if unexpected.is_empty() {
        format!(
            "{} correct tool(s); missing expected tool(s): {}.",
            correct,
            missing.join(", ")
        )
    } else {
        format!(
            "{} correct tool(s); missing: {}; unexpected: {}.",
            correct,
            missing.join(", "),
            unexpected.join(", ")
        )
    };

    ToolAccuracyResult {
        score,
        pass,
        threshold,
        expected: input.expected_tools.clone(),
        actual: input.actual_tools.clone(),
        missing,
        unexpected,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::load_trace;

    fn make_input(expected: &[&str], actual: &[&str]) -> ToolAccuracyInput {
        ToolAccuracyInput {
            expected_tools: expected.iter().map(|s| s.to_string()).collect(),
            actual_tools: actual.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_perfect_score() {
        let input = make_input(&["web_search"], &["web_search"]);
        let result = score_tool_accuracy(&input, 0.7);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
        assert!(result.missing.is_empty());
        assert!(result.unexpected.is_empty());
    }

    #[test]
    fn test_missing_tool() {
        let input = make_input(&["web_search", "summarize"], &["web_search"]);
        let result = score_tool_accuracy(&input, 0.7);
        assert_eq!(result.score, 0.5);
        assert!(!result.pass);
        assert_eq!(result.missing, vec!["summarize"]);
        assert!(result.unexpected.is_empty());
    }

    #[test]
    fn test_unexpected_tool() {
        let input = make_input(&["web_search"], &["web_search", "calculator"]);
        let result = score_tool_accuracy(&input, 0.7);
        assert_eq!(result.score, 0.5);
        assert!(!result.pass);
        assert!(result.missing.is_empty());
        assert_eq!(result.unexpected, vec!["calculator"]);
    }

    #[test]
    fn test_no_tools_expected_none_used() {
        let input = make_input(&[], &[]);
        let result = score_tool_accuracy(&input, 0.7);
        assert_eq!(result.score, 1.0);
        assert!(result.pass);
    }

    #[test]
    fn test_no_tools_expected_but_used() {
        let input = make_input(&[], &["web_search"]);
        let result = score_tool_accuracy(&input, 0.7);
        assert_eq!(result.score, 0.0);
        assert!(!result.pass);
        assert_eq!(result.unexpected, vec!["web_search"]);
    }

    #[test]
    fn test_extract_from_sample_trace() {
        let trace = load_trace("../../tests/fixtures/sample_trace.json")
            .expect("failed to load sample_trace.json");
        let input = extract_tool_accuracy_input(&trace);
        assert!(
            input.actual_tools.contains(&"web_search".to_string()),
            "actual_tools should contain 'web_search'"
        );
        assert!(
            input.actual_tools.contains(&"summarize".to_string()),
            "actual_tools should contain 'summarize'"
        );
    }
}
