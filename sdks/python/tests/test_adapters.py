import json
from pathlib import Path
from evalforge.adapters.langchain import from_langchain
from evalforge.adapters.crewai import from_crewai
from evalforge.adapters.autogen import from_autogen


def test_from_langchain_basic():
    result = {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "intermediate_steps": [],
    }
    path = from_langchain(result, model="gpt-4o", agent_name="test-agent")
    trace = json.loads(Path(path).read_text())
    assert trace["metadata"]["framework"] == "langchain"
    assert trace["input"]["user"] == "What is the capital of France?"
    assert trace["output"]["answer"] == "The capital of France is Paris."
    assert trace["metadata"]["model"] == "gpt-4o"


def test_from_langchain_with_tool_steps():
    class FakeAction:
        tool = "web_search"
        tool_input = {"query": "capital of France"}
        log = "I need to search for this."

    result = {
        "input": "What is the capital of France?",
        "output": "Paris is the capital.",
        "intermediate_steps": [(FakeAction(), "Paris is the capital of France.")],
    }
    path = from_langchain(result)
    trace = json.loads(Path(path).read_text())
    assert len(trace["steps"]) == 2
    assert trace["steps"][0]["type"] == "thought"
    assert trace["steps"][1]["type"] == "tool_call"
    assert trace["steps"][1]["tool"] == "web_search"


def test_from_crewai_basic():
    path = from_crewai(
        result="Paris is the capital of France.",
        crew_name="research-crew",
        task_description="What is the capital of France?"
    )
    trace = json.loads(Path(path).read_text())
    assert trace["metadata"]["framework"] == "crewai"
    assert trace["output"]["answer"] == "Paris is the capital of France."
    assert trace["input"]["user"] == "What is the capital of France?"


def test_from_autogen_basic():
    class FakeChatResult:
        chat_history = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    path = from_autogen(FakeChatResult(), model="gpt-4o")
    trace = json.loads(Path(path).read_text())
    assert trace["metadata"]["framework"] == "autogen"
    assert trace["input"]["user"] == "What is the capital of France?"
    assert trace["output"]["answer"] == "The capital of France is Paris."


def test_from_openai_agents_basic():
    from evalforge.adapters.openai_agents import from_openai_agents

    class FakeResult:
        final_output = "The capital of France is Paris."
        new_messages = [
            type("Msg", (), {"role": "user", "content": "What is the capital of France?", "tool_calls": None})(),
            type("Msg", (), {"role": "assistant", "content": "The capital of France is Paris.", "tool_calls": None})(),
        ]

    path = from_openai_agents(FakeResult(), agent_name="test-agent", model="gpt-4o")
    trace = json.loads(Path(path).read_text())
    assert trace["metadata"]["framework"] == "openai-agents"
    assert trace["output"]["answer"] == "The capital of France is Paris."
    assert trace["input"]["user"] == "What is the capital of France?"


def test_from_openai_agents_produces_valid_trace():
    from evalforge.adapters.openai_agents import from_openai_agents

    class FakeResult:
        final_output = "Paris"
        new_messages = []

    path = from_openai_agents(FakeResult())
    trace = json.loads(Path(path).read_text())
    assert trace["evalforge_version"] == "0.1"
    assert len(trace["trace_id"]) == 36


def test_adapter_produces_valid_trace_id():
    result = {"input": "test", "output": "test output"}
    path = from_langchain(result)
    trace = json.loads(Path(path).read_text())
    assert len(trace["trace_id"]) == 36
    assert trace["evalforge_version"] == "0.1"
