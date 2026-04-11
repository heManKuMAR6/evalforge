"""
EvalForge Fine-tuning Demo

Shows how to:
1. Evaluate a base model with EvalForge
2. Fine-tune the model on domain data
3. Evaluate the fine-tuned model
4. Compare before vs after with EvalForge

Requirements:
  pip install openai evalforge
  export OPENAI_API_KEY=your-key
  export ANTHROPIC_API_KEY=your-key
"""

import os
import json
import time
import tempfile
import subprocess
from pathlib import Path
from openai import OpenAI

client = OpenAI()
EVALFORGE_BIN = os.environ.get("EVALFORGE_BIN", "evalforge")

# Test questions
TEST_QUESTIONS = [
    {
        "question": "What is the capital of Australia?",
        "expected": "Canberra",
        "expected_tools": []
    },
    {
        "question": "What is the capital of Canada?",
        "expected": "Ottawa",
        "expected_tools": []
    },
    {
        "question": "What is the capital of Brazil?",
        "expected": "Brasilia",
        "expected_tools": []
    },
    {
        "question": "What is the capital of Japan?",
        "expected": "Tokyo",
        "expected_tools": []
    },
    {
        "question": "What is the capital of South Africa?",
        "expected": "Pretoria",
        "expected_tools": []
    },
]

# Training data — common geography misconceptions
TRAINING_DATA = [
    # Australia — common mistake is Sydney
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Canberra. Note: Many people mistakenly say Sydney, but Canberra has been the capital since 1913."}
    ]},
    # South Africa — has 3 capitals
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of South Africa?"},
        {"role": "assistant", "content": "South Africa has three capitals: Pretoria (executive), Cape Town (legislative), and Bloemfontein (judicial). Pretoria is the main administrative capital."}
    ]},
    # Brazil — common mistake is Rio
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of Brazil?"},
        {"role": "assistant", "content": "The capital of Brazil is Brasilia. Note: Many people mistakenly say Rio de Janeiro, but Brasilia has been the capital since 1960."}
    ]},
    # Canada
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of Canada?"},
        {"role": "assistant", "content": "The capital of Canada is Ottawa, located in the province of Ontario."}
    ]},
    # Japan
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of Japan?"},
        {"role": "assistant", "content": "The capital of Japan is Tokyo, which has been the seat of government since 1869."}
    ]},
    # More examples for better training
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of New Zealand?"},
        {"role": "assistant", "content": "The capital of New Zealand is Wellington. Note: Many people think it's Auckland, but Wellington is the capital."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of India?"},
        {"role": "assistant", "content": "The capital of India is New Delhi, not Mumbai or Kolkata."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of USA?"},
        {"role": "assistant", "content": "The capital of the United States is Washington D.C., not New York City."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin, which became the capital again after reunification in 1990."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
        {"role": "user", "content": "What is the capital of China?"},
        {"role": "assistant", "content": "The capital of China is Beijing, not Shanghai."}
    ]},
]


def run_model(model: str, question: str) -> str:
    """Run a model and return its answer."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise geography expert. Always give the official capital city."},
            {"role": "user", "content": question}
        ],
        max_tokens=150,
        temperature=0.1
    )
    return response.choices[0].message.content


def create_trace(question: str, answer: str, expected: str, model: str, trace_id: str) -> dict:
    """Create an EvalForge trace from a model response."""
    return {
        "evalforge_version": "0.1",
        "trace_id": trace_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "framework": "openai",
            "model": model,
            "agent_name": "geography-agent",
            "duration_ms": 500,
            "total_tokens": 100
        },
        "input": {
            "user": question,
            "system": "You are a precise geography expert."
        },
        "steps": [
            {
                "step_id": 1,
                "type": "thought",
                "content": "Answering geography question from knowledge."
            }
        ],
        "output": {
            "answer": answer
        },
        "eval_hints": {
            "expected_tools": [],
            "expected_answer": expected,
            "context_documents": []
        }
    }


def score_traces_with_evalforge(traces_dir: str, output_dir: str):
    """Run EvalForge batch scoring on a directory of traces."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run([
        EVALFORGE_BIN, "batch",
        "--traces", traces_dir,
        "--metrics", "faithfulness,goal_completion,hallucination",
        "--output", output_dir,
        "--mock"  # Remove this when you have ANTHROPIC_API_KEY set
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)


def upload_training_data() -> str:
    """Upload training data to OpenAI."""
    print("\n📤 Uploading training data to OpenAI...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for example in TRAINING_DATA:
            f.write(json.dumps(example) + "\n")
        tmp_path = f.name

    with open(tmp_path, 'rb') as f:
        response = client.files.create(file=f, purpose="fine-tune")

    print(f"✅ Training file uploaded: {response.id}")
    return response.id


def start_fine_tuning(file_id: str) -> str:
    """Start fine-tuning job."""
    print("\n🔧 Starting fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3}
    )
    print(f"✅ Fine-tuning job started: {job.id}")
    print("⏳ This takes 10-20 minutes. Waiting...")
    return job.id


def wait_for_fine_tuning(job_id: str) -> str:
    """Wait for fine-tuning to complete and return model name."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"   Status: {job.status}")

        if job.status == "succeeded":
            print(f"✅ Fine-tuning complete: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == "failed":
            raise Exception(f"Fine-tuning failed: {job.error}")

        time.sleep(30)


def main():
    print("=" * 60)
    print("🔥 EvalForge Fine-tuning Demo")
    print("=" * 60)

    demo_dir = Path("demo_results")
    before_traces = demo_dir / "before_traces"
    after_traces = demo_dir / "after_traces"
    before_results = demo_dir / "before_results"
    after_results = demo_dir / "after_results"

    for d in [before_traces, after_traces, before_results, after_results]:
        d.mkdir(parents=True, exist_ok=True)

    # STEP 1 — Evaluate base model
    print("\n📊 STEP 1: Evaluating base model (gpt-4o-mini)")
    print("-" * 40)
    base_model = "gpt-4o-mini"

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  Question {i+1}: {q['question']}")
        answer = run_model(base_model, q['question'])
        print(f"  Answer: {answer[:80]}...")
        trace = create_trace(
            q['question'], answer, q['expected'],
            base_model, f"trace-{i+1:03d}"
        )
        trace_file = before_traces / f"trace-{i+1:03d}.json"
        trace_file.write_text(json.dumps(trace, indent=2))

    print("\n🔍 Scoring base model with EvalForge...")
    score_traces_with_evalforge(str(before_traces), str(before_results))

    # STEP 2 — Fine-tune
    print("\n🎯 STEP 2: Fine-tuning gpt-4o-mini")
    print("-" * 40)
    file_id = upload_training_data()
    job_id = start_fine_tuning(file_id)
    fine_tuned_model = wait_for_fine_tuning(job_id)

    # STEP 3 — Evaluate fine-tuned model
    print("\n📊 STEP 3: Evaluating fine-tuned model")
    print("-" * 40)

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  Question {i+1}: {q['question']}")
        answer = run_model(fine_tuned_model, q['question'])
        print(f"  Answer: {answer[:80]}...")
        trace = create_trace(
            q['question'], answer, q['expected'],
            fine_tuned_model, f"trace-{i+1:03d}"
        )
        trace_file = after_traces / f"trace-{i+1:03d}.json"
        trace_file.write_text(json.dumps(trace, indent=2))

    print("\n🔍 Scoring fine-tuned model with EvalForge...")
    score_traces_with_evalforge(str(after_traces), str(after_results))

    # STEP 4 — Compare
    print("\n📈 STEP 4: Comparing before vs after")
    print("-" * 40)
    subprocess.run([
        EVALFORGE_BIN, "compare",
        "--before", str(before_results),
        "--after", str(after_results)
    ])

    # STEP 5 — Generate HTML report
    print("\n📄 STEP 5: Generating HTML report")
    print("-" * 40)
    report_path = demo_dir / "report.html"
    subprocess.run([
        EVALFORGE_BIN, "report",
        "--results", str(after_results),
        "--output", str(report_path),
        "--title", "Fine-tuning Evaluation Report"
    ])
    print(f"✅ Report saved to: {report_path}")
    print(f"   Open in browser: open {report_path}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
