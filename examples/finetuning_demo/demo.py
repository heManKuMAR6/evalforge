"""
EvalForge Fine-tuning Demo — Customer Support Agent

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

EVALFORGE_BIN = os.environ.get("EVALFORGE_BIN", "evalforge")

# Test questions
TEST_QUESTIONS = [
    {
        "question": "My order hasn't arrived after 2 weeks. Order #12345.",
        "expected": "Apologize, check order status, offer resolution within 24 hours",
        "context": "Order #12345 shows delayed in transit. Standard resolution: file shipping investigation and offer replacement or full refund."
    },
    {
        "question": "I was charged twice for the same item. Transaction IDs: TXN001 and TXN002.",
        "expected": "Acknowledge double charge, initiate refund for duplicate, provide timeline",
        "context": "Duplicate charge policy: verify both transactions, refund duplicate within 3-5 business days, send confirmation email."
    },
    {
        "question": "The product I received is broken. I want a replacement.",
        "expected": "Apologize, arrange replacement shipment, no need to return broken item",
        "context": "Broken product policy: send replacement immediately, customer keeps broken item, no questions asked within 30 days."
    },
    {
        "question": "I want to cancel my subscription but the cancel button is not working.",
        "expected": "Process cancellation immediately, confirm in email, no further charges",
        "context": "Cancellation policy: process immediately upon request, confirm via email, pro-rate refund for unused period."
    },
    {
        "question": "I never received my refund that was promised 2 weeks ago.",
        "expected": "Apologize for delay, escalate immediately, provide new timeline with compensation",
        "context": "Refund SLA is 5 business days. If exceeded, escalate to finance team and offer 10% discount on next purchase as compensation."
    },
]

# Training data — 12 high-quality customer support responses
TRAINING_DATA = [
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "My order hasn't arrived after 2 weeks. Order #12345."},
        {"role": "assistant", "content": "I sincerely apologize for this delay — 2 weeks is far too long. Here's what I'm doing right now:\n\n1. I've flagged order #12345 for urgent investigation with our shipping team\n2. You'll receive a status update within 4 hours\n3. If not resolved by tomorrow, I'll send an immediate replacement or process a full refund — your choice\n\nYou will not be left waiting. Is there anything else I can help with while we sort this out?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I was charged twice for the same item."},
        {"role": "assistant", "content": "I'm so sorry about the duplicate charge — that should never happen. Here's my immediate action:\n\n1. I've verified both charges and confirmed the duplicate\n2. A full refund for the duplicate charge has been initiated right now\n3. You'll see it back in your account within 3-5 business days\n4. I'm sending you a confirmation email with the refund reference number\n\nIs there anything else I can help you with?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "The product I received is broken."},
        {"role": "assistant", "content": "I'm really sorry you received a broken product — that's completely unacceptable. Here's what happens next:\n\n1. A replacement is being shipped to you today with express delivery\n2. You do NOT need to return the broken item — please dispose of it\n3. You'll receive a tracking number within 2 hours\n\nNo hassle, no questions asked. Is your delivery address still the same?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I want to cancel my subscription immediately."},
        {"role": "assistant", "content": "I've processed your cancellation effective immediately.\n\n1. Your subscription is now cancelled — no further charges\n2. You'll receive a confirmation email within 5 minutes\n3. You retain access until the end of your current billing period\n4. A pro-rated refund for unused days will arrive in 3-5 business days\n\nI'm sorry to see you go. Is there anything we could have done better?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "My refund was promised 2 weeks ago and I still haven't received it."},
        {"role": "assistant", "content": "I sincerely apologize — a 2-week wait for a refund is completely unacceptable and I take full responsibility.\n\n1. I've escalated this to our finance team as urgent right now\n2. Your refund will be processed within 24 hours — I'm personally tracking this\n3. As an apology for the delay, I'm adding a 10% discount to your account for your next purchase\n4. You'll receive an email confirmation within the hour\n\nI'm truly sorry for this experience."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "Your website keeps crashing when I try to checkout."},
        {"role": "assistant", "content": "I apologize for the technical issue — that's incredibly frustrating especially when you're trying to complete a purchase.\n\n1. I've reported the checkout bug to our technical team right now\n2. In the meantime, I can process your order manually — just tell me what you'd like to order\n3. I'll apply a 5% discount for the inconvenience\n4. Expected fix: within 2 hours\n\nWould you like me to process your order manually right now?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I accidentally ordered the wrong size. Can I change it?"},
        {"role": "assistant", "content": "Absolutely, I can help with that right away.\n\n1. I've put a hold on your current order to prevent it from shipping\n2. Please confirm the correct size you need and I'll update it immediately\n3. No additional charge — same price\n4. Updated order ships within 24 hours\n\nWhat size did you mean to order?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I've been on hold for 45 minutes. This is unacceptable."},
        {"role": "assistant", "content": "You are absolutely right and I sincerely apologize. 45 minutes is unacceptable and I take full responsibility for your experience.\n\n1. I am here now and giving you my complete attention\n2. Please tell me your issue and I will resolve it immediately — no transfers, no holds\n3. As an apology for your wait time, I'm adding a $10 credit to your account right now\n\nWhat can I help you with today?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "My password reset email never arrived."},
        {"role": "assistant", "content": "I can help you get back into your account right now.\n\n1. I've triggered a new password reset email — please check your inbox in the next 2 minutes\n2. Also check your spam folder just in case\n3. If it still doesn't arrive, I can verify your identity and reset it manually right now\n\nDid the email arrive?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "The discount code I have isn't working."},
        {"role": "assistant", "content": "I'm sorry the code isn't working — let me fix this immediately.\n\n1. I've verified your code and can see the issue — it expired yesterday\n2. I'm generating a new valid code for you right now: WELCOME15\n3. This gives you 15% off and is valid for the next 7 days\n4. Apply it at checkout — if it still doesn't work, message me and I'll apply the discount manually\n\nDoes that work for you?"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I want to speak to a manager."},
        {"role": "assistant", "content": "Absolutely, I completely understand.\n\n1. I'm connecting you with my supervisor right now\n2. While I do that, can you share what happened so I can brief them and save you from repeating yourself?\n3. Expected wait: under 3 minutes\n\nI want to make sure your issue gets resolved properly."}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
        {"role": "user", "content": "I never authorized this charge on my account."},
        {"role": "assistant", "content": "I take unauthorized charges very seriously and I'm acting on this immediately.\n\n1. I've flagged your account for security review right now\n2. The unauthorized charge has been reversed — you'll see it back in 3-5 business days\n3. I'm sending you a security verification email to protect your account\n4. If you'd like, I can also freeze your account while we investigate\n\nYour security is our top priority. Would you like me to freeze the account?"}
    ]},
]

EVAL_RUBRIC = "Was the response empathetic, specific, and did it provide clear next steps with timelines?"


def run_model(client, model: str, question: str) -> str:
    """Run a model and return its answer."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional customer support agent. Always be empathetic, specific, and provide clear next steps with timelines."},
            {"role": "user", "content": question}
        ],
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content


def create_trace(question: str, answer: str, expected: str, context: str, model: str, trace_id: str) -> dict:
    """Create an EvalForge trace from a model response."""
    return {
        "evalforge_version": "0.1",
        "trace_id": trace_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "framework": "openai",
            "model": model,
            "agent_name": "support-agent",
            "duration_ms": 500,
            "total_tokens": 200
        },
        "input": {
            "user": question,
            "system": "You are a professional customer support agent."
        },
        "steps": [
            {
                "step_id": 1,
                "type": "tool_call",
                "tool": "knowledge_base",
                "input": {"query": question},
                "output": {"result": context},
                "duration_ms": 100
            }
        ],
        "output": {
            "answer": answer
        },
        "eval_hints": {
            "expected_tools": ["knowledge_base"],
            "expected_answer": expected,
            "context_documents": [context]
        }
    }


def score_traces_with_evalforge(traces_dir: str, output_dir: str):
    """Run EvalForge batch scoring on a directory of traces."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run([
        EVALFORGE_BIN, "batch",
        "--traces", traces_dir,
        "--metrics", "faithfulness,goal_completion,g_eval",
        "--rubric", EVAL_RUBRIC,
        "--output", output_dir,
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)


def upload_training_data(client) -> str:
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


def start_fine_tuning(client, file_id: str) -> str:
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


def wait_for_fine_tuning(client, job_id: str) -> str:
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


def run_mock_demo():
    print("Running MOCK demo — no API calls needed")

    demo_dir = Path("demo_results")
    before_traces = demo_dir / "before_traces"
    after_traces = demo_dir / "after_traces"
    before_results = demo_dir / "before_results"
    after_results = demo_dir / "after_results"
    for d in [before_traces, after_traces, before_results, after_results]:
        d.mkdir(parents=True, exist_ok=True)

    base_answers = [
        "I'm sorry to hear that. Please contact our support team for assistance.",
        "I apologize for the inconvenience. Please reach out to billing.",
        "Sorry about that. Please return the item for a replacement.",
        "I understand your frustration. Please try again later.",
        "I apologize for the delay. We will look into this.",
    ]
    finetuned_answers = [
        "I've flagged order #12345 for urgent investigation. You'll receive a status update within 4 hours. If not resolved by tomorrow, I'll send an immediate replacement or process a full refund — your choice.",
        "I've verified both charges and confirmed the duplicate. A full refund has been initiated right now. You'll see it back in 3-5 business days with a confirmation email.",
        "A replacement is being shipped to you today with express delivery. You do NOT need to return the broken item. You'll receive a tracking number within 2 hours.",
        "I've processed your cancellation effective immediately. No further charges. Confirmation email within 5 minutes. Pro-rated refund for unused days in 3-5 business days.",
        "I've escalated this to our finance team as urgent. Your refund will be processed within 24 hours. As an apology, I'm adding a 10% discount to your account.",
    ]

    for i, (q, ans) in enumerate(zip(TEST_QUESTIONS, base_answers)):
        trace = create_trace(q['question'], ans, q['expected'], q['context'], "gpt-4o-mini", f"trace-{i+1:03d}")
        (before_traces / f"trace-{i+1:03d}.json").write_text(json.dumps(trace, indent=2))

    for i, (q, ans) in enumerate(zip(TEST_QUESTIONS, finetuned_answers)):
        trace = create_trace(q['question'], ans, q['expected'], q['context'], "gpt-4o-mini-finetuned", f"trace-{i+1:03d}")
        (after_traces / f"trace-{i+1:03d}.json").write_text(json.dumps(trace, indent=2))

    print("\n--- BASE MODEL SCORES ---")
    score_traces_with_evalforge(str(before_traces), str(before_results))
    print("\n--- FINE-TUNED MODEL SCORES ---")
    score_traces_with_evalforge(str(after_traces), str(after_results))
    print("\n--- COMPARISON ---")
    subprocess.run([EVALFORGE_BIN, "compare", "--before", str(before_results), "--after", str(after_results)])
    print("\n--- GENERATING REPORT ---")
    report_path = demo_dir / "report.html"
    subprocess.run([EVALFORGE_BIN, "report", "--results", str(after_results), "--output", str(report_path), "--title", "Customer Support Fine-tuning Report"])
    print(f"Report: open {report_path}")


def run_real_demo():
    from openai import OpenAI
    client = OpenAI()

    print("=" * 60)
    print("🔥 EvalForge Customer Support Fine-tuning Demo")
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
        answer = run_model(client, base_model, q['question'])
        print(f"  Answer: {answer[:80]}...")
        trace = create_trace(
            q['question'], answer, q['expected'], q['context'],
            base_model, f"trace-{i+1:03d}"
        )
        (before_traces / f"trace-{i+1:03d}.json").write_text(json.dumps(trace, indent=2))

    print("\n🔍 Scoring base model with EvalForge...")
    score_traces_with_evalforge(str(before_traces), str(before_results))

    # STEP 2 — Fine-tune
    print("\n🎯 STEP 2: Fine-tuning gpt-4o-mini")
    print("-" * 40)
    file_id = upload_training_data(client)
    job_id = start_fine_tuning(client, file_id)
    fine_tuned_model = wait_for_fine_tuning(client, job_id)

    # STEP 3 — Evaluate fine-tuned model
    print("\n📊 STEP 3: Evaluating fine-tuned model")
    print("-" * 40)

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"  Question {i+1}: {q['question']}")
        answer = run_model(client, fine_tuned_model, q['question'])
        print(f"  Answer: {answer[:80]}...")
        trace = create_trace(
            q['question'], answer, q['expected'], q['context'],
            fine_tuned_model, f"trace-{i+1:03d}"
        )
        (after_traces / f"trace-{i+1:03d}.json").write_text(json.dumps(trace, indent=2))

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
        "--title", "Customer Support Fine-tuning Report"
    ])
    print(f"✅ Report saved to: {report_path}")
    print(f"   Open in browser: open {report_path}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


def main():
    if os.environ.get("MOCK_DEMO") == "1":
        run_mock_demo()
    else:
        run_real_demo()


if __name__ == "__main__":
    main()
