# EvalForge Fine-tuning Demo

End-to-end demo showing fine-tuning + evaluation with EvalForge.

## What this demo does

1. Runs base gpt-4o-mini on geography questions
2. Scores with EvalForge (faithfulness, goal_completion, hallucination)
3. Fine-tunes gpt-4o-mini on capitals dataset
4. Scores fine-tuned model with EvalForge
5. Compares before vs after
6. Generates HTML report

## Setup

```bash
pip install openai evalforge
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
export EVALFORGE_BIN=/path/to/evalforge/target/debug/evalforge
```

## Run

```bash
python demo.py
```

## Cost

- OpenAI fine-tuning: ~$2-5
- Anthropic eval calls: ~$0.50
- Total: ~$3-6

## Expected output

Before fine-tuning: faithfulness ~0.65-0.75
After fine-tuning: faithfulness ~0.85-0.95
Improvement: +0.15 to +0.25
