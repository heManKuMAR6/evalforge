# EvalForge Python SDK

pip install evalforge

## Quick Start
import evalforge
result = evalforge.run("trace.json", metrics=["faithfulness"])
print(result.passed)
print(result.metrics[0].score)
