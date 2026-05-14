# Evaluation Module

This module integrates benchmark datasets from LATAM GPT and other sources for evaluating model performance on Latin American cultural and political knowledge.

## LATAM GPT Datasets

| Dataset | Purpose | Size | Notes |
|---------|---------|------|-------|
| `latam-gpt/Trueque-Benchmark-beta-0.1` | Latin American cultural knowledge QA | 500 questions | 20 countries, human-reviewed |
| `latam-gpt/CHOCLO` | Latin American cultural knowledge | 104K+ rows | 18 countries, 3 difficulty levels |
| `latam-gpt/personas-instruct-messages` | Instruction tuning | 393K messages | - |

## Usage

```python
from evaluation.latam_gpt_benchmark import run_trueque_benchmark

# Run Trueque benchmark across all available models
results = await run_trueque_benchmark(model_keys=None)
print(f"Accuracy by region: {results['accuracy_by_region']}")
```

## Benchmark Integration Status

- [ ] Trueque-Benchmark download and runner
- [ ] CHOCLO dataset integration
- [ ] Results analysis pipeline
- [ ] Visualization of model performance by region

## Intercalation Protocol

### Assign to Claude Code
- Benchmark design and statistical validation
- Question formulation and bias mitigation
- Results interpretation

### Assign to OpenCode
- Dataset loading scripts
- Batch execution pipeline
- Results storage and serialization
