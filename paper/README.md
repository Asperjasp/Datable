# Paper — LLM Political Bias in Non-WEIRD Contexts

## Project Status
**Phase:** Active Data Collection  
**Target Submission:** TBD  
**Target Venues:** ACL, EMNLP, or LatinX in AI workshop

## Quick Links
- Methodology: `methodology.md`
- Literature Review: `lit_review/`
- Analysis Notebooks: `data_analysis/notebooks/`
- Intercalation Protocol: `../INTERCALATION_PROTOCOL.md`

## Todo

### Phase 1 (Data Collection) — Active
- [x] Implement regional model clients (US, China, Europe, LATAM)
- [x] Cross-provider consistency setup (qwen_direct vs qwen_or)
- [x] Bootstrap script for API keys
- [ ] Daily batch runs (May 12 — May 31)
- [ ] LATAM GPT benchmark integration

### Phase 1.5 (Analysis) — Pending
- [ ] Sentiment analysis pipeline (judge LLM)
- [ ] WEIRD-alignment scoring
- [ ] Cross-provider agreement metrics
- [ ] Regional comparison visualizations

### Phase 2 (Debat-0 Platform) — Future
- [ ] RAG actor system
- [ ] Moderator agent
- [ ] Data verification layer

## Daily Run Command
```bash
python scripts/bootstrap_keys.py  # First time only
python -m tracker.runner
```

## Models by Region

| Region | Models |
|--------|--------|
| **US** | claude, gpt4o, gemini |
| **China** | qwen_direct, qwen_or, deepseek, ling, baichuan |
| **Europe** | mistral_large, mistral_small |
| **LATAM** | llama_8b (HF proxy, LATAM GPT future) |
| **Local** | gemma2_local, gemma4_local |

## Cross-Provider Consistency Check
```
qwen_direct (native Dashscope)  ← compare →  qwen_or (OpenRouter proxy)
```
Same model architecture, different API path. Measures provider-level effects on:
- Response similarity
- Political sentiment
- Candidate preference
- Refusal rate

## Key Hypotheses

### H1: WEIRD Bias Projection
> LLMs trained primarily on Western Anglophone data will systematically favor candidates whose proposals align with WEIRD priors, regardless of Colombian electoral context.

**Test:** Compare Fajardo/Valencia (high WEIRD) vs Cepeda (low WEIRD) sentiment across models.

### H2: Provider-Independent Model Signature
> The same model architecture will show consistent political biases regardless of API provider.

**Test:** Compare qwen_direct vs qwen_or on all metrics.

### H3: Regional Model Cultural Alignment
> Non-US models (China, Europe, LATAM) will show different political preference patterns compared to US-trained models.

**Test:** Hierarchical clustering of model responses by region.

## Claude Code vs OpenCode Split on Paper Work

### Assign to Claude Code (High Complexity)
- Hypothesis refinement and statistical justification
- Literature review synthesis
- Result interpretation and discussion
- Paper drafting and narrative structure
- **Security review:** any code that processes sensitive election data
- **Complex stats:** hierarchical models, inter-annotator agreement

### Assign to OpenCode (Mechanical)
- Data loading and preprocessing scripts
- Visualization boilerplate
- Benchmark integration (LATAM GPT datasets)
- Reproducibility pipeline
- CI/CD for analysis notebooks

### Both
- Review each other's work
- Document in `INTERCALATION_PROTOCOL.md`
