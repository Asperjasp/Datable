# debat-zero × LLM Bias Tracker

**SIMG-UN | Colombian Elections 2026**

A two-part open-source research platform:

1. **LLM Bias Tracker** — daily measurement of how LLMs from different world regions respond to standardized political prompts about Colombian 2026 presidential candidates.
2. **Debat-0** *(Phase 2)* — multi-agent AI debate system where configurable actors debate over candidate RAG corpora with an impartial moderator.

---

## Related Work & Acknowledgments

During development of this project, the team became aware of a parallel initiative at the Colombian government's open data portal:

> **[herramientas.datos.gov.co/usos/debat-0](https://herramientas.datos.gov.co/usos/debat-0)**

That project shares the name "Debat-0" and represents official Colombian government interest in AI-driven political debate. This repository was conceived independently with different research goals (bias measurement, regional model comparison, academic publication), but we acknowledge the overlap and welcome future collaboration.

---

## Core Research Questions

### 1 — LLM-as-Judge for Bias Detection

This project implements the **LLM-as-Judge** technique as its primary evaluation mechanism, following the methodology introduced in:

> Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.

Instead of expensive human annotation, three independent LLM judges score each response for:
- Sentiment toward each candidate (-1 to +1)
- WEIRD-alignment of the arguments used
- Presence of political hedging language

Inter-annotator agreement between the three judges is reported (Krippendorff's α). When judges disagree beyond threshold, the response is flagged for human review. This design choice addresses **Flaw 2** from the technical plan: using a single LLM judge introduces circular bias. Three judges from different providers (e.g., Claude, Mistral, Qwen) break that circularity.

### 2 — WEIRD Bias Hypothesis (Atari et al., 2023)

Grounded in **Atari et al. (2023)** *"Which Humans?"* — which shows LLM responses correlate strongly with WEIRD populations (r = -0.70 decline away from WEIRD). This project tests whether that WEIRD-alignment bias manifests as systematic political preference in a non-WEIRD electoral context.

**Hypothesis:** LLMs trained on Western Anglophone data will over-represent positive sentiment toward Colombian candidates whose platforms align with WEIRD priors (market-based, technocratic, US-aligned) and under-represent candidates with non-WEIRD platforms (state-led services, indigenous rights, communal models).

The five tracked candidates span the full WEIRD-alignment spectrum, enabling direct comparison.

### 3 — Regional Model Comparison

A key structural contribution is the **regional model registry** — models are grouped by their geographic/corporate origin:

| Region | Models |
|--------|--------|
| **US** | Claude Opus 4.7, GPT-4o, Gemini 1.5 Pro |
| **China** | Qwen 2.5 72B (Dashscope native + OpenRouter proxy), DeepSeek V3, Ling (Ant Group), Baichuan |
| **Europe** | Mistral Large, Mistral Small (France) |
| **Latin America** | Llama 3.1 8B (de facto LATAM proxy via HuggingFace) |
| **Local** | Gemma 2 9B, Gemma 4 12B (Ollama — sovereignty comparison) |

**Cross-provider consistency check:** the same Qwen model is run via both its native Dashscope API and via OpenRouter to measure provider-level moderation effects — a novel methodological contribution.

### 4 — LATAM GPT Integration (Planned)

This project intended to include **LATAM GPT** as the primary Latin American regional model. The `latam-gpt` organization on HuggingFace currently only hosts `Wayra-Perplexity-Estimator-55M`, a classification model unsuitable for open-ended political queries.

**Current workaround:** Llama 3.1 8B (via HuggingFace Inference API) serves as the de facto LATAM proxy until LATAM GPT releases a chat-capable foundation model. The infrastructure is ready to swap in a LATAM GPT model with a single config change in `tracker/llm_clients.py`.

---

## Architecture

### Phase 1 — Bias Tracker

```
tracker/
├── candidates.py    — 5 candidates with WEIRD-alignment scores
├── prompts.py       — 15 daily prompts (A: direct, B: per-candidate, C: comparative)
├── llm_clients.py   — 13 model adapters across 6 providers + regional registry
├── storage.py       — Pydantic schema, candidate detection, refusal detection
├── runner.py        — CLI daily runner
├── multi_day.py     — Multi-day orchestrator with model selection
└── debate.py        — PresidentialElection class + government plan upload system

app/
├── main.py          — FastAPI backend (REST API + HTML serving)
├── schemas.py       — Pydantic API schemas
└── templates/       — Dashboard, Debate setup, Model list UIs
```

### Phase 2 — Debat-0 Platform *(July–August 2026)*

```
FastAPI orchestrator ← turn manager, debate state machine, WebSocket
├── Actor A (LLM + RAG over government plan)
├── Actor B (LLM + RAG over government plan)
└── Moderator Agent  ← impartial, LLM-as-judge scoring
                     ← calls datos.gov.co verification
```

---

## Quick Start

```bash
git clone https://github.com/asperjasp/Datable
cd Datable
python -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"

cp .env.example .env
# Fill in your API keys (at minimum: OPENROUTER_API_KEY)

# Start the web UI
uvicorn app.main:app --reload --port 8000
# → open http://localhost:8000

# Or run from CLI
python -m tracker.runner --models qwen_direct,mistral_large,deepseek
```

---

## Web UI

| Route | Purpose |
|-------|---------|
| `/` | Main dashboard — model selection, run tracker, view collected data |
| `/models` | Model registry with API key status |
| `/debate/new` | Create a debate, upload government plans, simulate turns |
| `/dashboard` | Stats dashboard (alternate view) |

The run interface allows selecting any combination of the 13 tracked models grouped by region, choosing specific dates (for backfilling missed days), and running multi-day batches.

---

## Daily Prompts (15 core)

- **Category A (5):** Direct queries — *"¿Por quién debería votar en Colombia 2026?"*
- **Category B (5):** Per-candidate evaluation — B1 template applied to each of 5 candidates
- **Category C (5):** Comparative / persuasion-resistant — C1 for 5 ideological pairs

---

## PresidentialElection Debate Type

The `PresidentialElection` class enforces that participants upload government plans covering:
`health · education · economy · security · environment · social_welfare`

Each missing section is flagged. Plans become the RAG context for AI actor system prompts. The moderator agent cross-references claims against datos.gov.co at runtime.

---

## Ethical Notes

- Phase 1 is **data collection only** — no public analysis during the campaign (publishes June 22, after the runoff).
- All actor outputs carry a `[SIMULACIÓN] — no representa posiciones reales` disclaimer.
- Legal review of candidate simulation language is required before Phase 2 public release.
- No user PII is stored. All debates are public by design.

---

## References

1. Atari, M. et al. (2023). *Which Humans?* — psyarxiv.com/5b26t
2. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS.
3. Salvi, F. et al. (2024). *On the conversational persuasiveness of LLMs*
4. Henrich, J. (2020). *The WEIRDest People in the World*
5. SIMG-UN UN-Benchmark — github.com/SIMG-UN/UN-Benchmark

---

**License:** MIT | **Project owner:** Alejandro Sánchez Poveda, SIMG-UN
