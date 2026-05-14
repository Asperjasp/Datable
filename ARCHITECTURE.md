# Debat-Zero Architecture

## Version 0.2.0

## Overview

Debat-Zero is a research platform for studying LLM political bias in non-WEIRD electoral contexts (Colombia 2026). It has two phases:

- **Phase 1 (Active):** Bias tracker — daily LLM querying across regional models
- **Phase 2 (Future):** Debat-0 platform — multi-agent AI debates

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  PHASE 1                                      │
│                            BIAS TRACKER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Models by Region                                │  │
│  ├─────────────┬─────────────┬─────────────┬─────────────┬────────────┤  │
│  │    US       │   China     │  Europe     │   LATAM     │   Local    │  │
│  ├─────────────┼─────────────┼─────────────┼─────────────┼────────────┤  │
│  │ Claude      │ qwen_direct │ mistral_    │ llama_8b    │ gemma2_    │  │
│  │ GPT-4o      │ qwen_or     │ large       │ (HF)        │ local      │  │
│  │ Gemini      │ deepseek    │ mistral_    │ ─────────── │ gemma4_    │  │
│  │             │ ling        │ small       │ LATAM GPT   │ local      │  │
│  │             │ baichuan    │             │ (future)    │            │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┴────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Daily Runner (tracker/runner.py)                 │  │
│  │  - 15 prompts × N models = 15N queries/day                           │  │
│  │  - Prompts: Category A (general), B (per-candidate), C (comparative)│  │
│  │  - Saves to data/raw/{date}/{model}.json                              │  │
│  │  - Appends to data/processed/timeseries.csv                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE 2 (FUTURE)                             │
│                             DEBAT-0 PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Multi-Agent Debate Orchestrator                   │  │
│  ├───────────┬───────────┬───────────┬─────────────────────────────────┤  │
│  │ Actor 1   │ Actor 2   │ Actor 3   │  Moderator Agent                │  │
│  │ (Fajardo) │ (Cepeda)  │ (Valencia)│  (Impartial Judge)              │  │
│  ├───────────┴───────────┴───────────┴─────────────────────────────────┤  │
│  │                        RAG per actor                                  │  │
│  │  - Campaign documents                                                │  │
│  │  - Speeches                                                          │  │
│  │  - Voting records                                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
debat-zero/
├── tracker/                    # Core library (Phase 1)
│   ├── __init__.py
│   ├── llm_clients.py         # LLM clients + regional model registry
│   ├── candidates.py          # 5 Colombian 2026 candidates
│   ├── prompts.py              # 15 daily prompts (A1-A5, B1×5, C1×5)
│   ├── debate.py               # Phase 2 classes (PresidentialElection)
│   ├── runner.py               # Daily batch runner
│   ├── multi_day.py            # Multi-day orchestration
│   └── storage.py              # JSON + CSV persistence
│
├── app/                        # FastAPI web app
│   ├── main.py                 # API endpoints
│   ├── schemas.py              # Pydantic models
│   └── templates/              # HTML frontend
│
├── paper/                      # Academic paper materials
│   ├── README.md               # Paper status and todo
│   ├── methodology.md          # Full methodology (WEIRD framework)
│   ├── lit_review/             # Literature notes
│   │   └── which_humans_atari_2023.md
│   └── data_analysis/
│       ├── notebooks/
│       └── figures/
│
├── evaluation/                 # Benchmark integration
│   ├── README.md               # LATAM GPT benchmarks
│   └── results/
│
├── scripts/
│   ├── run_daily.sh            # Cron entrypoint
│   └── bootstrap_keys.py       # Secure API key setup
│
├── data/
│   ├── raw/                    # Raw JSON responses per model per day
│   ├── processed/              # Aggregated timeseries.csv
│   └── debates/                # Saved debate JSONs (Phase 2)
│
├── tests/                      # pytest tests
├── INTERCALATION_PROTOCOL.md  # Claude Code ↔ OpenCode workflow
├── ARCHITECTURE.md             # This file
├── pyproject.toml
├── .env.example
└── .env                        # gitignored, actual keys
```

## Key Design Decisions

### 1. Regional Model Classification

Models grouped by training region for WEIRD bias analysis:

| Region | Models | Rationale |
|--------|--------|-----------|
| **US** | claude, gpt4o, gemini | Western commercial, WEIRD alignment baseline |
| **China** | qwen_direct, qwen_or, deepseek, ling, baichuan | Non-WEIRD, alternative perspective |
| **Europe** | mistral_large, mistral_small | France, intermediate between US and non-WEIRD |
| **LATAM** | llama_8b (proxy), LATAM GPT (future) | De facto regional model, future LATAM GPT |
| **Local** | gemma2_local, gemma4_local | Sovereignty comparison, runs on user hardware |

### 2. Cross-Provider Consistency Check

**Critical methodological decision:** Run the same model (Qwen 2.5 72B) via two different API paths:

- **qwen_direct:** Native Alibaba Dashscope API (`DASHSCOPE_API_KEY`)
- **qwen_or:** OpenRouter proxy (`OPENROUTER_API_KEY`)

**Why?** To isolate provider-level effects from model architecture effects.
- Do moderation layers vary by provider?
- Do system prompts differ?
- Is political bias introduced at the API layer?

See `paper/methodology.md` for full details.

### 3. Candidate WEIRD Alignment

Based on Atari et al. (2023) framework:

| Candidate | Coalition | Political Leaning | WEIRD Alignment |
|-----------|-----------|-------------------|-----------------|
| Iván Cepeda | Pacto Histórico | Left | **Low** |
| Abelardo de la Espriella | Defensores de la Patria | Right | **Medium** |
| Paloma Valencia | Centro Democrático | Center-Right | **High** |
| Sergio Fajardo | Compromiso Ciudadano | Center | **High** |
| Claudia López | Independiente | Center-Left | **Medium** |

**Core Hypothesis:** US models will prefer high-WEIRD candidates (Fajardo, Valencia) over low-WEIRD candidates (Cepeda).

### 4. Prompt Categories

| Category | Count | Purpose |
|----------|-------|---------|
| **A** (Direct) | 5 | "Who should I vote for?" — direct preference elicitation |
| **B** (Per-candidate) | 5 (B1 only) | Strengths/weaknesses per candidate |
| **C** (Comparative) | 5 (C1 only) | Cross-candidate comparison, designed to stress-test bias |

**Total:** 15 prompts per day per model.

## Security Model

### API Key Handling

- **.env file:** Mode 600 (user-readonly) after bootstrap
- **Secure input:** Uses `getpass` (no terminal echo)
- **Key verification:** Minimal API call before saving
- **Fingerprinting:** SHA256 hashes stored in `.key_status.json` (never plaintext keys)
- **Logging:** Keys never logged, only masked (`sk-or...123`)

### Security Review Items for Claude Code

| File | Risk |
|------|------|
| `tracker/llm_clients.py` | HuggingFaceClient accepts arbitrary model_id — could be prompt injection vector via crafted repo names |
| `scripts/bootstrap_keys.py` | Key handling, .env permissions, logging |
| `tracker/storage.py` | CSV injection prevention, JSON escaping |

## Intercalation Protocol

See `INTERCALATION_PROTOCOL.md` for the full workflow between Claude Code and OpenCode.

**Quick summary:**

| Task Type | Owner |
|-----------|-------|
| Architecture, security | Claude Code |
| Mechanical wiring, tests | OpenCode |
| Paper writing, stats | Claude Code |
| Scripts, config | OpenCode |

## Dependencies

### Core
```
anthropic>=0.30.0      # Claude
openai>=1.35.0         # GPT-4o, OpenRouter
google-generativeai>=0.7.0  # Gemini
ollama>=0.3.0          # Local Gemma
httpx>=0.27.0          # Mistral, Qwen, HuggingFace (no SDK needed)
pydantic>=2.7.0
python-dotenv>=1.0.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
```

### Analysis (Optional)
```
pandas>=2.2.0
jupyter>=1.0.0
matplotlib>=3.9.0
seaborn>=0.13.0
scipy>=1.13.0
datasets>=2.19.0      # HuggingFace datasets (LATAM GPT benchmarks)
```

## Quick Start

```bash
# 1. Enter API keys (securely)
python scripts/bootstrap_keys.py

# 2. Run daily batch
python -m tracker.runner

# 3. Web app (optional)
uvicorn app.main:app --reload
```

## Next Steps

### Immediate
- [x] Create bootstrap script
- [x] Add Mistral, Qwen, HuggingFace clients
- [x] Remove Together AI (US lab)
- [x] Add regional model classification
- [x] Create paper/ directory
- [x] Create evaluation/ directory
- [x] Create intercalation protocol
- [ ] User enters API keys via bootstrap script
- [ ] Create GitHub repo

### Short-term (1-3 days)
- [ ] Claude Code security review of `llm_clients.py`
- [ ] Write tests for new clients
- [ ] First daily run with all available models

### Medium-term (1-2 weeks)
- [ ] LATAM GPT benchmark integration
- [ ] Phase 1.5 judge pipeline for sentiment analysis
- [ ] First analysis notebook

### Long-term (Summer 2026)
- [ ] Phase 2: RAG actor system
- [ ] Phase 2: Moderator agent
- [ ] Phase 2: Data verification layer
