# Which Humans? (Atari et al., 2023)

## Core Paper
- **Title:** Which Humans? AI responses reflect WEIRD populations
- **URL:** https://psyarxiv.com/5b26t
- **Authors:** Atari, M., Xue, M.J., Park, P.S., Blasi, D.E., Henrich, J.
- **Year:** 2023

## Key Finding

> "LLM responses correlate strongly with WEIRD populations (r = -0.70 decline as you move away from WEIRD)."

The paper shows that:
1. LLMs systematically favor responses that align with WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural values
2. This correlation decreases as you move away from WEIRD populations
3. LLMs may be projecting Western values onto non-Western contexts

## Our Extension

We extend this methodology to:
1. **Political context** rather than general cultural questions
2. **Colombian 2026 election** — a non-WEIRD electoral context
3. **Multi-model comparison** across regions (US, China, Europe, LATAM)

## Specific Hypothesis

See `../methodology.md` for full details.

**Testable Prediction:** US models will prefer Fajardo/Valencia (high WEIRD alignment) over Cepeda (low WEIRD alignment).

## Candidate WEIRD Alignment Classification (per plan)

| Candidate | WEIRD Alignment | Policy Basis |
|-----------|-----------------|--------------|
| Cepeda | **Low** | State-led services, indigenous rights, non-WEIRD priors |
| De la Espriella | **Medium** | Hard-right but market-friendly (mixed signal) |
| Valencia | **High** | Market-friendly, US-aligned, conservative-WEIRD |
| Fajardo | **High** | Technocratic-centrist, high-WEIRD signal |
| López | **Medium** | Center-left, mixed signals |

## Related Literature

### The WEIRDest People in the World (Henrich, 2020)
- Book that establishes the WEIRD framework
- Shows how Western psychology is unusual in global context

### Moral Machine (Awad et al., 2018)
- Shows cross-cultural variation in moral decision-making
- LLM biases may not account for this variation

### On the conversational persuasiveness of LLMs (Salvi et al., 2024)
- Shows GPT-4 with personalized info is more persuasive than humans
- Relevant for our timing hypothesis: does persuasive bias intensify as election approaches?

## To Read for Claude Code

- [ ] Full text of Atari et al. (2023)
- [ ] Henrich (2020) relevant sections
- [ ] Salvi et al. (2024)
- [ ] Related work on LLM political bias

## To Code for OpenCode

- [x] Implement candidate metadata (`tracker/candidates.py` — DONE)
- [ ] Implement WEIRD-alignment as a feature in analysis
- [ ] Build regional classification in `llm_clients.py` (DONE)
