# Git Workflow — Debat-Zero

## Branch strategy

```
main          ← stable, tested, always deployable
dev           ← integration branch (merge here before main)
feature/xxx   ← individual features (branch from dev)
hotfix/xxx    ← urgent production fixes (branch from main)
```

## When to commit

Commit **one logical change at a time**. A good commit answers:
> "What was broken / missing before this, and what is it now?"

Commit when you have:
- A working feature (even if incomplete at system level)
- A bug fix that passes its test
- A new knowledge base file
- A schema or config change
- A documentation update

**Never commit:**
- `.env` (API keys — already in `.gitignore`)
- `*.key_status.json`
- Half-broken code that breaks the server startup

## Commit message format

```
<type>(<scope>): <short summary>

[optional body — what problem this solves]
```

Types:
| Type | Use for |
|---|---|
| `feat` | New feature or endpoint |
| `fix` | Bug fix |
| `data` | Knowledge base / candidate data updates |
| `eval` | Evaluation scripts, judge pipeline |
| `docs` | README, paper, methodology docs |
| `refactor` | Internal restructure, no behavior change |
| `test` | Test additions or fixes |
| `chore` | Dependency bumps, config, CI |

Examples:
```
feat(simulator): add system_prompt support to all LLM clients
fix(routes): rename /api/debates/presets to avoid wildcard conflict
data(fajardo): add platform.md from 2026 campaign website
eval(judge): add Krippendorff alpha inter-annotator scorer
```

## When to push

Push to `origin` when:
- You finish a feature and it's tested locally
- Before taking a break / end of session
- Before sharing a demo URL

Do NOT push:
- Directly to `main` without going through `dev`
- Force-push to `main` ever

## Structured commit order for large features

When adding a major feature (e.g., new debate module), commit in this order:

1. **Data** — knowledge bases, fixtures, seed files
2. **Core logic** — tracker/, simulator, etc.
3. **API** — new endpoints in main.py, schemas
4. **Frontend** — HTML/JS/CSS changes
5. **Tests** — any new test coverage
6. **Docs** — README, paper updates

This order means each commit builds on stable prior layers.

## Preserving trial results / experimental branches

When running a model evaluation trial that you want to preserve:

```bash
# Tag the run before overwriting
git tag trial/2026-05-15-fajardo-vs-espriella

# Save raw outputs to data/runs/ before next run
cp -r data/raw/$(date +%Y-%m-%d) data/runs/trial-$(date +%Y-%m-%d)/
```

Tag naming: `trial/YYYY-MM-DD-description`

## Clean architecture rules

- **`tracker/`** — pure logic, no FastAPI imports
- **`app/`** — FastAPI layer only, thin wrappers over tracker/
- **`data/knowledge_bases/`** — one folder per candidate, markdown only
- **`data/raw/`** — daily bias collection output, gitignored (large)
- **`scripts/`** — one-off tools (bootstrap, data fetchers)
- **`paper/`** — LaTeX/markdown research paper, methodology docs
- **`tests/`** — pytest, mirrors tracker/ structure

Never import from `app/` inside `tracker/`. Data flows one way: tracker → app.

## Recommended daily session flow

```bash
# 1. Pull latest
git pull origin dev

# 2. Work in feature branch
git checkout -b feature/my-thing

# 3. Commit as you go (small, often)
git add tracker/debate_simulator.py
git commit -m "feat(simulator): add round-robin mode"

# 4. When done, merge to dev
git checkout dev
git merge feature/my-thing

# 5. Push dev
git push origin dev

# 6. When dev is stable, merge to main
git checkout main
git merge dev
git push origin main
git tag v0.x.0
```
