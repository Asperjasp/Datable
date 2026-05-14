# Claude Code ↔ OpenCode Intercalation Protocol

> Version 1.0 — for use across all projects

## Purpose

Maximize quality and cost-efficiency by assigning tasks to the right tool, while maintaining full traceability and collaboration.

## Core Principle

| Tool | Strength | Use When |
|------|----------|----------|
| **Claude Code** | High-complexity reasoning, security, architecture | Planning, security reviews, complex logic, paper writing |
| **OpenCode** | Mechanical implementation, speed, cost | Wiring, config, tests, scripts, data pipelines |

---

## Task Classification Matrix

### Phase: Architecture & Design

| Task Type | Primary Owner | Secondary | Documentation Marker |
|-----------|---------------|-----------|----------------------|
| System architecture decisions | Claude Code | Review | `[CLAUDE-DESIGNED]` |
| Security model design | Claude Code | Review | `[SECURITY-REVIEW]` |
| Database schema design | Claude Code | Review | `[CLAUDE-DESIGNED]` |
| API contract design | Claude Code | Implementation | `[CLAUDE-DESIGNED]` |
| Prompt engineering (research) | Claude Code | OpenCode (execution) | `[PROMPT-ENGINEERED]` |
| Paper methodology design | Claude Code | - | `[METHODOLOGY-NOTE]` |

### Phase: Implementation

| Task Type | Primary Owner | Secondary | Documentation Marker |
|-----------|---------------|-----------|----------------------|
| Boilerplate code | OpenCode | Claude Code (review if complex) | `[OPENCODE-WIRED]` |
| Configuration files | OpenCode | Simple review | `[CONFIG-MANUAL-EDIT]` |
| Test scaffolding | OpenCode | Claude Code (logic tests) | `[TEST-SCAFFOLD]` |
| Data pipelines | OpenCode | Claude Code (stats review) | `[OPENCODE-WIRED]` |
| Scripts and utilities | OpenCode | - | `[OPENCODE-WIRED]` |
| CSS/UI components | OpenCode | - | `[OPENCODE-WIRED]` |
| CI/CD pipeline | OpenCode | Security review | `[OPENCODE-WIRED]` |

### Phase: Security Review

**Always assign to Claude Code:**

| Task | Rationale |
|------|-----------|
| Authentication code | Security-critical, requires deep reasoning |
| Authorization logic | Complex edge cases |
| Input validation & sanitization | Injection prevention |
| Secret handling | Key management, exposure risks |
| API key storage | Secure configuration patterns |
| Database query construction | SQL injection risks |
| RAG pipeline with sensitive data | Data leakage prevention |

### Phase: Research & Writing

| Task Type | Primary Owner | Notes |
|-----------|---------------|-------|
| Literature review | Claude Code | Synthesis, not just search |
| Paper drafting | Claude Code | Narrative structure |
| Result interpretation | Claude Code | Statistical reasoning |
| Documentation | Claude Code | Quality of explanation |
| README first draft | OpenCode + Claude | OpenCode drafts, Claude polishes |

---

## Inline Comment Markers

Every file should have ownership markers. Add these as comments:

```python
# [CLAUDE-DESIGNED] Security: HF model_id must be validated against
# an allowlist to prevent prompt injection via crafted repo names.
# See paper/methodology.md for the regional model classification.

# [OPENCODE-WIRED] HuggingFaceClient implementation follows TogetherClient pattern.
# Test coverage in tests/test_llm_clients.py.

# [SECURITY-REVIEW] Never log full API keys. Use masking: key[:8] + "..."
# Key fingerprinting in scripts/bootstrap_keys.py for verification without storage.

# [METHODOLOGY-NOTE] Cross-provider consistency check:
# qwen_direct (native) vs qwen_or (OpenRouter)
# Measures provider-level effects vs architecture effects.
# See paper/methodology.md for full rationale.
```

### Markers Cheatsheet

| Marker | Owner | Purpose |
|--------|-------|---------|
| `[CLAUDE-DESIGNED]` | Claude Code | Architecture/security decisions |
| `[OPENCODE-WIRED]` | OpenCode | Mechanical implementation |
| `[SECURITY-REVIEW]` | Claude Code | Security-critical code |
| `[METHODOLOGY-NOTE]` | Claude Code | Research methodology |
| `[CONFIG-MANUAL-EDIT]` | Both | User-specific config |
| `[TEST-SCAFFOLD]` | OpenCode | Test infrastructure |

---

## Workflow: Starting a New Project

### Step 1: Claude Code Phase (Planning)

1. **Claude Code** creates:
   - Architecture document (`ARCHITECTURE.md`)
   - Security model
   - Data model design
   - API contract

2. **Claude Code** reviews with user and documents decisions.

### Step 2: OpenCode Phase (Implementation)

1. **OpenCode** reviews architecture docs
2. **OpenCode** implements:
   - Config files
   - Boilerplate code
   - Test scaffolding
   - Data pipelines
3. **OpenCode** adds `[OPENCODE-WIRED]` markers

### Step 3: Claude Code Phase (Review)

1. **Claude Code** reviews:
   - Security-critical sections
   - Complex logic
   - Test logic
2. **Claude Code** adds `[SECURITY-REVIEW]` markers where reviewed
3. **Claude Code** suggests fixes or approves

### Step 4: OpenCode Phase (Polish)

1. **OpenCode** addresses feedback
2. **OpenCode** runs tests
3. **OpenCode** commits and opens PR

### Step 5: Claude Code Phase (Merge)

1. **Claude Code** does final security scan
2. **Claude Code** approves merge (if user wants)

---

## Workflow: Ongoing Project (Like This One)

### Current State: Debat-Zero

| Component | Status | Next |
|-----------|--------|------|
| `llm_clients.py` | OpenCode updated | Claude Code reviews security |
| Bootstrap script | OpenCode created | Claude Code reviews key handling |
| Paper structure | OpenCode created | Claude Code expands methodology |
| Intercalation protocol | This document | Both use |

### Next Actions

1. **You (user)** run `python scripts/bootstrap_keys.py` to enter API keys
2. **You (user)** create the GitHub repo or tell me to help
3. **Claude Code** should review `llm_clients.py` for security
4. **OpenCode** (me) is ready to implement remaining features

---

## Project-Specific: Debat-Zero Assignments

### For This Project

| Task | Primary Tool | Rationale |
|------|--------------|-----------|
| Add more LLM clients | OpenCode | Mechanical, follows existing patterns |
| Security review of LLM clients | Claude Code | Key handling, input validation |
| Daily runner execution | Either | Simple scripts |
| Judge LLM pipeline | Claude Code | Complex logic, inter-annotator agreement |
| Paper drafting | Claude Code | Academic writing |
| Data visualization | OpenCode | Mechanical, well-defined |
| Candidate document RAG | Claude Code | Security and retrieval logic |
| Moderator agent | Claude Code | Complex multi-agent logic |
| Bootstrap script | OpenCode (done) | Mechanical key handling |
| Config files | OpenCode (done) | Mechanical |

### Security Items for Claude Code Review

| File | What to Review |
|------|----------------|
| `llm_clients.py` | API key exposure, model_id validation (HuggingFaceClient) |
| `scripts/bootstrap_keys.py` | Secure key handling, .env permissions, fingerprinting |
| `tracker/storage.py` | Data persistence, CSV injection prevention |

---

## Handoff Protocol

When switching tools, leave:

### OpenCode → Claude Code

```
# [HANDOFF: OpenCode → Claude Code]
# OpenCode completed:
# - Added MistralClient, QwenClient, HuggingFaceClient
# - Updated AVAILABLE_MODELS with regional classification
# - Created bootstrap script
# - Created paper/ structure
#
# Claude Code should:
# - Review security of llm_clients.py (especially HuggingFaceClient model_id)
# - Review key handling in bootstrap_keys.py
# - Expand paper/methodology.md with stats methodology
```

### Claude Code → OpenCode

```
# [HANDOFF: Claude Code → OpenCode]
# Claude Code reviewed:
# - Security: API key exposure is minimal
# - Architecture: regional model classification is sound
#
# OpenCode should:
# - Implement the judge LLM pipeline per spec in paper/methodology.md
# - Add more models to AVAILABLE_MODELS from OpenRouter
# - Write test_llm_clients.py tests for new clients
```

---

## Repository Structure for Collaboration

```
project/
├── ARCHITECTURE.md          # Claude Code owned
├── INTERCALATION_PROTOCOL.md  # This file, both own
├── SECURITY_REVIEW.md        # Claude Code owned
├── .github/
│   └── CODEOWNERS            # Auto-enforce task routing
├── paper/
│   ├── methodology.md        # Claude Code owned
│   └── lit_review/           # Claude Code owned
├── tracker/
│   ├── llm_clients.py        # Both review, security by Claude
│   └── prompts.py            # OpenCode implements, Claude designs
├── scripts/
│   └── bootstrap_keys.py     # OpenCode implements, Claude reviews
└── tests/
    └── test_*.py             # OpenCode implements, Claude reviews logic
```

---

## CODEOWNERS Template

Create `.github/CODEOWNERS`:

```
# Architecture/security decisions require Claude review
/ARCHITECTURE.md               @claude-owner
/SECURITY_REVIEW.md            @claude-owner
*/SECURITY*                    @claude-owner

# Paper/methodology requires Claude review
/paper/                       @claude-owner

# Security-critical code
/tracker/*llm*.py             @claude-owner  @opencode-owner
/scripts/bootstrap_keys.py    @claude-owner  @opencode-owner

# Tests and config can be OpenCode
/tests/                       @opencode-owner
*.config.js                   @opencode-owner
*.toml                        @opencode-owner
.env.example                  @opencode-owner
```

---

## Quick Reference Cheatsheet

### Ask Claude Code When...
- You're unsure about security
- You need architecture decisions
- You're writing a paper or research
- You need complex logic design
- You want statistical reasoning

### Ask OpenCode When...
- You need wiring/boilerplate
- You want test scaffolding
- You need data pipelines
- You want config files edited
- You need scripts created
- You want cost-efficient mechanical work

### Always Document...
- Security decisions with `[SECURITY-REVIEW]`
- Architecture with `[CLAUDE-DESIGNED]`
- Methodology with `[METHODOLOGY-NOTE]`
- Implementation with `[OPENCODE-WIRED]`

---

## Next Steps for This Project

1. **User:** Run `python scripts/bootstrap_keys.py` to enter API keys
2. **User:** Create GitHub repo at `github.com/asperjasp/Datable`
3. **Claude Code:** Review `llm_clients.py` and `bootstrap_keys.py` for security
4. **OpenCode:** If needed, implement more features
5. **Both:** Use this protocol for collaboration

---

## Contribution to OpenSource

This protocol could be generalized into:
1. A template repo for Claude+OpenCode collaboration
2. A VS Code extension that suggests task routing
3. A git hook that enforces documentation markers

**User's idea:** You have a fork where you want to start planning this for contribution.

For this repo, we should:
- Document the protocol (this file)
- Use the markers consistently
- Create `ARCHITECTURE.md` and `SECURITY_REVIEW.md`
- Consider a `CODEOWNERS` file

When ready to generalize, we can extract this into a standalone template project.
