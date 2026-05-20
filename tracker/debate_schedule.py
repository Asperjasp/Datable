"""
Daily debate schedule — candidates, topics, judge rotation.

Judge rotation: 3-day blocks alternating Claude ↔ Qwen
Starting 2026-05-20 (Wednesday): Claude block, then Qwen block, repeat.

Actor assignment: Each candidate is played by a different LLM to measure
cross-model consistency and avoid single-model bias.
"""
from datetime import date

_EPOCH = date(2026, 5, 20)
_BLOCK_SIZE = 3

_JUDGE_BLOCKS = ["claude", "qwen_or"]  # alternating


def get_judge_for_date(d: date = None) -> str:
    if d is None:
        d = date.today()
    days = max(0, (d - _EPOCH).days)
    return _JUDGE_BLOCKS[(days // _BLOCK_SIZE) % len(_JUDGE_BLOCKS)]


# Each matchup rotates daily. Actors are assigned to cross different LLM families.
_MATCHUPS = [
    {
        "candidate_a": "cepeda",       "actor_a": "qwen_or",
        "candidate_b": "de_la_espriella", "actor_b": "openai",
        "topic": "Salud y sistema de salud en Colombia",
    },
    {
        "candidate_a": "fajardo",      "actor_a": "claude",
        "candidate_b": "valencia",     "actor_b": "qwen_or",
        "topic": "Economía y modelo de Estado",
    },
    {
        "candidate_a": "lopez",        "actor_a": "openai",
        "candidate_b": "cepeda",       "actor_b": "claude",
        "topic": "Seguridad y paz total",
    },
    {
        "candidate_a": "valencia",     "actor_a": "qwen_or",
        "candidate_b": "de_la_espriella", "actor_b": "openai",
        "topic": "Educación y reforma educativa",
    },
    {
        "candidate_a": "fajardo",      "actor_a": "claude",
        "candidate_b": "lopez",        "actor_b": "qwen_or",
        "topic": "Medio ambiente y transición energética",
    },
    {
        "candidate_a": "cepeda",       "actor_a": "openai",
        "candidate_b": "valencia",     "actor_b": "claude",
        "topic": "Bienestar social y reducción de pobreza",
    },
    {
        "candidate_a": "de_la_espriella", "actor_a": "qwen_or",
        "candidate_b": "lopez",        "actor_b": "openai",
        "topic": "Infraestructura y desarrollo regional",
    },
]


def get_matchup_for_date(d: date = None) -> dict:
    if d is None:
        d = date.today()
    days = max(0, (d - _EPOCH).days)
    return _MATCHUPS[days % len(_MATCHUPS)]


def get_schedule_summary(num_days: int = 7) -> list[dict]:
    """Return the schedule for the next N days."""
    today = date.today()
    summary = []
    for i in range(num_days):
        from datetime import timedelta
        d = today + timedelta(days=i)
        m = get_matchup_for_date(d)
        summary.append({
            "date": d.isoformat(),
            "candidate_a": m["candidate_a"],
            "candidate_b": m["candidate_b"],
            "actor_a": m["actor_a"],
            "actor_b": m["actor_b"],
            "topic": m["topic"],
            "judge": get_judge_for_date(d),
        })
    return summary
