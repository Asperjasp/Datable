#!/usr/bin/env python3
"""
Run today's scheduled debate — CLI runner for cron / GitHub Actions.

Usage:
    python scripts/run_daily_debate.py               # today's schedule
    python scripts/run_daily_debate.py --date 2026-05-22   # specific date
    python scripts/run_daily_debate.py --questions 10      # full 10-question debate
"""
import asyncio
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from tracker.debate_schedule import get_judge_for_date, get_matchup_for_date
from tracker.debate_orchestrator import (
    OrchestratedDebate, ParticipantConfig,
    run_orchestrated_debate, save_orchestrated_debate,
)
from tracker.llm_clients import build_selected_clients


async def main(run_date: date, max_questions: int = 5):
    judge_key = get_judge_for_date(run_date)
    matchup = get_matchup_for_date(run_date)

    print(f"\n{'='*60}")
    print(f"  Debat-Zero — {run_date}")
    print(f"{'='*60}")
    print(f"  Topic:       {matchup['topic']}")
    print(f"  Candidate A: {matchup['candidate_a']} (actor: {matchup['actor_a']})")
    print(f"  Candidate B: {matchup['candidate_b']} (actor: {matchup['actor_b']})")
    print(f"  Judge:       {judge_key}")
    print(f"  Questions:   {max_questions}")
    print(f"{'='*60}\n")

    debate = OrchestratedDebate(
        debate_id=f"{run_date.isoformat()}-{matchup['candidate_a']}-vs-{matchup['candidate_b']}",
        topic=matchup["topic"],
        participant_a=ParticipantConfig(
            candidate_key=matchup["candidate_a"],
            mode="ai",
            actor_model_key=matchup["actor_a"],
        ),
        participant_b=ParticipantConfig(
            candidate_key=matchup["candidate_b"],
            mode="ai",
            actor_model_key=matchup["actor_b"],
        ),
        judge_model_key=judge_key,
        max_questions=max_questions,
    )

    models_needed = {judge_key, matchup["actor_a"], matchup["actor_b"]}
    clients = build_selected_clients(list(models_needed))

    # Fallback missing models to claude
    available = set(clients.keys())
    missing = models_needed - available
    if missing:
        print(f"  WARNING: Models unavailable: {missing}")
        fallback = "claude" if "claude" in available else next(iter(available), None)
        if not fallback:
            print("  ERROR: No LLM clients available. Check API keys.")
            sys.exit(1)
        print(f"  Falling back to: {fallback}")
        if matchup["actor_a"] not in available:
            debate.participant_a.actor_model_key = fallback
        if matchup["actor_b"] not in available:
            debate.participant_b.actor_model_key = fallback
        if judge_key not in available:
            debate.judge_model_key = fallback

    print("  Running debate...\n")
    result = await run_orchestrated_debate(debate, clients)

    path = save_orchestrated_debate(result)

    print(f"\n{'='*60}")
    print(f"  Status:  {result.status}")
    print(f"  Tokens:  {result.total_tokens}")
    print(f"  Saved:   {path}")

    if result.analytics:
        summary = result.analytics.get("summary", "")
        if summary:
            print(f"\n  SUMMARY:\n  {summary}")
        ev = result.analytics.get("evidence_quality", {})
        sp = result.analytics.get("specificity", {})
        if ev:
            print(f"\n  Evidence quality: {ev}")
        if sp:
            print(f"  Specificity:      {sp}")

    if result.error:
        print(f"\n  ERROR: {result.error}")
        sys.exit(1)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--questions", type=int, default=5, help="Max questions (1-10)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    asyncio.run(main(run_date, max(1, min(10, args.questions))))
