#!/usr/bin/env python3
"""
Run today's scheduled debate — CLI runner for cron / GitHub Actions.

Match format (12 minutes, 3 questions):
  Per question: A-position → B-position → B-rebuttal → A-rebuttal
  Judge: generates questions + HOW/WHY/WHO analytics (no winner verdict)

Usage:
    python scripts/run_daily_debate.py               # today's schedule
    python scripts/run_daily_debate.py --date 2026-05-27
    python scripts/run_daily_debate.py --questions 3   # default
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
    TURN_POSITION, TURN_REBUTTAL,
)
from tracker.llm_clients import build_selected_clients
from tracker.candidates import CANDIDATE_BY_KEY


async def main(run_date: date, max_questions: int = 3):
    judge_key = get_judge_for_date(run_date)
    matchup   = get_matchup_for_date(run_date)

    cand_a = CANDIDATE_BY_KEY.get(matchup["candidate_a"])
    cand_b = CANDIDATE_BY_KEY.get(matchup["candidate_b"])

    print(f"\n{'='*70}")
    print(f"  DEBAT-ZERO — {run_date}  |  12-min match, {max_questions} questions")
    print(f"{'='*70}")
    print(f"  Tema:        {matchup['topic']}")
    print(f"  Candidato A: {cand_a.full_name if cand_a else matchup['candidate_a']}")
    print(f"               Actor: {matchup['actor_a']}")
    print(f"  Candidato B: {cand_b.full_name if cand_b else matchup['candidate_b']}")
    print(f"               Actor: {matchup['actor_b']}")
    print(f"  Juez:        {judge_key}")
    print(f"  Formato:     pos-A → pos-B → réplica-B → réplica-A (×{max_questions})")
    print(f"{'='*70}\n")

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

    available = set(clients.keys())
    missing   = models_needed - available
    if missing:
        print(f"  WARNING: Models unavailable: {missing}")
        fallback = "claude" if "claude" in available else next(iter(available), None)
        if not fallback:
            print("  ERROR: No LLM clients available. Check API keys.")
            sys.exit(1)
        print(f"  Falling back to: {fallback}")
        if matchup["actor_a"] not in available: debate.participant_a.actor_model_key = fallback
        if matchup["actor_b"] not in available: debate.participant_b.actor_model_key = fallback
        if judge_key not in available:           debate.judge_model_key = fallback

    print("  [FASE 1] Juez generando preguntas...\n")
    result = await run_orchestrated_debate(debate, clients, run_date=run_date)

    # ── Print full transcript ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TRANSCRIPCION COMPLETA")
    print(f"{'='*70}")

    for i, q in enumerate(result.questions):
        print(f"\n  PREGUNTA {i+1}: {q}\n")
        q_turns = sorted([t for t in result.turns if t.question_idx == i],
                         key=lambda t: t.turn_order)
        for turn in q_turns:
            label = "POSICIÓN" if turn.turn_type == TURN_POSITION else "RÉPLICA"
            print(f"  [{turn.speaker_display.upper()} — {label} | {turn.model_used}]")
            for line in turn.text.split("\n"):
                print(f"  {line}")
            if turn.artifacts:
                for art in turn.artifacts:
                    print(f"  📊 {art.get('title','')}: {art.get('content','')[:100]}")
            print(f"  [tokens: {turn.tokens_used}]\n")

    # ── Print analytics ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ANALISIS DEL JUEZ — HOW / WHY / WHO")
    print(f"{'='*70}")

    a = result.analytics or {}
    if a.get("raw"):
        print(f"\n  {a.get('summary','')[:600]}")
    else:
        for section, label in [("how","CÓMO"), ("why","POR QUÉ"), ("who","PARA QUIÉN")]:
            d = a.get(section, {})
            if d:
                print(f"\n  {label}:")
                for k, v in d.items():
                    c = CANDIDATE_BY_KEY.get(k)
                    print(f"    {c.display_name if c else k}: {v}")

        for section, label in [("strongest_arguments","ARGUMENTOS MÁS SÓLIDOS"),
                                ("weakest_points","PUNTOS MÁS DÉBILES")]:
            d = a.get(section, {})
            if d:
                print(f"\n  {label}:")
                for k, v in d.items():
                    c = CANDIDATE_BY_KEY.get(k)
                    print(f"    {c.display_name if c else k}: {v}")

        tensions = a.get("key_tensions", [])
        if tensions:
            print("\n  TENSIONES CLAVE:")
            for t in tensions: print(f"    · {t}")

        unanswered = a.get("unanswered", [])
        if unanswered:
            print("\n  SIN RESPUESTA CONCRETA:")
            for u in unanswered: print(f"    · {u}")

        ev = a.get("evidence_quality", {})
        sp = a.get("specificity", {})
        rb = a.get("rebuttal_effectiveness", {})
        if ev:
            print("\n  PUNTUACIONES (0-10):")
            for k in ev:
                c = CANDIDATE_BY_KEY.get(k)
                n = c.display_name if c else k
                print(f"    {n}: evidencia={ev.get(k)}  especificidad={sp.get(k)}  réplica={rb.get(k)}")

        if a.get("summary"):
            print(f"\n  RESUMEN:\n  {a['summary']}")
        if a.get("disclaimer"):
            print(f"\n  {a['disclaimer']}")

    path = save_orchestrated_debate(result)
    print(f"\n{'='*70}")
    print(f"  STATUS: {result.status}   TOKENS: {result.total_tokens}")
    print(f"  GUARDADO: {path}")
    print(f"{'='*70}\n")

    if result.error:
        print(f"  ERROR: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",      default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--questions", type=int, default=3, help="Questions per debate (default: 3)")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    asyncio.run(main(run_date, max(1, min(5, args.questions))))
