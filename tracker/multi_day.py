"""
Multi-day runner for Debat-Zero.

Orchestrates running the daily bias tracker across multiple consecutive days,
accumulates results, and generates summary reports.
"""

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from tracker.llm_clients import BaseLLMClient, build_selected_clients
from tracker.prompts import DAILY_PROMPTS, Prompt
from tracker.storage import TrackerRecord, append_to_timeseries, build_record, save_records

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("multi_day")

BASE_PATH = Path(__file__).parent.parent
TEMPERATURE = 0.7
MAX_TOKENS = 1000
CONCURRENCY_PER_MODEL = 3


async def _run_model(
    model_key: str,
    client: BaseLLMClient,
    prompts: List[Prompt],
) -> List[TrackerRecord]:
    sem = asyncio.Semaphore(CONCURRENCY_PER_MODEL)

    async def _query(prompt: Prompt) -> TrackerRecord:
        async with sem:
            log.info(f"  [{model_key}] → {prompt.id}")
            resp = await client.query(prompt.text, TEMPERATURE, MAX_TOKENS)
            if resp.error:
                log.warning(f"  [{model_key}] {prompt.id} ERROR: {resp.error}")
            return build_record(
                model=client.model_id,
                provider=client.provider,
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                raw_response=resp.raw_text,
                token_count=resp.output_tokens,
                latency_ms=resp.latency_ms,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                error=resp.error,
            )

    return list(await asyncio.gather(*[_query(p) for p in prompts]))


async def run_single_day(
    day: date,
    prompts: List[Prompt] = DAILY_PROMPTS,
    model_keys: Optional[List[str]] = None,
) -> List[TrackerRecord]:
    """Run one day of queries across selected (or all) models."""
    log.info(f"=== debat-zero run: {day.isoformat()} ===")

    if model_keys is None:
        from tracker.llm_clients import AVAILABLE_MODELS
        model_keys = list(AVAILABLE_MODELS.keys())

    clients = build_selected_clients(model_keys)
    if not clients:
        log.error("No LLM clients available — check API keys in .env")
        return []

    total = len(prompts) * len(clients)
    log.info(f"{len(prompts)} prompts × {len(clients)} models = {total} queries")

    all_records: List[TrackerRecord] = []

    async def _run_and_save(model_key: str, client: BaseLLMClient) -> List[TrackerRecord]:
        log.info(f"Starting {model_key}")
        records = await _run_model(model_key, client, prompts)
        path = save_records(records, day.isoformat(), model_key, BASE_PATH)
        successes = sum(1 for r in records if not r.error)
        log.info(f"[{model_key}] {successes}/{len(records)} OK → {path}")
        return records

    results = await asyncio.gather(
        *[_run_and_save(k, c) for k, c in clients.items()],
        return_exceptions=False,
    )

    for batch in results:
        all_records.extend(batch)

    append_to_timeseries(all_records, BASE_PATH)

    refusals = sum(1 for r in all_records if r.refused_to_answer)
    errors = sum(1 for r in all_records if r.error)
    log.info(
        f"=== Done: {len(all_records)} records | "
        f"{refusals} refusals | {errors} errors ==="
    )
    return all_records


async def run_multi_day(
    start_date: date,
    num_days: int,
    prompts: List[Prompt] = DAILY_PROMPTS,
    model_keys: Optional[List[str]] = None,
    delay_seconds: float = 0,
) -> dict:
    """
    Run the tracker across `num_days` consecutive days.

    Args:
        start_date: First day to run (date object).
        num_days: Total number of days to run.
        prompts: Prompts to use each day.
        model_keys: Model keys to query (None = all available).
        delay_seconds: Wait between days (useful for rate limits).

    Returns:
        Summary dict with per-day and aggregate stats.
    """
    summary: dict = {"days": [], "total_records": 0, "total_errors": 0, "total_refusals": 0}

    for day_offset in range(num_days):
        current_day = start_date + timedelta(days=day_offset)
        log.info(f"\n{'='*60}")
        log.info(f"DAY {day_offset + 1}/{num_days} — {current_day.isoformat()}")
        log.info(f"{'='*60}")

        import time as _time
        if day_offset > 0 and delay_seconds > 0:
            log.info(f"Waiting {delay_seconds}s before next day...")
            _time.sleep(delay_seconds)

        records = await run_single_day(current_day, prompts, model_keys)
        day_stats = {
            "date": current_day.isoformat(),
            "records": len(records),
            "errors": sum(1 for r in records if r.error),
            "refusals": sum(1 for r in records if r.refused_to_answer),
            "successes": sum(1 for r in records if not r.error and not r.refused_to_answer),
        }
        summary["days"].append(day_stats)
        summary["total_records"] += len(records)
        summary["total_errors"] += day_stats["errors"]
        summary["total_refusals"] += day_stats["refusals"]

    summary["total_successes"] = summary["total_records"] - summary["total_errors"] - summary["total_refusals"]
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-day LLM bias tracker")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to run")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between days")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys (default: all)")
    parser.add_argument("--prompts", choices=["daily", "all"], default="daily", help="Prompt set")
    args = parser.parse_args()

    from tracker.prompts import ALL_PROMPTS, DAILY_PROMPTS

    start = date.today() if args.start is None else date.fromisoformat(args.start)
    model_keys = args.models.split(",") if args.models else None
    prompts = ALL_PROMPTS if args.prompts == "all" else DAILY_PROMPTS

    summary = asyncio.run(
        run_multi_day(start, args.days, prompts, model_keys, args.delay)
    )

    log.info("\n" + "=" * 60)
    log.info("MULTI-DAY SUMMARY")
    log.info("=" * 60)
    for d in summary["days"]:
        log.info(f"  {d['date']}: {d['records']} records, {d['errors']} errors, {d['refusals']} refusals")
    log.info(f"\nTOTAL: {summary['total_records']} records | "
             f"{summary['total_successes']} successes | "
             f"{summary['total_errors']} errors | "
             f"{summary['total_refusals']} refusals")


if __name__ == "__main__":
    main()