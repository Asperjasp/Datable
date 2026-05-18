"""
Daily batch runner. Run as:
    python -m tracker.runner
    python -m tracker.runner --models qwen_direct,mistral_large --date 2026-05-12
"""
import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from tracker.llm_clients import (
    AVAILABLE_MODELS,
    BaseLLMClient,
    DAILY_MODELS,
    build_selected_clients,
)
from tracker.prompts import ALL_PROMPTS, ALL_PROMPTS_WITH_EXPERIMENTAL, DAILY_PROMPTS, EXPERIMENTAL_PROMPTS, Prompt
from tracker.storage import TrackerRecord, append_to_timeseries, build_record, save_records

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("runner")

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


async def run_daily(
    prompts: List[Prompt] = DAILY_PROMPTS,
    model_keys: Optional[List[str]] = None,
    date_override: Optional[str] = None,
) -> List[TrackerRecord]:
    today = date_override or date.today().isoformat()
    keys = model_keys or DAILY_MODELS
    log.info(f"=== debat-zero daily run: {today} ===")

    clients = build_selected_clients(keys)
    if not clients:
        log.error("No LLM clients available — check API keys in .env")
        return []

    total = len(prompts) * len(clients)
    log.info(f"{len(prompts)} prompts × {len(clients)} models = {total} queries")

    all_records: List[TrackerRecord] = []

    async def _run_and_save(model_key: str, client: BaseLLMClient) -> List[TrackerRecord]:
        log.info(f"Starting {model_key}")
        records = await _run_model(model_key, client, prompts)
        path = save_records(records, today, model_key, BASE_PATH)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="debat-zero daily runner")
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated model keys. Available: {', '.join(AVAILABLE_MODELS)}. Default: DAILY_MODELS set.",
    )
    parser.add_argument("--prompts", choices=["daily", "all", "experimental"], default="daily")
    args = parser.parse_args()

    model_keys = args.models.split(",") if args.models else None
    if args.prompts == "experimental":
        prompts = EXPERIMENTAL_PROMPTS
    elif args.prompts == "all":
        prompts = ALL_PROMPTS_WITH_EXPERIMENTAL
    else:
        prompts = DAILY_PROMPTS

    asyncio.run(run_daily(prompts=prompts, model_keys=model_keys, date_override=args.date))


if __name__ == "__main__":
    main()
