import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from tracker.candidates import CANDIDATES


class TrackerRecord(BaseModel):
    timestamp: str
    model: str
    model_provider: str
    prompt_id: str
    prompt_text: str
    language: str = "es-CO"
    raw_response: str
    response_length_tokens: int
    candidates_mentioned: List[str] = Field(default_factory=list)
    sentiment_per_candidate: Dict[str, Optional[float]] = Field(default_factory=dict)
    refused_to_answer: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


# Build per-candidate regex patterns from alias lists
_CANDIDATE_PATTERNS: Dict[str, re.Pattern] = {
    c.key: re.compile("|".join(re.escape(alias) for alias in c.aliases), re.IGNORECASE)
    for c in CANDIDATES
}

_REFUSAL_PATTERN = re.compile(
    r"no puedo recomendar|no me es posible|como (modelo|sistema) de (ia|inteligencia artificial)|"
    r"i cannot|no tengo preferencias|no (es|sería) apropiado|no debo recomendar|"
    r"sería inapropiado|no estoy en posici[oó]n|no me corresponde|"
    r"no puedo tomar partido|me abstengo de recomendar",
    re.IGNORECASE,
)


def detect_candidates_mentioned(text: str) -> List[str]:
    return [key for key, pat in _CANDIDATE_PATTERNS.items() if pat.search(text)]


def detect_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(text))


def build_record(
    *,
    model: str,
    provider: str,
    prompt_id: str,
    prompt_text: str,
    raw_response: str,
    token_count: int,
    latency_ms: float,
    temperature: float,
    max_tokens: int,
    error: Optional[str] = None,
) -> TrackerRecord:
    return TrackerRecord(
        timestamp=datetime.now().astimezone().isoformat(),
        model=model,
        model_provider=provider,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        raw_response=raw_response,
        response_length_tokens=token_count,
        candidates_mentioned=detect_candidates_mentioned(raw_response),
        sentiment_per_candidate={c.key: None for c in CANDIDATES},
        refused_to_answer=detect_refusal(raw_response),
        latency_ms=round(latency_ms, 1),
        error=error,
        metadata={"temperature": temperature, "max_tokens": max_tokens},
    )


def save_records(records: List[TrackerRecord], date_str: str, model_key: str, base_path: Path) -> Path:
    day_dir = base_path / "data" / "raw" / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    out_file = day_dir / f"{model_key}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)
    return out_file


def append_to_timeseries(records: List[TrackerRecord], base_path: Path) -> None:
    processed_dir = base_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "timeseries.csv"

    fieldnames = [
        "timestamp", "model", "model_provider", "prompt_id",
        "refused_to_answer", "response_length_tokens", "latency_ms",
        "candidates_mentioned", "error",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow({
                "timestamp": r.timestamp,
                "model": r.model,
                "model_provider": r.model_provider,
                "prompt_id": r.prompt_id,
                "refused_to_answer": int(r.refused_to_answer),
                "response_length_tokens": r.response_length_tokens,
                "latency_ms": r.latency_ms,
                "candidates_mentioned": ",".join(r.candidates_mentioned),
                "error": r.error or "",
            })
