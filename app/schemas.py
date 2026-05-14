"""
API request/response schemas for the Debat-Zero web app.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Model / LLM ──────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    key: str
    display_name: str
    provider: str
    group: str
    region: str = ""
    env_key: Optional[str]
    available: bool


class QueryRequest(BaseModel):
    prompt: str
    model: str = "claude"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)


class QueryResponse(BaseModel):
    raw_text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    error: Optional[str] = None


class OpenRouterModel(BaseModel):
    id: str
    name: str
    context_length: int
    pricing: Dict[str, float]


# ── Debate ────────────────────────────────────────────────────────────────────

class DebatePhase(str, Enum):
    preparation = "preparation"
    opening = "opening"
    rebuttal = "rebuttal"
    q_and_de = "q_and_de"
    closing = "closing"
    judging = "judging"
    complete = "complete"


class GovernmentPlanTopic(str, Enum):
    health = "health"
    education = "education"
    economy = "economy"
    security = "security"
    environment = "environment"
    social_welfare = "social_welfare"
    infrastructure = "infrastructure"
    technology_and_innovation = "technology_and_innovation"
    foreign_policy = "foreign_policy"
    governance_and_institutions = "governance_and_institutions"


class GovernmentPlanSectionCreate(BaseModel):
    topic: GovernmentPlanTopic
    summary: str
    page_reference: Optional[str] = None
    key_proposals: List[str] = Field(default_factory=list)


class GovernmentPlanUploadCreate(BaseModel):
    participant_name: str
    plan_title: str
    document_url: Optional[str] = None
    sections: List[GovernmentPlanSectionCreate] = Field(default_factory=list)


class DebateParticipantCreate(BaseModel):
    name: str
    candidate_key: Optional[str] = None
    role: str = "candidate"


class DebateCreate(BaseModel):
    debate_id: str
    title: str
    description: Optional[str] = None
    election_year: int = 2026
    country: str = "Colombia"
    participants: List[DebateParticipantCreate] = Field(default_factory=list)


class DebateStateResponse(BaseModel):
    debate_id: str
    title: str
    description: str
    type: str
    phase: str
    participants: List[str]
    candidates: Dict[str, str]
    coalitions: Dict[str, str]
    election_year: int
    country: str
    government_plans: Dict[str, Any]
    scoring_criteria: Dict[str, float]
    num_turns: int
    created_at: str


class DebateTurnCreate(BaseModel):
    phase: DebatePhase
    speaker: str
    text: str
    llm_model: Optional[str] = None


# ── Run / Batch ──────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    date: Optional[str] = None          # YYYY-MM-DD, None = today
    model_keys: Optional[List[str]] = None
    prompt_set: str = "daily"           # "daily" or "all"


class MultiDayRunRequest(BaseModel):
    start_date: str                     # YYYY-MM-DD
    num_days: int = Field(default=1, ge=1)
    delay_seconds: float = Field(default=0, ge=0)
    model_keys: Optional[List[str]] = None
    prompt_set: str = "daily"


class DaySummary(BaseModel):
    date: str
    records: int
    errors: int
    refusals: int
    successes: int


class MultiDaySummary(BaseModel):
    days: List[DaySummary]
    total_records: int
    total_successes: int
    total_errors: int
    total_refusals: int


# ── Candidate ────────────────────────────────────────────────────────────────

class CandidateInfo(BaseModel):
    key: str
    full_name: str
    display_name: str
    coalition: str
    political_leaning: str
    weird_alignment: str
    aliases: List[str]