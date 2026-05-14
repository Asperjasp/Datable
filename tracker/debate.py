"""
Debate class hierarchy for Debat-Zero.

Defines the abstract Debate base class and concrete PresidentialElection
subclass. Each debate type specifies required participant uploads (e.g.
government plans) and validates them against a topic checklist.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from tracker.candidates import CANDIDATES, CANDIDATE_BY_KEY


# ── Enums ────────────────────────────────────────────────────────────────────

class DebatePhase(str, Enum):
    PREPARATION = "preparation"
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    QANDE = "q_and_de"
    CLOSING = "closing"
    JUDGING = "judging"
    COMPLETE = "complete"


class GovernmentPlanTopic(str, Enum):
    HEALTH = "health"
    EDUCATION = "education"
    ECONOMY = "economy"
    SECURITY = "security"
    ENVIRONMENT = "environment"
    SOCIAL_WELFARE = "social_welfare"
    INFRASTRUCTURE = "infrastructure"
    TECHNOLOGY_AND_INNOVATION = "technology_and_innovation"
    FOREIGN_POLICY = "foreign_policy"
    GOVERNANCE_AND_INSTITUTIONS = "governance_and_institutions"


# ── Pydantic models (API / storage) ──────────────────────────────────────────

class DebateParticipantCreate(BaseModel):
    name: str
    candidate_key: Optional[str] = None
    role: str = "candidate"


class GovernmentPlanSection(BaseModel):
    topic: GovernmentPlanTopic
    summary: str
    page_reference: Optional[str] = None
    key_proposals: List[str] = Field(default_factory=list)


class GovernmentPlanUpload(BaseModel):
    participant_name: str
    plan_title: str
    document_url: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    sections: List[GovernmentPlanSection] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "participant_name": self.participant_name,
            "plan_title": self.plan_title,
            "document_url": self.document_url,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "sections": [
                {
                    "topic": s.topic.value,
                    "summary": s.summary,
                    "page_reference": s.page_reference,
                    "key_proposals": s.key_proposals,
                }
                for s in self.sections
            ],
        }


class DebateTurn(BaseModel):
    phase: DebatePhase
    speaker: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None


# ── Core debate classes ──────────────────────────────────────────────────────

@dataclass
class Debate(ABC):
    """Abstract base class for any debate type."""

    debate_id: str
    title: str
    description: str
    participants: List[str] = field(default_factory=list)
    phase: DebatePhase = DebatePhase.PREPARATION
    turns: List[DebateTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    REQUIRED_TOPICS: ClassVar[List[GovernmentPlanTopic]] = []

    @abstractmethod
    def validate_participant_upload(self, upload: GovernmentPlanUpload) -> List[str]:
        """Return list of missing required topics."""
        ...

    @abstractmethod
    def scoring_criteria(self) -> Dict[str, float]:
        """Return weights for each evaluation dimension."""
        ...

    def add_participant(self, name: str) -> None:
        if name not in self.participants:
            self.participants.append(name)

    def add_turn(self, turn: DebateTurn) -> None:
        self.turns.append(turn)

    def advance_phase(self) -> DebatePhase:
        order = list(DebatePhase)
        idx = order.index(self.phase)
        if idx < len(order) - 1:
            self.phase = order[idx + 1]
        return self.phase

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "title": self.title,
            "description": self.description,
            "participants": self.participants,
            "phase": self.phase.value,
            "num_turns": len(self.turns),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PresidentialElection(Debate):
    """
    Concrete debate type: Presidential Election.

    Participants must upload government plans covering:
    - Health (salud)
    - Education (educación)
    - Economy
    - Security
    - Environment
    - Social welfare

    Additional election-specific features:
    - Links to Candidate objects
    - Coalition tracking
    - WEIRD-alignment tagging
    """

    REQUIRED_TOPICS: ClassVar[List[GovernmentPlanTopic]] = [
        GovernmentPlanTopic.HEALTH,
        GovernmentPlanTopic.EDUCATION,
        GovernmentPlanTopic.ECONOMY,
        GovernmentPlanTopic.SECURITY,
        GovernmentPlanTopic.ENVIRONMENT,
        GovernmentPlanTopic.SOCIAL_WELFARE,
    ]

    election_year: int = 2026
    country: str = "Colombia"
    description: str = ""
    candidates: Dict[str, str] = field(default_factory=dict)
    coalitions: Dict[str, str] = field(default_factory=dict)
    government_plans: Dict[str, GovernmentPlanUpload] = field(default_factory=dict)
    weird_alignments: Dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return "presidential_election"

    def __post_init__(self) -> None:
        self.title = self.title or f"Colombian Presidential Election {self.election_year}"
        self.description = (
            self.description
            or f"AI-powered debate simulation for the {self.election_year} Colombian presidential election. "
            f"Participants must submit government plans covering: "
            f"{', '.join(t.value for t in self.REQUIRED_TOPICS)}."
        )

    def register_candidate(
        self,
        participant_name: str,
        candidate_key: str,
        coalition: Optional[str] = None,
        weird_alignment: Optional[str] = None,
    ) -> None:
        """Link a participant to a known Candidate from tracker.candidates."""
        self.candidates[participant_name] = candidate_key
        if candidate_key in CANDIDATE_BY_KEY:
            c = CANDIDATE_BY_KEY[candidate_key]
            self.coalitions[candidate_key] = coalition or c.coalition
            self.weird_alignments[candidate_key] = weird_alignment or c.weird_alignment
        self.add_participant(participant_name)

    def upload_government_plan(self, upload: GovernmentPlanUpload) -> List[str]:
        """Store a participant's government plan. Returns list of missing topics."""
        missing = self.validate_participant_upload(upload)
        self.government_plans[upload.participant_name] = upload
        return missing

    def validate_participant_upload(self, upload: GovernmentPlanUpload) -> List[str]:
        """Check that all REQUIRED_TOPICS appear in the uploaded sections."""
        covered = {s.topic for s in upload.sections}
        return [t.value for t in self.REQUIRED_TOPICS if t not in covered]

    def scoring_criteria(self) -> Dict[str, float]:
        return {
            "policy_specificity": 0.25,
            "feasibility": 0.20,
            "coverage_of_required_topics": 0.20,
            "electoral_viability": 0.15,
            "originality_vs_llm_bias": 0.10,
            "rhetorical_persuasiveness": 0.10,
        }

    def get_alliance_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "election": f"{self.country} {self.election_year}",
            "phase": self.phase.value,
            "participants": [],
        }
        for name, ckey in self.candidates.items():
            plan = self.government_plans.get(name)
            missing = self.validate_participant_upload(plan) if plan else [t.value for t in self.REQUIRED_TOPICS]
            report["participants"].append({
                "name": name,
                "candidate_key": ckey,
                "coalition": self.coalitions.get(ckey, "Unknown"),
                "weird_alignment": self.weird_alignments.get(ckey, "Unknown"),
                "plan_uploaded": plan is not None,
                "missing_topics": missing,
                "num_plan_sections": len(plan.sections) if plan else 0,
            })
        return report

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "type": "presidential_election",
            "election_year": self.election_year,
            "country": self.country,
            "candidates": self.candidates,
            "coalitions": self.coalitions,
            "weird_alignments": self.weird_alignments,
            "government_plans": {
                name: upload.to_dict()
                for name, upload in self.government_plans.items()
            },
            "scoring_criteria": self.scoring_criteria(),
            "alliance_report": self.get_alliance_report(),
        })
        return base


# ── Utility ──────────────────────────────────────────────────────────────────

def build_presidential_election(
    year: int = 2026,
    include_all_candidates: bool = True,
) -> PresidentialElection:
    debate = PresidentialElection(
        debate_id=f"colombia-{year}-presidential",
        title=f"Colombian Presidential Election {year}",
        description=(
            f"AI debate simulation for the {year} Colombian presidential election. "
            f"Each candidate must submit a government plan covering health, "
            f"education, economy, security, environment, and social welfare."
        ),
        election_year=year,
        country="Colombia",
    )
    if include_all_candidates:
        for c in CANDIDATES:
            debate.register_candidate(
                participant_name=c.display_name,
                candidate_key=c.key,
                coalition=c.coalition,
                weird_alignment=c.weird_alignment,
            )
    return debate


def save_debate(debate: Debate, base_path: Path, date_str: str) -> Path:
    out_dir = base_path / "data" / "debates" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{debate.debate_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(debate.to_dict(), f, ensure_ascii=False, indent=2)
    return out_file


def load_debate(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)