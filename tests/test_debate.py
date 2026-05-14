"""
Tests for tracker.debate — PresidentialElection and GovernmentPlan logic.
"""

import pytest
from datetime import datetime

from tracker.candidates import CANDIDATES, CANDIDATE_BY_KEY
from tracker.debate import (
    build_presidential_election,
    Debate,
    DebatePhase,
    DebateTurn,
    GovernmentPlanSection,
    GovernmentPlanTopic,
    GovernmentPlanUpload,
    PresidentialElection,
    save_debate,
)


class TestGovernmentPlanSection:
    def test_creation(self):
        s = GovernmentPlanSection(topic=GovernmentPlanTopic.HEALTH, summary="Free clinics for all")
        assert s.topic == GovernmentPlanTopic.HEALTH
        assert s.summary == "Free clinics for all"


class TestGovernmentPlanUpload:
    def test_to_dict_roundtrip(self):
        upload = GovernmentPlanUpload(
            participant_name="Test Candidate",
            plan_title="Plan 2026-2030",
            sections=[
                GovernmentPlanSection(topic=GovernmentPlanTopic.HEALTH, summary="Universal health"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.EDUCATION, summary="Free tuition"),
            ],
        )
        d = upload.to_dict()
        assert d["participant_name"] == "Test Candidate"
        assert len(d["sections"]) == 2


class TestDebateBase:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Debate(
                debate_id="test",
                title="Test Debate",
                description="Testing",
            )


class TestPresidentialElection:
    def test_build_with_all_candidates(self):
        debate = build_presidential_election()
        assert debate.type == "presidential_election"
        assert debate.country == "Colombia"
        assert debate.election_year == 2026
        assert len(debate.candidates) == len(CANDIDATES)
        assert len(debate.participants) == len(CANDIDATES)

    def test_required_topics(self):
        assert len(PresidentialElection.REQUIRED_TOPICS) == 6
        topics = {t.value for t in PresidentialElection.REQUIRED_TOPICS}
        assert "health" in topics
        assert "education" in topics
        assert "economy" in topics
        assert "security" in topics
        assert "environment" in topics
        assert "social_welfare" in topics

    def test_register_candidate(self):
        debate = PresidentialElection(debate_id="test", title="Test")
        debate.register_candidate("Juan Perez", "cepeda")
        assert "Juan Perez" in debate.participants
        assert debate.candidates["Juan Perez"] == "cepeda"
        assert debate.coalitions["cepeda"] == "Pacto Histórico"
        assert debate.weird_alignments["cepeda"] == "low"

    def test_upload_government_plan_all_topics(self):
        debate = build_presidential_election()
        debate.register_candidate("Iván Cepeda", "cepeda")

        upload = GovernmentPlanUpload(
            participant_name="Iván Cepeda",
            plan_title="Plan de Gobierno Cepeda 2026",
            sections=[
                GovernmentPlanSection(topic=GovernmentPlanTopic.HEALTH, summary="Universal health"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.EDUCATION, summary="Free education"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.ECONOMY, summary="Economic reform"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.SECURITY, summary="Security plan"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.ENVIRONMENT, summary="Green energy"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.SOCIAL_WELFARE, summary="Social programs"),
            ],
        )
        missing = debate.upload_government_plan(upload)
        assert missing == []
        assert "Iván Cepeda" in debate.government_plans

    def test_upload_government_plan_missing_topics(self):
        debate = build_presidential_election()
        debate.register_candidate("Iván Cepeda", "cepeda")

        upload = GovernmentPlanUpload(
            participant_name="Iván Cepeda",
            plan_title="Incomplete Plan",
            sections=[
                GovernmentPlanSection(topic=GovernmentPlanTopic.HEALTH, summary="Universal health"),
                GovernmentPlanSection(topic=GovernmentPlanTopic.EDUCATION, summary="Free education"),
            ],
        )
        missing = debate.upload_government_plan(upload)
        assert len(missing) == 4  # economy, security, environment, social_welfare
        assert "economy" in missing

    def test_validate_incomplete_upload(self):
        debate = build_presidential_election()
        upload = GovernmentPlanUpload(
            participant_name="Someone",
            plan_title="Empty Plan",
            sections=[],
        )
        missing = debate.validate_participant_upload(upload)
        assert len(missing) == len(PresidentialElection.REQUIRED_TOPICS)

    def test_scoring_criteria(self):
        criteria = PresidentialElection.scoring_criteria(None)  # type: ignore
        assert "policy_specificity" in criteria
        assert "feasibility" in criteria
        assert abs(sum(criteria.values()) - 1.0) < 0.001

    def test_add_turn(self):
        debate = build_presidential_election()
        turn = DebateTurn(
            phase=DebatePhase.OPENING,
            speaker="Moderator",
            text="Bienvenidos al debate.",
        )
        debate.add_turn(turn)
        assert len(debate.turns) == 1
        assert debate.turns[0].speaker == "Moderator"

    def test_advance_phase(self):
        debate = build_presidential_election()
        assert debate.phase == DebatePhase.PREPARATION
        debate.advance_phase()
        assert debate.phase == DebatePhase.OPENING
        debate.advance_phase()
        assert debate.phase == DebatePhase.REBUTTAL

    def test_alliance_report(self):
        debate = build_presidential_election()
        report = debate.get_alliance_report()
        assert report["election"] == "Colombia 2026"
        assert len(report["participants"]) == len(CANDIDATES)
        for p in report["participants"]:
            assert "name" in p
            assert "candidate_key" in p
            assert "coalition" in p
            assert "weird_alignment" in p

    def test_to_dict_serializes(self):
        debate = build_presidential_election()
        d = debate.to_dict()
        assert d["type"] == "presidential_election"
        assert d["election_year"] == 2026
        assert "scoring_criteria" in d
        assert "alliance_report" in d


class TestSaveDebate:
    def test_save_to_disk(self, tmp_path):
        debate = build_presidential_election()
        out_dir = tmp_path / "data" / "debates" / "2026-05-11"
        out_dir.mkdir(parents=True, exist_ok=True)

        from pathlib import Path as P
        path = save_debate(debate, P(tmp_path), "2026-05-11")
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["debate_id"] == "colombia-2026-presidential"
        assert data["type"] == "presidential_election"