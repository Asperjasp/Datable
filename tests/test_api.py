"""
Tests for app.main — FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRoot:
    def test_dashboard_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert b"Debat-Zero" in resp.content


class TestModels:
    def test_models_page(self):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_api_models(self):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # At least the local Gemma should be listed
        model_keys = [m["key"] for m in data["models"]]
        assert "gemma2_local" in model_keys
        assert "gemma4_local" in model_keys

    def test_api_models_status_availability(self):
        resp = client.get("/api/models")
        data = resp.json()
        for m in data["models"]:
            assert "available" in m
            assert "display_name" in m
            assert "provider" in m


class TestCandidates:
    def test_candidates_list(self):
        resp = client.get("/api/candidates")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["candidates"]) == 5  # 5 Colombian candidates
        c0 = data["candidates"][0]
        assert "key" in c0
        assert "display_name" in c0
        assert "coalition" in c0
        assert "weird_alignment" in c0

    def test_specific_candidate(self):
        resp = client.get("/api/candidates")
        data = resp.json()
        cepeda = [c for c in data["candidates"] if c["key"] == "cepeda"][0]
        assert cepeda["political_leaning"] == "left"
        assert cepeda["weird_alignment"] == "low"


class TestDebateEndpoints:
    def test_create_debate(self):
        resp = client.post("/api/debates", json={
            "debate_id": "test-debate-1",
            "title": "Test Debate",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["debate_id"] == "test-debate-1"
        assert data["type"] == "presidential_election"
        assert data["phase"] == "preparation"

    def test_create_duplicate_debate_409(self):
        # First create
        resp = client.post("/api/debates", json={
            "debate_id": "test-debate-dup",
            "title": "Test Debate Dup",
        })
        assert resp.status_code == 200
        # Second create should fail
        resp = client.post("/api/debates", json={
            "debate_id": "test-debate-dup",
            "title": "Test Debate Dup 2",
        })
        assert resp.status_code == 409

    def test_get_debate_not_found(self):
        resp = client.get("/api/debates/nonexistent")
        assert resp.status_code == 404

    def test_list_debates(self):
        resp = client.get("/api/debates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_upload_government_plan(self):
        # First create debate
        client.post("/api/debates", json={
            "debate_id": "test-plan-debate",
            "title": "Plan Test",
        })
        # Upload plan
        resp = client.post("/api/debates/test-plan-debate/upload", json={
            "participant_name": "Test User",
            "plan_title": "My Government Plan",
            "sections": [
                {"topic": "health", "summary": "Universal healthcare", "key_proposals": ["Clinics in every municipality"]},
                {"topic": "education", "summary": "Free university", "key_proposals": ["Scholarships for all"]},
                {"topic": "economy", "summary": "Grow GDP", "key_proposals": ["Tax reform"]},
                {"topic": "security", "summary": "Safe streets", "key_proposals": ["More police"]},
                {"topic": "environment", "summary": "Clean energy", "key_proposals": ["Solar farms"]},
                {"topic": "social_welfare", "summary": "Help poor", "key_proposals": ["Cash transfers"]},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["missing_topics"] == []

    def test_upload_incomplete_plan(self):
        client.post("/api/debates", json={
            "debate_id": "test-incomplete-plan",
            "title": "Incomplete Plan Test",
        })
        resp = client.post("/api/debates/test-incomplete-plan/upload", json={
            "participant_name": "Lazy User",
            "plan_title": "My Half Plan",
            "sections": [
                {"topic": "health", "summary": "Only health"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["missing_topics"]) == 5  # education, economy, security, environment, social_welfare

    def test_add_debate_turn(self):
        client.post("/api/debates", json={
            "debate_id": "test-turn-debate",
            "title": "Turn Test",
        })
        resp = client.post("/api/debates/test-turn-debate/turn", json={
            "phase": "opening",
            "speaker": "Moderator",
            "text": "Welcome to the debate.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_turns"] == 1

    def test_advance_phase(self):
        client.post("/api/debates", json={
            "debate_id": "test-advance-debate",
            "title": "Advance Test",
        })
        resp = client.post("/api/debates/test-advance-debate/advance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_phase"] == "opening"

    def test_save_debate(self):
        client.post("/api/debates", json={
            "debate_id": "test-save-debate",
            "title": "Save Test",
        })
        resp = client.post("/api/debates/test-save-debate/save")
        assert resp.status_code == 200
        data = resp.json()
        assert "path" in data

    def test_advance_past_complete(self):
        client.post("/api/debates", json={
            "debate_id": "test-advance-complete",
            "title": "Complete Test",
        })
        # Advance through all phases
        phases = ["preparation", "opening", "rebuttal", "q_and_de", "closing", "judging"]
        for expected in phases:
            resp = client.post("/api/debates/test-advance-complete/advance")
            assert resp.status_code == 200
            data = resp.json()
            # The phase advances to the next one, or stays at complete
            # After complete, it stays at complete
        resp = client.post("/api/debates/test-advance-complete/advance")
        # Should be at "complete" now and stay there
        assert resp.status_code == 200


class TestQueryEndpoint:
    def test_query_bad_model(self):
        resp = client.post("/api/query", json={
            "prompt": "Hello",
            "model": "nonexistent_model_xyz",
        })
        assert resp.status_code == 400


class TestDates:
    def test_list_dates_empty(self):
        resp = client.get("/api/dates")
        assert resp.status_code == 200
        assert "dates" in resp.json()

    def test_get_data_not_found(self):
        resp = client.get("/api/data/2099-01-01")
        assert resp.status_code == 404


class TestSchemas:
    def test_query_request_validation(self):
        from app.schemas import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(prompt="test", temperature=5.0)  # temperature > 2.0

    def test_multi_day_request_validation(self):
        from app.schemas import MultiDayRunRequest
        with pytest.raises(Exception):
            MultiDayRunRequest(start_date="2026-05-11", num_days=0)  # num_days < 1

    def test_candidate_info_schema(self):
        from app.schemas import CandidateInfo
        ci = CandidateInfo(
            key="test", full_name="Test Name", display_name="Test",
            coalition="Test Coalition", political_leaning="center",
            weird_alignment="medium", aliases=["T"]
        )
        assert ci.display_name == "Test"