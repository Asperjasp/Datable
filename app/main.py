"""
FastAPI web application for Debat-Zero.

Endpoints:
  GET  /                     → Dashboard (recent data, status)
  GET  /models               → Available LLM models with status
  POST /query                → Send a prompt to a model (quick test)
  POST /openrouter/test      → Test OpenRouter connectivity
  GET  /api/candidates       → List all tracked candidates

  # Debate endpoints
  POST /api/debates              → Create a new debate
  GET  /api/debates/{debate_id}  → Get debate state
  GET  /api/debates              → List all debates
  POST /api/debates/{id}/upload  → Upload a government plan
  POST /api/debates/{id}/turn    → Add a debate turn
  POST /api/debates/{id}/advance → Advance debate phase
  POST /api/debates/{id}/save    → Save debate to disk

  # Simulation endpoints
  GET  /api/simulate/presets     → List preset debates
  POST /api/simulate             → Run a full AI debate simulation
  GET  /api/simulate/results     → List all simulation results

  # Document upload
  POST /api/documents/{candidate_key}  → Upload documents for a candidate
  GET  /api/documents/{candidate_key}  → List documents for a candidate
  DELETE /api/documents/{candidate_key}/{doc_id} → Delete a document

  # Batch / multi-day
  POST /api/run                  → Run single-day batch
  POST /api/run/multi-day        → Run multi-day batch
  GET  /api/dates                → List dates with data available
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse

from tracker.debate import (
    Debate,
    DebatePhase,
    DebateTurn,
    GovernmentPlanTopic,
    GovernmentPlanUpload,
    PresidentialElection,
    build_presidential_election,
    save_debate,
)
from tracker.llm_clients import (
    AVAILABLE_MODELS,
    LLMResponse,
    build_selected_clients,
)
from tracker.multi_day import run_multi_day
from tracker.prompts import ALL_PROMPTS, DAILY_PROMPTS
from tracker.storage import append_to_timeseries, build_record, save_records
from tracker.debate_simulator import (
    PRESET_DEBATES,
    run_debate,
    save_debate_result,
    load_candidate_documents,
    save_candidate_documents,
)
from tracker.document_ingestion import (
    process_uploaded_file,
    add_document_to_candidate,
    remove_document_from_candidate,
    get_candidate_documents_summary,
)

from app.schemas import (
    CandidateInfo,
    DebateCreate,
    DebateStateResponse,
    DebateTurnCreate,
    DaySummary,
    GovernmentPlanUploadCreate,
    MultiDayRunRequest,
    MultiDaySummary,
    ModelInfo,
    QueryRequest,
    QueryResponse,
    RunRequest,
    SimulateRequest,
    SimulationResultOut,
    SimulationTurnOut,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("app")

BASE_PATH = Path(__file__).parent.parent

# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Debat-Zero",
    description="LLM Bias Tracker & AI Debate Platform — Colombian Elections 2026",
    version="0.2.0",
)

# In-memory debate store (persisted to disk on save)
_debates: dict[str, Debate] = {}
_simulation_results: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_html(path: str) -> HTMLResponse:
    full = BASE_PATH / "app" / "templates" / path
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"Template not found: {path}")
    return HTMLResponse(content=full.read_text(encoding="utf-8"))


def _debate_to_response(debate: Debate) -> DebateStateResponse:
    d = debate.to_dict()
    return DebateStateResponse(**d)


# ── Dashboard ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return _get_html("index.html")


@app.get("/api/dates")
async def list_dates():
    data_dir = BASE_PATH / "data" / "raw"
    dates: list[str] = []
    if data_dir.exists():
        dates = sorted([d.name for d in data_dir.iterdir() if d.is_dir()], reverse=True)
    return {"dates": dates}


@app.get("/api/data/{date_str}")
async def get_data(date_str: str):
    raw_dir = BASE_PATH / "data" / "raw" / date_str
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"No data for {date_str}")
    records = []
    for f in raw_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            records.extend(__import__("json").load(fh))
    return {"date": date_str, "records": records}


# ── Models ──────────────────────────────────────────────────────────────

@app.get("/models", response_class=HTMLResponse)
async def models_page():
    return _get_html("models.html")


@app.get("/api/models")
async def list_models():
    models: list[ModelInfo] = []
    for key, cfg in AVAILABLE_MODELS.items():
        env_val = os.environ.get(cfg.env_key) if cfg.env_key else "always"
        models.append(ModelInfo(
            key=key,
            display_name=cfg.display_name,
            provider=cfg.provider,
            group=cfg.group,
            region=cfg.region,
            env_key=cfg.env_key,
            available=bool(env_val),
        ))
    return {"models": models}


@app.get("/api/openrouter/models")
async def openrouter_models():
    """Fetch available models from OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=403, detail="OPENROUTER_API_KEY not set")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/SIMG-UN/debat-zero",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    return {"models": data.get("data", [])}


# ── Quick query ─────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def quick_query(req: QueryRequest):
    """Send a single prompt to any available model."""
    clients = build_selected_clients([req.model])
    if req.model not in clients:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' unavailable")
    client = clients[req.model]
    resp = await client.query(req.prompt, req.temperature, req.max_tokens)
    if resp.error:
        raise HTTPException(status_code=502, detail=resp.error)
    return resp


@app.post("/api/openrouter/test")
async def openrouter_test():
    """Quick connectivity test for OpenRouter using Qwen."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=403, detail="OPENROUTER_API_KEY not set in .env")
    import openai
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        resp = await client.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct",
            messages=[{"role": "user", "content": "Say 'OpenRouter is working' and nothing else."}],
            max_tokens=20,
        )
        text = resp.choices[0].message.content
        return {"status": "ok", "model": "qwen/qwen-2.5-72b-instruct", "response": text}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Single-day run ──────────────────────────────────────────────────────

@app.post("/api/run", response_model=MultiDaySummary)
async def run_batch(req: RunRequest):
    """Run the daily batch (single day) for selected or all models."""
    day = date.today() if req.date is None else date.fromisoformat(req.date)
    prompts = ALL_PROMPTS if req.prompt_set == "all" else DAILY_PROMPTS
    summary = await run_multi_day(day, 1, prompts, req.model_keys, delay_seconds=0)
    return summary


# ── Multi-day run ───────────────────────────────────────────────────────

@app.post("/api/run/multi-day", response_model=MultiDaySummary)
async def run_multi(req: MultiDayRunRequest):
    """Run the tracker across multiple consecutive days."""
    start = date.fromisoformat(req.start_date)
    prompts = ALL_PROMPTS if req.prompt_set == "all" else DAILY_PROMPTS
    summary = await run_multi_day(
        start_date=start,
        num_days=req.num_days,
        prompts=prompts,
        model_keys=req.model_keys,
        delay_seconds=req.delay_seconds,
    )
    return summary


# ── Candidates ─────────────────────────────────────────────────────────

@app.get("/api/candidates")
async def list_candidates():
    from tracker.candidates import CANDIDATES
    return {
        "candidates": [
            CandidateInfo(
                key=c.key,
                full_name=c.full_name,
                display_name=c.display_name,
                coalition=c.coalition,
                political_leaning=c.political_leaning,
                weird_alignment=c.weird_alignment,
                aliases=c.aliases,
            )
            for c in CANDIDATES
        ]
    }


# ── Debate CRUD ─────────────────────────────────────────────────────────

@app.post("/api/debates", response_model=DebateStateResponse)
async def create_debate(req: DebateCreate):
    """Create a new debate (defaults to PresidentialElection)."""
    if req.debate_id in _debates:
        raise HTTPException(status_code=409, detail=f"Debate {req.debate_id} already exists")
    debate = PresidentialElection(
        debate_id=req.debate_id,
        title=req.title,
        description=req.description or "",
        election_year=req.election_year,
        country=req.country,
    )
    for p in req.participants:
        debate.register_candidate(
            participant_name=p.name,
            candidate_key=p.candidate_key or "",
        )
    _debates[req.debate_id] = debate
    return _debate_to_response(debate)


@app.get("/api/debates/{debate_id}", response_model=DebateStateResponse)
async def get_debate(debate_id: str):
    if debate_id not in _debates:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    return _debate_to_response(_debates[debate_id])


@app.get("/api/debates", response_model=List[DebateStateResponse])
async def list_debates():
    return [_debate_to_response(d) for d in _debates.values()]


@app.post("/api/debates/{debate_id}/upload")
async def upload_plan(debate_id: str, upload: GovernmentPlanUploadCreate):
    """Upload a government plan for a participant. Returns missing topics."""
    if debate_id not in _debates:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    debate = _debates[debate_id]
    if not isinstance(debate, PresidentialElection):
        raise HTTPException(status_code=400, detail="Government plans only supported for PresidentialElection")
    plan = GovernmentPlanUpload(
        participant_name=upload.participant_name,
        plan_title=upload.plan_title,
        document_url=upload.document_url,
        uploaded_at=datetime.now(),
        sections=[
            {
                "topic": GovernmentPlanTopic(s.topic.value),
                "summary": s.summary,
                "page_reference": s.page_reference,
                "key_proposals": s.key_proposals,
            }
            for s in upload.sections
        ],
    )
    missing = debate.upload_government_plan(plan)
    return {"participant": upload.participant_name, "missing_topics": missing}


@app.post("/api/debates/{debate_id}/turn")
async def add_turn(debate_id: str, turn: DebateTurnCreate):
    """Add a debate turn (speaker text)."""
    if debate_id not in _debates:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    debate = _debates[debate_id]
    dturn = DebateTurn(
        phase=DebatePhase(turn.phase.value),
        speaker=turn.speaker,
        text=turn.text,
        llm_model=turn.llm_model,
        llm_provider="llm",
    )
    debate.add_turn(dturn)
    return {"status": "added", "num_turns": len(debate.turns)}


@app.post("/api/debates/{debate_id}/advance")
async def advance_phase(debate_id: str):
    """Advance the debate to the next phase."""
    if debate_id not in _debates:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    debate = _debates[debate_id]
    new_phase = debate.advance_phase()
    return {"debate_id": debate_id, "new_phase": new_phase.value}


@app.post("/api/debates/{debate_id}/save")
async def save_debate_endpoint(debate_id: str):
    """Persist the debate state to disk."""
    if debate_id not in _debates:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")
    today = date.today().isoformat()
    path = save_debate(_debates[debate_id], BASE_PATH, today)
    return {"status": "saved", "path": str(path)}


# ── Simulation ─────────────────────────────────────────────────────────

@app.get("/api/simulate/presets")
async def list_presets():
    """Return the four pre-configured debate matchups."""
    return {"presets": PRESET_DEBATES}


@app.post("/api/simulate", response_model=SimulationResultOut)
async def simulate_debate_endpoint(req: SimulateRequest):
    """
    Run a full AI debate simulation between two candidates.
    Actor model argues as each candidate using their platform.
    Judge model scores the exchange.
    """
    # Load documents for each candidate
    docs_a = [d.get("content", "") for d in load_candidate_documents(req.candidate_a_key)]
    docs_b = [d.get("content", "") for d in load_candidate_documents(req.candidate_b_key)]
    
    result = await run_debate(
        candidate_a_key=req.candidate_a_key,
        candidate_b_key=req.candidate_b_key,
        actor_model_key=req.actor_model_key,
        judge_model_key=req.judge_model_key,
        topic=req.topic,
        documents_a=docs_a,
        documents_b=docs_b,
    )
    
    # Save result
    save_debate_result(result)
    _simulation_results[result.debate_id] = result.to_dict()
    
    turns_out = [
        SimulationTurnOut(
            round=t.round,
            speaker_key=t.speaker_key,
            speaker_display=t.speaker_display,
            text=t.text,
            model_used=t.model_used,
            tokens_used=t.tokens_used,
            error=t.error,
        )
        for t in result.turns
    ]
    
    judgment_str = ""
    if result.judgment:
        judgment_str = f"Winner: {result.judgment.winner}\n{result.judgment.summary}"
    
    return SimulationResultOut(
        topic=req.topic,
        candidate_a=result.candidate_a_display,
        candidate_b=result.candidate_b_display,
        actor_model=req.actor_model_key,
        judge_model=req.judge_model_key,
        turns=turns_out,
        judgment=judgment_str,
        total_tokens=result.total_tokens,
        error=result.error,
    )


@app.get("/api/simulate/results")
async def list_simulation_results():
    """List all simulation results."""
    # Load from disk
    results = []
    debates_dir = BASE_PATH / "data" / "debates"
    if debates_dir.exists():
        for date_dir in sorted(debates_dir.iterdir(), reverse=True):
            if date_dir.is_dir():
                for f in date_dir.glob("*.json"):
                    with open(f, encoding="utf-8") as fh:
                        results.append(json.load(fh))
    return {"results": results}


# ── Document Upload ────────────────────────────────────────────────────

@app.post("/api/documents/{candidate_key}")
async def upload_documents(candidate_key: str, files: List[UploadFile] = File(...)):
    """
    Upload documents for a candidate's knowledge base.
    Supports: .txt, .pdf, .doc, .docx
    """
    valid_candidates = ["cepeda", "de_la_espriella", "valencia", "fajardo", "lopez"]
    if candidate_key not in valid_candidates:
        raise HTTPException(status_code=400, detail=f"Unknown candidate: {candidate_key}. Valid: {valid_candidates}")
    
    saved_files = []
    
    for file in files:
        # Save to temp file for processing
        temp_path = BASE_PATH / "data" / "documents" / f"temp_{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process the file
        doc = process_uploaded_file(temp_path, file.filename)
        add_document_to_candidate(candidate_key, doc)
        saved_files.append(file.filename)
        
        # Clean up temp file
        temp_path.unlink()
    
    summary = get_candidate_documents_summary(candidate_key)
    
    return {
        "candidate": candidate_key,
        "uploaded": saved_files,
        "total_documents": summary["total_documents"],
    }


@app.get("/api/documents/{candidate_key}")
async def list_documents(candidate_key: str):
    """List all documents for a candidate."""
    valid_candidates = ["cepeda", "de_la_espriella", "valencia", "fajardo", "lopez"]
    if candidate_key not in valid_candidates:
        raise HTTPException(status_code=400, detail=f"Unknown candidate: {candidate_key}. Valid: {valid_candidates}")
    
    return get_candidate_documents_summary(candidate_key)


@app.delete("/api/documents/{candidate_key}/{doc_index}")
async def delete_document(candidate_key: str, doc_index: int):
    """Delete a document from a candidate's knowledge base."""
    valid_candidates = ["cepeda", "de_la_espriella", "valencia", "fajardo", "lopez"]
    if candidate_key not in valid_candidates:
        raise HTTPException(status_code=400, detail=f"Unknown candidate: {candidate_key}. Valid: {valid_candidates}")
    
    remove_document_from_candidate(candidate_key, doc_index)
    return get_candidate_documents_summary(candidate_key)


# ── Static HTML pages ──────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    return _get_html("dashboard.html")


@app.get("/debate/new", response_class=HTMLResponse)
async def debate_page():
    return _get_html("debate_setup.html")


@app.get("/debate/{debate_id}", response_class=HTMLResponse)
async def debate_live_page(debate_id: str):
    return _get_html("debate_live.html")
