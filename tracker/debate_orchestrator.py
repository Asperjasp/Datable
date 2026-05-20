"""
Orchestrated Debate Engine — Judge-First Protocol
==================================================

Flow:
  1. Judge reads both candidates' platforms
  2. Judge generates up to N impartial questions (CÓMO/POR QUÉ/PARA QUIÉN)
  3. For each question: Candidate A answers → Candidate B answers
     - AI mode: LLM auto-generates answer (max 300 tokens ≈ 1 minute)
     - Human mode: question is returned, caller supplies answer via API
  4. Judge produces structured HOW/WHY/WHO analytics (no winner verdict)

The judge's role is analysis, not adjudication.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracker.candidates import CANDIDATE_BY_KEY
from tracker.llm_clients import build_selected_clients

log = logging.getLogger("debate_orchestrator")

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_bases"

MAX_TOKENS_QUESTIONS = 900
MAX_TOKENS_ANSWER = 300   # ~1 minute of debate speaking
MAX_TOKENS_ANALYTICS = 1400
TEMP_JUDGE = 0.3
TEMP_ACTOR = 0.75

DISCLAIMER = "[SIMULACIÓN ACADÉMICA — no representa posiciones reales]"


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class ParticipantConfig:
    candidate_key: str
    mode: str = "ai"          # "ai" | "human"
    actor_model_key: str = "claude"


@dataclass
class DebateAnswer:
    question_idx: int
    speaker_key: str
    speaker_display: str
    text: str
    mode: str
    model_used: Optional[str] = None
    tokens_used: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratedDebate:
    debate_id: str
    topic: str
    participant_a: ParticipantConfig
    participant_b: ParticipantConfig
    judge_model_key: str
    max_questions: int = 5

    questions: List[str] = field(default_factory=list)
    answers: List[DebateAnswer] = field(default_factory=list)
    analytics: Optional[Dict[str, Any]] = None

    status: str = "pending"   # pending|questioning|answering|judging|complete|error
    current_question_idx: int = 0
    awaiting_human: Optional[str] = None  # "a" | "b" | None

    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "participant_a": self.participant_a.candidate_key,
            "participant_b": self.participant_b.candidate_key,
            "mode_a": self.participant_a.mode,
            "mode_b": self.participant_b.mode,
            "actor_a": self.participant_a.actor_model_key,
            "actor_b": self.participant_b.actor_model_key,
            "judge_model": self.judge_model_key,
            "max_questions": self.max_questions,
            "status": self.status,
            "current_question_idx": self.current_question_idx,
            "awaiting_human": self.awaiting_human,
            "questions": self.questions,
            "answers": [
                {
                    "question_idx": a.question_idx,
                    "speaker": a.speaker_key,
                    "speaker_display": a.speaker_display,
                    "text": a.text,
                    "mode": a.mode,
                    "model": a.model_used,
                    "tokens": a.tokens_used,
                    "error": a.error,
                    "timestamp": a.timestamp,
                }
                for a in self.answers
            ],
            "analytics": self.analytics,
            "total_tokens": self.total_tokens,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ── Platform loading ───────────────────────────────────────────────────────

def _load_platform(candidate_key: str, extra_docs: list[str] = None) -> str:
    path = KB_DIR / candidate_key / "platform.md"
    base = path.read_text(encoding="utf-8")[:4000] if path.exists() else ""
    if not base:
        c = CANDIDATE_BY_KEY.get(candidate_key)
        base = f"Candidato: {c.full_name if c else candidate_key}. (Sin programa disponible.)"
    if extra_docs:
        base += "\n\nDOCUMENTOS ADICIONALES:\n" + "\n\n".join(d[:1000] for d in extra_docs[:3])
    return base


def _short(text: str, n: int = 800) -> str:
    return text[:n] + "..." if len(text) > n else text


# ── Judge: generate questions ──────────────────────────────────────────────

async def _generate_questions(
    judge_client,
    topic: str,
    a_key: str, b_key: str,
    platform_a: str, platform_b: str,
    max_questions: int,
) -> list[str]:
    cand_a = CANDIDATE_BY_KEY.get(a_key)
    cand_b = CANDIDATE_BY_KEY.get(b_key)
    a_name = cand_a.full_name if cand_a else a_key
    b_name = cand_b.full_name if cand_b else b_key
    a_coalition = cand_a.coalition if cand_a else ""
    b_coalition = cand_b.coalition if cand_b else ""

    system = (
        "Eres el moderador imparcial de un debate presidencial colombiano 2026. "
        "Generas preguntas que revelan diferencias reales, cuestionan viabilidad, "
        "y exploran consecuencias. No tomas partido. "
        "Enfocas en CÓMO (implementación), POR QUÉ (motivación), PARA QUIÉN (beneficiarios)."
    )

    prompt = (
        f"DEBATE SOBRE: {topic}\n\n"
        f"CANDIDATO A: {a_name} — {a_coalition}\n"
        f"PROGRAMA:\n{_short(platform_a, 1600)}\n\n"
        f"CANDIDATO B: {b_name} — {b_coalition}\n"
        f"PROGRAMA:\n{_short(platform_b, 1600)}\n\n"
        f"Genera exactamente {max_questions} preguntas de debate. Cada pregunta debe:\n"
        f"1. Ser dirigida a AMBOS candidatos (cada uno responde la misma)\n"
        f"2. Enfocarse en CÓMO, POR QUÉ o PARA QUIÉN\n"
        f"3. Explorar tensiones reales entre sus propuestas\n"
        f"4. Pedir especificidad: presupuestos, plazos, mecanismos, evidencia\n"
        f"5. Ser absolutamente imparcial\n\n"
        f"Responde SOLO con JSON válido:\n"
        f'{{ "questions": ["Pregunta 1", "Pregunta 2", ..., "Pregunta {max_questions}"] }}'
    )

    resp = await judge_client.query(
        prompt=prompt,
        temperature=TEMP_JUDGE,
        max_tokens=MAX_TOKENS_QUESTIONS,
        system_prompt=system,
    )

    try:
        text = resp.raw_text or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            questions = [q for q in data.get("questions", []) if q]
            if questions:
                return questions[:max_questions]
    except Exception as e:
        log.warning(f"Could not parse judge questions JSON: {e}")

    # Fallback generic questions
    return [
        f"¿Cuál es su propuesta concreta para {topic} y cuál es el presupuesto estimado?",
        f"¿Qué mecanismos de implementación usaría y en qué plazo?",
        f"¿Quiénes se benefician y quiénes asumen los costos de su propuesta?",
        f"¿Qué evidencia respalda que su enfoque funcionaría en el contexto colombiano?",
        f"¿Cómo difiere su propuesta de la del candidato oponente y por qué es superior?",
    ][:max_questions]


# ── Actor: AI answer ───────────────────────────────────────────────────────

async def _ai_answer(
    actor_client,
    candidate_key: str,
    platform: str,
    question: str,
    opponent_key: str,
    topic: str,
) -> tuple[str, int, Optional[str]]:
    cand = CANDIDATE_BY_KEY.get(candidate_key)
    opp = CANDIDATE_BY_KEY.get(opponent_key)
    if not cand:
        return f"[{candidate_key}: candidato no encontrado]", 0, "Not found"

    system = (
        f"Eres {cand.full_name}, candidato presidencial de Colombia 2026 por {cand.coalition}.\n"
        f"Orientación: {cand.political_leaning}.\n\n"
        f"TU PROGRAMA:\n{_short(platform, 2000)}\n\n"
        f"REGLAS:\n"
        f"- Responde en primera persona\n"
        f"- Máximo 200 palabras (≈1 minuto de debate)\n"
        f"- Fundamenta CADA argumento en tu programa\n"
        f"- Cita cifras y fuentes cuando las tengas\n"
        f"- Español colombiano formal\n"
        f"- Última línea: {DISCLAIMER}"
    )

    opp_name = opp.display_name if opp else opponent_key
    prompt = (
        f"El moderador pregunta sobre {topic}:\n\n"
        f"{question}\n\n"
        f"({opp_name} también responderá esta pregunta. Defiende tu posición con datos.)"
    )

    resp = await actor_client.query(
        prompt=prompt,
        temperature=TEMP_ACTOR,
        max_tokens=MAX_TOKENS_ANSWER,
        system_prompt=system,
    )
    return resp.raw_text or "[Sin respuesta]", resp.input_tokens + resp.output_tokens, resp.error


# ── Judge: HOW/WHY/WHO analytics ──────────────────────────────────────────

async def _generate_analytics(
    judge_client,
    debate: OrchestratedDebate,
    platform_a: str, platform_b: str,
) -> dict:
    cand_a = CANDIDATE_BY_KEY.get(debate.participant_a.candidate_key)
    cand_b = CANDIDATE_BY_KEY.get(debate.participant_b.candidate_key)
    a_name = cand_a.display_name if cand_a else debate.participant_a.candidate_key
    b_name = cand_b.display_name if cand_b else debate.participant_b.candidate_key
    a_key = debate.participant_a.candidate_key
    b_key = debate.participant_b.candidate_key

    transcript_parts = []
    for q_idx, question in enumerate(debate.questions):
        transcript_parts.append(f"\nPREGUNTA {q_idx + 1}: {question}")
        for ans in debate.answers:
            if ans.question_idx == q_idx:
                transcript_parts.append(f"\n{ans.speaker_display}: {ans.text}")
    transcript = "\n".join(transcript_parts)

    system = (
        "Eres analista político imparcial. Tu trabajo es entender las propuestas de cada candidato "
        "y ayudar a los ciudadanos a evaluarlas. NO eliges ganador. "
        "Produces análisis estructurado sobre CÓMO, POR QUÉ y PARA QUIÉN."
    )

    prompt = (
        f"DEBATE PRESIDENCIAL COLOMBIA 2026\n"
        f"TEMA: {debate.topic}\n"
        f"CANDIDATOS: {a_name} vs {b_name}\n\n"
        f"TRANSCRIPCIÓN COMPLETA:\n{transcript}\n\n"
        f"Produce análisis IMPARCIAL en JSON:\n"
        "{\n"
        f'  "how": {{ "{a_key}": "¿Cómo implementa {a_name}?", "{b_key}": "¿Cómo implementa {b_name}?" }},\n'
        f'  "why": {{ "{a_key}": "¿Por qué {a_name} prioriza este enfoque?", "{b_key}": "¿Por qué {b_name}?" }},\n'
        f'  "who": {{ "{a_key}": "¿Quiénes se benefician con {a_name}?", "{b_key}": "¿Quiénes con {b_name}?" }},\n'
        '  "key_tensions": ["Tensión real entre ambas propuestas"],\n'
        '  "unanswered": ["Pregunta que quedó sin respuesta clara"],\n'
        f'  "evidence_quality": {{ "{a_key}": 0, "{b_key}": 0 }},\n'
        f'  "specificity": {{ "{a_key}": 0, "{b_key}": 0 }},\n'
        '  "summary": "3-4 oraciones. Sin elegir ganador. Qué propuso cada uno y qué quedó pendiente.",\n'
        f'  "disclaimer": "{DISCLAIMER}"\n'
        "}\n\n"
        "Scores de 0-10. Sé riguroso e imparcial con ambos candidatos."
    )

    resp = await judge_client.query(
        prompt=prompt,
        temperature=TEMP_JUDGE,
        max_tokens=MAX_TOKENS_ANALYTICS,
        system_prompt=system,
    )

    try:
        text = resp.raw_text or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        log.warning(f"Could not parse analytics JSON: {e}")

    return {"summary": resp.raw_text or "[Sin análisis]", "raw": True}


# ── Main orchestration ─────────────────────────────────────────────────────

async def run_orchestrated_debate(
    debate: OrchestratedDebate,
    clients: dict,
) -> OrchestratedDebate:
    """
    Run a full debate with AI actors. Human-mode participants are skipped
    (their answers must be submitted separately via submit_human_answer).
    """
    judge_client = clients.get(debate.judge_model_key)
    if not judge_client:
        # Fallback to any available client
        judge_client = next(iter(clients.values()), None)
        if judge_client:
            log.warning(f"Judge model {debate.judge_model_key} unavailable, using fallback")
        else:
            debate.status = "error"
            debate.error = "No LLM clients available"
            return debate

    platform_a = _load_platform(debate.participant_a.candidate_key)
    platform_b = _load_platform(debate.participant_b.candidate_key)

    cand_a = CANDIDATE_BY_KEY.get(debate.participant_a.candidate_key)
    cand_b = CANDIDATE_BY_KEY.get(debate.participant_b.candidate_key)
    a_display = cand_a.display_name if cand_a else debate.participant_a.candidate_key
    b_display = cand_b.display_name if cand_b else debate.participant_b.candidate_key

    # Step 1: Judge generates questions
    debate.status = "questioning"
    log.info(f"[ORCH] {debate.debate_id} — judge generating {debate.max_questions} questions")
    debate.questions = await _generate_questions(
        judge_client=judge_client,
        topic=debate.topic,
        a_key=debate.participant_a.candidate_key,
        b_key=debate.participant_b.candidate_key,
        platform_a=platform_a,
        platform_b=platform_b,
        max_questions=debate.max_questions,
    )
    log.info(f"[ORCH] {len(debate.questions)} questions generated")

    # Step 2: Each candidate answers each question
    debate.status = "answering"
    for q_idx, question in enumerate(debate.questions):
        debate.current_question_idx = q_idx
        log.info(f"[ORCH] Q{q_idx + 1}: {question[:70]}...")

        # Candidate A
        if debate.participant_a.mode == "ai":
            actor_a = clients.get(debate.participant_a.actor_model_key) or judge_client
            text, tokens, err = await _ai_answer(
                actor_a, debate.participant_a.candidate_key,
                platform_a, question,
                debate.participant_b.candidate_key, debate.topic,
            )
            debate.answers.append(DebateAnswer(
                question_idx=q_idx, speaker_key=debate.participant_a.candidate_key,
                speaker_display=a_display, text=text, mode="ai",
                model_used=actor_a.model_id, tokens_used=tokens, error=err,
            ))
            debate.total_tokens += tokens

        # Candidate B
        if debate.participant_b.mode == "ai":
            actor_b = clients.get(debate.participant_b.actor_model_key) or judge_client
            text, tokens, err = await _ai_answer(
                actor_b, debate.participant_b.candidate_key,
                platform_b, question,
                debate.participant_a.candidate_key, debate.topic,
            )
            debate.answers.append(DebateAnswer(
                question_idx=q_idx, speaker_key=debate.participant_b.candidate_key,
                speaker_display=b_display, text=text, mode="ai",
                model_used=actor_b.model_id, tokens_used=tokens, error=err,
            ))
            debate.total_tokens += tokens

    # Step 3: Judge analytics
    debate.status = "judging"
    log.info(f"[ORCH] Generating HOW/WHY/WHO analytics")
    debate.analytics = await _generate_analytics(judge_client, debate, platform_a, platform_b)

    debate.status = "complete"
    debate.completed_at = datetime.now().isoformat()
    log.info(f"[ORCH] {debate.debate_id} complete — {debate.total_tokens} tokens")
    return debate


def submit_human_answer(debate: OrchestratedDebate, speaker: str, text: str) -> OrchestratedDebate:
    """Add a human participant's answer for the current question."""
    q_idx = debate.current_question_idx
    candidate_key = (
        debate.participant_a.candidate_key if speaker == "a"
        else debate.participant_b.candidate_key
    )
    cand = CANDIDATE_BY_KEY.get(candidate_key)
    display = cand.display_name if cand else candidate_key

    debate.answers.append(DebateAnswer(
        question_idx=q_idx,
        speaker_key=candidate_key,
        speaker_display=display,
        text=text[:1200],  # cap human input too
        mode="human",
    ))
    debate.awaiting_human = None
    return debate


def save_orchestrated_debate(debate: OrchestratedDebate) -> Path:
    base = Path(__file__).parent.parent
    results_dir = (
        Path("/tmp/debat-zero/orchestrated")
        if not (base / "data").exists()
        else base / "data" / "orchestrated_debates" / date.today().isoformat()
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{debate.debate_id}.json"
    path.write_text(json.dumps(debate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
