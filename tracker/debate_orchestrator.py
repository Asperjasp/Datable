"""
Orchestrated Debate Engine — Judge-First, 4-Turn Protocol
==========================================================

Match structure (12 minutes total, 3 questions):
  For each question the judge asks:
    Turn 1 (1 min): Candidate A — presents their own position
    Turn 2 (1 min): Candidate B — presents their own position
    Turn 3 (1 min): Candidate B — rebuts Candidate A's position
    Turn 4 (1 min): Candidate A — rebuts Candidate B's position + rebuttal

  Judge receives all 4 documents per question and produces:
    HOW  — implementation mechanics for each candidate
    WHY  — motivation and theory of change
    WHO  — beneficiaries and cost-bearers
    + evidence quality scores, key tensions, unanswered questions

Judge role: analytical, never picks a winner.
Each turn: ≈300 tokens (≈1 minute of debate speaking).
Each LLM actor also generates a supporting artifact (stat / citation / analysis).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracker.candidates import CANDIDATE_BY_KEY

log = logging.getLogger("debate_orchestrator")

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_bases"

MAX_TOKENS_QUESTIONS  = 1000
MAX_TOKENS_TURN       = 320   # ≈1 minute of debate speaking
MAX_TOKENS_ARTIFACT   = 200   # supporting stat / citation
MAX_TOKENS_ANALYTICS  = 3000
TEMP_JUDGE  = 0.3
TEMP_ACTOR  = 0.75

DISCLAIMER = "[SIMULACIÓN ACADÉMICA — no representa posiciones reales de los candidatos]"

# ── Turn types ─────────────────────────────────────────────────────────────────

TURN_POSITION = "position"   # candidate argues FOR own platform
TURN_REBUTTAL = "rebuttal"   # candidate argues AGAINST opponent


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ParticipantConfig:
    candidate_key: str
    mode: str = "ai"          # "ai" | "human"
    actor_model_key: str = "claude"


@dataclass
class DebateArtifact:
    """A data point / citation the LLM actor generates to support a turn."""
    artifact_type: str   # "statistic" | "citation" | "analysis" | "comparison"
    title: str
    content: str
    source: str = ""


@dataclass
class DebateTurn:
    question_idx: int
    speaker_key: str
    speaker_display: str
    turn_type: str             # TURN_POSITION | TURN_REBUTTAL
    turn_order: int            # 1=A_pos, 2=B_pos, 3=B_reb, 4=A_reb
    text: str
    mode: str                  # "ai" | "human"
    model_used: Optional[str] = None
    tokens_used: int = 0
    artifacts: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratedDebate:
    debate_id: str
    topic: str
    participant_a: ParticipantConfig
    participant_b: ParticipantConfig
    judge_model_key: str
    max_questions: int = 3

    questions: List[str] = field(default_factory=list)
    turns: List[DebateTurn] = field(default_factory=list)
    analytics: Optional[Dict[str, Any]] = None

    status: str = "pending"   # pending|questioning|debating|judging|complete|error
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
            "turns": [
                {
                    "question_idx": t.question_idx,
                    "speaker": t.speaker_key,
                    "speaker_display": t.speaker_display,
                    "turn_type": t.turn_type,
                    "turn_order": t.turn_order,
                    "text": t.text,
                    "mode": t.mode,
                    "model": t.model_used,
                    "tokens": t.tokens_used,
                    "artifacts": t.artifacts,
                    "error": t.error,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
            "analytics": self.analytics,
            "total_tokens": self.total_tokens,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_platform(candidate_key: str) -> str:
    path = KB_DIR / candidate_key / "platform.md"
    if path.exists():
        return path.read_text(encoding="utf-8")[:4000]
    c = CANDIDATE_BY_KEY.get(candidate_key)
    return f"Candidato: {c.full_name if c else candidate_key}. (Sin programa disponible.)"


def _short(text: str, n: int = 800) -> str:
    return text[:n] + "..." if len(text) > n else text


def _extract_json(text: str) -> str:
    """Strip markdown code fences then return the outermost JSON object."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    return text[start:end] if start >= 0 and end > start else text


def _parse_json_safe(raw: str, label: str) -> Optional[dict]:
    if not raw:
        log.warning(f"{label}: empty response")
        return None
    try:
        return json.loads(_extract_json(raw))
    except Exception as e:
        log.warning(f"{label}: JSON parse failed — {e} — raw[:150]: {raw[:150]}")
        return None


# ── PHASE 1: Judge generates questions ────────────────────────────────────────

QUESTION_SYSTEM = (
    "Eres el moderador imparcial de un debate presidencial colombiano 2026. "
    "Cada pregunta debe forzar a los candidatos a ser específicos sobre "
    "CÓMO (mecanismos de implementación, cronograma, presupuesto), "
    "POR QUÉ (evidencia empírica, teoría del cambio) y "
    "PARA QUIÉN (beneficiarios concretos, costos y quién los asume). "
    "No tomas partido. Generas tensiones reales entre las propuestas."
)

QUESTION_PROMPT_TEMPLATE = """
DEBATE PRESIDENCIAL COLOMBIA 2026
Fecha: {date}
Tema: {topic}

CANDIDATO A — {a_name} ({a_coalition}):
{platform_a}

CANDIDATO B — {b_name} ({b_coalition}):
{platform_b}

Genera exactamente {n} preguntas de debate para este tema.
Cada pregunta debe:
1. Dirigirse a AMBOS candidatos (cada uno responde la misma)
2. Enfocarse en UNA dimensión: CÓMO, POR QUÉ o PARA QUIÉN
3. Nombrar propuestas concretas de ambos candidatos y preguntar por diferencias
4. Pedir especificidad: cifras, plazos, mecanismos, evidencia
5. Ser absolutamente imparcial

Responde SOLO con JSON válido (sin markdown):
{{ "questions": ["Pregunta 1", "Pregunta 2", "Pregunta {n}"] }}
""".strip()

GENERIC_QUESTIONS = [
    "¿Cómo implementarían su propuesta en los primeros 100 días y con qué presupuesto exacto?",
    "¿Qué evidencia nacional o internacional respalda que su enfoque funciona, y cómo medirían el éxito a 4 años?",
    "¿Quiénes se benefician concretamente de su propuesta, quiénes asumen los costos, y cómo llegan a territorios olvidados?",
]


async def _generate_questions(judge_client, topic: str, a_key: str, b_key: str,
                               platform_a: str, platform_b: str, n: int, run_date: date) -> list[str]:
    cand_a = CANDIDATE_BY_KEY.get(a_key)
    cand_b = CANDIDATE_BY_KEY.get(b_key)

    prompt = QUESTION_PROMPT_TEMPLATE.format(
        date=run_date.isoformat(),
        topic=topic,
        a_name=cand_a.full_name if cand_a else a_key,
        a_coalition=cand_a.coalition if cand_a else "",
        platform_a=_short(platform_a, 1600),
        b_name=cand_b.full_name if cand_b else b_key,
        b_coalition=cand_b.coalition if cand_b else "",
        platform_b=_short(platform_b, 1600),
        n=n,
    )

    resp = await judge_client.query(prompt, TEMP_JUDGE, MAX_TOKENS_QUESTIONS, QUESTION_SYSTEM)
    data = _parse_json_safe(resp.raw_text or "", "judge questions")
    if data:
        questions = [q for q in data.get("questions", []) if q]
        if questions:
            log.info(f"[ORCH] Judge generated {len(questions)} questions")
            return questions[:n]

    log.warning("[ORCH] Falling back to generic questions")
    return GENERIC_QUESTIONS[:n]


# ── PHASE 2: Actor turns ───────────────────────────────────────────────────────

POSITION_SYSTEM_TEMPLATE = (
    "Eres {full_name}, candidato presidencial de Colombia 2026 por {coalition}.\n"
    "Orientación política: {leaning}.\n\n"
    "TU PROGRAMA:\n{platform}\n\n"
    "REGLAS:\n"
    "- Habla en primera persona como el candidato\n"
    "- Máximo 200 palabras (≈1 minuto de debate)\n"
    "- Fundamenta CADA argumento en tu programa\n"
    "- Cita cifras y fuentes cuando las tengas\n"
    "- Español colombiano formal\n"
    "- NO menciones todavía la propuesta del rival\n"
    "- Última línea obligatoria: {disclaimer}"
)

REBUTTAL_SYSTEM_TEMPLATE = (
    "Eres {full_name}, candidato presidencial de Colombia 2026 por {coalition}.\n"
    "Orientación política: {leaning}.\n\n"
    "TU PROGRAMA:\n{platform}\n\n"
    "REGLAS DE RÉPLICA:\n"
    "- Habla en primera persona como el candidato\n"
    "- Máximo 200 palabras (≈1 minuto)\n"
    "- Señala al menos 2 problemas concretos en la propuesta rival\n"
    "- Cita evidencia o datos que contradigan al rival\n"
    "- Contrasta con tu propuesta que es superior\n"
    "- Español colombiano formal\n"
    "- Última línea obligatoria: {disclaimer}"
)

ARTIFACT_PROMPT_TEMPLATE = (
    "Genera UN artefacto de apoyo (estadística, cita, análisis comparativo) "
    "que {name} usaría en un debate sobre '{topic}' para reforzar su argumento. "
    "Responde SOLO con JSON (sin markdown):\n"
    '{{"type": "statistic|citation|analysis", "title": "...", "content": "...", "source": "..."}}'
)


async def _ai_position(actor_client, candidate_key: str, platform: str,
                        question: str, topic: str, opponent_key: str) -> tuple[str, int, list[dict], Optional[str]]:
    cand = CANDIDATE_BY_KEY.get(candidate_key)
    if not cand:
        return f"[{candidate_key}: no encontrado]", 0, [], "Not found"

    system = POSITION_SYSTEM_TEMPLATE.format(
        full_name=cand.full_name, coalition=cand.coalition,
        leaning=cand.political_leaning, platform=_short(platform, 2000),
        disclaimer=DISCLAIMER,
    )
    opp = CANDIDATE_BY_KEY.get(opponent_key)
    prompt = (
        f"El moderador pregunta sobre {topic}:\n\n{question}\n\n"
        f"Presenta tu posición propia. (Tu rival es {opp.display_name if opp else opponent_key}; "
        f"no respondas a su propuesta aún.)"
    )

    resp = await actor_client.query(prompt, TEMP_ACTOR, MAX_TOKENS_TURN, system)
    if not resp.raw_text:
        log.warning(f"[ORCH] Actor {candidate_key} position empty — {resp.error}")

    # Generate supporting artifact
    artifact_resp = await actor_client.query(
        ARTIFACT_PROMPT_TEMPLATE.format(name=cand.display_name, topic=topic),
        TEMP_ACTOR, MAX_TOKENS_ARTIFACT,
    )
    artifacts = []
    art = _parse_json_safe(artifact_resp.raw_text or "", f"artifact {candidate_key}")
    if art:
        artifacts.append(art)

    tokens = resp.input_tokens + resp.output_tokens + artifact_resp.output_tokens
    return resp.raw_text or "[Sin respuesta]", tokens, artifacts, resp.error


async def _ai_rebuttal(actor_client, candidate_key: str, platform: str,
                        question: str, opponent_text: str, topic: str,
                        prior_rebuttal: str = "") -> tuple[str, int, list[dict], Optional[str]]:
    cand = CANDIDATE_BY_KEY.get(candidate_key)
    if not cand:
        return f"[{candidate_key}: no encontrado]", 0, [], "Not found"

    system = REBUTTAL_SYSTEM_TEMPLATE.format(
        full_name=cand.full_name, coalition=cand.coalition,
        leaning=cand.political_leaning, platform=_short(platform, 2000),
        disclaimer=DISCLAIMER,
    )

    opp_section = f'Tu rival argumentó:\n"{_short(opponent_text, 600)}"'
    if prior_rebuttal:
        opp_section += f'\n\nAdemás, tu rival ya rebatió tu posición:\n"{_short(prior_rebuttal, 400)}"'

    prompt = (
        f"El moderador preguntó sobre {topic}:\n{question}\n\n"
        f"{opp_section}\n\n"
        f"En tu tiempo de réplica, señala los problemas en la propuesta rival y "
        f"defiende tu posición con datos específicos."
    )

    resp = await actor_client.query(prompt, TEMP_ACTOR, MAX_TOKENS_TURN, system)
    if not resp.raw_text:
        log.warning(f"[ORCH] Actor {candidate_key} rebuttal empty — {resp.error}")

    # Artifact for rebuttal
    artifact_resp = await actor_client.query(
        ARTIFACT_PROMPT_TEMPLATE.format(name=cand.display_name, topic=topic),
        TEMP_ACTOR, MAX_TOKENS_ARTIFACT,
    )
    artifacts = []
    art = _parse_json_safe(artifact_resp.raw_text or "", f"rebuttal artifact {candidate_key}")
    if art:
        artifacts.append(art)

    tokens = resp.input_tokens + resp.output_tokens + artifact_resp.output_tokens
    return resp.raw_text or "[Sin respuesta]", tokens, artifacts, resp.error


# ── PHASE 3: Judge analytics ───────────────────────────────────────────────────

ANALYTICS_SYSTEM = (
    "Eres analista político imparcial. Produces análisis estructurado "
    "sobre CÓMO, POR QUÉ y PARA QUIÉN propone cada candidato. "
    "NO eliges ganador. Evalúas calidad de evidencia y especificidad. "
    "Tu análisis ayuda a ciudadanos a comparar propuestas objetivamente."
)

ANALYTICS_PROMPT_TEMPLATE = """
DEBATE PRESIDENCIAL COLOMBIA 2026
Tema: {topic}
{a_name} vs {b_name}

TRANSCRIPCIÓN POR PREGUNTA:
{transcript}

Produce análisis IMPARCIAL en JSON (sin markdown):
{{
  "how": {{
    "{a_key}": "¿Cómo implementa {a_name}? Mecanismos, plazos, presupuesto.",
    "{b_key}": "¿Cómo implementa {b_name}? Mecanismos, plazos, presupuesto."
  }},
  "why": {{
    "{a_key}": "¿Por qué {a_name} prioriza este enfoque? Teoría del cambio.",
    "{b_key}": "¿Por qué {b_name}? Motivación e ideología."
  }},
  "who": {{
    "{a_key}": "¿Quiénes se benefician con {a_name}? ¿Quiénes pagan los costos?",
    "{b_key}": "¿Quiénes con {b_name}? ¿Quiénes pagan?"
  }},
  "strongest_arguments": {{
    "{a_key}": "El argumento más sólido que presentó {a_name}.",
    "{b_key}": "El argumento más sólido que presentó {b_name}."
  }},
  "weakest_points": {{
    "{a_key}": "El punto más débil o sin responder de {a_name}.",
    "{b_key}": "El punto más débil de {b_name}."
  }},
  "key_tensions": ["Tensión real entre sus propuestas", "..."],
  "unanswered": ["Pregunta que quedó sin respuesta concreta"],
  "evidence_quality": {{"{a_key}": 0, "{b_key}": 0}},
  "specificity": {{"{a_key}": 0, "{b_key}": 0}},
  "rebuttal_effectiveness": {{"{a_key}": 0, "{b_key}": 0}},
  "summary": "3-4 oraciones. Sin elegir ganador. Qué propuso cada uno y qué quedó pendiente.",
  "disclaimer": "{disclaimer}"
}}
Scores 0-10. Sé riguroso e imparcial.
""".strip()


async def _generate_analytics(judge_client, debate: OrchestratedDebate,
                               platform_a: str, platform_b: str) -> dict:
    cand_a = CANDIDATE_BY_KEY.get(debate.participant_a.candidate_key)
    cand_b = CANDIDATE_BY_KEY.get(debate.participant_b.candidate_key)
    a_name = cand_a.display_name if cand_a else debate.participant_a.candidate_key
    b_name = cand_b.display_name if cand_b else debate.participant_b.candidate_key
    a_key  = debate.participant_a.candidate_key
    b_key  = debate.participant_b.candidate_key

    # Build transcript grouped by question
    transcript_parts = []
    for q_idx, question in enumerate(debate.questions):
        transcript_parts.append(f"\n── PREGUNTA {q_idx + 1}: {question}")
        q_turns = sorted([t for t in debate.turns if t.question_idx == q_idx],
                         key=lambda t: t.turn_order)
        for turn in q_turns:
            label = ("POSICIÓN" if turn.turn_type == TURN_POSITION else "RÉPLICA")
            transcript_parts.append(f"\n  [{turn.speaker_display} — {label}]\n  {turn.text}")

    prompt = ANALYTICS_PROMPT_TEMPLATE.format(
        topic=debate.topic,
        a_name=a_name, b_name=b_name, a_key=a_key, b_key=b_key,
        transcript="\n".join(transcript_parts),
        disclaimer=DISCLAIMER,
    )

    resp = await judge_client.query(prompt, TEMP_JUDGE, MAX_TOKENS_ANALYTICS, ANALYTICS_SYSTEM)
    data = _parse_json_safe(resp.raw_text or "", "analytics")
    if data:
        return data

    return {"summary": resp.raw_text or "[Sin análisis]", "raw": True}


# ── Main orchestration ─────────────────────────────────────────────────────────

async def run_orchestrated_debate(debate: OrchestratedDebate, clients: dict,
                                   run_date: date = None) -> OrchestratedDebate:
    if run_date is None:
        run_date = date.today()

    judge_client = clients.get(debate.judge_model_key) or next(iter(clients.values()), None)
    if not judge_client:
        debate.status = "error"
        debate.error = "No LLM clients available"
        return debate
    if debate.judge_model_key not in clients:
        log.warning(f"Judge {debate.judge_model_key} unavailable, using fallback")

    platform_a = _load_platform(debate.participant_a.candidate_key)
    platform_b = _load_platform(debate.participant_b.candidate_key)

    cand_a = CANDIDATE_BY_KEY.get(debate.participant_a.candidate_key)
    cand_b = CANDIDATE_BY_KEY.get(debate.participant_b.candidate_key)
    a_display = cand_a.display_name if cand_a else debate.participant_a.candidate_key
    b_display = cand_b.display_name if cand_b else debate.participant_b.candidate_key

    # Phase 1: Judge generates questions
    debate.status = "questioning"
    log.info(f"[ORCH] {debate.debate_id} — generating {debate.max_questions} questions")
    debate.questions = await _generate_questions(
        judge_client, debate.topic,
        debate.participant_a.candidate_key, debate.participant_b.candidate_key,
        platform_a, platform_b, debate.max_questions, run_date,
    )

    # Phase 2: 4 turns per question
    debate.status = "debating"
    actor_a = clients.get(debate.participant_a.actor_model_key) or judge_client
    actor_b = clients.get(debate.participant_b.actor_model_key) or judge_client

    for q_idx, question in enumerate(debate.questions):
        debate.current_question_idx = q_idx
        log.info(f"[ORCH] Q{q_idx + 1}/{len(debate.questions)}: {question[:80]}...")

        # Turn 1: A position
        if debate.participant_a.mode == "ai":
            text, tokens, arts, err = await _ai_position(
                actor_a, debate.participant_a.candidate_key, platform_a,
                question, debate.topic, debate.participant_b.candidate_key,
            )
            debate.turns.append(DebateTurn(
                question_idx=q_idx, speaker_key=debate.participant_a.candidate_key,
                speaker_display=a_display, turn_type=TURN_POSITION, turn_order=1,
                text=text, mode="ai", model_used=actor_a.model_id,
                tokens_used=tokens, artifacts=arts, error=err,
            ))
            debate.total_tokens += tokens
            log.info(f"[ORCH]   Turn 1 ({a_display} pos): {tokens} tokens")

        # Turn 2: B position
        if debate.participant_b.mode == "ai":
            text, tokens, arts, err = await _ai_position(
                actor_b, debate.participant_b.candidate_key, platform_b,
                question, debate.topic, debate.participant_a.candidate_key,
            )
            debate.turns.append(DebateTurn(
                question_idx=q_idx, speaker_key=debate.participant_b.candidate_key,
                speaker_display=b_display, turn_type=TURN_POSITION, turn_order=2,
                text=text, mode="ai", model_used=actor_b.model_id,
                tokens_used=tokens, artifacts=arts, error=err,
            ))
            debate.total_tokens += tokens
            log.info(f"[ORCH]   Turn 2 ({b_display} pos): {tokens} tokens")

        # Get position texts for rebuttals
        a_pos_text = next((t.text for t in debate.turns
                           if t.question_idx == q_idx and t.speaker_key == debate.participant_a.candidate_key
                           and t.turn_type == TURN_POSITION), "")
        b_pos_text = next((t.text for t in debate.turns
                           if t.question_idx == q_idx and t.speaker_key == debate.participant_b.candidate_key
                           and t.turn_type == TURN_POSITION), "")

        # Turn 3: B rebuts A
        if debate.participant_b.mode == "ai" and a_pos_text:
            text, tokens, arts, err = await _ai_rebuttal(
                actor_b, debate.participant_b.candidate_key, platform_b,
                question, a_pos_text, debate.topic,
            )
            debate.turns.append(DebateTurn(
                question_idx=q_idx, speaker_key=debate.participant_b.candidate_key,
                speaker_display=b_display, turn_type=TURN_REBUTTAL, turn_order=3,
                text=text, mode="ai", model_used=actor_b.model_id,
                tokens_used=tokens, artifacts=arts, error=err,
            ))
            debate.total_tokens += tokens
            log.info(f"[ORCH]   Turn 3 ({b_display} rebuttal): {tokens} tokens")

        # Turn 4: A rebuts B (sees B's position + B's rebuttal)
        b_reb_text = next((t.text for t in debate.turns
                           if t.question_idx == q_idx and t.speaker_key == debate.participant_b.candidate_key
                           and t.turn_type == TURN_REBUTTAL), "")
        if debate.participant_a.mode == "ai" and b_pos_text:
            text, tokens, arts, err = await _ai_rebuttal(
                actor_a, debate.participant_a.candidate_key, platform_a,
                question, b_pos_text, debate.topic, prior_rebuttal=b_reb_text,
            )
            debate.turns.append(DebateTurn(
                question_idx=q_idx, speaker_key=debate.participant_a.candidate_key,
                speaker_display=a_display, turn_type=TURN_REBUTTAL, turn_order=4,
                text=text, mode="ai", model_used=actor_a.model_id,
                tokens_used=tokens, artifacts=arts, error=err,
            ))
            debate.total_tokens += tokens
            log.info(f"[ORCH]   Turn 4 ({a_display} rebuttal): {tokens} tokens")

    # Phase 3: Judge HOW/WHY/WHO analytics
    debate.status = "judging"
    log.info("[ORCH] Judge generating HOW/WHY/WHO analytics...")
    debate.analytics = await _generate_analytics(judge_client, debate, platform_a, platform_b)

    debate.status = "complete"
    debate.completed_at = datetime.now().isoformat()
    log.info(f"[ORCH] {debate.debate_id} complete — {debate.total_tokens} total tokens")
    return debate


def submit_human_answer(debate: OrchestratedDebate, speaker: str, text: str) -> OrchestratedDebate:
    """Add a human participant's answer for the current question."""
    q_idx = debate.current_question_idx
    is_a = speaker == "a"
    candidate_key = (debate.participant_a.candidate_key if is_a
                     else debate.participant_b.candidate_key)
    cand = CANDIDATE_BY_KEY.get(candidate_key)
    display = cand.display_name if cand else candidate_key
    # Determine turn_order: if A, check if we already have B's position (rebuttal phase)
    existing_q = [t for t in debate.turns if t.question_idx == q_idx]
    has_b_pos = any(t.speaker_key == debate.participant_b.candidate_key
                    and t.turn_type == TURN_POSITION for t in existing_q)
    turn_type = TURN_REBUTTAL if (is_a and has_b_pos) else TURN_POSITION
    turn_order = 4 if (is_a and has_b_pos) else (1 if is_a else 2)
    debate.turns.append(DebateTurn(
        question_idx=q_idx, speaker_key=candidate_key, speaker_display=display,
        turn_type=turn_type, turn_order=turn_order,
        text=text[:1200], mode="human",
    ))
    debate.awaiting_human = None
    return debate


def save_orchestrated_debate(debate: OrchestratedDebate) -> Path:
    base = Path(__file__).parent.parent
    date_str = debate.debate_id[:10] if len(debate.debate_id) >= 10 else date.today().isoformat()
    results_dir = (
        Path("/tmp/debat-zero/orchestrated") / date_str
        if not (base / "data").exists()
        else base / "data" / "orchestrated_debates" / date_str
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{debate.debate_id}.json"
    path.write_text(json.dumps(debate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
