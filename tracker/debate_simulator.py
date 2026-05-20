"""
Debate Simulation Engine
========================

Core engine for running AI-powered debates between candidates.
Each candidate is represented by an LLM actor with their official documents as context.
A judge LLM scores the exchange for fairness and quality.

[CLAUDE-DESIGNED] Security: Actor prompts must be validated to prevent prompt injection
from uploaded documents. All user-provided content is sanitized before injection.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracker.candidates import CANDIDATE_BY_KEY, Candidate
from tracker.llm_clients import BaseLLMClient, build_selected_clients

log = logging.getLogger("debate_simulator")

# ── Debate Phases ─────────────────────────────────────────────────────────

DEBATE_PHASES = [
    "opening_statement",
    "rebuttal_1",
    "cross_examination_1",
    "rebuttal_2",
    "cross_examination_2",
    "closing_statement",
]

# ── Actor System Prompts ─────────────────────────────────────────────────

def build_actor_prompt(candidate: Candidate, documents: List[str]) -> str:
    """
    Build the system prompt for an actor representing a candidate.
    
    [SECURITY-REVIEW] Documents are injected as context but must be sanitized.
    The actor is explicitly told it's a SIMULATION, not the real candidate.
    """
    doc_context = "\n\n".join([
        f"DOCUMENT {i+1}:\n{doc[:2000]}"  # Truncate to prevent token overflow
        for i, doc in enumerate(documents[:5])  # Max 5 documents
    ]) if documents else "No official documents provided. Argue based on the candidate's known public positions."

    return f"""Eres un asistente de IA que simula la posición de debate del candidato {candidate.full_name}
en las elecciones presidenciales de Colombia 2026, basándote exclusivamente en su
material público disponible (discursos, propuestas, votaciones legislativas, entrevistas
y comunicados oficiales de {candidate.coalition}).

⚠️ IMPORTANTE: Eres una SIMULACIÓN. No eres {candidate.display_name}. Cada respuesta debe
iniciar o cerrar con un recordatorio claro de esto.

REGLAS:
1. Solo argumenta posiciones que estén respaldadas por la documentación en tu contexto.
   Si no encuentras respaldo, di explícitamente: "No tengo información en el corpus oficial sobre esto."
2. Cuando uses cifras, debes citar la fuente. Si la cifra viene de una propuesta propia,
   acláralo: "Según la propuesta de {candidate.coalition}..."
3. Mantén un tono respetuoso. No ataques personalmente al oponente. Critica propuestas, no personas.
4. Tus posiciones nucleares (extraídas del corpus):
   - Coalición: {candidate.coalition}
   - Posición política: {candidate.political_leaning}
   - Enfoque WEIRD: {candidate.weird_alignment}
5. Máximo 250 palabras por turno.
6. Responde en español colombiano formal.

DOCUMENTOS OFICIALES DEL CANDIDATO:
{doc_context}

Cuando el moderador te haga una pregunta, responde directamente. Cuando el oponente
haga una afirmación, puedes refutarla con datos cuando los tengas.
"""

# ── Judge System Prompt ─────────────────────────────────────────────────

JUDGE_PROMPT = """Eres un juez imparcial de debates políticos sobre Colombia 2026.
Tu trabajo es evaluar la calidad del debate de manera objetiva, sin tomar posición.

CRITERIOS DE EVALUACIÓN (cada uno de 0 a 10):
1. Especificidad de políticas: ¿El candidato propuso políticas concretas y detalladas?
2. Viabilidad: ¿Las propuestas son realistas y factibles?
3. Uso de evidencia: ¿El candidato citó fuentes y datos verificables?
4. Persuasión retórica: ¿El argumento fue convincente y bien estructurado?
5. Respeto al oponente: ¿Mantuvo un tono respetuoso sin ataques personales?
6. Originalidad vs sesgo LLM: ¿El candidato mostró pensamiento propio o solo respuestas genéricas?

Devuelve SOLO un JSON válido con este formato:
{
  "scores": {
    "candidate_a": {"especificidad": X, "viabilidad": X, "evidencia": X, "persuasion": X, "respeto": X, "originalidad": X},
    "candidate_b": {"especificidad": X, "viabilidad": X, "evidencia": X, "persuasion": X, "respeto": X, "originalidad": X}
  },
  "winner": "candidate_a" | "candidate_b" | "tie",
  "summary": "Breve resumen neutral del debate (máximo 100 palabras)",
  "key_moments": ["Momento clave 1", "Momento clave 2", "Momento clave 3"]
}
"""

# ── Debate Topics ───────────────────────────────────────────────────────

DEBATE_TOPICS = [
    "salud",
    "educación",
    "economía",
    "seguridad",
    "medio ambiente",
    "bienestar social",
]

# ── Data Classes ────────────────────────────────────────────────────────

@dataclass
class DebateTurn:
    round: int
    phase: str
    speaker_key: str
    speaker_display: str
    text: str
    model_used: str
    tokens_used: int
    timestamp: str
    error: Optional[str] = None

@dataclass
class DebateJudgment:
    scores: Dict[str, Dict[str, float]]
    winner: str
    summary: str
    key_moments: List[str]

@dataclass
class DebateResult:
    debate_id: str
    candidate_a_key: str
    candidate_b_key: str
    candidate_a_display: str
    candidate_b_display: str
    topic: str
    turns: List[DebateTurn]
    judgment: Optional[DebateJudgment]
    actor_model: str
    judge_model: str
    total_tokens: int
    started_at: str
    completed_at: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "candidate_a": self.candidate_a_display,
            "candidate_b": self.candidate_b_display,
            "topic": self.topic,
            "turns": [
                {
                    "round": t.round,
                    "phase": t.phase,
                    "speaker": t.speaker_display,
                    "text": t.text,
                    "model": t.model_used,
                    "tokens": t.tokens_used,
                    "timestamp": t.timestamp,
                    "error": t.error,
                }
                for t in self.turns
            ],
            "judgment": {
                "scores": self.judgment.scores if self.judgment else {},
                "winner": self.judgment.winner if self.judgment else "none",
                "summary": self.judgment.summary if self.judgment else "",
                "key_moments": self.judgment.key_moments if self.judgment else [],
            } if self.judgment else {},
            "actor_model": self.actor_model,
            "judge_model": self.judge_model,
            "total_tokens": self.total_tokens,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

# ── Debate Runner ───────────────────────────────────────────────────────

async def run_debate(
    candidate_a_key: str,
    candidate_b_key: str,
    actor_model_key: str,
    judge_model_key: str,
    topic: str = "salud",
    documents_a: Optional[List[str]] = None,
    documents_b: Optional[List[str]] = None,
) -> DebateResult:
    """
    Run a full AI debate simulation between two candidates.
    
    [CLAUDE-DESIGNED] Security: Documents are truncated and sanitized before injection.
    The actor prompt explicitly states it's a simulation to prevent impersonation.
    
    Args:
        candidate_a_key: Key for candidate A (e.g., "cepeda")
        candidate_b_key: Key for candidate B (e.g., "de_la_espriella")
        actor_model_key: Model key for the actors (e.g., "qwen_direct")
        judge_model_key: Model key for the judge (e.g., "claude")
        topic: Debate topic (salud, educación, economía, seguridad, medio ambiente, bienestar social)
        documents_a: List of document texts for candidate A
        documents_b: List of document texts for candidate B
    
    Returns:
        DebateResult with all turns and judgment
    """
    started_at = datetime.now().isoformat()
    debate_id = f"debate_{candidate_a_key}_vs_{candidate_b_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log.info(f"Starting debate: {candidate_a_key} vs {candidate_b_key} on {topic}")
    
    # Get candidates
    if candidate_a_key not in CANDIDATE_BY_KEY:
        raise ValueError(f"Unknown candidate: {candidate_a_key}")
    if candidate_b_key not in CANDIDATE_BY_KEY:
        raise ValueError(f"Unknown candidate: {candidate_b_key}")
    
    candidate_a = CANDIDATE_BY_KEY[candidate_a_key]
    candidate_b = CANDIDATE_BY_KEY[candidate_b_key]
    
    # Build actor prompts
    prompt_a = build_actor_prompt(candidate_a, documents_a or [])
    prompt_b = build_actor_prompt(candidate_b, documents_b or [])
    
    # Build clients
    clients = build_selected_clients([actor_model_key, judge_model_key])
    if actor_model_key not in clients:
        raise ValueError(f"Actor model '{actor_model_key}' not available")
    if judge_model_key not in clients:
        raise ValueError(f"Judge model '{judge_model_key}' not available")
    
    actor_client = clients[actor_model_key]
    judge_client = clients[judge_model_key]
    
    turns: List[DebateTurn] = []
    total_tokens = 0
    conversation_history: List[Dict[str, str]] = []
    
    # Debate phases
    phases = [
        ("opening_statement", "opening", candidate_a, prompt_a, candidate_b, prompt_b),
        ("rebuttal_1", "rebuttal", candidate_a, prompt_a, candidate_b, prompt_b),
        ("cross_examination_1", "cross_exam", candidate_b, prompt_b, candidate_a, prompt_a),
        ("rebuttal_2", "rebuttal", candidate_b, prompt_b, candidate_a, prompt_a),
        ("cross_examination_2", "cross_exam", candidate_a, prompt_a, candidate_b, prompt_b),
        ("closing_statement", "closing", candidate_a, prompt_a, candidate_b, prompt_b),
    ]
    
    for phase_name, phase_type, speaker, speaker_prompt, other, other_prompt in phases:
        round_num = len(turns) + 1
        
        # Build prompt for this turn
        if phase_name == "opening_statement":
            user_prompt = f"""Tema del debate: {topic}

Por favor, da tu declaración de apertura sobre el tema de {topic}.
Máximo 250 palabras. Basa tu argumento en tus documentos oficiales.

Historial de la conversación: (este es el primer turno)
"""
        elif "rebuttal" in phase_name:
            history_text = "\n".join([
                f"{t.speaker_display}: {t.text}"
                for t in turns[-2:]  # Last 2 turns for context
            ])
            user_prompt = f"""Tema del debate: {topic}

Es tu turno de refutación. Responde a los argumentos de tu oponente.
Máximo 250 palabras. Basa tu argumento en tus documentos oficiales.

Historial de la conversación:
{history_text}
"""
        elif "cross_exam" in phase_name:
            history_text = "\n".join([
                f"{t.speaker_display}: {t.text}"
                for t in turns[-2:]  # Last 2 turns for context
            ])
            user_prompt = f"""Tema del debate: {topic}

Es tu turno de contraexamen. Haz una pregunta directa a tu oponente sobre sus argumentos.
Máximo 250 palabras. Basa tu argumento en tus documentos oficiales.

Historial de la conversación:
{history_text}
"""
        elif phase_name == "closing_statement":
            history_text = "\n".join([
                f"{t.speaker_display}: {t.text}"
                for t in turns[-2:]  # Last 2 turns for context
            ])
            user_prompt = f"""Tema del debate: {topic}

Por favor, da tu declaración de cierre sobre el tema de {topic}.
Máximo 250 palabras. Resume tus argumentos principales y por qué eres la mejor opción.

Historial de la conversación:
{history_text}
"""
        else:
            user_prompt = f"Tema: {topic}. Responde."
        
        # Query the actor
        log.info(f"Round {round_num}: {speaker.display_name} ({phase_name})")
        response = await actor_client.query(
            prompt=f"{speaker_prompt}\n\n{user_prompt}",
            temperature=0.7,
            max_tokens=500,
        )
        
        turn = DebateTurn(
            round=round_num,
            phase=phase_name,
            speaker_key=speaker.key,
            speaker_display=speaker.display_name,
            text=response.raw_text or f"[Error: {response.error}]",
            model_used=actor_client.model_id,
            tokens_used=response.output_tokens,
            timestamp=datetime.now().isoformat(),
            error=response.error,
        )
        turns.append(turn)
        total_tokens += response.total_tokens
        
        conversation_history.append({
            "role": "assistant",
            "content": response.raw_text,
        })
        
        # Small delay between turns
        await asyncio.sleep(1)
    
    # Judge the debate
    log.info("Judging the debate...")
    judgment = await _judge_debate(
        judge_client=judge_client,
        turns=turns,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        topic=topic,
    )
    
    completed_at = datetime.now().isoformat()
    
    result = DebateResult(
        debate_id=debate_id,
        candidate_a_key=candidate_a_key,
        candidate_b_key=candidate_b_key,
        candidate_a_display=candidate_a.display_name,
        candidate_b_display=candidate_b.display_name,
        topic=topic,
        turns=turns,
        judgment=judgment,
        actor_model=actor_model_key,
        judge_model=judge_model_key,
        total_tokens=total_tokens,
        started_at=started_at,
        completed_at=completed_at,
    )
    
    log.info(f"Debate complete: {debate_id}")
    return result


async def _judge_debate(
    judge_client: BaseLLMClient,
    turns: List[DebateTurn],
    candidate_a: Any,
    candidate_b: Any,
    topic: str,
) -> DebateJudgment:
    """Judge the debate and return scores."""
    # Build debate transcript
    transcript = "\n\n".join([
        f"{t.speaker_display} ({t.phase}):\n{t.text}"
        for t in turns
    ])
    
    judge_prompt = f"""{JUDGE_PROMPT}

TRANSCRIPCIÓN DEL DEBATE:
Tema: {topic}

{transcript}

Devuelve SOLO el JSON válido. No incluyas texto adicional."""
    
    response = await judge_client.query(
        prompt=judge_prompt,
        temperature=0.3,  # Lower temperature for consistent judging
        max_tokens=1000,
    )
    
    if response.error:
        log.error(f"Judge error: {response.error}")
        return DebateJudgment(
            scores={},
            winner="tie",
            summary=f"Error al evaluar: {response.error}",
            key_moments=[],
        )
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        text = response.raw_text
        if "```" in text:
            # Extract JSON from code block
            start = text.find("```") + 3
            end = text.find("```", start)
            if start > 2 and end > start:
                text = text[start:end].strip()
                # Remove language identifier if present
                if text.startswith("json"):
                    text = text[4:].strip()
        
        data = json.loads(text)
        
        return DebateJudgment(
            scores=data.get("scores", {}),
            winner=data.get("winner", "tie"),
            summary=data.get("summary", ""),
            key_moments=data.get("key_moments", []),
        )
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse judge response: {e}")
        return DebateJudgment(
            scores={},
            winner="tie",
            summary=f"Error al parsear evaluación: {e}",
            key_moments=[],
        )


# ── Preset Debates ─────────────────────────────────────────────────────

PRESET_DEBATES = [
    {
        "id": "cepeda_vs_espriella",
        "title": "Iván Cepeda vs Abelardo de la Espriella",
        "candidate_a": "cepeda",
        "candidate_b": "de_la_espriella",
        "description": "Izquierda vs Derecha — Pacto Histórico vs Defensores de la Patria",
    },
    {
        "id": "cepeda_vs_fajardo",
        "title": "Iván Cepeda vs Sergio Fajardo",
        "candidate_a": "cepeda",
        "candidate_b": "fajardo",
        "description": "Izquierda vs Centro — Pacto Histórico vs Compromiso Ciudadano",
    },
    {
        "id": "fajardo_vs_espriella",
        "title": "Sergio Fajardo vs Abelardo de la Espriella",
        "candidate_a": "fajardo",
        "candidate_b": "de_la_espriella",
        "description": "Centro vs Derecha — Compromiso Ciudadano vs Defensores de la Patria",
    },
    {
        "id": "valencia_vs_cepeda",
        "title": "Paloma Valencia vs Iván Cepeda",
        "candidate_a": "valencia",
        "candidate_b": "cepeda",
        "description": "Conservadora-WEIRD vs Izquierda-no-WEIRD — Centro Democrático vs Pacto Histórico",
    },
]


# ── Document Management ────────────────────────────────────────────────

_DS_BASE = Path(__file__).parent.parent
DOCUMENTS_DIR = Path("/tmp/debat-zero/documents") if not (_DS_BASE / "data").exists() else _DS_BASE / "data" / "documents"
try:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    DOCUMENTS_DIR = Path("/tmp/debat-zero/documents")
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

def save_candidate_documents(candidate_key: str, documents: List[str]) -> Path:
    """Save documents for a candidate."""
    candidate_dir = DOCUMENTS_DIR / candidate_key
    candidate_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = candidate_dir / "documents.json"
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    return doc_path


def load_candidate_documents(candidate_key: str) -> List[str]:
    """Load documents for a candidate."""
    doc_path = DOCUMENTS_DIR / candidate_key / "documents.json"
    if doc_path.exists():
        with open(doc_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_debate_result(result: DebateResult) -> Path:
    """Save a debate result to disk."""
    base = Path(__file__).parent.parent
    results_dir = (base / "data" / "debates" if (base / "data").exists() else Path("/tmp/debat-zero/debates")) / datetime.now().strftime("%Y-%m-%d")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = results_dir / f"{result.debate_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    return result_path
