"""
Debate simulation engine.

Runs LLM actors as Colombian presidential candidates and a separate judge LLM.
Each actor is an LLM prompted with the candidate's platform as system context.
The judge scores the debate for argument quality, not political position.

Usage:
    from tracker.debate_simulator import run_simulation
    result = await run_simulation(SimulationConfig(...))
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from tracker.candidates import CANDIDATE_BY_KEY
from tracker.llm_clients import AVAILABLE_MODELS, build_selected_clients

log = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent.parent
KB_DIR = BASE_PATH / "data" / "knowledge_bases"

DISCLAIMER = "[SIMULACIÓN ACADÉMICA — no representa posiciones reales]"


@dataclass
class SimulationConfig:
    candidate_a_key: str
    candidate_b_key: str
    judge_model_key: str
    topic: str
    actor_model_key: str = "claude"
    num_rounds: int = 2
    max_tokens_actor: int = 600
    max_tokens_judge: int = 800
    temperature_actor: float = 0.75
    temperature_judge: float = 0.5


@dataclass
class SimulationTurnResult:
    round: int
    speaker_key: str
    speaker_display: str
    text: str
    model_used: str
    tokens_used: int
    error: Optional[str] = None


@dataclass
class SimulationResult:
    config: SimulationConfig
    turns: List[SimulationTurnResult] = field(default_factory=list)
    judgment: str = ""
    judgment_model: str = ""
    total_tokens: int = 0
    error: Optional[str] = None

    @property
    def candidate_a_display(self) -> str:
        c = CANDIDATE_BY_KEY.get(self.config.candidate_a_key)
        return c.display_name if c else self.config.candidate_a_key

    @property
    def candidate_b_display(self) -> str:
        c = CANDIDATE_BY_KEY.get(self.config.candidate_b_key)
        return c.display_name if c else self.config.candidate_b_key


def _load_platform(candidate_key: str) -> str:
    path = KB_DIR / candidate_key / "platform.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    c = CANDIDATE_BY_KEY.get(candidate_key)
    name = c.full_name if c else candidate_key
    return f"Candidato: {name}. (Base de conocimiento no disponible.)"


def _actor_system_prompt(candidate_key: str, platform: str, topic: str, opponent_key: str) -> str:
    c = CANDIDATE_BY_KEY.get(candidate_key)
    opp = CANDIDATE_BY_KEY.get(opponent_key)
    if not c:
        return f"Eres un candidato presidencial. Debate sobre: {topic}."
    opp_name = opp.display_name if opp else opponent_key

    return (
        f"Eres {c.full_name}, candidato presidencial de Colombia 2026 por {c.coalition}.\n"
        f"Tu orientación política: {c.political_leaning}.\n\n"
        f"TU PROGRAMA DE GOBIERNO (contexto clave):\n{platform}\n\n"
        f"INSTRUCCIONES DE ROL:\n"
        f"- Habla en primera persona como {c.display_name}\n"
        f"- Fundamenta CADA argumento en propuestas concretas de tu programa\n"
        f"- Cita cifras, políticas específicas y plazos cuando los tengas\n"
        f"- Cuando respondas a {opp_name}, señala diferencias reales entre sus propuestas y las tuyas\n"
        f"- Tono presidencial pero accesible, en español colombiano\n"
        f"- Máximo 3 párrafos por intervención\n"
        f"- Termina siempre con: {DISCLAIMER}"
    )


def _judge_system_prompt(candidate_a_display: str, candidate_b_display: str) -> str:
    return (
        f"Eres el moderador-juez imparcial de un debate presidencial colombiano 2026 "
        f"entre {candidate_a_display} y {candidate_b_display}.\n\n"
        "Tu función es evaluar la CALIDAD de los argumentos, no la posición política.\n\n"
        "DIMENSIONES DE EVALUACIÓN (0–10 cada una):\n"
        "1. Consistencia con el programa del candidato\n"
        "2. Calidad argumentativa: lógica, evidencia, especificidad\n"
        "3. Relevancia al tema del debate\n"
        "4. Claridad y comunicación\n\n"
        "FORMATO DE RESPUESTA:\n"
        "- Puntajes por candidato y por dimensión\n"
        "- Análisis narrativo (2-3 párrafos)\n"
        "- Veredicto: ¿Quién presentó los argumentos más sólidos y concretos?\n"
        "- Imparcialidad: No juzgues la posición política, solo la calidad del argumento\n\n"
        f"Termina con: {DISCLAIMER}"
    )


def _judge_turn_prompt(turns: List[SimulationTurnResult], topic: str) -> str:
    transcript = "\n".join(
        f"\n--- {t.speaker_display} (Ronda {t.round + 1}) ---\n{t.text}"
        for t in turns
    )
    return (
        f"TEMA DEL DEBATE: {topic}\n\n"
        f"TRANSCRIPCIÓN COMPLETA:\n{transcript}\n\n"
        "Por favor evalúa este debate según las dimensiones indicadas y emite tu veredicto."
    )


async def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    result = SimulationResult(config=cfg)

    # Resolve which models to load
    models_needed = {cfg.actor_model_key}
    if cfg.judge_model_key != cfg.actor_model_key:
        models_needed.add(cfg.judge_model_key)

    clients = build_selected_clients(list(models_needed))

    if cfg.actor_model_key not in clients:
        result.error = f"Actor model '{cfg.actor_model_key}' unavailable — check API key."
        return result

    actor = clients[cfg.actor_model_key]
    judge_client = clients.get(cfg.judge_model_key) or clients[cfg.actor_model_key]
    judge_model_used = cfg.judge_model_key if cfg.judge_model_key in clients else cfg.actor_model_key

    platform_a = _load_platform(cfg.candidate_a_key)
    platform_b = _load_platform(cfg.candidate_b_key)

    cand_a = CANDIDATE_BY_KEY.get(cfg.candidate_a_key)
    cand_b = CANDIDATE_BY_KEY.get(cfg.candidate_b_key)
    a_display = cand_a.display_name if cand_a else cfg.candidate_a_key
    b_display = cand_b.display_name if cand_b else cfg.candidate_b_key

    sys_a = _actor_system_prompt(cfg.candidate_a_key, platform_a, cfg.topic, cfg.candidate_b_key)
    sys_b = _actor_system_prompt(cfg.candidate_b_key, platform_b, cfg.topic, cfg.candidate_a_key)

    history: list[str] = []

    for round_num in range(cfg.num_rounds):
        # --- Candidate A speaks ---
        if round_num == 0:
            prompt_a = (
                f"Estamos debatiendo sobre: {cfg.topic}.\n"
                f"Presenta tu posición inicial. Sé específico con tus propuestas."
            )
        else:
            last_b = result.turns[-1].text if result.turns else ""
            prompt_a = (
                f"Tu oponente {b_display} acaba de decir:\n\n{last_b}\n\n"
                f"Responde a sus argumentos y refuerza tu posición sobre: {cfg.topic}."
            )

        log.info(f"[SIM] Ronda {round_num + 1} — {a_display} hablando...")
        resp_a = await actor.query(
            prompt=prompt_a,
            temperature=cfg.temperature_actor,
            max_tokens=cfg.max_tokens_actor,
            system_prompt=sys_a,
        )
        result.turns.append(SimulationTurnResult(
            round=round_num,
            speaker_key=cfg.candidate_a_key,
            speaker_display=a_display,
            text=resp_a.raw_text or "[Sin respuesta]",
            model_used=actor.model_id,
            tokens_used=resp_a.output_tokens,
            error=resp_a.error,
        ))
        result.total_tokens += resp_a.input_tokens + resp_a.output_tokens

        # --- Candidate B speaks ---
        last_a = result.turns[-1].text
        prompt_b = (
            f"{a_display} acaba de decir:\n\n{last_a}\n\n"
            f"Responde a sus argumentos y defiende tu posición sobre: {cfg.topic}."
        )

        log.info(f"[SIM] Ronda {round_num + 1} — {b_display} hablando...")
        resp_b = await actor.query(
            prompt=prompt_b,
            temperature=cfg.temperature_actor,
            max_tokens=cfg.max_tokens_actor,
            system_prompt=sys_b,
        )
        result.turns.append(SimulationTurnResult(
            round=round_num,
            speaker_key=cfg.candidate_b_key,
            speaker_display=b_display,
            text=resp_b.raw_text or "[Sin respuesta]",
            model_used=actor.model_id,
            tokens_used=resp_b.output_tokens,
            error=resp_b.error,
        ))
        result.total_tokens += resp_b.input_tokens + resp_b.output_tokens

    # --- Judge evaluates ---
    log.info(f"[SIM] Juez {judge_model_used} evaluando...")
    judge_sys = _judge_system_prompt(a_display, b_display)
    judge_prompt = _judge_turn_prompt(result.turns, cfg.topic)
    judge_resp = await judge_client.query(
        prompt=judge_prompt,
        temperature=cfg.temperature_judge,
        max_tokens=cfg.max_tokens_judge,
        system_prompt=judge_sys,
    )
    result.judgment = judge_resp.raw_text or "[Sin veredicto]"
    result.judgment_model = judge_model_used
    result.total_tokens += judge_resp.input_tokens + judge_resp.output_tokens

    log.info(f"[SIM] Debate completado — {result.total_tokens} tokens totales")
    return result


# ── Pre-configured debate suite ───────────────────────────────────────────────

PRESET_DEBATES = [
    {
        "id": "debate-1-fajardo-vs-espriella",
        "label": "Fajardo vs De la Espriella",
        "candidate_a": "fajardo",
        "candidate_b": "de_la_espriella",
        "judge": "mistral_large",
        "topic": "economía y modelo de Estado",
    },
    {
        "id": "debate-2-valencia-vs-cepeda",
        "label": "Valencia vs Cepeda",
        "candidate_a": "valencia",
        "candidate_b": "cepeda",
        "judge": "claude",
        "topic": "seguridad y paz",
    },
    {
        "id": "debate-3-fajardo-vs-lopez",
        "label": "Fajardo vs Claudia López",
        "candidate_a": "fajardo",
        "candidate_b": "lopez",
        "judge": "qwen_direct",
        "topic": "medio ambiente y transición energética",
    },
    {
        "id": "debate-4-lopez-vs-espriella",
        "label": "Claudia López vs De la Espriella",
        "candidate_a": "lopez",
        "candidate_b": "de_la_espriella",
        "judge": "llama_8b",
        "topic": "salud y bienestar social",
    },
]
