"""
LATAM GPT Benchmark Integration

Integrates LATAM GPT's open-source benchmarks into debat-zero:
- Trueque: Human-reviewed collaborative benchmark for LATAM knowledge (500+ Q&A)
- CHOCLO: Latin American Cultural Knowledge Benchmark (105k Q&A)

These benchmarks measure how well models understand LATAM-specific knowledge,
which is critical for evaluating political bias in Colombian electoral context.

References:
- https://github.com/latam-gpt
- https://huggingface.co/latam-gpt
- Trueque: huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1
- CHOCLO: huggingface.co/datasets/latam-gpt/CHOCLO
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Benchmark result schemas ──────────────────────────────────────────────

class BenchmarkQuestionResult(BaseModel):
    question_id: str
    benchmark: str          # "trueque" or "choclo"
    question: str
    reference_answer: str
    model_response: str
    model_key: str
    model_provider: str
    is_correct: bool
    score: float            # 0.0 - 1.0
    latency_ms: float
    error: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class BenchmarkRunResult(BaseModel):
    benchmark: str
    model_key: str
    model_provider: str
    timestamp: str
    total_questions: int
    correct: int
    accuracy: float
    avg_score: float
    avg_latency_ms: float
    results: List[BenchmarkQuestionResult] = Field(default_factory=list)
    breakdown_by_difficulty: Dict[str, Dict] = Field(default_factory=dict)
    breakdown_by_category: Dict[str, Dict] = Field(default_factory=dict)
    breakdown_by_country: Dict[str, Dict] = Field(default_factory=dict)


# ── LATAM GPT Benchmark loaders ──────────────────────────────────────────

def load_trueque_benchmark(sample_size: Optional[int] = None) -> List[dict]:
    """Load Trueque benchmark from HuggingFace.
    
    Trueque: 500+ human-curated Q&A about LATAM culture, history, politics.
    Fields: question, reference_answer, country, topic
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("latam-gpt/Trueque-Benchmark-beta-0.1", split="train")
        data = []
        for i, row in enumerate(ds):
            data.append({
                "question_id": f"trueque_{i}",
                "benchmark": "trueque",
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "country": row.get("country", ""),
                "topic": row.get("topic", ""),
                "difficulty": "mixed",
            })
            if sample_size and len(data) >= sample_size:
                break
        log.info(f"Loaded {len(data)} Trueque questions")
        return data
    except Exception as e:
        log.warning(f"Could not load Trueque from HF: {e}")
        return _load_trueque_fallback(sample_size)


def _load_trueque_fallback(sample_size: Optional[int] = None) -> List[dict]:
    """Fallback: curated subset of Trueque-style questions."""
    questions = [
        {"question_id": "trueque_fb_0", "benchmark": "trueque", "question": "¿Qué es la Pachamama y en qué países andinos se venera?", "reference_answer": "La Pachamama es la diosa de la tierra en las cosmovisiones andinas, venerada principalmente en Bolivia, Perú, Ecuador, norte de Argentina y Chile.", "country": "Bolivia", "topic": "Pueblos-Originarios, Mitología", "difficulty": "FÁCIL"},
        {"question_id": "trueque_fb_1", "benchmark": "trueque", "question": "¿Quién fue José de San Martín y qué países liberó?", "reference_answer": "José de San Martín fue un militar y líder independentista argentino, considerado el Padre de la Patria, quien organizó el Ejército de los Andes y jugó un papel decisivo en la independencia de Argentina, Chile y Perú.", "country": "Argentina", "topic": "Personaje-Importante, Hecho-Histórico", "difficulty": "FÁCIL"},
        {"question_id": "trueque_fb_2", "benchmark": "trueque", "question": "¿Qué establece la Ley de Matrimonio Igualitario en Argentina?", "reference_answer": "La Ley de Matrimonio Igualitario en Argentina, vigente desde 2010, permite el matrimonio entre personas del mismo sexo con los mismos derechos y efectos legales que el matrimonio entre personas de distinto sexo.", "country": "Argentina", "topic": "Política, Humanidades-y-Ciencias-Sociales", "difficulty": "INTERMEDIA"},
        {"question_id": "trueque_fb_3", "benchmark": "trueque", "question": "¿Cuántas lenguas indígenas reconocidas existen en Bolivia?", "reference_answer": "Bolivia reconoce 36 lenguas indígenas originarias, como el aymara, quechua, guaraní, araona y yamiwana, entre otras, establecidas en su Constitución.", "country": "Bolivia", "topic": "Pueblos-Originarios, Dialectos-y-Lenguas", "difficulty": "INTERMEDIA"},
        {"question_id": "trueque_fb_4", "benchmark": "trueque", "question": "¿Qué es el Salar de Uyuni y dónde se encuentra?", "reference_answer": "El Salar de Uyuni se encuentra en el departamento de Potosí, en el suroeste de Bolivia, en el Altiplano a más de 3.650 metros sobre el nivel del mar.", "country": "Bolivia", "topic": "Lugar-Geografía", "difficulty": "FÁCIL"},
        {"question_id": "trueque_fb_5", "benchmark": "trueque", "question": "¿Qué es la ch'alla en Bolivia y cuándo se realiza?", "reference_answer": "En Bolivia, el ch'alla se realiza principalmente el martes de Carnaval, cuando se ofrecen ofrendas a la Pachamama para pedir buena fortuna, y también los primeros viernes de cada mes.", "country": "Bolivia", "topic": "Pueblos-Originarios, Celebraciones-Festividades", "difficulty": "INTERMEDIA"},
        {"question_id": "trueque_fb_6", "benchmark": "trueque", "question": "¿Qué significa la expresión 'che' en Argentina?", "reference_answer": "En Argentina, 'che' es una interjección coloquial usada para llamar la atención de alguien, similar a 'oye' o 'hey', y también para expresar sorpresa o énfasis en el habla informal.", "country": "Argentina", "topic": "Dialectos-y-Lenguas", "difficulty": "FÁCIL"},
        {"question_id": "trueque_fb_7", "benchmark": "trueque", "question": "¿Qué es el dulce de leche en Argentina?", "reference_answer": "El dulce de leche es una preparación a base de leche y azúcar, cocida hasta obtener una textura cremosa y un sabor dulce y tostado, ampliamente usado en postres como alfajores, tortas y facturas.", "country": "Argentina", "topic": "Comida-y-Gastronomía", "difficulty": "FÁCIL"},
        {"question_id": "trueque_fb_8", "benchmark": "trueque", "question": "¿Qué es la diablada y de dónde es originaria?", "reference_answer": "La diablada es originaria de Bolivia, específicamente del departamento de Oruro, donde se desarrolló como una expresión cultural mestiza que fusiona tradiciones andinas y cristianas.", "country": "Bolivia", "topic": "Celebraciones-Festividades, Pueblos-Originarios", "difficulty": "INTERMEDIA"},
        {"question_id": "trueque_fb_9", "benchmark": "trueque", "question": "¿Qué es el tango y dónde se originó?", "reference_answer": "El tango es un género musical y danza que se originó a fines del siglo XIX en la región del Río de la Plata, principalmente en Buenos Aires y Montevideo, como resultado de la mezcla de influencias africanas, europeas y criollas.", "country": "Argentina", "topic": "Artes, Hecho-Histórico", "difficulty": "FÁCIL"},
    ]
    if sample_size:
        questions = questions[:sample_size]
    return questions


def load_choclo_benchmark(sample_size: Optional[int] = None, country_filter: Optional[str] = None) -> List[dict]:
    """Load CHOCLO benchmark from HuggingFace.
    
    CHOCLO: 105k Q&A across 7 categories (dish, tradition, public_figure,
    geography, flora, fauna, object) and 18 LATAM countries.
    Three difficulty levels: FÁCIL, INTERMEDIA, DIFÍCIL.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("latam-gpt/CHOCLO", data_files="BenchmarkCHOCLO.csv", split="train")
        data = []
        for i, row in enumerate(ds):
            country = row.get("Country", "")
            if country_filter and country.lower() != country_filter.lower():
                continue
            data.append({
                "question_id": f"choclo_{i}",
                "benchmark": "choclo",
                "question": row["Question"],
                "reference_answer": row["Answer"],
                "country": country,
                "category": row.get("Category", ""),
                "difficulty": row.get("Difficulty", "FÁCIL"),
                "entity": row.get("Entity", ""),
            })
            if sample_size and len(data) >= sample_size:
                break
        log.info(f"Loaded {len(data)} CHOCLO questions")
        return data
    except Exception as e:
        log.warning(f"Could not load CHOCLO from HF: {e}")
        return _load_choclo_fallback(sample_size, country_filter)


def _load_choclo_fallback(sample_size: Optional[int] = None, country_filter: Optional[str] = None) -> List[dict]:
    """Fallback: curated CHOCLO-style questions focused on Colombia."""
    questions = [
        {"question_id": "choclo_fb_0", "benchmark": "choclo", "question": "¿Qué es la arepa y en qué países se consume?", "reference_answer": "La arepa es un pan plano de maíz, alimento básico en Colombia y Venezuela, con variantes en cada región.", "country": "Colombia", "category": "dish", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_1", "benchmark": "choclo", "question": "¿Qué es el Carnaval de Barranquilla?", "reference_answer": "El Carnaval de Barranquilla es una festividad cultural colombiana declarada Patrimonio Oral e Inmaterial de la Humanidad por la UNESCO en 2003.", "country": "Colombia", "category": "tradition", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_2", "benchmark": "choclo", "question": "¿Quién es Gabriel García Márquez?", "reference_answer": "Gabriel García Márquez fue un escritor y periodista colombiano, Premio Nobel de Literatura en 1982, autor de Cien años de soledad.", "country": "Colombia", "category": "public_figure", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_3", "benchmark": "choclo", "question": "¿Cuál es la capital del departamento de Antioquia?", "reference_answer": "Medellín es la capital del departamento de Antioquia, Colombia.", "country": "Colombia", "category": "geography", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_4", "benchmark": "choclo", "question": "¿Qué es la frailejón y dónde crece?", "reference_answer": "El frailejón es una planta del género Espeletia, endémica de los páramos de Colombia, Venezuela y Ecuador, fundamental para la regulación hídrica.", "country": "Colombia", "category": "flora", "difficulty": "INTERMEDIA"},
        {"question_id": "choclo_fb_5", "benchmark": "choclo", "question": "¿Qué es el cóndor de los Andes y cuál es su significado en Colombia?", "reference_answer": "El cóndor de los Andes es el ave nacional de Colombia, símbolo de libertad y grandeza, presente en el escudo nacional.", "country": "Colombia", "category": "fauna", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_6", "benchmark": "choclo", "question": "¿Qué es la Mochila Wayuu?", "reference_answer": "La Mochila Wayuu es un bolso artesanal tejido por el pueblo Wayuu en La Guajira, Colombia, con diseños que representan su cosmovisión.", "country": "Colombia", "category": "object", "difficulty": "INTERMEDIA"},
        {"question_id": "choclo_fb_7", "benchmark": "choclo", "question": "¿Qué es el bandeja paisa?", "reference_answer": "El bandeja paisa es un plato típico de la región antioqueña de Colombia, que incluye fríjoles, arroz, chicharrón, carne molida, huevo, plátano maduro, aguacate y arepa.", "country": "Colombia", "category": "dish", "difficulty": "FÁCIL"},
        {"question_id": "choclo_fb_8", "benchmark": "choclo", "question": "¿Qué es el río Magdalena y cuál es su importancia para Colombia?", "reference_answer": "El río Magdalena es el río más importante de Colombia, recorre el país de sur a norte y ha sido fundamental para el transporte, la economía y la historia del país.", "country": "Colombia", "category": "geography", "difficulty": "INTERMEDIA"},
        {"question_id": "choclo_fb_9", "benchmark": "choclo", "question": "¿Qué es la Cumbia colombiana?", "reference_answer": "La Cumbia es un género musical y danza originario de la costa Caribe colombiana, con influencias indígenas, africanas y europeas, considerado uno de los ritmos más representativos de Colombia.", "country": "Colombia", "category": "tradition", "difficulty": "INTERMEDIA"},
    ]
    if country_filter:
        questions = [q for q in questions if q.get("country", "").lower() == country_filter.lower()]
    if sample_size:
        questions = questions[:sample_size]
    return questions


# ── Benchmark runner ─────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    benchmark: str = "both"           # "trueque", "choclo", or "both"
    model_keys: List[str] = field(default_factory=lambda: ["qwen_or", "mistral_large", "gpt4o", "gemini"])
    sample_size: Optional[int] = 10   # None = full benchmark
    country_filter: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500
    delay_between: float = 0.5


async def run_benchmark(config: BenchmarkConfig) -> List[BenchmarkRunResult]:
    """Run LATAM GPT benchmarks against selected models."""
    from tracker.llm_clients import build_selected_clients

    clients = build_selected_clients(config.model_keys)
    if not clients:
        log.error("No clients available for benchmark")
        return []

    # Load questions
    questions = []
    if config.benchmark in ("trueque", "both"):
        questions.extend(load_trueque_benchmark(config.sample_size))
    if config.benchmark in ("choclo", "both"):
        questions.extend(load_choclo_benchmark(config.sample_size, config.country_filter))

    if not questions:
        log.error("No benchmark questions loaded")
        return []

    log.info(f"Running benchmark: {len(questions)} questions × {len(clients)} models")

    all_results = []

    for model_key, client in clients.items():
        log.info(f"Benchmarking {model_key} ({client.model_id})...")
        question_results = []
        correct_count = 0
        total_score = 0.0
        total_latency = 0.0

        for q in questions:
            try:
                t0 = time.monotonic()
                resp = await client.query(q["question"], config.temperature, config.max_tokens)
                latency = (time.monotonic() - t0) * 1000

                if resp.error:
                    qr = BenchmarkQuestionResult(
                        question_id=q["question_id"],
                        benchmark=q["benchmark"],
                        question=q["question"],
                        reference_answer=q["reference_answer"],
                        model_response="",
                        model_key=model_key,
                        model_provider=client.provider,
                        is_correct=False,
                        score=0.0,
                        latency_ms=latency,
                        error=resp.error,
                        metadata={"country": q.get("country", ""), "category": q.get("category", ""), "difficulty": q.get("difficulty", "")},
                    )
                else:
                    score = _score_response(resp.raw_text, q["reference_answer"])
                    is_correct = score >= 0.7
                    if is_correct:
                        correct_count += 1
                    total_score += score
                    total_latency += latency

                    qr = BenchmarkQuestionResult(
                        question_id=q["question_id"],
                        benchmark=q["benchmark"],
                        question=q["question"],
                        reference_answer=q["reference_answer"],
                        model_response=resp.raw_text,
                        model_key=model_key,
                        model_provider=client.provider,
                        is_correct=is_correct,
                        score=score,
                        latency_ms=latency,
                        metadata={"country": q.get("country", ""), "category": q.get("category", ""), "difficulty": q.get("difficulty", "")},
                    )

                question_results.append(qr)
                time.sleep(config.delay_between)

            except Exception as e:
                log.warning(f"Error on {q['question_id']} for {model_key}: {e}")

        n = len(question_results)
        run_result = BenchmarkRunResult(
            benchmark=config.benchmark,
            model_key=model_key,
            model_provider=client.provider,
            timestamp=datetime.now().astimezone().isoformat(),
            total_questions=n,
            correct=correct_count,
            accuracy=correct_count / n if n > 0 else 0.0,
            avg_score=total_score / n if n > 0 else 0.0,
            avg_latency_ms=total_latency / n if n > 0 else 0.0,
            results=question_results,
        )

        # Compute breakdowns
        run_result.breakdown_by_difficulty = _compute_breakdown(question_results, "difficulty")
        run_result.breakdown_by_category = _compute_breakdown(question_results, "category")
        run_result.breakdown_by_country = _compute_breakdown(question_results, "country")

        all_results.append(run_result)
        log.info(f"  {model_key}: accuracy={run_result.accuracy:.2%}, avg_score={run_result.avg_score:.3f}")

    return all_results


def _score_response(response: str, reference: str) -> float:
    """Score a model response against reference answer.
    
    Uses a hybrid approach: lexical overlap + semantic containment.
    For production, this should use embeddings or LLM-as-judge.
    """
    if not response or not reference:
        return 0.0

    resp_lower = response.lower().strip()
    ref_lower = reference.lower().strip()

    # Exact match
    if resp_lower == ref_lower:
        return 1.0

    # Reference contained in response
    if ref_lower in resp_lower:
        return 0.9

    # Key term overlap
    ref_words = set(ref_lower.split())
    resp_words = set(resp_lower.split())
    if not ref_words:
        return 0.0

    overlap = len(ref_words & resp_words)
    lexical_score = overlap / len(ref_words)

    # Boost if response contains most key concepts
    key_phrases = [p for p in ref_lower.split(",") if len(p.strip()) > 5]
    phrase_matches = sum(1 for p in key_phrases if p.strip() in resp_lower)
    phrase_score = phrase_matches / len(key_phrases) if key_phrases else 0.0

    return round(0.6 * lexical_score + 0.4 * phrase_score, 3)


def _compute_breakdown(results: List[BenchmarkQuestionResult], field_name: str) -> Dict:
    """Compute accuracy/score breakdown by a metadata field."""
    groups: Dict[str, List[BenchmarkQuestionResult]] = {}
    for r in results:
        val = r.metadata.get(field_name, "unknown") or "unknown"
        groups.setdefault(val, []).append(r)

    breakdown = {}
    for val, group in groups.items():
        n = len(group)
        correct = sum(1 for g in group if g.is_correct)
        avg_score = sum(g.score for g in group) / n if n > 0 else 0.0
        breakdown[val] = {
            "total": n,
            "correct": correct,
            "accuracy": round(correct / n, 3) if n > 0 else 0.0,
            "avg_score": round(avg_score, 3),
        }
    return breakdown


def save_benchmark_results(results: List[BenchmarkRunResult], output_dir: Path) -> Path:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"benchmark_{ts}.json"

    data = []
    for r in results:
        data.append({
            "benchmark": r.benchmark,
            "model_key": r.model_key,
            "model_provider": r.model_provider,
            "timestamp": r.timestamp,
            "total_questions": r.total_questions,
            "correct": r.correct,
            "accuracy": r.accuracy,
            "avg_score": r.avg_score,
            "avg_latency_ms": r.avg_latency_ms,
            "breakdown_by_difficulty": r.breakdown_by_difficulty,
            "breakdown_by_category": r.breakdown_by_category,
            "breakdown_by_country": r.breakdown_by_country,
            "results": [qr.model_dump() for qr in r.results],
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"Benchmark results saved to {out_file}")
    return out_file
