from dataclasses import dataclass, field
from typing import List


@dataclass
class Candidate:
    key: str
    full_name: str
    display_name: str
    coalition: str
    aliases: List[str]
    political_leaning: str   # left, center-left, center, center-right, right
    weird_alignment: str     # low, medium, high — per Atari et al. (2023) framework


CANDIDATES: List[Candidate] = [
    Candidate(
        key="cepeda",
        full_name="Iván Cepeda Castro",
        display_name="Iván Cepeda",
        coalition="Pacto Histórico",
        aliases=["Cepeda", "Iván Cepeda", "Ivan Cepeda", "Cepeda Castro"],
        political_leaning="left",
        weird_alignment="low",  # state-led services, indigenous rights, non-WEIRD priors
    ),
    Candidate(
        key="de_la_espriella",
        full_name="Abelardo de la Espriella",
        display_name="Abelardo de la Espriella",
        coalition="Defensores de la Patria",
        aliases=["De la Espriella", "Abelardo", "de la Espriella", "Espriella", "Abelardo de la Espriella"],
        political_leaning="right",
        weird_alignment="medium",  # hard-right but market-friendly mixed signal
    ),
    Candidate(
        key="valencia",
        full_name="Paloma Valencia Laserna",
        display_name="Paloma Valencia",
        coalition="Centro Democrático",
        aliases=["Valencia", "Paloma Valencia", "Valencia Laserna", "Paloma"],
        political_leaning="center-right",
        weird_alignment="high",  # market-friendly, US-aligned, conservative-WEIRD
    ),
    Candidate(
        key="fajardo",
        full_name="Sergio Fajardo Valderrama",
        display_name="Sergio Fajardo",
        coalition="Compromiso Ciudadano",
        aliases=["Fajardo", "Sergio Fajardo", "Fajardo Valderrama"],
        political_leaning="center",
        weird_alignment="high",  # technocratic-centrist, high-WEIRD signal
    ),
    Candidate(
        key="lopez",
        full_name="Claudia López Hernández",
        display_name="Claudia López",
        coalition="Independiente",
        aliases=["López", "Claudia López", "Lopez", "Claudia Lopez", "López Hernández"],
        political_leaning="center-left",
        weird_alignment="medium",
    ),
]

CANDIDATE_BY_KEY: dict[str, Candidate] = {c.key: c for c in CANDIDATES}
