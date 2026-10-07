"""HR / life-event model: the *stressors* and *triggers* of the Critical Pathway.

Shaw & Sellers' Critical Pathway to Insider Risk holds that predispositions become
dangerous when activated by **stressors**. This module generates, per employee, a
timeline of stressors (personal and professional) and the subset that is *observable*
to the organisation (HR records, manager notes). Latent stressors drive the
psychology model (:mod:`synthitd.psych`); observable ones also become HR-channel
events. Benign-anomaly triggers (travel, on-call, promotion, re-org) live here too,
so that the hard-negative population has a *cause* for its anomalous behaviour
(requirement R4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RNG, SimConfig
from .org import Organization, Employee


# stressor_type -> (base magnitude 0..1, observable_by_org)
STRESSOR_TYPES: dict[str, tuple[float, bool]] = {
    "negative_review": (0.45, True),
    "passed_over_promotion": (0.40, True),
    "formal_sanction": (0.60, True),
    "demotion": (0.55, True),
    "layoff_rumor": (0.35, False),
    "resignation_notice": (0.70, True),
    "financial_hardship": (0.50, False),
    "interpersonal_conflict": (0.35, True),
    "personal_crisis": (0.45, False),
    "role_change": (0.20, True),
    "reorg": (0.25, True),
}

# benign-anomaly triggers: (type, observable) — these justify anomalous-but-benign
# behaviour and are emitted as HR context so detectors *could* exonerate.
BENIGN_TRIGGERS: dict[str, bool] = {
    "business_travel": True,
    "oncall_rotation": True,
    "promotion": True,
    "project_crunch": True,
    "approved_bulk_export": True,
    "team_transfer": True,
}


@dataclass
class Stressor:
    day: int
    kind: str
    magnitude: float
    observable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "kind": self.kind,
            "magnitude": round(self.magnitude, 3),
            "observable": self.observable,
        }


@dataclass
class BenignTrigger:
    start_day: int
    end_day: int
    kind: str
    observable: bool

    def active(self, day: int) -> bool:
        return self.start_day <= day <= self.end_day

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_day": self.start_day,
            "end_day": self.end_day,
            "kind": self.kind,
            "observable": self.observable,
        }


@dataclass
class LifeHistory:
    emp_id: str
    stressors: list[Stressor]
    benign_triggers: list[BenignTrigger]

    def stress_level(self, day: int, window: int = 30, decay: float = 0.06) -> float:
        """Exponentially-decaying accumulation of recent stressors (0..~1+)."""
        total = 0.0
        for s in self.stressors:
            if 0 <= day - s.day <= window * 3:
                total += s.magnitude * float(np.exp(-decay * max(0, day - s.day)))
        return total

    def active_benign(self, day: int) -> list[BenignTrigger]:
        return [t for t in self.benign_triggers if t.active(day)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "emp_id": self.emp_id,
            "stressors": [s.to_dict() for s in self.stressors],
            "benign_triggers": [t.to_dict() for t in self.benign_triggers],
        }


def generate_histories(cfg: SimConfig, rng: RNG, org: Organization) -> dict[str, LifeHistory]:
    histories: dict[str, LifeHistory] = {}
    H = cfg.horizon_days
    for emp in org.employees:
        g = rng.stream("life", emp.emp_id)
        stressors = _gen_stressors(g, emp, H)
        triggers = _gen_benign_triggers(g, emp, H)
        histories[emp.emp_id] = LifeHistory(emp.emp_id, stressors, triggers)
    return histories


def _gen_stressors(g: np.random.Generator, emp: Employee, H: int) -> list[Stressor]:
    out: list[Stressor] = []
    # everyone gets a scheduled performance review mid-horizon; outcome depends on
    # conscientiousness + noise. Low-conscientiousness → more likely negative.
    review_day = int(g.integers(max(1, H // 3), max(2, 2 * H // 3)))
    if g.random() < (0.5 - 0.35 * emp.ocean["conscientiousness"]):
        out.append(Stressor(review_day, "negative_review", _mag(g, "negative_review"), True))

    # baseline stressor arrivals scale with predisposition (predisposed people lead
    # more turbulent working lives in this model) but are not deterministic.
    rate = 0.5 + 1.5 * emp.predisposition
    n = int(g.poisson(rate))
    kinds = [k for k in STRESSOR_TYPES if k != "resignation_notice"]
    for _ in range(n):
        day = int(g.integers(0, max(1, H)))
        kind = kinds[g.integers(len(kinds))]
        out.append(Stressor(day, kind, _mag(g, kind), STRESSOR_TYPES[kind][1]))

    # insiders may file resignation late in the horizon (a classic precursor signal)
    if emp.population == "insider" and g.random() < 0.45:
        day = int(g.integers(max(1, 3 * H // 4), max(2, H)))
        out.append(Stressor(day, "resignation_notice", _mag(g, "resignation_notice"), True))
    out.sort(key=lambda s: s.day)
    return out


def _gen_benign_triggers(g: np.random.Generator, emp: Employee, H: int) -> list[BenignTrigger]:
    out: list[BenignTrigger] = []
    # normal employees occasionally travel / go on-call; anomalous-benign population
    # gets a *strong* trigger that will drive genuinely anomalous behaviour.
    base_rate = 0.6 if emp.population != "anomalous_benign" else 1.6
    n = int(g.poisson(base_rate))
    kinds = list(BENIGN_TRIGGERS)
    for _ in range(n):
        kind = kinds[g.integers(len(kinds))]
        start = int(g.integers(0, max(1, H)))
        dur = int(g.integers(2, 12))
        out.append(BenignTrigger(start, min(H, start + dur), kind, BENIGN_TRIGGERS[kind]))

    if emp.population == "anomalous_benign" and not out:
        kind = kinds[g.integers(len(kinds))]
        start = int(g.integers(0, max(1, H - 6)))
        out.append(BenignTrigger(start, min(H, start + 6), kind, BENIGN_TRIGGERS[kind]))
    out.sort(key=lambda t: t.start_day)
    return out


def _mag(g: np.random.Generator, kind: str) -> float:
    base = STRESSOR_TYPES[kind][0]
    return float(np.clip(base * (0.7 + 0.6 * g.random()), 0, 1))
