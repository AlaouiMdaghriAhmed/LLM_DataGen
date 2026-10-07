"""Critical-Pathway latent risk model (requirement R1 — the core contribution).

Instead of *injecting* a scripted attack into a chosen user (as CERT, Chimera, and
OrgForge-IT all do), we **grow** intent from a per-day latent risk trajectory driven
by the Critical Pathway to Insider Risk (Shaw & Sellers): a function of the person's
static *predisposition*, their accumulated *stressors*, and a feedback term for
*concerning behaviour*. For the insider population, a hazard model decides *whether*
and *when* a pathway activates; the pathway is never guaranteed to fire, and a
predisposed, stressed person who never activates stays a (hard) negative.

The latent ``risk`` trajectory is the ground truth for early-warning / risk-ranking
tasks, and it is what decides the graded event labels:

* ``risk`` below the concern threshold on a normal day  -> ``benign``
* an active benign trigger drives the anomaly             -> ``anomalous_benign``
* insider, pathway armed, before the act window           -> ``precursor``
* insider, inside the act window                          -> ``malicious``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RNG, SimConfig
from .lifecycle import LifeHistory
from .org import Organization, Employee


CONCERN_THRESHOLD = 0.55   # risk above this = observable "concerning behaviour"
ACTIVATION_FLOOR = 0.52    # hazard only accrues above this
PRECURSOR_LEAD_DAYS = 21   # max precursor window length before the act


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RiskTrajectory:
    """Per-employee latent state over the horizon."""

    emp_id: str
    population: str
    risk: np.ndarray                   # shape (H,), values 0..1
    onset_day: int | None = None       # day the pathway activates (insiders only)
    pathway: str = ""                  # chosen playbook family (set later)
    precursor_start: int | None = None # first day of the precursor window
    episode_end: int | None = None     # last active day of the episode

    def state_on(self, day: int) -> str:
        """Coarse latent state used by the labeller."""
        if self.onset_day is not None and self.onset_day <= day <= (self.episode_end or day):
            return "act"
        if (
            self.precursor_start is not None
            and self.precursor_start <= day < (self.onset_day or 10**9)
        ):
            return "precursor"
        return "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "emp_id": self.emp_id,
            "population": self.population,
            "onset_day": self.onset_day,
            "pathway": self.pathway,
            "precursor_start": self.precursor_start,
            "episode_end": self.episode_end,
            "risk_mean": float(self.risk.mean()),
            "risk_max": float(self.risk.max()),
        }


def build_trajectories(
    cfg: SimConfig,
    rng: RNG,
    org: Organization,
    histories: dict[str, LifeHistory],
) -> dict[str, RiskTrajectory]:
    H = cfg.horizon_days
    trajectories: dict[str, RiskTrajectory] = {}
    for emp in org.employees:
        g = rng.stream("psych", emp.emp_id)
        hist = histories[emp.emp_id]
        risk = np.zeros(H, dtype=float)

        # Latent "intent pressure" for the insider population: a grievance that
        # consolidates over time. It is *latent* — it is never emitted as an
        # observable feature, only reflected in the risk target and in whether/when
        # a pathway activates — so it creates no shortcut in the observed telemetry.
        # Normal and anomalous-benign employees have zero intent pressure but can
        # still reach high risk through stressors, so high risk != insider.
        intent = _intent_pressure(g, emp, H)

        # slow idiosyncratic drift (Ornstein-Uhlenbeck-ish) so trajectories are
        # autocorrelated rather than i.i.d. noise.
        drift = 0.0
        concern_accum = 0.0
        for d in range(H):
            stress = hist.stress_level(d)
            drift = 0.92 * drift + 0.08 * (g.random() - 0.5)
            # logit-linear combination of the CPIR factors
            z = (
                -1.9
                + 2.4 * emp.predisposition
                + 1.3 * min(stress, 1.5)
                + 1.1 * concern_accum
                + 1.6 * intent[d]
                + 0.6 * drift
            )
            r = _sigmoid(z)
            risk[d] = r
            # concerning-behaviour feedback: high risk begets rumination/grievance,
            # a modest self-reinforcing term (bounded).
            if r > CONCERN_THRESHOLD:
                concern_accum = min(0.8, concern_accum + 0.03)
            else:
                concern_accum = max(0.0, concern_accum - 0.02)

        traj = RiskTrajectory(emp_id=emp.emp_id, population=emp.population, risk=risk)

        if emp.population == "insider":
            _resolve_activation(cfg, g, emp, traj)

        trajectories[emp.emp_id] = traj
    return trajectories


def _intent_pressure(g: np.random.Generator, emp: Employee, H: int) -> np.ndarray:
    """Latent grievance-consolidation ramp for the insider population (0..~1)."""
    out = np.zeros(H, dtype=float)
    if emp.population != "insider":
        return out
    # grievance begins somewhere in the first ~60% of the horizon and ramps with a
    # random slope to a random ceiling; some insiders ramp slowly (never activate).
    begin = int(g.integers(5, max(6, int(0.6 * H))))
    slope = 0.02 + 0.06 * g.random()
    ceiling = float(np.clip(0.4 + 0.6 * g.random(), 0.3, 1.0))
    for d in range(begin, H):
        out[d] = min(ceiling, slope * (d - begin))
    return out


def _resolve_activation(
    cfg: SimConfig, g: np.random.Generator, emp: Employee, traj: RiskTrajectory
) -> None:
    """Hazard model for pathway activation.

    Each day above the activation floor contributes a hazard proportional to how far
    risk exceeds the floor. The first "success" is the onset day. Many insiders will
    *not* activate within the horizon — those remain latent risks (and negatives),
    which is exactly the realism CERT lacks.
    """
    H = len(traj.risk)
    onset = None
    for d in range(10, H - 2):  # need a little runway and a little episode room
        excess = traj.risk[d] - ACTIVATION_FLOOR
        if excess <= 0:
            continue
        hazard = 0.08 * excess + 0.02  # per-day probability once armed
        if g.random() < hazard:
            onset = d
            break
    if onset is None:
        return

    traj.onset_day = onset
    # precursor window: walk back from onset while risk stays notably elevated
    start = onset
    while start > 0 and start > onset - PRECURSOR_LEAD_DAYS and traj.risk[start - 1] > 0.45:
        start -= 1
    traj.precursor_start = start

    # episode length depends on pathway tempo (set later); default a short burst
    dur = int(np.clip(g.geometric(0.35), 1, H - onset - 1))
    traj.episode_end = min(H - 1, onset + dur)
