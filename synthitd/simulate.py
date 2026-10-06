"""Top-level orchestrator: turn a :class:`SimConfig` into a labelled dataset.

Pipeline (all deterministic given the seed):

1. build the organisation (:mod:`synthitd.org`)
2. generate life histories / stressors (:mod:`synthitd.lifecycle`)
3. grow latent critical-pathway risk trajectories (:mod:`synthitd.psych`)
4. assign pathway families to activated insiders (:mod:`synthitd.threats`)
5. emit observable HR events + per-day multi-surface behaviour + on-pathway events
6. finalize the deterministic event bus (full ground truth)
7. apply the observability filter to get the observed stream

The LLM renderer does **not** run here — it is a separate, label-blind pass over the
finished stream (:mod:`synthitd.render`), preserving the guarantee that prose can
never leak the label.
"""

from __future__ import annotations

from dataclasses import dataclass

from .behavior import BehaviorModel
from .config import RNG, SimConfig
from .events import Channel, Event, EventBus, Label
from .lifecycle import generate_histories, LifeHistory
from .observability import apply_observability, ObservabilityReport
from .org import build_org, Organization
from .psych import build_trajectories, RiskTrajectory
from .simclock import WorkCalendar
from .threats import assign_pathways, ThreatEngine


@dataclass
class SimResult:
    config: SimConfig
    org: Organization
    histories: dict[str, LifeHistory]
    trajectories: dict[str, RiskTrajectory]
    events: list[Event]            # full ground-truth stream (all labels)
    observed: list[Event]          # after observability filter
    report: ObservabilityReport

    # -- convenience summaries -------------------------------------------
    def label_counts(self, observed: bool = False) -> dict[str, int]:
        stream = self.observed if observed else self.events
        out: dict[str, int] = {}
        for ev in stream:
            out[ev.label.value] = out.get(ev.label.value, 0) + 1
        return out

    def insiders(self) -> list[str]:
        return [e for e, t in self.trajectories.items() if t.onset_day is not None]

    def summary(self) -> dict:
        return {
            "name": self.config.name,
            "domain": self.config.domain,
            "n_employees": len(self.org.employees),
            "horizon_days": self.config.horizon_days,
            "n_events_full": len(self.events),
            "n_events_observed": len(self.observed),
            "activated_insiders": len(self.insiders()),
            "insider_population": sum(
                1 for e in self.org.employees if e.population == "insider"
            ),
            "anomalous_benign_population": sum(
                1 for e in self.org.employees if e.population == "anomalous_benign"
            ),
            "label_counts_full": self.label_counts(False),
            "label_counts_observed": self.label_counts(True),
            "observability": self.report.to_dict(),
        }


def simulate(cfg: SimConfig) -> SimResult:
    rng = RNG(cfg.seed)
    cal = WorkCalendar(
        cfg.start_date, cfg.workday_start_hour, cfg.workday_end_hour, cfg.lunch_hours
    )

    org = build_org(cfg, rng)
    histories = generate_histories(cfg, rng, org)
    trajectories = build_trajectories(cfg, rng, org, histories)
    assign_pathways(cfg, rng, org, trajectories)

    bus = EventBus()
    behavior = BehaviorModel(cfg, rng, cal)
    threats = ThreatEngine(cfg, rng, cal, org)

    # observable HR / lifecycle events (justify anomalies; mark precursors)
    for emp in org.employees:
        for s in histories[emp.emp_id].stressors:
            if not s.observable or not (0 <= s.day < cfg.horizon_days):
                continue
            label = Label.BENIGN
            # a resignation during an armed precursor window is itself a precursor
            traj = trajectories[emp.emp_id]
            if s.kind == "resignation_notice" and traj.state_on(s.day) in ("precursor", "act"):
                label = Label.PRECURSOR
            bus.emit(
                cal.ts(s.day, 10.0), Channel.HR, emp.emp_id, f"hr.{s.kind}",
                attrs={"magnitude": round(s.magnitude, 2)},
                label=label,
                risk_state=float(traj.risk[s.day]),
            )

    # per-day activity
    for day in range(cfg.horizon_days):
        for emp in org.employees:
            traj = trajectories[emp.emp_id]
            behavior.emit_day(bus, emp, day, histories[emp.emp_id], traj)
            threats.emit_day(bus, emp, day, traj)

    events = bus.finalize()
    observed, report = apply_observability(cfg, rng, events)

    return SimResult(
        config=cfg,
        org=org,
        histories=histories,
        trajectories=trajectories,
        events=events,
        observed=observed,
        report=report,
    )
