"""Sensor-degradation / partial-visibility model (requirement R8).

Real SOCs never see everything: some channels are poorly instrumented, some events
are dropped, some arrive late. CERT and ChimeraLog assume a perfect sensor. Here we
apply an explicit observability filter to the *ground-truth* stream to produce the
*observed* stream a detector actually gets. The full labelled stream is always kept
separately, so evaluation can measure how much malicious signal was rendered
invisible by instrumentation gaps — a quantity no prior dataset exposes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RNG, SimConfig
from .events import Channel, Event


@dataclass
class ObservabilityReport:
    total: int
    observed: int
    dropped: int
    dropped_malicious: int
    per_channel_coverage: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "total_events": self.total,
            "observed_events": self.observed,
            "dropped_events": self.dropped,
            "dropped_malicious_events": self.dropped_malicious,
            "effective_recall_ceiling": round(
                1 - (self.dropped_malicious / max(1, self._total_malicious)), 4
            ),
            "per_channel_coverage": self.per_channel_coverage,
        }

    _total_malicious: int = 0


def apply_observability(
    cfg: SimConfig, rng: RNG, events: list[Event]
) -> tuple[list[Event], ObservabilityReport]:
    g = rng.stream("observability")
    coverage = {c.value: float(cfg.channel_coverage.get(c.value, 1.0)) for c in Channel}
    observed: list[Event] = []
    dropped = 0
    dropped_mal = 0
    total_mal = 0
    for ev in events:
        if ev.label.is_positive:
            total_mal += 1
        cov = coverage.get(ev.channel.value, 1.0) * (1.0 - cfg.sensor_dropout)
        if g.random() <= cov:
            observed.append(ev)
        else:
            dropped += 1
            if ev.label.is_positive:
                dropped_mal += 1
    report = ObservabilityReport(
        total=len(events),
        observed=len(observed),
        dropped=dropped,
        dropped_malicious=dropped_mal,
        per_channel_coverage=coverage,
    )
    report._total_malicious = total_mal
    return observed, report
