"""synthitd — a reproducible generator and benchmark for insider-threat detection.

This is a *defensive / academic* research tool. It models workplace behaviour at
the level of detection-relevant observables and MITRE ATT&CK technique *labels* in
order to produce labelled telemetry for training and evaluating insider-threat
detection (ITD) systems. It does not contain operational attack instructions.

See ``docs/LITERATURE_REVIEW.md`` for the gap analysis that motivates the design
and ``docs/DESIGN.md`` for the architecture.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .events import Event, Label, Channel, EventBus  # noqa: E402,F401
from .config import SimConfig, load_config  # noqa: E402,F401

__all__ = [
    "Event",
    "Label",
    "Channel",
    "EventBus",
    "SimConfig",
    "load_config",
    "__version__",
]
