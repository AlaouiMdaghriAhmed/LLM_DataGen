"""Configuration and deterministic randomness (requirement R7: reproducibility).

A whole dataset is a pure function of :class:`SimConfig` — org size, horizon,
prevalence, noise, and seed. Change the seed and nothing else, and you get an
independent draw from the *same distribution*; change the size/prevalence and you
get a comparable dataset at a new operating point. Everything downstream draws from
named sub-streams of a single seeded generator so that, e.g., adding a channel does
not perturb the random decisions of unrelated channels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

try:  # PyYAML is optional at import time
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


@dataclass
class SimConfig:
    """Top-level dataset specification."""

    # --- identity / reproducibility ---
    seed: int = 1234
    name: str = "synthitd-sample"

    # --- organisation ---
    domain: str = "tech"             # one of org.DOMAINS
    n_employees: int = 200
    # fraction of the workforce that ever becomes a malicious insider over the
    # horizon. 0.02 ~ CERT r5.2 density; 0.001 ~ r6.2 sparse realism.
    insider_prevalence: float = 0.02
    # fraction of employees that are benign-but-anomalous hard negatives (R4).
    anomalous_benign_rate: float = 0.08

    # --- time ---
    start_date: str = "2024-01-01"   # ISO date; sim runs forward from here
    horizon_days: int = 120
    workday_start_hour: float = 8.0
    workday_end_hour: float = 18.0
    lunch_hours: tuple[float, float] = (12.0, 13.0)

    # --- behaviour / noise ---
    # multiplier on baseline per-user daily event volume
    activity_scale: float = 1.0
    # probability a given scheduled baseline action is dropped (behavioural noise)
    behavior_dropout: float = 0.05

    # --- threat model ---
    # cap on simultaneously-active insider episodes (None = unbounded)
    max_active_episodes: int | None = None
    # allowed pathway families (subset of threats.PLAYBOOKS keys); empty = all
    enabled_playbooks: list[str] = field(default_factory=list)

    # --- observability (requirement R8) ---
    # per-channel coverage probability (1.0 = perfect sensor). Missing keys = 1.0
    channel_coverage: dict[str, float] = field(default_factory=dict)
    # global extra random drop applied on top of coverage
    sensor_dropout: float = 0.0

    # --- rendering (requirement R2 / R7) ---
    renderer: str = "template"       # "template" (free) or "anthropic"
    render_fraction: float = 1.0     # fraction of renderable events to render
    model: str = "claude-opus-4-8"
    render_max_events: int | None = None

    def with_overrides(self, **kw: Any) -> "SimConfig":
        d = asdict(self)
        d.update(kw)
        # tuples survive asdict as lists; coerce back where needed
        if isinstance(d.get("lunch_hours"), list):
            d["lunch_hours"] = tuple(d["lunch_hours"])
        return SimConfig(**d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lunch_hours"] = list(self.lunch_hours)
        return d


def load_config(path_or_dict: str | dict[str, Any]) -> SimConfig:
    """Load a :class:`SimConfig` from a YAML/JSON file path or a dict."""
    if isinstance(path_or_dict, dict):
        data = dict(path_or_dict)
    else:
        with open(path_or_dict, "r", encoding="utf-8") as fh:
            text = fh.read()
        if yaml is not None:
            data = yaml.safe_load(text)
        else:  # pragma: no cover - fallback if PyYAML missing
            import json

            data = json.loads(text)
    if "lunch_hours" in data and isinstance(data["lunch_hours"], list):
        data["lunch_hours"] = tuple(data["lunch_hours"])
    known = set(SimConfig().to_dict().keys())
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return SimConfig(**data)


class RNG:
    """Deterministic named-substream RNG.

    ``rng.stream("behavior", employee_id)`` returns a NumPy ``Generator`` whose seed
    is a stable hash of the base seed and the name components. Independent concerns
    draw from independent streams, so the dataset stays stable under unrelated code
    changes.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._cache: dict[tuple[Any, ...], np.random.Generator] = {}

    def stream(self, *name: Any) -> np.random.Generator:
        key = (self.seed, *name)
        gen = self._cache.get(key)
        if gen is None:
            digest = hashlib.blake2b(
                "::".join(str(x) for x in key).encode("utf-8"), digest_size=8
            ).digest()
            sub_seed = int.from_bytes(digest, "big")
            gen = np.random.default_rng(sub_seed)
            self._cache[key] = gen
        return gen
