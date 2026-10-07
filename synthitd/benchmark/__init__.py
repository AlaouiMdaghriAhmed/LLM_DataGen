"""Benchmark suite: honest splits, leakage-safe features, baselines, metrics, audits.

See ``docs/BENCHMARK.md``. The suite is deliberately adversarial toward its own
dataset: it ships a *shortcut audit* that quantifies how much of the label can be
recovered from a single obvious rule, and a *stylized-facts* check that the benign
background is distributionally plausible.
"""

from __future__ import annotations

from .features import build_userday_features, FeatureMatrix, FEATURE_NAMES  # noqa: F401
from .splits import temporal_split, user_holdout_split, scenario_holdout_split  # noqa: F401
from .baselines import run_baselines, BASELINES  # noqa: F401
from .sequence import sequence_logreg, ewma_selfbaseline, augment_sequence  # noqa: F401
from .metrics import evaluate, Metrics  # noqa: F401
from .audits import shortcut_audit, stylized_facts  # noqa: F401

__all__ = [
    "build_userday_features", "FeatureMatrix", "FEATURE_NAMES",
    "temporal_split", "user_holdout_split", "scenario_holdout_split",
    "run_baselines", "BASELINES", "evaluate", "Metrics",
    "shortcut_audit", "stylized_facts",
]
