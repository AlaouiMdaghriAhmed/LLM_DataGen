"""Run the full benchmark: 3 splits x N baselines + audits, into one report dict."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..simulate import SimResult
from ..threats import PLAYBOOKS
from .audits import shortcut_audit, stylized_facts
from .baselines import run_baselines
from .features import build_userday_features, FeatureMatrix
from .metrics import evaluate
from .splits import scenario_holdout_split, temporal_split, user_holdout_split


def _eval_split(fm: FeatureMatrix, tr, te, budget: float) -> dict[str, Any]:
    if len(te) == 0 or fm.y[te].sum() == 0 or len(tr) == 0:
        return {"skipped": "no positives in test or empty split",
                "n_test": int(len(te)), "n_positive": int(fm.y[te].sum()) if len(te) else 0}
    scores = run_baselines(fm, tr, te)
    return {name: evaluate(fm, te, s, budget=budget).to_dict() for name, s in scores.items()}


def run_benchmark(
    result: SimResult, budget: float = 0.01, observed: bool = True
) -> dict[str, Any]:
    fm = build_userday_features(result, observed=observed)

    report: dict[str, Any] = {
        "dataset": result.config.name,
        "n_userdays": int(len(fm)),
        "n_malicious_userdays": int(fm.y.sum()),
        "userday_prevalence": round(float(fm.y.mean()), 6),
        "review_budget": budget,
        "splits": {},
        "audits": {
            "shortcut": shortcut_audit(fm),
            "stylized_facts": stylized_facts(result),
        },
    }

    tr, te = temporal_split(fm, 0.7)
    report["splits"]["temporal"] = _eval_split(fm, tr, te, budget)

    tr, te = user_holdout_split(fm, 0.3, seed=result.config.seed)
    report["splits"]["user_holdout"] = _eval_split(fm, tr, te, budget)

    # scenario-held-out: pick the most-common present pathway to hold out
    present = [p for p in fm.pathway.tolist() if p]
    if present:
        vals, cnts = np.unique(np.array(present), return_counts=True)
        holdout = [str(vals[int(np.argmax(cnts))])]
        tr, te = scenario_holdout_split(fm, holdout)
        res = _eval_split(fm, tr, te, budget)
        res["_held_out_pathways"] = holdout
        report["splits"]["scenario_holdout"] = res
    else:
        report["splits"]["scenario_holdout"] = {"skipped": "no activated pathways"}

    return report
