"""Head-to-head: SOTA CERT-style ITD detectors on CERT-regime vs realistic data.

Controlled ablation — the two datasets share org, seed, prevalence, horizon, and
everything else; the ONLY difference is ``legacy_cert_mode``, which toggles the
three anti-shortcut mechanisms (benign power-user uploads/removable, benign-anomaly
exfil-looking bursts, insider stealth blending). So any drop in a detector's score
is attributable to those mechanisms, not to a confound.

Run:  python scripts/sota_experiment.py [n_employees] [horizon_days] [n_seeds]
Runs n_seeds independent seeds per regime and reports mean +/- std, so the
comparison is not a single-draw anecdote. Writes reference/sota_comparison.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from synthitd.config import SimConfig
from synthitd.simulate import simulate
from synthitd.benchmark.features import build_userday_features
from synthitd.benchmark.splits import temporal_split, user_holdout_split
from synthitd.benchmark.sota import SOTA_MODELS
from synthitd.benchmark.metrics import evaluate


def _run(cfg, budget=0.01):
    res = simulate(cfg)
    fm = build_userday_features(res, observed=True)
    out = {"activated_insiders": len(res.insiders()),
           "n_userdays": int(len(fm)), "n_malicious": int(fm.y.sum()),
           "prevalence": round(float(fm.y.mean()), 6), "splits": {}}
    for split_name, (tr, te) in {
        "temporal": temporal_split(fm, 0.7),
        "user_holdout": user_holdout_split(fm, 0.3, seed=cfg.seed),
    }.items():
        y = fm.y[te]
        split_res = {"n_test": int(len(te)), "n_positive": int(y.sum())}
        if y.sum() == 0 or y.sum() == len(y):
            split_res["skipped"] = "no positives in test"
            out["splits"][split_name] = split_res
            continue
        for name, fn in SOTA_MODELS.items():
            s = fn(fm, tr, te)
            m = evaluate(fm, te, s, budget=budget).to_dict()
            split_res[name] = {
                "roc_auc": round(float(roc_auc_score(y, s)), 4),
                "pr_auc": round(float(average_precision_score(y, s)), 4),
                "recall_at_1pct": m["recall_at_budget"],
                "precision_at_1pct": m["precision_at_budget"],
                "fp_anomalous_benign_share": m["fp_anomalous_benign_share"],
                "median_lead_days": m["median_detection_lead_days"],
            }
        out["splits"][split_name] = split_res
    return out


def _aggregate(per_seed_runs):
    """Aggregate per-seed run dicts into mean/std per detector per split per metric."""
    agg = {"splits": {}}
    splits = per_seed_runs[0]["splits"].keys()
    for sp in splits:
        agg["splits"][sp] = {}
        models = [k for k in per_seed_runs[0]["splits"][sp] if k in SOTA_MODELS]
        agg["splits"][sp]["n_positive_mean"] = float(np.mean(
            [r["splits"][sp].get("n_positive", 0) for r in per_seed_runs]))
        for mdl in models:
            for metric in ("roc_auc", "pr_auc", "recall_at_1pct"):
                vals = [r["splits"][sp][mdl][metric] for r in per_seed_runs
                        if mdl in r["splits"][sp]]
                agg["splits"][sp].setdefault(mdl, {})[metric + "_mean"] = round(float(np.mean(vals)), 4)
                agg["splits"][sp][mdl][metric + "_std"] = round(float(np.std(vals)), 4)
    agg["activated_insiders_mean"] = float(np.mean([r["activated_insiders"] for r in per_seed_runs]))
    agg["prevalence_mean"] = round(float(np.mean([r["prevalence"] for r in per_seed_runs])), 6)
    return agg


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    seeds = [13 + 7 * i for i in range(n_seeds)]
    base = dict(domain="tech", n_employees=n, horizon_days=d,
                insider_prevalence=0.05, anomalous_benign_rate=0.10)

    cert_runs, real_runs = [], []
    for s in seeds:
        print(f"[sota] seed={s} CERT-regime ...", flush=True)
        cert_runs.append(_run(SimConfig(name="cert_regime", legacy_cert_mode=True, seed=s, **base)))
        print(f"[sota] seed={s} realistic ...", flush=True)
        real_runs.append(_run(SimConfig(name="synthitd_realistic", legacy_cert_mode=False, seed=s, **base)))

    report = {
        "config": {**base, "seeds": seeds},
        "note": ("Controlled ablation: both regimes share org/seed/prevalence/splits and "
                 "differ only in legacy_cert_mode, which degenerately toggles the whole "
                 "anti-shortcut regime (benign upload/removable + benign after-hours + "
                 "benign-anomaly bursts + insider stealth). SOTA detectors are reference "
                 "reimplementations of published CERT recipes over the same leakage-safe "
                 "user-day features. The isolated signature effect is in the shortcut_audit."),
        "cert_regime_aggregate": _aggregate(cert_runs),
        "realistic_aggregate": _aggregate(real_runs),
        "cert_regime_per_seed": cert_runs,
        "realistic_per_seed": real_runs,
    }
    with open("reference/sota_comparison.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n=== Temporal split, {n_seeds} seeds: ROC / PR-AUC  mean±std  (CERT -> realistic) ===", flush=True)
    ct, rt = report["cert_regime_aggregate"]["splits"]["temporal"], report["realistic_aggregate"]["splits"]["temporal"]
    for name in SOTA_MODELS:
        c, r = ct.get(name, {}), rt.get(name, {})
        if c and r:
            print(f"  {name:20s} ROC {c['roc_auc_mean']:.3f}±{c['roc_auc_std']:.2f} -> "
                  f"{r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.2f}   "
                  f"PR {c['pr_auc_mean']:.3f} -> {r['pr_auc_mean']:.3f}", flush=True)
    print("\nwrote reference/sota_comparison.json", flush=True)


if __name__ == "__main__":
    main()
