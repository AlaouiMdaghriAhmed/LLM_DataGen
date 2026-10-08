"""Run the reimplemented UBS-Transformer SOTA on CERT-regime vs realistic data.

Reproduces the best-reported recent CERT SOTA architecture (arXiv:2506.23446) and
evaluates it per user, exactly as the paper does, on:
  1. CERT-regime data (legacy_cert_mode=True) — a faithfulness check: the architecture
     should approach its reported CERT numbers when the data has CERT-style separability.
  2. the realistic synthitd data — the test of whether that reported performance holds.

Writes reference/ubs_transformer_comparison.json.

Run:  PYTHONPATH=. python scripts/ubs_experiment.py [n_employees] [horizon] [seed] [epochs]
"""

from __future__ import annotations

import json
import sys

from synthitd.config import SimConfig
from synthitd.simulate import simulate
from synthitd.benchmark.ubs_transformer import run_ubs_transformer, UBSConfig


# the paper's best reported numbers (Test-4 combined r4.2/r5.2/r6.2, Transformer+iForest)
PAPER_REPORTED = {
    "source": "arXiv:2506.23446 (2025), UBS-Transformer, Test-4 combined, Transformer+iForest",
    "accuracy": 0.9661, "precision": 0.9351, "recall": 0.9943, "f1": 0.9638, "auroc": 0.9500,
    "also": "Transformer+OCSVM reported perfect recall / 0% FNR on Test-4",
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 13
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12

    base = dict(domain="tech", n_employees=n, horizon_days=h,
                insider_prevalence=0.08, anomalous_benign_rate=0.10, seed=seed)
    cfg = UBSConfig(d_model=512, n_layers=6, n_heads=8, ffn=2048, dropout=0.1,
                    epochs=epochs, lr=5e-4, batch_size=64, seed=0)

    results = {}
    for regime, legacy in [("cert_regime", True), ("realistic", False)]:
        print(f"\n[ubs] === {regime} (legacy_cert_mode={legacy}) {n}x{h} seed={seed} ===", flush=True)
        res = simulate(SimConfig(name=regime, legacy_cert_mode=legacy, **base))
        print(f"[ubs] simulated {len(res.events):,} events, activated={len(res.insiders())}", flush=True)
        out = run_ubs_transformer(res, cfg)
        results[regime] = out
        print(f"[ubs] {regime}: test_users={out['n_test_users']} insiders={out['n_test_insiders']}", flush=True)
        for det, m in out["detectors"].items():
            if "auroc" in m:
                print(f"    {det:18s} AUROC {m['auroc']:.3f}  F1 {m['f1']:.3f}  "
                      f"P {m['precision']:.3f}  R {m['recall']:.3f}", flush=True)

    report = {
        "architecture": "UBS-Transformer (6-layer d_model=512 8-head reconstruction "
                         "Transformer encoder + OCSVM/LOF/iForest on reconstruction errors), "
                         "reference reimplementation of arXiv:2506.23446",
        "note": ("Faithful architecture, adapted to this repo's day-granularity user-day "
                 "features; per-user evaluation like the paper. epochs reduced for CPU. "
                 "CERT-regime is this repo's legacy_cert_mode ablation, not the real CERT "
                 "corpus (egress-blocked). Single seed; magnitudes are indicative."),
        "config": {**base, "ubs": vars(cfg)},
        "paper_reported_on_cert": PAPER_REPORTED,
        "cert_regime": results["cert_regime"],
        "realistic": results["realistic"],
    }
    with open("reference/ubs_transformer_comparison.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== UBS-Transformer: reported-on-CERT vs reimpl ===", flush=True)
    pr = PAPER_REPORTED
    print(f"  paper (CERT, Transformer+iForest):   AUROC {pr['auroc']:.3f}  F1 {pr['f1']:.3f}  R {pr['recall']:.3f}", flush=True)
    for regime in ["cert_regime", "realistic"]:
        m = results[regime]["detectors"].get("iforest", {})
        if "auroc" in m:
            print(f"  reimpl ({regime:12s}, +iForest):  AUROC {m['auroc']:.3f}  F1 {m['f1']:.3f}  R {m['recall']:.3f}", flush=True)
    print("\nwrote reference/ubs_transformer_comparison.json", flush=True)


if __name__ == "__main__":
    main()
