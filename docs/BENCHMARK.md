# Benchmark

An evaluation harness shipped *with* the data, designed to avoid the reporting
practices that inflate CERT numbers (random splits, balanced subsets, a single
headline F1) and to be self-critical about its own dataset.

Run it:

```bash
python -m synthitd.cli benchmark --data out/tech --budget 0.01
# or
python -c "from synthitd.simulate import simulate; from synthitd.config import load_config; \
from synthitd.benchmark.run import run_benchmark; import json; \
print(json.dumps(run_benchmark(simulate(load_config('configs/tech_small.yaml'))), indent=2))"
```

## Unit of analysis

Per **(user, day)**. Features are built only from the *observed* stream and only from
attributes a real sensor could report; generator-internal flags that encode the label
(`out_of_pattern`, `staging_dir`, `record_modification`, `elevated`, …) are excluded
by an explicit denylist (`benchmark/features.DENYLISTED_ATTRS`), and a test asserts
that injecting them changes no feature. The legitimate contextual signal — the
observable `hr.context` event — *is* a feature, so a good model can learn to exonerate
explained anomalies.

## Splits

| split | what it tests | why it matters |
|---|---|---|
| `temporal` | train early days, test later days | prevents future→past leakage that random splits cause |
| `user_holdout` | disjoint employees in train/test | generalisation to unseen *people* |
| `scenario_holdout` | one pathway *family* held out of training entirely | generalisation to an *unseen kind* of insider — a test prior datasets don't offer |

## Metrics

- **ROC-AUC** and **AUC-PR** at the true (sparse) prevalence — not on a balanced
  subset. AUC-PR is the headline because prevalence is ~0.1–0.4 %.
- **precision@budget / recall@budget** — flag the top `budget` fraction of user-days
  (default 1 %); reports how many insiders a budget-limited SOC catches and how noisy
  the queue is.
- **detection earliness** — for each activated insider, days between first flag and
  onset (positive ⇒ caught in the precursor window). Enabled by the precursor labels.
- **false-positive attribution** — share of flagged negatives that are the engineered
  *anomalous-benign* hard negatives. High values mean the detector is fooled by
  explained anomalies.

## Audits (self-critical, shipped with the data)

- **shortcut audit** — AUC of a single trivial rule ("after-hours + removable +
  upload") vs a full model, plus `signal_depth_pr_gap = full_pr − trivial_pr`. On
  CERT the trivial rule is near-perfect; here it should be near-zero with a positive
  gap, demonstrating the signature shortcut is defeated.
- **stylized facts** — events/user-day distribution, after-hours fraction, work-hour
  concentration, weekend fraction, per-channel share, and a `plausible` flag.

## Reference baselines

`volume_rule` (the CERT shortcut, a floor), `user_zscore` (per-user self-baseline),
`logreg` (supervised, class-balanced), `isolation_forest` (unsupervised). They are
deliberately simple — the deliverable is the *benchmark*, not a new SOTA detector.

## Reading the sample results

From `sample/benchmark.json` (120×70, tech, 0.38 % prevalence):

- The `volume_rule` shortcut gets **PR-AUC 0.009** and its false positives are
  84–100 % hard negatives — it is actively trapped by the benign-anomaly population.
- A full model reaches **PR-AUC 0.35–0.40** and ROC 0.74–0.88 — real signal, far
  below CERT's ~0.99, i.e. a genuinely unsolved benchmark.
- Earliness ranges from catching insiders ~8 days before the act (scenario-holdout,
  logreg) to a few days late — the precursor labels make this measurable at all.

## Adding your detector

```python
from synthitd.benchmark.features import build_userday_features
from synthitd.benchmark.splits import temporal_split
from synthitd.benchmark.metrics import evaluate

fm = build_userday_features(result)          # result = simulate(cfg)
tr, te = temporal_split(fm, 0.7)
scores = my_detector(fm.X[tr], fm.y[tr], fm.X[te])   # -> per-test-row score
print(evaluate(fm, te, scores, budget=0.01).to_dict())
```

Or register it in `benchmark/baselines.BASELINES` to have it run across all splits by
`run_benchmark`.
