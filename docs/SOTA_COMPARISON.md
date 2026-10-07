# SOTA CERT Detectors: CERT-Regime vs. Realistic Data

Two experiments:

- **Part A — the single best-reported SOTA architecture, reimplemented.** We take the
  strongest recent CERT architecture with a concretely specified design (the
  UBS-Transformer, arXiv:2506.23446), reimplement it faithfully, run it, and compare
  what it scores against the numbers that paper reports on CERT.
- **Part B — the SOTA recipe family, ablated.** We run the dominant CERT detector
  families (RF / GBDT / MLP / autoencoder / temporal) across a controlled CERT-regime
  vs realistic ablation, over 5 seeds.

The question for both is the same: **does a detector that scores ~0.95–0.99 on CERT
keep that score when the data stops leaking the generator's signature?**

---

# Part A — Best-reported SOTA architecture: UBS-Transformer

**Model.** "Enhancing Insider Threat Detection Using User-Based Sequencing and
Transformer Encoders" (arXiv:2506.23446, 2025) — a recent best-in-class CERT result
with a fully specified architecture. **User-Based Sequencing** turns each user into
one ordered sequence of behavioural vectors; a **6-layer, d_model=512, 8-head
Transformer encoder** is trained with **MSE reconstruction loss on normal users**;
then **One-Class SVM / LOF / Isolation Forest** score the per-user reconstruction
errors; evaluation is **per user** (benign vs malicious). Reported best (Test-4,
combined r4.2/r5.2/r6.2, Transformer + iForest): **accuracy 0.966, precision 0.935,
recall 0.994, F1 0.964, AUROC 0.950** (FNR 0.0057, FPR 0.0571).

**Our reimplementation** (`synthitd/benchmark/ubs_transformer.py`) reproduces that
architecture exactly (6×512×8 encoder, reconstruction loss, the three outlier
detectors on reconstruction-error summaries, per-user evaluation), adapted to this
repo's day-granularity user-day feature tokens. We then run it on (1) **CERT-regime**
data — a faithfulness check — and (2) **realistic** synthitd data — the real test.
Both are 400 employees × 120 days, seed 13, 19 activated insiders, 134 held-out test
users (19 positive), 12 training epochs on CPU.

**Result (per-user):**

| Setting | AUROC | F1 | precision | recall |
|---|---|---|---|---|
| **Paper, reported on CERT** (Transformer + iForest) | **0.950** | **0.964** | 0.935 | 0.994 |
| Reimpl on **CERT-regime** — iForest | 0.874 | 0.681 | 0.571 | 0.842 |
| Reimpl on **CERT-regime** — OCSVM | 0.892 | 0.750 | 0.714 | 0.789 |
| Reimpl on **CERT-regime** — LOF | 0.878 | 0.750 | 0.923 | 0.632 |
| Reimpl on **realistic** — iForest | **0.643** | **0.425** | 0.357 | 0.526 |
| Reimpl on **realistic** — OCSVM | 0.533 | 0.340 | 0.265 | 0.474 |
| Reimpl on **realistic** — LOF | 0.703 | 0.516 | 0.667 | 0.421 |

Two readings, both honest:

1. **Faithfulness.** On CERT-regime data the reimplementation reaches AUROC
   **0.87–0.89** — the same regime as the paper's 0.95, landing a little lower as
   expected for a stylized in-pipeline CERT proxy trained on far less data (hundreds
   of users vs thousands), 12 epochs, and day- rather than session-granularity tokens.
   Close enough to confirm the architecture is reproduced correctly.
2. **Transfer.** The *same code at the same scale* drops from AUROC 0.874 → **0.643**
   and F1 0.681 → **0.425** (iForest) moving from CERT-regime to realistic data; the
   bare reconstruction-error signal falls to AUROC 0.573, barely above chance. The
   best-reported CERT SOTA architecture **loses ~0.23 AUROC and ~0.26 F1** on
   realistic telemetry — its headline number does not transfer.

The within-reimplementation CERT-regime → realistic gap (same architecture, same
scale, only the data regime changes) is the clean, confound-free measurement; the
paper's 0.95 is the external anchor showing our CERT-regime number is in the right
place.

**Reproduce:** `PYTHONPATH=. python scripts/ubs_experiment.py 400 120 13 12`
(writes `reference/ubs_transformer_comparison.json`; needs PyTorch, CPU is fine).
Caveats: single seed, 19 test insiders, 12 epochs, in-pipeline CERT proxy not the
real corpus — magnitudes are indicative, the direction (large drop) is the result.

---

# Part B — SOTA recipe family, ablated (5 seeds)

A head-to-head comparison of the dominant CERT detector *families*, run on the two
regimes this repository produces from one shared config. The question is narrow and
operational: **does a detector that scores ~0.99 on CERT keep that score when the
data stops leaking the generator's signature?**

> All numbers are measured over **5 seeds** (13, 20, 27, 34, 41) at 300 employees ×
> 120 days, reported as mean ± std, in
> [`reference/sota_comparison.json`](../reference/sota_comparison.json). They are
> reference reimplementations of published CERT recipes on *synthetic* data, not the
> published values themselves. Read [§5 Threats to validity](#5-threats-to-validity)
> before quoting any figure — the magnitudes are under-powered; the *direction* is
> the result.

---

## 1. Question

The CERT r4.2/r5.2 literature reports near-ceiling numbers for feature-engineered
supervised ITD. Two evaluation regimes produce wildly different "SOTA", and
conflating them is the field's central reporting pitfall:

- **Full-population, FPR/budget-based (credible).** Random Forest on user-day
  features reaches ROC-AUC ≈ **0.979** (user-week 0.987) detecting ~85 % of insiders
  at a 0.78 % false-positive rate, on the real imbalanced population with a
  chronological split — Le, Zincir-Heywood & Heywood, *IEEE TNSM* 2020.
- **Balanced-subset / resampled (inflated, non-transferable).** Under-sampling benign
  and/or SMOTE-oversampling malicious yields 96–100 %: XGBoost ROC-AUC **0.997**
  (Sivakrishna et al., *Security & Privacy* 2025), a stack-classifier AUC **0.988**
  (Hall et al., *IEEE BigData* 2018), one-hot + SMOTE trees AUC ≈ **1.00** (Al-Shehari
  & Alsowail, *Entropy* 2021), an 88-algorithm sweep topping **>98 %** accuracy with
  RF best (Noever, arXiv 2019).

On the realistic, sparse r6.2 release the honest numbers are far lower (Log2vec ≈
0.86, DeepLog ≈ 0.86 AUC; a GCN+Bi-LSTM scores 0.986 on *dense* r5.2 but 0.885 on
*sparse* r6.2), confirming the score is produced by dense releases, class balancing,
random (non-temporal) splits, and a scripted malicious population a detector can
fingerprint. **We test whether that score survives when the generator signature is
removed and nothing else changes.**

## 2. Method

A controlled ablation. Both datasets are generated from one base config (tech, 300
employees, 120-day horizon, 5 % insider selection, 10 % anomalous-benign rate), over
5 seeds. The **only** difference is the `legacy_cert_mode` flag, which degenerately
toggles the *whole* anti-shortcut regime — **four coupled behavioural axes**, not a
single "shortcut":

1. benign power-user cloud uploads / removable-media copies → 0;
2. benign after-hours baseline → suppressed;
3. benign-anomaly exfil-looking bursts (hard negatives) → removed;
4. insider **stealth** → 0 (obvious routing + after-hours timing + larger volume).

Because both regimes share seed, org, insider selection, onset, labels, prevalence,
and splits (verified: the flag appears only in `config.py`, `behavior.py`,
`threats.py` — never in `org.py`/`lifecycle.py`), this is a cleaner apples-to-apples
ablation than fetching the real corpus. We could **not** fetch the multi-GB real CERT
release here (arXiv/IEEE/PDF hosts are egress-blocked), so we reproduce the CERT
*regime* in-pipeline and confirm the *mechanism*, not any published value. Both
regimes report (mean over seeds) 11.4 activated insiders, 36,000 user-days,
prevalence **0.0017**, 32 positive user-days in the temporal test split, 19 in the
user-holdout split.

> **Important:** this flag changes four axes at once, so a drop here reflects the
> combined regime — it does **not** isolate "the signature shortcut". The isolated
> signature effect is measured separately by the [shortcut audit](BENCHMARK.md)
> (the single after-hours+removable+upload rule: PR-AUC ≈ 0.009 on realistic data).

The detectors are reference reimplementations of the dominant published CERT recipes,
all over the same leakage-safe user-day feature matrix
(`synthitd/benchmark/features.py`, with a denylist removing trivial label leakage):

| Detector | Stands in for | Representative reported CERT number |
|---|---|---|
| `sota_random_forest` | RF on user-day features (credible classical SOTA) | ROC 0.979 full-pop (Le 2020); >98 % acc balanced (Noever 2019) |
| `sota_gbdt` (HistGradientBoosting) | XGBoost / LightGBM user-day | ROC 0.997 (Sivakrishna 2025); 0.988 stack (Hall 2018) |
| `sota_mlp` | deep feed-forward baseline | ~0.90–0.95 AUC balanced |
| `sota_autoencoder` (PCA reconstruction) | DNN/LSTM auto-encoder reconstruction-error | ~90 % acc (Tuor 2017; Sharma 2020; Pantelidis 2021) |
| `sota_seq_gbdt` (GBDT over causal sequence features) | LSTM / transformer-encoder temporal SOTA | LSTM-CNN 0.945 (Yuan 2018); transformer ~0.95 (2024–25) |

Graph (GNN) and LLM families are out of scope: the flat user-day representation here
does not carry the interaction graph or raw text those recipes consume.

## 3. Results

**Temporal split** (train early days → test later days), ROC-AUC and PR-AUC,
mean ± std over 5 seeds, CERT-regime → realistic:

| Detector | ROC (CERT → realistic) | PR-AUC (CERT → realistic) | ΔPR | recall@1 % (realistic) |
|---|---|---|---|---|
| `sota_random_forest` | 0.879 ± .06 → **0.773** ± .05 | 0.686 → **0.363** | −47 % | 0.44 |
| `sota_gbdt` | 0.860 ± .04 → 0.610 ± .08 | 0.598 → 0.049 | −92 % | 0.15 |
| `sota_mlp` | 0.825 ± .06 → 0.652 ± .05 | 0.642 → 0.341 | −47 % | 0.38 |
| `sota_autoencoder` | 0.880 ± .04 → 0.817 ± .05 | 0.679 → 0.281 | −59 % | 0.32 |
| `sota_seq_gbdt` | **0.966** ± .02 → **0.879** ± .05 | 0.445 → 0.111 | −75 % | 0.35 |

**User-holdout split** (disjoint employees train/test):

| Detector | ROC (CERT → realistic) | PR-AUC (CERT → realistic) | recall@1 % (realistic) |
|---|---|---|---|
| `sota_random_forest` | 0.744 ± .15 → 0.680 ± .11 | 0.447 → 0.235 | 0.27 |
| `sota_gbdt` | 0.729 ± .13 → 0.620 ± .08 | 0.259 → 0.057 | 0.18 |
| `sota_mlp` | 0.623 ± .24 → 0.508 ± .16 | 0.427 → 0.241 | 0.26 |
| `sota_autoencoder` | 0.760 ± .10 → 0.739 ± .06 | 0.432 → 0.217 | 0.25 |
| `sota_seq_gbdt` | **0.980** ± .02 → **0.935** ± .06 | 0.213 → 0.167 | **0.68** |

What is robust across all 5 seeds and both splits:

1. **Every classical CERT detector drops** going CERT-regime → realistic, on **both**
   ROC (≈ 0.1) and PR-AUC (**43–92 %**). The direction is consistent; the exact
   percentages are not (see §5).
2. **On realistic data, every detector is weak in PR-AUC** — 0.05–0.36 against a
   no-skill baseline of ≈ 0.0017. Realistic ITD at true prevalence is *not solved* by
   any of these recipes. The best realistic PR-AUC is the **random forest** (0.363),
   not the sequence model.
3. The CERT-regime ROCs here (0.82–0.98) already sit *below* the literature's 0.99,
   because we evaluate at true sparse prevalence with temporal/user splits and no
   balancing — matching the honest-SOTA camp (Le 2020), not the balanced-subset camp.
4. **`sota_seq_gbdt` is the only regime-robust and generalizing model — on ROC and
   budgeted recall, not PR.** It holds the top ROC in every cell (realistic 0.879
   temporal, 0.935 user-holdout) and, under a 1 % review budget on **unseen users**,
   catches **0.68** of insiders vs ~0.25 for the static detectors — ~2.5× more. Its
   PR-AUC is low everywhere because precision at 0.17 % prevalence is poor for all
   models; it wins on ranking and budgeted recall, which is what a review-limited SOC
   actually operates on.

## 4. Interpretation

- The field's ~0.99 on CERT is an artifact of dense releases, class balancing, random
  splits, and a fingerprintable scripted population. Removing the generator signature
  (plus restoring stealthy insiders and hard negatives) costs the classical recipes
  roughly half their PR-AUC and ~0.1 ROC — repeatably.
- At realistic prevalence, **flat user-day detection is weak for everyone.** The
  dataset is not "another CERT you can max out"; it is a hard benchmark where PR-AUC
  lives near the floor.
- Where signal remains, it is in the **behavioural trajectory**, not the single day:
  the temporal model is the only one that transfers across the ablation and across
  unseen users, and under a fixed review budget it is clearly the most useful — it
  also surfaces insiders with a median lead of ~2 weeks before the act (see the
  earliness column in the benchmark). That is exactly what the generator is built to
  reward: intent that escalates along the Critical Pathway, not a static signature.

## 5. Threats to validity

Incorporated verbatim from the workflow's adversarial audit — stated so the reader
can discount accordingly:

- **The ablation is a 4-way joint change, not an isolated shortcut.** Setting insider
  stealth to 0 also *attenuates the positive-class signal* (fewer, work-hours, larger
  events), so the collapse cannot be attributed purely to "signature removal". The
  isolated signature effect is the separate `shortcut_audit` (trivial-rule PR-AUC
  ≈ 0.009 on realistic data); read the two together.
- **Under-powered magnitudes.** ~11 insiders and 32 (temporal) / 19 (user-holdout)
  positive user-days per seed at prevalence 0.0017. Even over 5 seeds the std is large
  (e.g. MLP user-holdout ROC 0.623 ± 0.24). Treat the specific percentages as
  indicative; only the sign/direction is defensible. More seeds and larger orgs would
  tighten the intervals.
- **Don't cross metrics.** The classical collapse is reported on PR-AUC *and* ROC; the
  sequence model's advantage is on ROC and budgeted recall, **not** PR-AUC (where it
  sits near the floor like everything else, and below RF on realistic data). We do not
  claim it is "the best detector" — only the most regime-robust and the best at
  budgeted recall on unseen users.
- **PR-AUC baseline is ~0.0017.** The "surviving" realistic PR values (0.05–0.36) are
  all operationally poor; the honest framing is "all detectors are weak on realistic
  data", not "our sequence model works".
- **`sota_autoencoder` is a linear PCA reconstruction**, a *lower bound* on the
  nonlinear/temporal DNN/LSTM-AE family — its realistic drop partly reflects PCA's
  brittleness. Three non-AE classical models drop too, so the conclusion does not rest
  on it.
- **`sota_seq_gbdt` is gradient boosting over engineered causal rolling features**, a
  proxy for — not an instance of — an LSTM/transformer. Its regime-invariant ROC
  should be read as "temporal structure survives the ablation", and audited for any
  residual leakage from untouched temporal features (history length / escalation).
- **Synthetic regime, not the real corpus.** We reproduce the CERT *regime*
  in-pipeline; we did not run on the real CERT data (egress-blocked). The comparison
  is internally valid and non-confounded on labels/onset/prevalence/splits, but it is
  a stylized upper bound on the signature effect, not a measurement on CERT itself.

## 6. Reproduce

```bash
PYTHONPATH=. python scripts/sota_experiment.py 300 120 5   # n_employees horizon n_seeds
# writes reference/sota_comparison.json (per-seed + mean/std aggregate)
```

Detectors: `synthitd/benchmark/sota.py`. Features (leakage-safe):
`synthitd/benchmark/features.py`. Sequence features:
`synthitd/benchmark/sequence.py`. CERT-regime flag: `legacy_cert_mode` in
`synthitd/config.py`. Context: [`BENCHMARK.md`](BENCHMARK.md),
[`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).
