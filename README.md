# synthitd — Synthetic Insider-Threat Data & Benchmark

Generating synthetic insider data for behaviour-based models — a reproducible
generator *and* an honest benchmark that targets the documented shortcomings of the
CERT dataset and the recent LLM-based successors (ChimeraLog, OrgForge-IT).

> **Defensive / academic tool.** It models workplace behaviour as detection-relevant
> observables and MITRE ATT&CK technique *labels* to produce labelled telemetry for
> training and evaluating insider-threat detection (ITD). It contains no operational
> attack instructions. See [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md).

## Why another insider-threat dataset?

For a decade the field has trained and evaluated almost entirely on CERT, whose
hand-scripted malicious users let detectors learn the *generator's* fingerprint
rather than genuine intent — the root cause of the field's inflated, non-transferable
scores. The 2026 LLM datasets improve realism (ChimeraLog) and verifiability
(OrgForge-IT) but still **inject** a scripted attack into a chosen user, label at the
granularity of the injected steps, and ship no benign-anomaly population and no
self-audit. This project's contribution (full argument in the literature review):

1. **Endogenous intent, not injected scripts.** Each person carries a latent
   *Critical-Pathway* risk trajectory (predisposition → stressors → concerning
   behaviour). A hazard model decides *whether and when* a pathway activates; many
   predisposed, stressed employees never activate and remain hard negatives. The
   label tracks latent **intent**, not a template — so detectors can't shortcut it.
2. **Deterministic bus + label-blind renderer.** A typed event bus owns identities,
   timestamps, causal links, and the graded label. The LLM renders only free text and
   is **never shown the label**; a leakage linter enforces it. Ground truth is an
   architectural guarantee and prose can't leak the answer.
3. **Engineered hard negatives.** A benign-but-anomalous population (travel, on-call,
   promotion, re-org, approved bulk export) produces exfiltration-*looking* bursts
   paired with justifying HR context, so precision and false-positive attribution
   become measurable.
4. **Graded, multi-resolution labels + sensor degradation + modern surfaces**
   (logon, IdP, device, file, email, chat, ticketing, VCS, HTTP, DLP, endpoint, HR).
5. **An honest benchmark shipped with the data** — temporal / user-held-out /
   **scenario-held-out** splits, AUC-PR at true prevalence, detection **earliness**,
   review-budget recall, FP attribution, and a **shortcut audit** that quantifies how
   much of the signal is a generator artifact.
6. **A CERT r6.2-compatible exporter** so existing pipelines consume it unchanged.

## Install

```bash
pip install -e ".[dev]"      # numpy + PyYAML + scikit-learn + pytest
pip install -e ".[llm]"      # optional: Anthropic renderer (Claude Messages API)
```

Only `numpy` and `PyYAML` are required to generate data with the free template
renderer; `scikit-learn` powers the benchmark baselines; `anthropic` is optional.

## Quickstart

```bash
# 1. generate a dataset (free, deterministic template renderer)
python -m synthitd.cli generate --config configs/tech_small.yaml --out out/tech

# 2. run the benchmark (writes out/tech/benchmark.json)
python -m synthitd.cli benchmark --data out/tech

# 3. export a CERT r6.2-compatible view for legacy pipelines
python -m synthitd.cli export-cert --data out/tech --out out/tech/cert

# what's available
python -m synthitd.cli info
```

Python API:

```python
from synthitd.config import SimConfig
from synthitd.simulate import simulate
from synthitd.render import render_events
from synthitd.writers import write_dataset
from synthitd.benchmark.run import run_benchmark

cfg = SimConfig(domain="finance", n_employees=500, horizon_days=180,
                insider_prevalence=0.01, seed=42)      # a pure function of cfg
res = simulate(cfg)
render_events(res.events, res.org, cfg)                 # label-blind free text
write_dataset(res, "out/finance")
report = run_benchmark(res)                             # splits + audits
```

## What the benchmark shows (shipped sample, `sample/benchmark.json`)

120 employees · 70 days · tech · 8,400 user-days · **0.38 % malicious prevalence**.

| split | detector | ROC-AUC | **PR-AUC** | recall@1% | median lead (days) | FP share = hard-neg |
|---|---|---|---|---|---|---|
| temporal | `volume_rule` (CERT shortcut) | 0.61 | **0.009** | 0.00 | — | 0.84 |
| temporal | logistic regression | 0.75 | **0.39** | 0.41 | −3 | 0.06 |
| user-holdout | logistic regression | 0.88 | **0.40** | 0.63 | 0 | 0.10 |
| scenario-holdout (hold out `ip_theft`) | logistic regression | 0.74 | **0.35** | 0.35 | +8 | 0.00 |

**Shortcut audit:** the single "after-hours + removable + upload" rule that scores
near-perfect on CERT collapses to **PR-AUC 0.009** here, while a full model reaches
0.39 (signal-depth gap **+0.38**). The shortcut's false positives are 84–100 % the
*engineered hard negatives* — they trap the naive rule, by design. Detection requires
genuine behavioural modelling and is far from solved at realistic prevalence.

## Use the Claude renderer (optional)

```bash
export ANTHROPIC_API_KEY=...        # or `ant auth login`
python -m synthitd.cli generate --config configs/tech_small.yaml \
    --out out/tech --renderer anthropic
```

The Anthropic backend (`claude-opus-4-8` by default) renders only label-blind
`ContentPlan`s, with structured JSON output, a cached instruction prefix, chunked
requests, graceful fallback to the template renderer, and an offline
`render_via_batch_api` path (50 % cheaper). It can never see or emit the label.

## Documentation

- [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) — gap analysis vs CERT,
  ChimeraLog, OrgForge-IT, MOLE, and the behavioural-science basis.
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture and the determinism guarantees.
- [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) — schema, labels, fields, intended
  use, and limitations.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — tasks, splits, metrics, and how to add a
  detector.

## Repository layout

```
synthitd/
  events.py        canonical event schema + deterministic label-owning bus
  config.py        SimConfig + seeded named-substream RNG
  org.py           domains, roles, employees, OCEAN, population assignment
  lifecycle.py     stressors / triggers (Critical-Pathway inputs)
  psych.py         latent risk trajectories + hazard-based pathway activation
  behavior.py      baseline multi-surface behaviour + benign hard-negative bursts
  threats.py       abstract stage-based pathways (MITRE technique labels)
  observability.py sensor-degradation / partial-visibility filter
  simulate.py      orchestrator: SimConfig -> labelled dataset
  render/          label-blind renderer (template + Anthropic) + leakage linter
  writers.py       JSONL / CSV / ground-truth writers
  export_cert.py   CERT r6.2-compatible exporter
  benchmark/       features, splits, baselines, metrics, audits, runner
  cli.py           generate / benchmark / export-cert / info
tests/             pytest suite (determinism, label-blindness, anti-shortcut, ...)
configs/           example dataset specifications
sample/            compact reports + CERT heads from the shipped sample
```

## License

MIT.
