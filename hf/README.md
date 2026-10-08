---
license: mit
pretty_name: synthitd — Endogenous-Intent Synthetic Insider-Threat Benchmark
annotations_creators:
  - machine-generated
language:
  - en
tags:
  - insider-threat
  - security
  - anomaly-detection
  - synthetic-data
  - benchmark
  - user-behavior-analytics
  - cert
task_categories:
  - tabular-classification
  - time-series-forecasting
size_categories:
  - 100K<n<1M
configs:
  - config_name: user_day_labels
    default: true
    data_files:
      - split: all
        path: labels_userday.csv
  - config_name: cert_export
    data_files:
      - split: logon
        path: cert/logon.csv
      - split: file
        path: cert/file.csv
      - split: email
        path: cert/email.csv
      - split: http
        path: cert/http.csv
      - split: device
        path: cert/device.csv
      - split: labels
        path: cert/labels.csv
---

# synthitd — A Reproducible, Endogenous-Intent Synthetic Insider-Threat Benchmark

Synthetic, fully reproducible, multi-surface workplace telemetry for **insider-threat
detection (ITD)** research, with graded ground-truth labels **grown** from a
Critical-Pathway intent model rather than injected as scripted attacks. It targets the
documented shortcomings of the CERT dataset and recent LLM-based successors
(ChimeraLog, OrgForge-IT): signature shortcuts, balanced-subset reporting bias,
pre-LLM modalities, and the absence of a benign-anomaly population or a self-audit.

> **Defensive / academic use only.** Behaviour is modelled as detection-relevant
> observables and MITRE ATT&CK technique *labels*; there are no operational attack
> instructions. Fully synthetic — no real people, accounts, credentials, or messages.

Code, generator, benchmark, and paper: <https://github.com/AlaouiMdaghriAhmed/LLM_DataGen>

## What makes it different

- **Endogenous intent.** Each employee has a latent Critical-Pathway risk trajectory
  (predisposition → stressors → concerning behaviour); a hazard model decides *whether
  and when* a pathway activates. Many predisposed, stressed employees never activate
  and remain hard negatives. Labels track latent intent, not a template.
- **Deterministic bus + label-blind renderer.** A typed event bus owns identities,
  timestamps, causal links, and the graded label; free text is rendered by a pass that
  never sees the label (a leakage linter enforces it).
- **Engineered benign-anomaly hard negatives** (travel, on-call, promotion, approved
  bulk export) with justifying HR context, so precision/false-positive attribution are
  measurable.
- **Graded labels, modern surfaces, sensor-degradation model**, and a **CERT r6.2
  compatible export**.

## Files

| file | description |
|---|---|
| `events.jsonl` | full labelled ground-truth stream (one JSON event per line) |
| `observed.jsonl` | stream after the sensor-degradation filter (what a detector sees) |
| `labels_userday.csv` | **default viewer config** — per (user, day): graded label, latent risk, counts |
| `employees.json` | org: roles, access, OCEAN traits, predisposition, population |
| `episodes.json` | one record per activated insider episode (pathway, onset, MITRE techniques) |
| `ground_truth.json` | per-employee latent risk-trajectory summary |
| `manifest.json` | config + summary + observability/render reports |
| `cert/` | CERT r6.2-compatible export (logon/device/http/email/file + labels + insiders) |
| `cert/extended/` | modern surfaces (chat/ticket/vcs/idp/endpoint/hr) |

### Event schema (`events.jsonl`)
`event_id, ts (unix seconds), channel, actor, action, attrs (object), label, episode_id,
technique_ids (list), risk_state (latent 0–1), render (object|null), causal_parent`.

Channels: logon, idp, device, file, email, http, chat, ticket, vcs, dlp, endpoint, hr.

### Labels (graded)
`benign` · `anomalous_benign` (hard negative, paired with an HR-context event) ·
`precursor` (concerning activity on an armed pathway, before the act) · `malicious`.

## Reproduce / regenerate at any scale

Every instance is a pure function of its config and regenerates byte-for-byte:

```bash
pip install -e ".[dev]"
python -m synthitd.cli generate --domain tech --employees 60 --days 60 --seed 7 --out out/demo
python -m synthitd.cli benchmark   --data out/demo
python -m synthitd.cli export-cert --data out/demo --out out/demo/cert
```

## Benchmark & honest evaluation

Ships with temporal / user-held-out / **scenario-held-out** splits, AUC-PR at true
(sparse) prevalence, detection **earliness**, review-budget recall, false-positive
attribution, and a **shortcut audit**. On the reference instance the single
after-hours+removable+upload rule that nears ceiling on CERT collapses to PR-AUC
≈ 0.0004, and a faithful reimplementation of the best-reported CERT SOTA
(UBS-Transformer; reported AUROC 0.95 / F1 0.96) drops to AUROC 0.64 / F1 0.43.

## Citation

```bibtex
@misc{synthitd2026,
  title        = {synthitd: A Reproducible, Endogenous-Intent Synthetic Benchmark for Insider-Threat Detection},
  author       = {Alaoui Mdaghri, Ahmed},
  year         = {2026},
  howpublished = {\url{https://github.com/AlaouiMdaghriAhmed/LLM_DataGen}}
}
```
