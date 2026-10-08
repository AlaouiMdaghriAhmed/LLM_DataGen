"""Build a larger, multi-domain synthitd release ready for the Hugging Face Hub.

Generates one dataset instance per domain (tech, finance, healthcare) under a single
folder, each with the full file set + CERT export, and writes a multi-config HF
dataset card (one viewer config per domain) with real statistics filled in.

Run:  PYTHONPATH=. python scripts/build_release.py [employees] [days] [out_dir]
Then push with:  python scripts/publish_hf.py --repo-id <user>/synthitd --from-dir <out_dir>
"""

from __future__ import annotations

import json
import os
import sys

from synthitd.config import SimConfig
from synthitd.simulate import simulate
from synthitd.writers import write_dataset
from synthitd.export_cert import export_cert

DOMAIN_SEED = {"tech": 11, "finance": 20, "healthcare": 31}

CARD_HEADER = """---
license: mit
pretty_name: synthitd — Endogenous-Intent Synthetic Insider-Threat Benchmark (multi-domain)
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
  - 1M<n<10M
configs:
{configs}
---
"""


def _card(stats: dict, employees: int, days: int) -> str:
    cfgs = []
    for i, dom in enumerate(DOMAIN_SEED):
        default = "    default: true\n" if i == 0 else ""
        cfgs.append(
            f"  - config_name: {dom}\n{default}"
            f"    data_files:\n      - split: all\n        path: {dom}/labels_userday.csv\n"
        )
    header = CARD_HEADER.format(configs="".join(cfgs).rstrip())

    rows = "\n".join(
        f"| {dom} | {s['n_employees']} | {s['activated_insiders']}/{s['insider_population']} | "
        f"{s['anomalous_benign_population']} | {s['n_events_full']:,} | {s['n_events_observed']:,} | "
        f"{s['label_counts_full'].get('precursor', 0)} | {s['label_counts_full'].get('malicious', 0)} |"
        for dom, s in stats.items()
    )
    body = f"""
# synthitd — Multi-Domain Insider-Threat Benchmark

Synthetic, fully reproducible, multi-surface workplace telemetry for **insider-threat
detection (ITD)**, with graded ground-truth labels **grown** from a Critical-Pathway
intent model rather than injected as scripted attacks. This release contains three
data-sensitive organisations — **technology, finance, healthcare** — each
{employees} employees over {days} days. Pick a domain with the dataset viewer's config
selector (default: `tech`).

> **Defensive / academic use only.** Behaviour is modelled as detection-relevant
> observables and MITRE ATT&CK technique *labels*; no operational attack instructions.
> Fully synthetic — no real people, accounts, credentials, or messages.

Code, generator, benchmark, paper: <https://github.com/AlaouiMdaghriAhmed/LLM_DataGen>

## Instances in this release

| domain | employees | activated/insiders | anomalous-benign | events (full) | events (observed) | precursor | malicious |
|---|---|---|---|---|---|---|---|
{rows}

## Layout (per domain `<d>/`)

| file | description |
|---|---|
| `<d>/labels_userday.csv` | **viewer config** — per (user, day): graded label, latent risk, counts |
| `<d>/events.jsonl` | full labelled ground-truth stream (one JSON event per line) |
| `<d>/observed.jsonl` | stream after the sensor-degradation filter (what a detector sees) |
| `<d>/employees.json` | org: roles, access, OCEAN traits, predisposition, population |
| `<d>/episodes.json` | one record per activated insider episode (pathway, onset, MITRE techniques) |
| `<d>/ground_truth.json` | per-employee latent risk-trajectory summary |
| `<d>/manifest.json` | config + summary + observability reports |
| `<d>/cert/` | CERT r6.2-compatible export (+ `cert/extended/` modern surfaces) |

### Event schema (`events.jsonl`)
`event_id, ts, channel, actor, action, attrs, label, episode_id, technique_ids,
risk_state, render, causal_parent`. Channels: logon, idp, device, file, email, http,
chat, ticket, vcs, dlp, endpoint, hr.

### Labels (graded)
`benign` · `anomalous_benign` (hard negative, paired with an HR-context event) ·
`precursor` (concerning activity on an armed pathway, before the act) · `malicious`.

## Reproduce
Every instance is a pure function of its config and regenerates byte-for-byte:
```bash
pip install -e ".[dev]"
python scripts/build_release.py {employees} {days} out/synthitd_release
```

## Citation
```bibtex
@misc{{synthitd2026,
  title        = {{synthitd: A Reproducible, Endogenous-Intent Synthetic Benchmark for Insider-Threat Detection}},
  author       = {{Alaoui Mdaghri, Ahmed}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/AlaouiMdaghriAhmed/LLM_DataGen}}}}
}}
```
"""
    return header + body


def main() -> int:
    employees = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "out/synthitd_release"
    os.makedirs(out_dir, exist_ok=True)

    stats: dict[str, dict] = {}
    for dom, seed in DOMAIN_SEED.items():
        cfg = SimConfig(
            name=f"synthitd_{dom}", domain=dom, n_employees=employees, horizon_days=days,
            insider_prevalence=0.03, anomalous_benign_rate=0.08, seed=seed,
            channel_coverage={"endpoint": 0.65, "http": 0.9, "chat": 0.8, "idp": 0.95},
            sensor_dropout=0.02, renderer="template",
        )
        print(f"[release] {dom}: simulating {employees}x{days} seed={seed} ...", flush=True)
        res = simulate(cfg)
        dom_dir = os.path.join(out_dir, dom)
        man = write_dataset(res, dom_dir)          # no free-text render (bounds size)
        export_cert(res, os.path.join(dom_dir, "cert"))
        stats[dom] = man["summary"]
        s = stats[dom]
        print(f"[release] {dom}: {s['n_events_full']:,} events, "
              f"{s['activated_insiders']}/{s['insider_population']} insiders activated", flush=True)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_card(stats, employees, days))
    with open(os.path.join(out_dir, "release_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"employees": employees, "days": days, "domains": stats}, fh, indent=2)

    tot = sum(s["n_events_full"] for s in stats.values())
    print(f"\n[release] wrote {out_dir} — 3 domains, {tot:,} total events", flush=True)
    print(f"[release] push with: python scripts/publish_hf.py --repo-id <user>/synthitd "
          f"--from-dir {out_dir} --public", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
