# Design & Architecture

This document explains how the generator works and why each boundary exists. The
motivating gap analysis is in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md); the
requirement tags (R1–R9) used below are defined there.

## Pipeline

```
SimConfig ──► org ──► lifecycle ──► psych ──► threats.assign_pathways
 (seed)      (R3)      (stressors)   (R1 risk)     (pathway family)
                                        │
                                        ▼
                         per-day:  behavior.emit_day   (benign + hard negatives, R4)
                                   threats.emit_day     (precursor/malicious, R1/R5)
                                        │
                                        ▼
                              EventBus.finalize()       (full labelled ground truth, R2)
                                        │
                            observability.apply()       (observed stream, R8)
                                        │
                 ┌──────────────────────┼───────────────────────────┐
                 ▼                      ▼                            ▼
       render/ (label-blind, R2)   writers (JSONL/CSV/GT)   export_cert (R9)
                 │
         benchmark/ (splits, baselines, metrics, audits, R6)
```

Everything upstream of rendering is a **pure, deterministic function of
`SimConfig`**. Regenerating from the same config yields byte-identical events; the
`benchmark` and `export-cert` CLI commands rely on this — they re-simulate from the
stored config rather than reloading a giant JSONL.

## Determinism (R7)

A single seed feeds `RNG`, which hands out NumPy generators keyed by a BLAKE2 hash of
`(seed, *name)`. Each concern draws from its own named sub-stream
(`rng.stream("behavior", emp_id, day)`, `rng.stream("psych", emp_id)`, …), so adding
a channel or a playbook does not perturb unrelated draws. This keeps datasets stable
and diffs interpretable across code changes.

## The label-owning bus (R2)

`EventBus` is the single source of truth. Emitters pass it fully-specified, label-free
facts plus a graded `Label`; it stamps a monotonic id and stores the event. Labels are
set once, at emission, by the component that knows the latent state — never inferred
later from the surface text. This is what makes ground truth an *architectural*
guarantee rather than an emergent property of an LLM's behaviour (contrast Chimera,
where the same model both acts and logs).

## Endogenous intent (R1) — the core idea

Instead of injecting a scripted attack, the simulator **grows** intent:

- `org.py` gives every employee a static **predisposition** from OCEAN traits.
- `lifecycle.py` generates **stressors** (negative review, passed-over promotion,
  sanction, financial hardship, resignation) and benign **triggers** (travel,
  on-call, promotion, crunch, approved bulk export).
- `psych.py` integrates these into a per-day latent **risk trajectory** via a
  logit-linear Critical-Pathway model with autocorrelated drift and a
  concerning-behaviour feedback term. For the insider population a latent
  *intent-pressure* ramp (grievance consolidation) is added — latent only, never an
  observable feature, so it creates no shortcut in the telemetry.
- A **hazard model** converts elevated risk into a per-day probability of pathway
  activation. The first success is the onset day; many insiders never activate.

The label of an event therefore reflects *where on the pathway the actor is*
(`benign → anomalous_benign → precursor → malicious`), not which script was run.

## Breaking the signature shortcut (R6)

The failure mode of CERT is that a single rule ("after-hours + removable + upload")
nearly separates the classes. Three mechanisms prevent that here:

1. **Benign power users.** Many roles legitimately upload to cloud, use removable
   media, and work late (`behavior._profile`); the engineered hard negatives do so
   *heavily*. So those features are not markers of malice.
2. **Stealth modulation.** A per-insider stealth level (from conscientiousness and
   the risk-curve shape) pushes covert actors into work hours, small volumes, and
   cloud channels that look benign; only reckless actors use the obvious route.
3. **Variant sampling.** Target asset, channel mix, and staging order are randomised
   per episode, so the positive class is a distribution, not a template.

The shipped **shortcut audit** measures the result: the trivial rule's PR-AUC and the
gap to a full model. A near-zero trivial PR-AUC and a positive gap are the design
goal (sample: 0.009 vs 0.39, gap +0.38).

## Label-blind rendering (R2)

`render/base.collect_plans` builds a `ContentPlan` for each renderable event
containing only neutral business facts (role, team, a *generic* topic, an intent
hint, internal/external, formality). The plan deliberately hides sensitive asset
names behind generic topics and carries **no** label/technique/risk/episode field;
`to_prompt_payload()` is the entire view a renderer gets. A consequence worth stating:
the renderer produces ordinary prose for malicious events too, so text is
non-discriminative and NLP-only detectors cannot keyword-spot intent. The
`render/linter` scans output for label terms and technique ids and redacts leaks, so
even a misbehaving model cannot poison the corpus.

## Observability (R8)

`observability.apply_observability` filters the full stream to the *observed* stream
using per-channel coverage and a global dropout, and reports how much malicious
signal was rendered invisible (`effective_recall_ceiling`). The full stream is always
retained for evaluation. This lets a benchmark separate "the detector missed it" from
"the sensor never saw it" — a distinction CERT and ChimeraLog cannot make.

## Extensibility

- **New domain:** add a role list to `org.DOMAINS` and crown-jewel assets to
  `org.CROWN_JEWELS`.
- **New pathway:** add a `Playbook` of abstract `Stage`s to `threats.PLAYBOOKS`
  (technique-id labels + a `kind` handled by `ThreatEngine._emit_stage`). Keep stages
  at the level of detection observables.
- **New surface:** add a `Channel`, emit it from `behavior`/`threats`, map it in
  `benchmark/features` (observable attributes only) and optionally `export_cert`.
- **New detector:** add a scorer to `benchmark/baselines.BASELINES`.
