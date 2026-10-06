# Literature Review & Gap Analysis

*Scope: synthetic data for **insider-threat detection (ITD)** research. This document
motivates the design of the generator in this repository. It is a defensive /
academic artifact: it models behaviour at the level of **detection-relevant
observables** and **MITRE ATT&CK technique labels**, not operational tradecraft.*

---

## 1. Why the field is dataset-bound

Insider-threat detection is one of the few security problems where the *data*,
not the *model*, is the bottleneck. Real incidents are rare, legally radioactive,
and privacy-laden, so almost the entire published literature trains and evaluates
on a single family of **synthetic** corpora produced by CERT/SEI. A decade of
systematic review now shows that the resulting performance numbers are inflated and
do not transfer to operational settings (Jurišić & Tomičić, *The CERT Dataset
Decade*, 2026). Any contribution to the field has to start by being honest about
what is wrong with the data everyone uses.

---

## 2. The incumbent: CERT / ExactData

**Glasser & Lindauer (2013), "Bridging the Gap: A Pragmatic Approach to Generating
Insider Threat Data"** (IEEE S&P Workshops) describes the generator behind the CERT
Insider Threat Test Dataset (CERT Division + ExactData, DARPA I2O). It composes a
topic model, a social graph, per-user psychometrics (OCEAN), and an LDAP/role model
to emit multi-modal activity logs (logon, device, file, email, HTTP, plus, in later
releases, psychometric and LDAP snapshots). The authors are explicit that *realism
is bounded*: the background is statistical, the malicious scenarios are hand-scripted
and few, and "realism" is defined operationally (does it fool a detector / an
analyst) rather than being guaranteed.

Widely used releases:

| Release | Users | Insiders | Density | Typical use |
|---|---|---|---|---|
| r4.2 | 1,000 | 70 user-scenario instances | "dense" | most-cited; many positives |
| r5.2 | 2,000 | 99 | medium | 4 scenario types |
| r6.2 | 4,000 | 5 | "sparse" (~0.1% positive) | realistic prevalence |

### Documented shortcomings (synthesising the review literature)

1. **Signature leakage / shortcut learning.** Malicious users were scripted from a
   handful of templates, so a detector can learn the *generator's* artifacts
   (e.g. "after-hours logon + removable-device + upload" co-occurrence, a specific
   website) rather than genuine malicious intent. This is the root cause of the
   near-perfect F1 numbers.
2. **Performance-reporting bias.** The same model reports ~0.99 on a balanced r6.2
   subset but ~0.68 on the full 4,000-user population; papers quote the former.
   Random (non-temporal) splits leak future into past.
3. **No cross-artifact ground truth.** Artifacts are emitted independently with no
   factual-constraint engine, so a Slack thread and a ticket about "the same project"
   cannot be guaranteed consistent — because those modern artifacts do not exist in
   the data at all.
4. **Pre-LLM modalities only.** Logon/device/file/email/HTTP. No chat, tickets,
   code review, docs, IdP/SSO, or DLP — the surfaces where modern insider activity
   actually leaves traces.
5. **Thin psychology.** OCEAN scores are static per user; there is no *trajectory*
   (predisposition → stressor → concerning behaviour → act) and no benign
   explanation for anomalies, so "anomaly" collapses onto "malicious".
6. **Static and unversioned.** You cannot regenerate at a new org size, prevalence,
   or noise level; the scenarios never change, so the field overfits them.
7. **Weak hard negatives.** Benign users who are *legitimately* anomalous (travel,
   promotion, re-org, on-call, approved bulk export) are largely absent, so false-
   positive behaviour is untested.

---

## 3. The state of the art this work must beat

### 3.1 Chimera / ChimeraLog (Yu et al., NDSS 2026)

**Design.** An LLM multi-agent framework: each employee is an agent with a role,
MBTI/personality, tools, and an email/task system; agents hold weekly planning
meetings, self-schedule days, and act. A separate attack scheduler injects a chosen
scenario into a chosen agent's day, mapping steps to MITRE ATT&CK. Logs are captured
at the *system* level (sysdig `.scap` syscalls, `tcpdump` `.pcap`) plus structured
logon/file/HTTP/email. Three domains (tech, finance, healthcare), ~20 employees,
one month, 15 scenarios (12 individual + 3 hybrid).

**Results that set the bar.** Human experts rated realism 4.20/5 (vs TWOS 4.25, CERT
1.78). Classifiers that score ~0.99 F1 on CERT drop to ~0.83 on ChimeraLog — i.e.
ChimeraLog is a *harder, less-leaky* benchmark.

**Remaining gaps (where a new contribution can live).** From the paper and released
code (`fish98/Chimera`):

- **No endogenous motivation.** The attacker agent is *told* "you are the ATTACKER,
  your goal is X, your steps are `how`" and asked to weave those steps into its
  schedule (see `src/daily_attack_schedule.py`). The malice is injected, not *grown*
  from the person's psychology and situation. There is no predisposition → stressor →
  pathway. So the label still (partly) corresponds to "an agent was handed an attack
  script", a subtler version of CERT's signature problem.
- **Label granularity = the injected steps.** Ground truth is the scheduled "Attack:
  True" activities. There is no separate, latent *intent / risk state* to evaluate
  early-warning or risk-scoring against, and no graded label (benign-anomalous vs
  pre-attack vs attack).
- **Benign anomalies under-modelled.** Realism is judged by plausibility, but there
  is no deliberately *constructed* population of benign-but-anomalous users to
  measure precision/FP attribution.
- **Cost & reproducibility.** Every simulated day consumes substantial LLM tokens for
  *every* agent's *every* action (syscall-level). Regenerating at 1,000+ users for a
  month is expensive; the realism depends on the serving model.
- **Determinism / verifiability.** Because the LLM drives behaviour *and* logging,
  cross-artifact factual consistency is emergent, not guaranteed.

### 3.2 OrgForge-IT (2026) — the complementary school

A **deterministic Python engine owns a `SimEvent` ground-truth bus**; LLMs render
only surface prose. This makes cross-artifact consistency and labels an
*architectural guarantee*. It adds modern surfaces (Slack/JIRA/Confluence, IdP),
three threat classes (negligent / disgruntled / malicious) and eight injectable
behaviours (secret-in-commit, unusual-hours access, excessive repo cloning,
sentiment drift, cross-dept snooping, exfil email, host data hoarding, social
engineering, IdP anomaly), with a multi-model detection leaderboard. It is small by
design (51 days, ~2,904 records at 96.4% noise) and aimed at **LLM-as-detector**
triage, not at large-scale classical ITD.

**Takeaway.** OrgForge-IT fixes verifiability and modern surfaces but is small and
leaderboard-oriented; Chimera fixes realism but is injection-labelled, syscall-heavy,
and costly. **Neither grows malice endogenously from validated behavioural theory,
and neither ships a benign-anomaly hard-negative population with a leakage/earliness
benchmark.** That is the opening.

### 3.3 Other relevant threads

- **MOLE (2026)** reframes "insider" as *AI agents* operating shared services (150
  accounts, 9 services, 12 threats, monitors under review budgets). Confirms the
  field's move toward **budget-aware, review-cost metrics** — not just F1.
- **LLM-as-detector / zero-shot** work (structured risk indicators on CERT r5.2 +
  RAG over per-user history; ethically-grounded syslog synthesis with Claude) shows
  demand for (a) per-user behavioural *context* and (b) *graded risk indicators*,
  both of which a dataset should expose as first-class fields.
- **Behavioural theory.** SOFIT (Greitzer et al.) — 300+ socio-technical factors;
  the **Critical Pathway to Insider Risk** (Shaw & Sellers, 2015) — predispositions →
  stressors → concerning behaviours → maladaptive org response → act; MITRE CTID
  **Insider Threat TTP Knowledge Base v2** — 50+ technical techniques plus
  *Observable Human Indicators*. These give us a *validated generative story* for
  intent that CERT/Chimera only gesture at.

---

## 4. Gap matrix

| Capability | CERT r6.2 | ChimeraLog | OrgForge-IT | **This work** |
|---|---|---|---|---|
| Deterministic ground-truth bus | ✗ | ✗ (emergent) | ✓ | ✓ |
| Modern surfaces (chat/ticket/VCS/IdP/DLP) | ✗ | partial | ✓ | ✓ |
| **Endogenous intent** (critical-pathway latent state) | ✗ | ✗ (injected) | ✗ (injected) | **✓** |
| **Graded labels** (benign / anomalous-benign / pre-attack / attack) | ✗ | ✗ | partial | **✓** |
| **Benign-anomaly hard negatives w/ justifying evidence** | ✗ | ✗ | partial | **✓** |
| **Sensor degradation / missingness model** | ✗ | ✗ | ✗ | **✓** |
| Regeneratable at arbitrary size/prevalence/noise | ✗ | expensive | ✓ | ✓ |
| Cost-bounded LLM use (cache + batch + template fallback) | n/a | ✗ | partial | ✓ |
| **Leakage / shortcut audit shipped with data** | ✗ | ✗ | ✗ | **✓** |
| Earliness & review-budget metrics | ✗ | ✗ | partial | **✓** |
| CERT-compatible export (drop-in for existing pipelines) | n/a | ✗ | ✗ | **✓** |

---

## 5. Shortcomings → design requirements

- **R1 — Grow intent, don't inject it.** A latent per-user risk trajectory driven by
  a Critical-Pathway state machine (predisposition, stressors, concerning behaviours)
  gates whether/when/how a pathway activates. The *label* is the latent state, so
  detectors are scored against intent, not against a script fingerprint. (→ §2.1, §3.1)
- **R2 — Deterministic bus, LLM only renders.** A typed event bus owns identities,
  timestamps, causal links, and labels; the LLM backend renders only free-text
  surfaces and is **never shown the label**, so prose cannot leak it. (→ §3.2)
- **R3 — Modern multi-surface telemetry.** logon/IdP, file/DLP, email, chat, tickets,
  VCS, HTTP, endpoint — each as a typed schema. (→ §2.4)
- **R4 — Engineered hard negatives.** A benign-anomaly population (travel, promotion,
  re-org, on-call, approved bulk jobs) that *looks* malicious but carries explicit
  benign justifying evidence. Precision and FP-attribution become measurable. (→ §2.7)
- **R5 — Graded, multi-resolution labels.** Per event, per user-day, and per episode:
  `benign / anomalous_benign / precursor / malicious`, plus technique IDs and the
  latent risk score. Enables early-warning and risk-ranking tasks. (→ §3.1, §3.3)
- **R6 — Honest evaluation.** Temporal, user-held-out, and **scenario-held-out**
  splits; AUC-PR at realistic prevalence; detection **earliness**; review-budget
  recall; FP attribution to benign-anomaly type. A **shortcut audit** that tries to
  predict the label from generator-only features ships *with* the data. (→ §2.1, §2.2)
- **R7 — Reproducible & affordable.** Seeded RNG; regenerate at any org size,
  prevalence, horizon, noise; a deterministic template renderer so the whole pipeline
  runs with **zero API cost**, with Claude rendering as a drop-in quality upgrade
  (structured output, prompt caching, Batch API). (→ §3.1 cost)
- **R8 — Sensor realism.** Explicit observability layer: per-channel coverage,
  latency, and missingness, so detectors face realistic partial visibility. (→ new)
- **R9 — Interoperable.** A CERT r6.2-compatible exporter so existing ITD code can
  consume the new data unchanged, making the harder benchmark adoptable. (→ §4)

---

## 6. Stated contribution

A reproducible generator that (1) **grows** insider risk from a validated
Critical-Pathway model instead of injecting scripted attacks, so labels track latent
*intent*; (2) separates a **deterministic, label-owning event bus** from a
**label-blind LLM renderer**, giving verifiable ground truth without prose leakage;
(3) ships an **engineered benign-anomaly hard-negative population**, **graded
multi-resolution labels**, and a **sensor-degradation model**; and (4) ships its own
**honest benchmark** — scenario-held-out splits, earliness and review-budget metrics,
and a **leakage/shortcut audit** that quantifies how much of the signal is a
generator artifact. A CERT-compatible exporter makes it a drop-in for the existing
pipeline the field is trying to escape.

---

## 7. Key references

- Glasser, J. & Lindauer, B. (2013). *Bridging the Gap: A Pragmatic Approach to
  Generating Insider Threat Data.* IEEE S&P Workshops, 98–104.
- Yu, J. et al. (2026). *Chimera: Harnessing Multi-Agent LLMs for Automatic Insider
  Threat Simulation.* NDSS 2026. arXiv:2508.07745. Code: `github.com/fish98/Chimera`.
- *OrgForge-IT: A Verifiable Synthetic Benchmark for LLM-Based Insider Threat
  Detection.* (2026) arXiv:2603.22499.
- *MOLE: Detecting Insider Threats in AI Agents.* (2026) arXiv:2609.06966.
- Jurišić, M. & Tomičić, I. (2026). *The CERT Dataset Decade: A Systematic Review of
  Methodological Evolution and Performance Bias.*
- Greitzer, F. et al. *SOFIT: Sociotechnical and Organizational Factors for Insider
  Threat.* IEEE S&P Workshops (WRIT) 2018.
- Shaw, E. & Sellers, L. (2015). *Application of the Critical-Path Method to Evaluate
  Insider Risks.* Studies in Intelligence.
- MITRE Center for Threat-Informed Defense. *Insider Threat TTP Knowledge Base v2.*
- Camargo et al. *CERT Insider Threat Test Dataset* (r4.2/r5.2/r6.2).
- Harilal et al. (2017). *TWOS: A Dataset of Malicious Insider Behavior from a
  Gamified Competition.*
