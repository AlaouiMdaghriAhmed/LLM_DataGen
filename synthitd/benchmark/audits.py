"""Self-critical audits shipped *with* the dataset (requirement R6).

Two things no prior insider-threat dataset provides about itself:

* **shortcut_audit** — how much of the label can a trivially-leaky rule recover?
  On CERT, an "after-hours + removable + upload" rule alone scores near-perfect,
  which is why the field's numbers don't transfer. We report that rule's AUC here
  and the *gap* to a full-feature model: a small trivial-rule AUC and a large gap
  mean the signal is genuinely behavioural, not a generator fingerprint.
* **stylized_facts** — distributional sanity of the benign background, so realism is
  a measured property rather than a claim.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = roc_auc_score = None  # type: ignore

from ..events import Channel
from ..simclock import WorkCalendar
from ..simulate import SimResult
from .baselines import run_baselines, logreg, volume_rule, _HAVE_SK
from .features import build_userday_features, FeatureMatrix
from .splits import temporal_split


def shortcut_audit(fm: FeatureMatrix | None = None, result: SimResult | None = None) -> dict:
    if fm is None:
        assert result is not None, "pass fm or result"
        fm = build_userday_features(result)
    tr, te = temporal_split(fm, 0.7)
    y = fm.y[te]
    out: dict = {"n_test": int(len(te)), "n_positive": int(y.sum())}
    if roc_auc_score is None or not (0 < y.sum() < len(y)):
        out["note"] = "degenerate test split (no positives) — widen horizon/prevalence"
        return out

    trivial = volume_rule(fm, tr, te)
    out["trivial_rule_roc_auc"] = round(float(roc_auc_score(y, trivial)), 4)
    out["trivial_rule_pr_auc"] = round(float(average_precision_score(y, trivial)), 4)
    if _HAVE_SK:
        full = logreg(fm, tr, te)
        out["full_model_roc_auc"] = round(float(roc_auc_score(y, full)), 4)
        out["full_model_pr_auc"] = round(float(average_precision_score(y, full)), 4)
        out["signal_depth_pr_gap"] = round(out["full_model_pr_auc"] - out["trivial_rule_pr_auc"], 4)
    out["interpretation"] = (
        "Lower trivial_rule_* and a positive signal_depth_pr_gap indicate the dataset "
        "resists the single-rule shortcut that inflates CERT scores."
    )
    return out


def stylized_facts(result: SimResult) -> dict:
    cfg = result.config
    cal = WorkCalendar(cfg.start_date, cfg.workday_start_hour, cfg.workday_end_hour, cfg.lunch_hours)
    day0 = cal.day_start_ts(0)

    per_userday: dict[tuple[str, int], int] = {}
    hour_hist = np.zeros(24)
    chan_counts: dict[str, int] = {}
    after_hours = 0
    weekend = 0
    total = 0
    for ev in result.events:
        total += 1
        day = int((ev.ts - day0) // 86400)
        per_userday[(ev.actor, day)] = per_userday.get((ev.actor, day), 0) + 1
        h = cal.hour_of_ts(ev.ts)
        hour_hist[int(h) % 24] += 1
        if cal.is_after_hours(h):
            after_hours += 1
        if cal.is_weekend(day):
            weekend += 1
        chan_counts[ev.channel.value] = chan_counts.get(ev.channel.value, 0) + 1

    counts = np.array(list(per_userday.values()), dtype=float)
    work_peak = hour_hist[int(cfg.workday_start_hour):int(cfg.workday_end_hour)].sum() / max(1, hour_hist.sum())
    facts = {
        "events_total": total,
        "events_per_userday_mean": round(float(counts.mean()), 2),
        "events_per_userday_p50": float(np.quantile(counts, 0.5)),
        "events_per_userday_p95": float(np.quantile(counts, 0.95)),
        "after_hours_fraction": round(after_hours / max(1, total), 4),
        "weekend_fraction": round(weekend / max(1, total), 4),
        "work_hour_concentration": round(float(work_peak), 4),
        "channel_share": {k: round(v / total, 4) for k, v in sorted(chan_counts.items())},
    }
    facts["plausible"] = bool(
        0.02 <= facts["after_hours_fraction"] <= 0.45
        and facts["work_hour_concentration"] >= 0.45
        and facts["weekend_fraction"] <= 0.15
    )
    return facts
