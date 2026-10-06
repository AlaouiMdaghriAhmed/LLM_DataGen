"""Operationally-honest metrics (requirement R6).

Beyond ROC-AUC and AUC-PR at true prevalence, we report the things a SOC actually
cares about and that F1-on-a-balanced-subset hides:

* **precision/recall at a review budget** — if analysts can review the top *b* % of
  user-days, how many insiders do they catch and how much of their queue is noise?
* **detection earliness** — how many days before the act is the insider first
  flagged? (Enabled by the precursor labels and latent risk — unique to this data.)
* **false-positive attribution** — what share of the flagged negatives are the
  engineered *anomalous-benign* hard negatives vs ordinary users?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = roc_auc_score = None  # type: ignore

from .features import FeatureMatrix


@dataclass
class Metrics:
    n_test: int
    n_positive: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    budget: float
    precision_at_budget: float
    recall_at_budget: float
    insiders_detected: int
    insiders_total: int
    median_lead_days: float
    fp_anomalous_benign_share: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_test": self.n_test,
            "n_positive": self.n_positive,
            "prevalence": round(self.prevalence, 6),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "review_budget": self.budget,
            "precision_at_budget": round(self.precision_at_budget, 4),
            "recall_at_budget": round(self.recall_at_budget, 4),
            "insiders_detected": self.insiders_detected,
            "insiders_total": self.insiders_total,
            "median_detection_lead_days": self.median_lead_days,
            "fp_anomalous_benign_share": round(self.fp_anomalous_benign_share, 4),
            **self.extra,
        }


def evaluate(
    fm: FeatureMatrix,
    test_idx: np.ndarray,
    scores: np.ndarray,
    budget: float = 0.01,
) -> Metrics:
    y = fm.y[test_idx]
    s = scores
    n = len(test_idx)
    n_pos = int(y.sum())
    prev = n_pos / max(1, n)

    if roc_auc_score is not None and 0 < n_pos < n:
        roc = float(roc_auc_score(y, s))
        pr = float(average_precision_score(y, s))
    else:
        roc = pr = float("nan")

    # review budget: flag the top-`budget` fraction of user-days by score
    k = max(1, int(round(budget * n)))
    order = np.argsort(-s)
    flagged = order[:k]
    tp = int(y[flagged].sum())
    precision = tp / max(1, k)
    recall = tp / max(1, n_pos)

    # false-positive attribution
    fp_mask = (y[flagged] == 0)
    fp_pop = fm.population[test_idx][flagged][fp_mask]
    n_fp = int(fp_mask.sum())
    fp_ab = float(np.mean(fp_pop == "anomalous_benign")) if n_fp else 0.0

    # earliness: per activated insider in the test set, first flagged day vs onset
    detected, total, leads = _earliness(fm, test_idx, flagged)

    return Metrics(
        n_test=n, n_positive=n_pos, prevalence=prev, roc_auc=roc, pr_auc=pr,
        budget=budget, precision_at_budget=precision, recall_at_budget=recall,
        insiders_detected=detected, insiders_total=total,
        median_lead_days=float(np.median(leads)) if leads else float("nan"),
        fp_anomalous_benign_share=fp_ab,
    )


def _earliness(fm: FeatureMatrix, test_idx: np.ndarray, flagged_local: np.ndarray):
    flagged_global = set(test_idx[flagged_local].tolist())
    groups = fm.groups
    days = fm.days
    onset = fm.onset_day
    # single pass: insider onset per user + earliest flagged day per user
    insider_users: dict[str, int] = {}
    earliest_flag: dict[str, int] = {}
    for gi in test_idx:
        u = groups[gi]
        if onset[gi] >= 0 and fm.pathway[gi] != "":
            insider_users[u] = onset[gi]
        if gi in flagged_global:
            d = days[gi]
            if u not in earliest_flag or d < earliest_flag[u]:
                earliest_flag[u] = d
    leads: list[float] = []
    detected = 0
    for user, on in insider_users.items():
        if user in earliest_flag:
            detected += 1
            leads.append(float(on - earliest_flag[user]))  # >0 => caught before the act
    return detected, len(insider_users), leads
