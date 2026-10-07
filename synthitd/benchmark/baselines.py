"""Reference detectors. Intentionally simple — the point is the *benchmark*, not SOTA.

Each baseline maps (train X/y, test X) -> per-test-row anomaly/risk scores. They span
the usual ITD families: a supervised linear model, an unsupervised anomaly detector,
a per-user z-score peer/self baseline, and a trivial volume rule used by the shortcut
audit as a floor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

from .features import FeatureMatrix


def _scale(train_X, test_X):
    sc = StandardScaler().fit(train_X)
    return sc.transform(train_X), sc.transform(test_X)


def logreg(fm, tr, te) -> np.ndarray:
    Xtr, Xte = _scale(fm.X[tr], fm.X[te])
    y = fm.y[tr]
    if y.sum() == 0 or y.sum() == len(y):
        return np.zeros(len(te))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]


def isolation_forest(fm, tr, te) -> np.ndarray:
    Xtr, Xte = _scale(fm.X[tr], fm.X[te])
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=0)
    clf.fit(Xtr)
    return -clf.score_samples(Xte)  # higher = more anomalous


def user_zscore(fm: FeatureMatrix, tr, te) -> np.ndarray:
    """Per-user self-baseline: z-score of today's feature vector vs that user's own
    history, summed over a few exfil-relevant features. A classic ITD heuristic."""
    names = fm.feature_names
    cols = [names.index(n) for n in
            ["file_size_total_kb", "n_file_removable", "n_email_external",
             "n_http_upload", "http_bytes_total", "after_hours_ratio"]
            if n in names]
    # build per-user mean/std from training rows
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for u in set(fm.groups[tr].tolist()):
        rows = fm.X[tr][fm.groups[tr] == u][:, cols]
        if len(rows) >= 3:
            stats[u] = (rows.mean(0), rows.std(0) + 1e-6)
    out = np.zeros(len(te))
    glob_mean = fm.X[tr][:, cols].mean(0)
    glob_std = fm.X[tr][:, cols].std(0) + 1e-6
    for i, gi in enumerate(te):
        u = fm.groups[gi]
        mean, std = stats.get(u, (glob_mean, glob_std))
        z = (fm.X[gi, cols] - mean) / std
        out[i] = float(np.clip(z, 0, None).sum())
    return out


def volume_rule(fm: FeatureMatrix, tr, te) -> np.ndarray:
    """Trivial 'after-hours + removable + upload' rule — the CERT shortcut, as a floor."""
    names = fm.feature_names
    ah = fm.X[:, names.index("after_hours_ratio")]
    rem = fm.X[:, names.index("n_file_removable")]
    up = fm.X[:, names.index("n_http_upload")]
    score = ah + rem + up
    return score[te]


BASELINES: dict[str, Callable[[FeatureMatrix, np.ndarray, np.ndarray], np.ndarray]] = {
    "volume_rule": volume_rule,
    "user_zscore": user_zscore,
}
if _HAVE_SK:
    BASELINES["logreg"] = logreg
    BASELINES["isolation_forest"] = isolation_forest

# temporal / sequence baselines (registered here to keep a single registry; the
# sequence module imports only from .features, so there is no import cycle).
from .sequence import ewma_selfbaseline, sequence_logreg  # noqa: E402

BASELINES["ewma_selfbaseline"] = ewma_selfbaseline
if _HAVE_SK:
    BASELINES["sequence_logreg"] = sequence_logreg


def run_baselines(fm: FeatureMatrix, tr: np.ndarray, te: np.ndarray) -> dict[str, np.ndarray]:
    return {name: fn(fm, tr, te) for name, fn in BASELINES.items()}
