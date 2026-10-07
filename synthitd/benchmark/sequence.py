"""Sequence / temporal baselines.

The per-(user, day) matrix is really a set of per-user time series, and insider
pathways *escalate* — the Critical-Pathway risk ramps before the act. These baselines
exploit that structure causally (each row sees only that user's own past), which also
makes them the natural detectors for the earliness metric.

* ``sequence_logreg`` — supervised logistic regression on the base features augmented
  with per-user rolling deviations and short-term escalation (lagged context).
* ``ewma_selfbaseline`` — unsupervised: each user's own exponentially-weighted moving
  average of risk-relevant features; the score is today's positive deviation from it.

Both avoid leakage: rolling/EWMA state is built only from strictly earlier days of
the same user, and labels are never used to form features.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .features import FeatureMatrix

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

# risk-relevant features the temporal baselines track
_SEQ_FEATURES = [
    "file_size_total_kb", "n_file_removable", "n_email_external", "n_http_upload",
    "http_bytes_total", "after_hours_ratio", "n_file_copy", "n_idp_admin_app",
]


def _key_cols(fm: FeatureMatrix) -> list[int]:
    return [fm.feature_names.index(n) for n in _SEQ_FEATURES if n in fm.feature_names]


def _user_order(fm: FeatureMatrix) -> dict:
    by_user: dict[object, list[int]] = defaultdict(list)
    for i in range(len(fm)):
        by_user[fm.groups[i]].append(i)
    for u in by_user:
        by_user[u].sort(key=lambda i: fm.days[i])
    return by_user


def augment_sequence(fm: FeatureMatrix, window: int = 7) -> np.ndarray:
    """Return base features concatenated with causal per-user temporal features.

    Cached on the FeatureMatrix instance so repeated splits reuse it.
    """
    cached = getattr(fm, "_augmented", None)
    if cached is not None:
        return cached
    cols = _key_cols(fm)
    base = fm.X
    k = len(cols)
    extra = np.zeros((len(fm), 2 * k + 1), dtype=float)
    for u, idxs in _user_order(fm).items():
        hist: list[np.ndarray] = []
        for i in idxs:
            cur = base[i, cols]
            if hist:
                recent = np.array(hist[-window:])
                extra[i, :k] = cur - recent.mean(0)        # deviation from own recent mean
                extra[i, k:2 * k] = recent.max(0)          # recent-max context
            extra[i, -1] = len(hist)                        # observed history length
            hist.append(cur)
    out = np.hstack([base, extra])
    try:
        fm._augmented = out  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def sequence_logreg(fm: FeatureMatrix, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    if not _HAVE_SK:
        return ewma_selfbaseline(fm, tr, te)
    Xa = augment_sequence(fm)
    sc = StandardScaler().fit(Xa[tr])
    Xtr, Xte = sc.transform(Xa[tr]), sc.transform(Xa[te])
    y = fm.y[tr]
    if y.sum() == 0 or y.sum() == len(y):
        return np.zeros(len(te))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]


def ewma_selfbaseline(fm: FeatureMatrix, tr: np.ndarray, te: np.ndarray,
                      alpha: float = 0.3) -> np.ndarray:
    cols = _key_cols(fm)
    X = fm.X[:, cols]
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    te_set = set(int(i) for i in te.tolist())
    ewma: dict[object, np.ndarray] = {}
    score_map: dict[int, float] = {}
    for u, idxs in _user_order(fm).items():
        state = None
        for i in idxs:
            z = Z[i]
            if state is None:
                dev = np.zeros_like(z)
                state = z.copy()
            else:
                dev = z - state
                state = alpha * z + (1 - alpha) * state
            if i in te_set:
                score_map[i] = float(np.clip(dev, 0, None).sum())
    return np.array([score_map.get(int(i), 0.0) for i in te])
