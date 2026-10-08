"""State-of-the-art CERT-style ITD detectors, for head-to-head comparison.

The insider-threat literature's strongest results on CERT come from a small set of
recipes, nearly all over the engineered **user-day feature representation**:

* **Gradient-boosted / random-forest classifiers** on user-day features — the
  dominant supervised recipe; reports up to ~0.99 ROC/F1 on CERT r4.2/r5.2.
* **MLP / deep feed-forward** on the same features.
* **Autoencoder reconstruction-error** anomaly detection (trained on benign days) —
  the canonical "deep unsupervised" CERT recipe (DNN/LSTM auto-encoders). We use a
  PCA reconstruction autoencoder as a dependency-free, faithful stand-in for the
  reconstruction-error mechanism (no GPU/torch required here).
* **Temporal / sequence models** (LSTM, transformer encoders, 2024-25 SOTA) — we
  stand in with a gradient-boosted model over the causal per-user sequence features
  from :mod:`synthitd.benchmark.sequence`.

Every detector consumes the exact leakage-safe feature matrix the rest of the
benchmark uses, so the comparison is apples-to-apples. These are reference
reimplementations of the published *recipes*, not the original authors' code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .features import FeatureMatrix
from .sequence import augment_sequence

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


def _scaled(fm, tr, te):
    sc = StandardScaler().fit(fm.X[tr])
    return sc.transform(fm.X[tr]), sc.transform(fm.X[te]), fm.y[tr]


def sota_random_forest(fm: FeatureMatrix, tr, te) -> np.ndarray:
    # trees need no feature scaling — use raw features
    y = fm.y[tr]
    if y.sum() == 0:
        return np.zeros(len(te))
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=0)
    clf.fit(fm.X[tr], y)
    return clf.predict_proba(fm.X[te])[:, 1]


def sota_gbdt(fm: FeatureMatrix, tr, te) -> np.ndarray:
    y = fm.y[tr]
    if y.sum() == 0:
        return np.zeros(len(te))
    # class imbalance via sample weights (HGB has no class_weight)
    w = np.where(y == 1, max(1.0, (len(y) - y.sum()) / max(1, y.sum())), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         l2_regularization=1.0, min_samples_leaf=5,
                                         random_state=0)
    clf.fit(fm.X[tr], y, sample_weight=w)  # trees: raw features
    return clf.predict_proba(fm.X[te])[:, 1]


def sota_mlp(fm: FeatureMatrix, tr, te) -> np.ndarray:
    Xtr, Xte, y = _scaled(fm, tr, te)
    if y.sum() == 0:
        return np.zeros(len(te))
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                        random_state=0)
    # oversample positives so the deep net sees the minority class
    pos = np.where(y == 1)[0]
    if len(pos):
        reps = max(0, int((len(y) - len(pos)) / max(1, len(pos)) / 8))
        idx = np.concatenate([np.arange(len(y))] + [pos] * reps)
        clf.fit(Xtr[idx], y[idx])
    else:
        clf.fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]


def sota_autoencoder(fm: FeatureMatrix, tr, te, n_components: int = 8) -> np.ndarray:
    """Reconstruction-error anomaly detector trained on benign training days.

    A PCA reconstruction stands in for the DNN/LSTM auto-encoder recipe: fit the
    low-rank benign manifold, score each test day by how badly it reconstructs.
    """
    sc = StandardScaler().fit(fm.X[tr])
    Xtr, Xte = sc.transform(fm.X[tr]), sc.transform(fm.X[te])
    benign = Xtr[fm.y[tr] == 0]
    if len(benign) < n_components + 1:
        benign = Xtr
    k = min(n_components, benign.shape[1] - 1, max(1, len(benign) - 1))
    pca = PCA(n_components=k, random_state=0).fit(benign)
    recon = pca.inverse_transform(pca.transform(Xte))
    return np.sqrt(((Xte - recon) ** 2).sum(axis=1))


def sota_seq_gbdt(fm: FeatureMatrix, tr, te) -> np.ndarray:
    """Temporal SOTA proxy: gradient boosting over causal per-user sequence features
    (stand-in for LSTM / transformer-encoder ITD models)."""
    Xa = augment_sequence(fm)   # trees: raw augmented features, no scaling
    y = fm.y[tr]
    if y.sum() == 0:
        return np.zeros(len(te))
    w = np.where(y == 1, max(1.0, (len(y) - y.sum()) / max(1, y.sum())), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         l2_regularization=1.0, min_samples_leaf=5,
                                         random_state=0)
    clf.fit(Xa[tr], y, sample_weight=w)
    return clf.predict_proba(Xa[te])[:, 1]


SOTA_MODELS: dict[str, Callable[[FeatureMatrix, np.ndarray, np.ndarray], np.ndarray]] = {}
if _HAVE_SK:
    SOTA_MODELS.update({
        "sota_random_forest": sota_random_forest,
        "sota_gbdt": sota_gbdt,
        "sota_mlp": sota_mlp,
        "sota_autoencoder": sota_autoencoder,
        "sota_seq_gbdt": sota_seq_gbdt,
    })
