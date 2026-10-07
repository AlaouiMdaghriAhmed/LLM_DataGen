"""Faithful reimplementation of the UBS-Transformer insider-threat detector.

Reference: "Enhancing Insider Threat Detection Using User-Based Sequencing and
Transformer Encoders" (arXiv:2506.23446, 2025) — the best-reported recent SOTA on
CERT with a concretely specified architecture. Reported (Test-4, combined
r4.2/r5.2/r6.2, Transformer + iForest on reconstruction errors): accuracy 96.61%,
precision 93.51%, recall 99.43%, F1 96.38%, AUROC 95.00% (FNR 0.0057, FPR 0.0571).

Architecture reproduced here:
* **User-Based Sequencing (UBS):** each user becomes one ordered sequence of daily
  behavioural feature vectors (the paper used 9 sessions/day × 501 days of 35-feature
  session vectors; we use this repo's leakage-safe user-day feature vectors as the
  per-step tokens — the same "one sequence per user" formulation at day granularity).
* **Transformer encoder autoencoder:** linear input embedding -> positional encoding
  -> ``nn.TransformerEncoder`` (default 6 layers, d_model 512, 8 heads, FFN 2048,
  dropout 0.1) -> linear head back to feature dim, trained with **MSE reconstruction
  loss on NORMAL users only** (unsupervised).
* **Outlier detection on reconstruction errors:** per-user reconstruction-error
  summary features are scored by One-Class SVM, Local Outlier Factor, and Isolation
  Forest (the paper's three detectors); the continuous score gives AUROC and the
  threshold-based accuracy/precision/recall/F1.
* **Per-user evaluation** (benign vs insider), matching the paper's user-level labels.

This is a reference reimplementation of the *recipe/architecture*, adapted to this
repository's day-granularity telemetry; it is not the authors' code or hyper-tuning.
Requires PyTorch (optional dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..simulate import SimResult
from .features import build_userday_features, FeatureMatrix

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
)


@dataclass
class UBSConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn: int = 2048
    dropout: float = 0.1
    epochs: int = 40
    lr: float = 5e-4
    batch_size: int = 64
    seed: int = 0
    train_frac_normal: float = 0.7   # normal users used to train the reconstructor


def build_user_sequences(result: SimResult, observed: bool = True):
    """Return (X[n_users,T,F], user_ids, is_insider[bool]) via user-based sequencing."""
    fm: FeatureMatrix = build_userday_features(result, observed=observed)
    users = []
    seen = set()
    for u in fm.groups:
        if u not in seen:
            seen.add(u)
            users.append(u)
    T = int(result.config.horizon_days)
    F = fm.X.shape[1]
    idx_by_user = {u: np.zeros((T, F), dtype=np.float32) for u in users}
    for i in range(len(fm)):
        d = int(fm.days[i])
        if 0 <= d < T:
            idx_by_user[fm.groups[i]][d] = fm.X[i]
    X = np.stack([idx_by_user[u] for u in users])
    activated = set(result.insiders())
    is_insider = np.array([u in activated for u in users], dtype=bool)
    return X, users, is_insider


if _HAVE_TORCH:

    class _PosEnc(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1024):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerReconstructor(nn.Module):
        """Encoder-only Transformer autoencoder (reconstruct the input sequence)."""

        def __init__(self, n_features: int, cfg: UBSConfig):
            super().__init__()
            self.embed = nn.Linear(n_features, cfg.d_model)
            self.pos = _PosEnc(cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn,
                dropout=cfg.dropout, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.head = nn.Linear(cfg.d_model, n_features)

        def forward(self, x):
            h = self.pos(self.embed(x))
            h = self.encoder(h)
            return self.head(h)


def _train_reconstructor(Xn: np.ndarray, cfg: UBSConfig):
    torch.manual_seed(cfg.seed)
    device = "cpu"
    n_features = Xn.shape[2]
    model = TransformerReconstructor(n_features, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    model.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            bi = perm[i : i + cfg.batch_size]
            xb = Xt[bi]
            opt.zero_grad()
            recon = model(xb)
            loss = lossf(recon, xb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(bi)
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    [ubs] epoch {ep + 1}/{cfg.epochs} recon_mse={total / n:.4f}", flush=True)
    model.eval()
    return model


def _per_day_error(model, X: np.ndarray, batch: int = 64) -> np.ndarray:
    """Per-(user, day) reconstruction MSE -> array [n_users, T]."""
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i : i + batch], dtype=torch.float32)
            recon = model(xb)
            e = ((recon - xb) ** 2).mean(dim=2)  # [b, T]
            errs.append(e.numpy())
    return np.concatenate(errs, axis=0)


def _user_error_features(day_err: np.ndarray) -> np.ndarray:
    """Summarise each user's daily reconstruction errors (the paper aggregates per
    user before outlier detection)."""
    return np.stack([
        day_err.mean(axis=1),
        day_err.max(axis=1),
        np.percentile(day_err, 95, axis=1),
        day_err.std(axis=1),
        np.sort(day_err, axis=1)[:, -5:].mean(axis=1),  # mean of top-5 worst days
    ], axis=1)


def _best_f1_threshold(y, score):
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / np.maximum(1, tp + fn)
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    k = int(np.argmax(f1))
    return score[order][k], float(f1[k])


def _eval_scores(y_test: np.ndarray, score: np.ndarray) -> dict:
    auroc = float(roc_auc_score(y_test, score)) if 0 < y_test.sum() < len(y_test) else float("nan")
    thr, _ = _best_f1_threshold(y_test, score)
    pred = (score >= thr).astype(int)
    return {
        "auroc": round(auroc, 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }


def run_ubs_transformer(result: SimResult, cfg: UBSConfig | None = None,
                        observed: bool = True) -> dict:
    """Train on normal users, score held-out users, evaluate per user (paper protocol)."""
    if not _HAVE_TORCH:
        raise RuntimeError("PyTorch is required for the UBS-Transformer reimplementation")
    cfg = cfg or UBSConfig()
    X, users, is_insider = build_user_sequences(result, observed=observed)

    # standardise features using NON-insider day vectors
    normal_mask = ~is_insider
    scaler = StandardScaler().fit(X[normal_mask].reshape(-1, X.shape[2]))
    Xs = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape).astype(np.float32)

    rng = np.random.default_rng(cfg.seed)
    normal_idx = np.where(normal_mask)[0]
    rng.shuffle(normal_idx)
    n_tr = int(cfg.train_frac_normal * len(normal_idx))
    train_normal = normal_idx[:n_tr]
    test_idx = np.concatenate([normal_idx[n_tr:], np.where(is_insider)[0]])

    model = _train_reconstructor(Xs[train_normal], cfg)

    err_train = _per_day_error(model, Xs[train_normal])
    err_test = _per_day_error(model, Xs[test_idx])
    feat_train = _user_error_features(err_train)
    feat_test = _user_error_features(err_test)
    sc = StandardScaler().fit(feat_train)
    ftr, fte = sc.transform(feat_train), sc.transform(feat_test)
    y_test = is_insider[test_idx].astype(int)

    out = {
        "n_users": len(users),
        "n_train_normal": int(len(train_normal)),
        "n_test_users": int(len(test_idx)),
        "n_test_insiders": int(y_test.sum()),
        "detectors": {},
        "config": vars(cfg),
    }

    # raw reconstruction-error (max-day) score — the model's own signal
    out["detectors"]["recon_error_only"] = _eval_scores(y_test, feat_test[:, 1])

    # Isolation Forest on reconstruction-error features (paper's best combo)
    iforest = IsolationForest(n_estimators=300, random_state=cfg.seed).fit(ftr)
    out["detectors"]["iforest"] = _eval_scores(y_test, -iforest.score_samples(fte))

    # One-Class SVM
    ocsvm = OneClassSVM(gamma="scale", nu=0.1).fit(ftr)
    out["detectors"]["ocsvm"] = _eval_scores(y_test, -ocsvm.decision_function(fte))

    # Local Outlier Factor (novelty mode)
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(ftr) - 1), novelty=True).fit(ftr)
        out["detectors"]["lof"] = _eval_scores(y_test, -lof.decision_function(fte))
    except Exception as exc:  # pragma: no cover
        out["detectors"]["lof"] = {"error": str(exc)}

    return out
