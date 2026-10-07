"""Train/test splits — including the scenario-held-out split prior datasets lack.

* **temporal_split** — train on early days, test on later days. Prevents the random-
  split leakage that inflates CERT numbers (future bleeding into the past).
* **user_holdout_split** — disjoint employees in train vs test; measures whether a
  detector generalises to *people* it never trained on.
* **scenario_holdout_split** — hold out one or more pathway *families* entirely from
  training. This is the hard generalisation test: can a detector flag a kind of
  insider behaviour it has never seen? CERT/Chimera never isolate scenarios this way.
"""

from __future__ import annotations

import numpy as np

from .features import FeatureMatrix


def _mask_indices(mask: np.ndarray) -> np.ndarray:
    return np.where(mask)[0]


def temporal_split(fm: FeatureMatrix, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    cutoff = np.quantile(fm.days, train_frac)
    train = _mask_indices(fm.days <= cutoff)
    test = _mask_indices(fm.days > cutoff)
    return train, test


def user_holdout_split(
    fm: FeatureMatrix, test_frac: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    users = np.array(sorted(set(fm.groups.tolist())))
    g = np.random.default_rng(seed)
    g.shuffle(users)
    n_test = max(1, int(round(test_frac * len(users))))
    test_users = set(users[:n_test].tolist())
    test_mask = np.array([u in test_users for u in fm.groups])
    return _mask_indices(~test_mask), _mask_indices(test_mask)


def scenario_holdout_split(
    fm: FeatureMatrix, holdout_pathways: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Test set = user-days of insiders whose pathway is held out (plus all benign
    users for context); training excludes every user-day of a held-out-pathway user.

    Benign/anomalous-benign rows are split by user id hash so both sides have
    negatives, while every positive of a held-out family appears only at test time.
    """
    hold = set(holdout_pathways)
    is_holdout_user = np.array([p in hold for p in fm.pathway])
    # negatives (no pathway) distributed deterministically 70/30 by user hash
    neg_user_test = np.array(
        [(p == "") and (hash(u) % 10 >= 7) for p, u in zip(fm.pathway, fm.groups)]
    )
    test_mask = is_holdout_user | neg_user_test
    train_mask = ~test_mask
    # ensure no held-out-family positive leaks into train (already guaranteed)
    return _mask_indices(train_mask), _mask_indices(test_mask)
