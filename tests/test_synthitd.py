"""Test suite for synthitd.

Fast fixtures use a 40-employee / 40-day org that is known to activate insiders at
seed 3, so the positive-class invariants and the benchmark can be exercised quickly.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from synthitd.config import SimConfig, load_config, RNG
from synthitd.events import Channel, Label
from synthitd.simulate import simulate
from synthitd.render import render_events
from synthitd.render.base import collect_plans, _FORBIDDEN_PAYLOAD_KEYS
from synthitd.render.linter import lint_and_clean
from synthitd.writers import write_dataset, userday_labels, episodes
from synthitd.export_cert import export_cert
from synthitd.benchmark.features import build_userday_features, DENYLISTED_ATTRS, FEATURE_NAMES
from synthitd.benchmark.audits import shortcut_audit, stylized_facts
from synthitd.benchmark.run import run_benchmark


CFG = SimConfig(name="test", domain="tech", n_employees=40, horizon_days=40,
                insider_prevalence=0.12, seed=3)


@pytest.fixture(scope="module")
def result():
    return simulate(CFG)


# --- reproducibility --------------------------------------------------------
def test_determinism():
    a, b = simulate(CFG), simulate(CFG)
    assert len(a.events) == len(b.events)
    assert [e.event_id for e in a.events[:500]] == [e.event_id for e in b.events[:500]]
    assert a.label_counts(False) == b.label_counts(False)


def test_seed_changes_draw():
    a = simulate(CFG)
    b = simulate(CFG.with_overrides(seed=CFG.seed + 1))
    assert len(a.events) != len(b.events) or a.label_counts() != b.label_counts()


def test_rng_substreams_independent():
    rng = RNG(42)
    s1 = rng.stream("a", 1).random()
    s2 = rng.stream("b", 1).random()
    assert s1 != s2
    assert rng.stream("a", 1).random() != s1  # second draw advances the same stream


# --- population & label invariants -----------------------------------------
def test_insiders_activate(result):
    assert len(result.insiders()) >= 1


def test_malicious_only_from_activated_insiders(result):
    activated = set(result.insiders())
    for ev in result.events:
        if ev.label is Label.MALICIOUS:
            assert ev.actor in activated
            assert ev.episode_id.startswith(f"ep-{ev.actor}-")
            assert ev.technique_ids  # malicious events carry technique labels


def test_graded_labels_present(result):
    labels = set(result.label_counts().keys())
    assert "benign" in labels
    assert "anomalous_benign" in labels
    # precursor/malicious present because fixture activates insiders
    assert "malicious" in labels


def test_benign_events_have_no_technique_ids(result):
    for ev in result.events:
        if ev.label is Label.BENIGN:
            assert ev.technique_ids == []


def test_episodes_consistent(result):
    eps = episodes(result)
    assert len(eps) == len(result.insiders())
    for ep in eps:
        assert ep["onset_day"] is not None
        assert ep["n_malicious_events"] >= 0


# --- label-blind rendering (R2) --------------------------------------------
def test_content_plans_are_label_blind(result):
    plans = collect_plans(result.events, result.org)
    assert plans, "expected some renderable events"
    forbidden = _FORBIDDEN_PAYLOAD_KEYS | {"label", "risk_state", "technique_ids", "episode_id"}
    for p in plans:
        payload = p.to_prompt_payload()
        assert not (set(payload) & forbidden)
        # the payload must not smuggle any label value as a string
        assert "malicious" not in str(payload).lower()
        assert "precursor" not in str(payload).lower()


def test_render_sets_text_and_no_leak(result):
    res = simulate(CFG)  # fresh copy so we can mutate render fields
    rr = render_events(res.events, res.org, res.config)
    assert rr.n_rendered > 0
    assert rr.lint["leak_rate"] == 0.0
    rendered = [e for e in res.events if e.render]
    assert rendered and all(isinstance(v, str) for e in rendered for v in e.render.values())


def test_linter_catches_injected_leak():
    bad = {"e1": {"body": "this is an exfiltration attempt by a malicious insider T1052"}}
    cleaned, report = lint_and_clean(bad, redact=True)
    assert report.flagged == 1
    assert report.redactions == 1
    assert "exfiltrat" not in cleaned["e1"]["body"].lower()


# --- feature leakage safety (R6) -------------------------------------------
def test_feature_names_exclude_denylist():
    assert not (set(FEATURE_NAMES) & DENYLISTED_ATTRS)


def test_features_well_formed(result):
    fm = build_userday_features(result)
    assert fm.X.shape[0] == len(result.org.employees) * result.config.horizon_days
    assert fm.X.shape[1] == len(FEATURE_NAMES)
    assert not np.isnan(fm.X).any()
    assert fm.y.sum() >= 1  # some malicious user-days


def test_denylisted_attr_does_not_leak_into_features(result):
    """Injecting a denylisted flag onto a benign event must not change its features."""
    fm0 = build_userday_features(result)
    # mutate a copy
    res2 = simulate(CFG)
    for ev in res2.events[:200]:
        ev.attrs["out_of_pattern"] = True
        ev.attrs["staging_dir"] = True
    fm1 = build_userday_features(res2)
    assert np.allclose(fm0.X, fm1.X)


# --- observability (R8) -----------------------------------------------------
def test_observability_drops_events():
    cfg = CFG.with_overrides(channel_coverage={"endpoint": 0.0, "http": 0.5}, sensor_dropout=0.1)
    res = simulate(cfg)
    assert len(res.observed) < len(res.events)
    assert res.report.dropped > 0
    # endpoint fully uninstrumented
    assert not any(e.channel is Channel.ENDPOINT for e in res.observed)


# --- CERT export (R9) -------------------------------------------------------
def test_cert_export(result):
    with tempfile.TemporaryDirectory() as d:
        info = export_cert(result, d, observed=True)
        for f in ["logon.csv", "device.csv", "http.csv", "email.csv", "file.csv",
                  "insiders.csv", "labels.csv"]:
            assert os.path.exists(os.path.join(d, f))
        assert os.path.isdir(os.path.join(d, "extended"))
        # row count sanity: logon rows == #logon events in observed
        with open(os.path.join(d, "logon.csv")) as fh:
            rows = sum(1 for _ in fh) - 1
        assert rows == sum(1 for e in result.observed if e.channel is Channel.LOGON)
        with open(os.path.join(d, "insiders.csv")) as fh:
            ins_rows = sum(1 for _ in fh) - 1
        assert ins_rows == len(result.insiders())


# --- writers ----------------------------------------------------------------
def test_write_dataset(result):
    with tempfile.TemporaryDirectory() as d:
        manifest = write_dataset(result, d)
        for f in ["events.jsonl", "observed.jsonl", "employees.json", "episodes.json",
                  "labels_userday.csv", "ground_truth.json", "manifest.json"]:
            assert os.path.exists(os.path.join(d, f))
        rows = userday_labels(result)
        assert len(rows) == len(result.org.employees) * result.config.horizon_days
        assert manifest["summary"]["n_events_full"] == len(result.events)


# --- benchmark & anti-shortcut claim (R6) ----------------------------------
def test_shortcut_audit_defeats_trivial_rule(result):
    # Stable, size-independent claim: the single-rule CERT shortcut does NOT work
    # on this dataset (on CERT it approaches ~1.0). The full-vs-trivial gap is a
    # noisy statistic on a 40x40 toy, so it is asserted at realistic size below.
    audit = shortcut_audit(build_userday_features(result))
    assert "trivial_rule_pr_auc" in audit
    assert audit["trivial_rule_pr_auc"] < 0.2


@pytest.mark.parametrize("seed", [11])
def test_signal_depth_positive_at_scale(seed):
    """At realistic size the full model clearly beats the trivial shortcut (R6)."""
    res = simulate(SimConfig(domain="tech", n_employees=120, horizon_days=70,
                             insider_prevalence=0.08, seed=seed))
    audit = shortcut_audit(build_userday_features(res))
    assert audit["n_positive"] >= 5
    assert audit["full_model_pr_auc"] > audit["trivial_rule_pr_auc"]
    assert audit["signal_depth_pr_gap"] > 0.1


def test_stylized_facts_plausible(result):
    facts = stylized_facts(result)
    assert facts["plausible"] is True
    assert facts["work_hour_concentration"] >= 0.45


def test_benchmark_runs(result):
    rep = run_benchmark(result, budget=0.02)
    assert "temporal" in rep["splits"]
    assert "scenario_holdout" in rep["splits"]
    assert rep["n_malicious_userdays"] >= 1


# --- sequence / temporal baselines -----------------------------------------
def test_sequence_baselines_registered():
    from synthitd.benchmark.baselines import BASELINES
    assert "sequence_logreg" in BASELINES
    assert "ewma_selfbaseline" in BASELINES


def test_sequence_augmentation_causal_and_finite(result):
    from synthitd.benchmark.sequence import augment_sequence, _user_order
    fm = build_userday_features(result)
    Xa = augment_sequence(fm)
    assert Xa.shape[0] == len(fm)
    assert Xa.shape[1] > fm.X.shape[1]          # temporal columns appended
    assert np.isfinite(Xa).all()
    # the earliest day of each user must have zero temporal deviation (no past)
    k = len([n for n in ["file_size_total_kb", "n_file_removable", "n_email_external",
                         "n_http_upload", "http_bytes_total", "after_hours_ratio",
                         "n_file_copy", "n_idp_admin_app"] if n in fm.feature_names])
    base_w = fm.X.shape[1]
    for u, idxs in _user_order(fm).items():
        first = idxs[0]
        assert np.allclose(Xa[first, base_w:base_w + k], 0.0)
        assert Xa[first, -1] == 0.0              # history length 0 on day one


def test_sequence_baselines_score(result):
    from synthitd.benchmark.sequence import sequence_logreg, ewma_selfbaseline
    from synthitd.benchmark.splits import temporal_split
    fm = build_userday_features(result)
    tr, te = temporal_split(fm, 0.7)
    for fn in (sequence_logreg, ewma_selfbaseline):
        s = fn(fm, tr, te)
        assert s.shape[0] == len(te)
        assert np.isfinite(s).all()


# --- SOTA detectors & CERT-regime ablation ---------------------------------
def test_sota_models_score(result):
    from synthitd.benchmark.sota import SOTA_MODELS
    from synthitd.benchmark.splits import temporal_split
    assert SOTA_MODELS, "expected sklearn-backed SOTA models to be registered"
    fm = build_userday_features(result)
    tr, te = temporal_split(fm, 0.7)
    for name, fn in SOTA_MODELS.items():
        s = fn(fm, tr, te)
        assert s.shape[0] == len(te), name
        assert np.isfinite(s).all(), name


def test_legacy_cert_mode_removes_benign_shortcut_sources():
    """The ablation must make upload/removable malicious-only, reproducing CERT's
    signature separability; realistic mode must spread them across benign users."""
    from synthitd.events import Channel, Label
    base = dict(domain="tech", n_employees=40, horizon_days=40,
                insider_prevalence=0.12, seed=3)
    real = simulate(SimConfig(name="real", legacy_cert_mode=False, **base))
    cert = simulate(SimConfig(name="cert", legacy_cert_mode=True, **base))

    def benign_upload_removable(res):
        up = sum(1 for e in res.events if e.action == "http.upload" and e.label is Label.BENIGN)
        rem = sum(1 for e in res.events if e.channel is Channel.FILE
                  and e.attrs.get("removable") and e.label is Label.BENIGN)
        return up, rem

    r_up, r_rem = benign_upload_removable(real)
    c_up, c_rem = benign_upload_removable(cert)
    # realistic has benign power-user uploads/removable; CERT-regime has ~none
    assert r_up > 0 and r_rem > 0
    assert c_up == 0 and c_rem == 0


# --- UBS-Transformer reimplementation (torch optional) ---------------------
def test_ubs_transformer_runs_if_torch_available():
    try:
        import torch  # noqa: F401
    except Exception:
        import pytest as _pytest
        _pytest.skip("PyTorch not installed")
    from synthitd.benchmark.ubs_transformer import (
        run_ubs_transformer, UBSConfig, build_user_sequences)
    res = simulate(SimConfig(domain="tech", n_employees=60, horizon_days=40,
                             insider_prevalence=0.15, seed=3))
    X, users, is_insider = build_user_sequences(res)
    assert X.shape[0] == len(users) == 60
    assert X.shape[1] == 40  # one step per horizon day
    # tiny architecture / few epochs just to exercise the pipeline
    out = run_ubs_transformer(res, UBSConfig(d_model=32, n_layers=1, n_heads=2,
                                             ffn=64, epochs=2, seed=0))
    assert "iforest" in out["detectors"]
    for det, m in out["detectors"].items():
        if "auroc" in m and m["auroc"] == m["auroc"]:  # not NaN
            assert 0.0 <= m["auroc"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0


# --- config -----------------------------------------------------------------
def test_config_roundtrip():
    cfg = SimConfig(name="x", lunch_hours=(12.0, 13.5))
    d = cfg.to_dict()
    assert load_config(d).lunch_hours == (12.0, 13.5)


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        load_config({"nonexistent_key": 1})
