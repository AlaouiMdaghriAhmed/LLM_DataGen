"""Leakage-safe per-(user, day) feature extraction.

Features are built **only** from the observed stream and **only** from attributes a
real sensor could legitimately report. Generator-internal flags that trivially encode
the label (``out_of_pattern``, ``staging_dir``, ``record_modification``,
``critical_asset``, ``out_of_role``, ``justification_ref``, ``risk_state``, ...) are
excluded by an explicit denylist, so a baseline cannot cheat. The legitimate
contextual exoneration signal — the observable HR context event — *is* included, so a
good detector can learn to down-weight explained anomalies (requirement R4/R6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..events import Channel, Event, Label
from ..simclock import WorkCalendar
from ..simulate import SimResult


# attributes a detector may never see (they encode the answer)
DENYLISTED_ATTRS = {
    "out_of_pattern", "out_of_role", "elevated", "staging_dir", "record_modification",
    "critical_asset", "out_of_change_window", "justification_ref", "breadth",
    "fields_changed", "reversible",
}

FEATURE_NAMES: list[str] = [
    "n_events", "after_hours_ratio", "weekend",
    "n_logon", "n_idp", "distinct_geo_remote",
    "n_email", "n_email_external", "email_attach_total", "email_size_total_kb",
    "n_chat", "n_http", "n_http_upload", "http_filehost", "http_bytes_total",
    "n_file", "n_file_copy", "n_file_removable", "file_size_total_kb", "n_file_delete",
    "n_vcs", "vcs_files_total", "n_ticket", "n_endpoint", "n_device", "device_unapproved",
    "n_idp_admin_app", "hr_context_present", "hr_negative_present",
    "privilege_level",
]


@dataclass
class FeatureMatrix:
    X: np.ndarray                 # (n_rows, n_features)
    y: np.ndarray                 # binary malicious label per row
    y_graded: np.ndarray          # 0..3 graded label rank per row
    groups: np.ndarray            # emp_id per row (for user-held-out CV)
    days: np.ndarray              # day index per row
    population: np.ndarray        # population string per row
    onset_day: np.ndarray         # onset day of that user (or -1) per row
    pathway: np.ndarray           # pathway family per row ("" if none)
    risk: np.ndarray              # latent risk per row (for earliness eval only)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __len__(self) -> int:
        return self.X.shape[0]


def _safe(attrs: dict, key: str, default=0):
    if key in DENYLISTED_ATTRS:
        return default
    return attrs.get(key, default)


def build_userday_features(result: SimResult, observed: bool = True) -> FeatureMatrix:
    cfg = result.config
    cal = WorkCalendar(cfg.start_date, cfg.workday_start_hour, cfg.workday_end_hour, cfg.lunch_hours)
    stream = result.observed if observed else result.events

    # accumulate per (emp, day)
    acc: dict[tuple[str, int], dict[str, float]] = {}
    maxlabel: dict[tuple[str, int], int] = {}

    def row(emp_id: str, day: int) -> dict[str, float]:
        key = (emp_id, day)
        r = acc.get(key)
        if r is None:
            r = {name: 0.0 for name in FEATURE_NAMES}
            acc[key] = r
            maxlabel[key] = 0
        return r

    day0 = cal.day_start_ts(0)
    for ev in stream:
        day = int((ev.ts - day0) // 86400)
        if day < 0 or day >= cfg.horizon_days:
            continue
        key = (ev.actor, day)
        r = row(ev.actor, day)
        r["n_events"] += 1
        hour = cal.hour_of_ts(ev.ts)
        if cal.is_after_hours(hour):
            r["after_hours_ratio"] += 1  # converted to ratio below
        if maxlabel[key] < ev.label.rank:
            maxlabel[key] = ev.label.rank
        a = ev.attrs
        ch = ev.channel
        if ch is Channel.LOGON:
            r["n_logon"] += 1
            if a.get("geo") == "remote":
                r["distinct_geo_remote"] = 1
        elif ch is Channel.IDP:
            r["n_idp"] += 1
            if a.get("app") == "admin_console":
                r["n_idp_admin_app"] += 1
        elif ch is Channel.EMAIL:
            r["n_email"] += 1
            if not a.get("to_internal", True) or a.get("external_domain"):
                r["n_email_external"] += 1
            r["email_attach_total"] += float(a.get("n_attachments", 0) or 0)
            r["email_size_total_kb"] += float(a.get("size_kb", 0) or 0)
        elif ch is Channel.CHAT:
            r["n_chat"] += 1
        elif ch is Channel.HTTP:
            r["n_http"] += 1
            if ev.action.endswith("upload"):
                r["n_http_upload"] += 1
            if a.get("category") == "filehost":
                r["http_filehost"] += 1
            r["http_bytes_total"] += float(a.get("bytes", 0) or 0)
        elif ch is Channel.FILE:
            r["n_file"] += 1
            if ev.action.endswith("copy"):
                r["n_file_copy"] += 1
            if ev.action.endswith("delete"):
                r["n_file_delete"] += 1
            if a.get("removable"):
                r["n_file_removable"] += 1
            r["file_size_total_kb"] += float(a.get("size_kb", 0) or 0)
        elif ch is Channel.VCS:
            r["n_vcs"] += 1
            r["vcs_files_total"] += float(a.get("n_files", 0) or 0)
        elif ch is Channel.TICKET:
            r["n_ticket"] += 1
        elif ch is Channel.ENDPOINT:
            r["n_endpoint"] += 1
        elif ch is Channel.DEVICE:
            r["n_device"] += 1
            if a.get("approved") is False:
                r["device_unapproved"] += 1
        elif ch is Channel.HR:
            if ev.action in ("hr.context",):
                r["hr_context_present"] = 1
            elif ev.action in ("hr.negative_review", "hr.formal_sanction",
                               "hr.passed_over_promotion", "hr.demotion",
                               "hr.resignation_notice", "hr.interpersonal_conflict"):
                r["hr_negative_present"] = 1

    # finalise per-row vectors
    rows = []
    y, yg, groups, days, pop, onset, pathway, risk = [], [], [], [], [], [], [], []
    for emp in result.org.employees:
        traj = result.trajectories[emp.emp_id]
        for day in range(cfg.horizon_days):
            key = (emp.emp_id, day)
            r = acc.get(key)
            if r is None:
                r = {name: 0.0 for name in FEATURE_NAMES}
            # convert after-hours count to ratio
            n = max(1.0, r["n_events"])
            r["after_hours_ratio"] = r["after_hours_ratio"] / n
            r["weekend"] = 1.0 if cal.is_weekend(day) else 0.0
            r["privilege_level"] = float(emp.privilege)
            rows.append([r[name] for name in FEATURE_NAMES])
            lab = maxlabel.get(key, 0)
            y.append(1 if lab == Label.MALICIOUS.rank else 0)
            yg.append(lab)
            groups.append(emp.emp_id)
            days.append(day)
            pop.append(emp.population)
            onset.append(traj.onset_day if traj.onset_day is not None else -1)
            pathway.append(traj.pathway)
            risk.append(float(traj.risk[day]))

    return FeatureMatrix(
        X=np.asarray(rows, dtype=float),
        y=np.asarray(y, dtype=int),
        y_graded=np.asarray(yg, dtype=int),
        groups=np.asarray(groups, dtype=object),
        days=np.asarray(days, dtype=int),
        population=np.asarray(pop, dtype=object),
        onset_day=np.asarray(onset, dtype=int),
        pathway=np.asarray(pathway, dtype=object),
        risk=np.asarray(risk, dtype=float),
    )
