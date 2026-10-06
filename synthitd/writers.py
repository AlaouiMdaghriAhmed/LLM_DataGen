"""Dataset serialisation: JSONL event streams, CSV, and ground-truth files.

Everything needed to train and evaluate is written to a single output directory:

* ``events.jsonl``         full labelled ground-truth stream
* ``observed.jsonl``       the observed stream after the observability filter
* ``employees.json``       the org (roles, access, OCEAN, predisposition, population)
* ``episodes.json``        one record per activated insider episode (R5)
* ``labels_userday.csv``   per (user, day) graded label + latent risk (R5)
* ``ground_truth.json``    per-employee trajectory summary
* ``manifest.json``        config + summary + render/lint/observability reports
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from .events import Event, Label
from .simclock import WorkCalendar
from .simulate import SimResult


def _dump_jsonl(events: list[Event], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_record(), sort_keys=True))
            fh.write("\n")


def userday_labels(result: SimResult) -> list[dict[str, Any]]:
    """Aggregate events to a per-(user, day) graded label and latent risk target."""
    cfg = result.config
    cal = WorkCalendar(cfg.start_date, cfg.workday_start_hour, cfg.workday_end_hour, cfg.lunch_hours)
    # init grid
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for emp in result.org.employees:
        traj = result.trajectories[emp.emp_id]
        for day in range(cfg.horizon_days):
            key = (emp.emp_id, cal.day_key(cal.day_start_ts(day)))
            rows[key] = {
                "emp_id": emp.emp_id,
                "day": day,
                "date": cal.day_key(cal.day_start_ts(day)),
                "role": emp.role_abbr,
                "population": emp.population,
                "n_events": 0,
                "max_label": Label.BENIGN.value,
                "max_label_rank": 0,
                "risk_state": round(float(traj.risk[day]), 4),
                "is_malicious": 0,
                "is_positive_any": 0,
            }
    for ev in result.events:
        key = (ev.actor, cal.day_key(ev.ts))
        row = rows.get(key)
        if row is None:
            continue
        row["n_events"] += 1
        if ev.label.rank > row["max_label_rank"]:
            row["max_label_rank"] = ev.label.rank
            row["max_label"] = ev.label.value
        if ev.label is Label.MALICIOUS:
            row["is_malicious"] = 1
        if ev.label.rank >= Label.PRECURSOR.rank:
            row["is_positive_any"] = 1
    return list(rows.values())


def episodes(result: SimResult) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for emp_id, traj in result.trajectories.items():
        if traj.onset_day is None:
            continue
        emp = result.org.by_id(emp_id)
        ep_events = [e for e in result.events if e.episode_id == f"ep-{emp_id}-{traj.onset_day}"]
        tids = sorted({t for e in ep_events for t in e.technique_ids})
        out.append({
            "episode_id": f"ep-{emp_id}-{traj.onset_day}",
            "emp_id": emp_id,
            "role": emp.role_title,
            "pathway": traj.pathway,
            "onset_day": traj.onset_day,
            "precursor_start": traj.precursor_start,
            "episode_end": traj.episode_end,
            "n_precursor_events": sum(1 for e in ep_events if e.label is Label.PRECURSOR),
            "n_malicious_events": sum(1 for e in ep_events if e.label is Label.MALICIOUS),
            "technique_ids": tids,
            "predisposition": round(emp.predisposition, 3),
            "risk_at_onset": round(float(traj.risk[traj.onset_day]), 3),
        })
    return out


def write_dataset(result: SimResult, outdir: str, render_result: Any | None = None) -> dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)

    _dump_jsonl(result.events, os.path.join(outdir, "events.jsonl"))
    _dump_jsonl(result.observed, os.path.join(outdir, "observed.jsonl"))

    with open(os.path.join(outdir, "employees.json"), "w", encoding="utf-8") as fh:
        json.dump([e.to_dict() for e in result.org.employees], fh, indent=2)

    eps = episodes(result)
    with open(os.path.join(outdir, "episodes.json"), "w", encoding="utf-8") as fh:
        json.dump(eps, fh, indent=2)

    with open(os.path.join(outdir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump({eid: t.to_dict() for eid, t in result.trajectories.items()}, fh, indent=2)

    rows = userday_labels(result)
    ud_path = os.path.join(outdir, "labels_userday.csv")
    if rows:
        with open(ud_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = {
        "config": result.config.to_dict(),
        "summary": result.summary(),
        "n_episodes": len(eps),
        "technique_ids_in_data": sorted({t for ep in eps for t in ep["technique_ids"]}),
    }
    if render_result is not None:
        manifest["render"] = render_result.to_dict()
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
