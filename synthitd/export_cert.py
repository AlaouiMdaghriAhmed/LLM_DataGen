"""CERT r6.2-compatible exporter (requirement R9).

Maps the modern multi-surface stream onto the classic CERT activity CSVs so existing
insider-threat pipelines can consume this harder dataset unchanged. The five core
CERT files (logon, device, http, email, file) are produced with CERT-style columns
and timestamps; the modern-only surfaces (chat, ticket, vcs, idp, endpoint, hr, dlp)
are written to an ``extended/`` folder for pipelines that can use them. An
``insiders.csv`` answer key mirrors CERT's red-team answers.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Any

from .events import Channel, Event, Label
from .simulate import SimResult


def _cert_ts(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%m/%d/%Y %H:%M:%S")


def _writer(path: str, header: list[str]):
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(header)
    return fh, w


def export_cert(result: SimResult, outdir: str, observed: bool = True) -> dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    ext = os.path.join(outdir, "extended")
    os.makedirs(ext, exist_ok=True)
    stream = result.observed if observed else result.events

    files = {
        "logon": _writer(os.path.join(outdir, "logon.csv"),
                         ["id", "date", "user", "pc", "activity"]),
        "device": _writer(os.path.join(outdir, "device.csv"),
                          ["id", "date", "user", "pc", "activity"]),
        "http": _writer(os.path.join(outdir, "http.csv"),
                        ["id", "date", "user", "pc", "url", "activity"]),
        "email": _writer(os.path.join(outdir, "email.csv"),
                         ["id", "date", "user", "pc", "to", "from", "size", "attachments", "activity"]),
        "file": _writer(os.path.join(outdir, "file.csv"),
                        ["id", "date", "user", "pc", "filename", "activity"]),
    }
    ext_files = {
        "chat": _writer(os.path.join(ext, "chat.csv"), ["id", "date", "user", "channel", "dm"]),
        "ticket": _writer(os.path.join(ext, "ticket.csv"), ["id", "date", "user", "project", "activity"]),
        "vcs": _writer(os.path.join(ext, "vcs.csv"), ["id", "date", "user", "repo", "activity", "n_files"]),
        "idp": _writer(os.path.join(ext, "idp.csv"), ["id", "date", "user", "app", "elevated", "result"]),
        "endpoint": _writer(os.path.join(ext, "endpoint.csv"), ["id", "date", "user", "pc", "activity"]),
        "hr": _writer(os.path.join(ext, "hr.csv"), ["id", "date", "user", "event"]),
    }
    # a label sidecar keyed by event id preserves the graded ground truth CERT lacks
    label_fh, label_w = _writer(os.path.join(outdir, "labels.csv"),
                                ["id", "date", "user", "label", "technique_ids", "risk_state"])

    counts: dict[str, int] = {}
    org = result.org
    for ev in stream:
        try:
            pc = org.by_id(ev.actor).host
        except KeyError:
            pc = "UNKNOWN"
        d = _cert_ts(ev.ts)
        ch = ev.channel
        if ch is Channel.LOGON:
            files["logon"][1].writerow([ev.event_id, d, ev.actor, pc,
                                        "Logon" if ev.action.endswith("start") else "Logoff"])
        elif ch is Channel.DEVICE:
            files["device"][1].writerow([ev.event_id, d, ev.actor, pc,
                                         "Connect" if ev.action.endswith("connect") else "Disconnect"])
        elif ch is Channel.HTTP:
            files["http"][1].writerow([ev.event_id, d, ev.actor, pc,
                                       f"http://{ev.attrs.get('domain', 'unknown')}/",
                                       "Upload" if ev.action.endswith("upload") else "Visit"])
        elif ch is Channel.EMAIL:
            to = ev.attrs.get("external_domain", "internal@corp.example") if not ev.attrs.get("to_internal", True) \
                else "team@corp.example"
            files["email"][1].writerow([ev.event_id, d, ev.actor, pc, to,
                                        ev.actor + "@corp.example", ev.attrs.get("size_kb", 0),
                                        ev.attrs.get("n_attachments", 0), "Send"])
        elif ch is Channel.FILE:
            files["file"][1].writerow([ev.event_id, d, ev.actor, pc,
                                       ev.attrs.get("path", ev.attrs.get("asset", "file")),
                                       ev.action.split(".", 1)[-1].capitalize()])
        elif ch is Channel.CHAT:
            ext_files["chat"][1].writerow([ev.event_id, d, ev.actor,
                                           ev.attrs.get("channel", ""), ev.attrs.get("dm", False)])
        elif ch is Channel.TICKET:
            ext_files["ticket"][1].writerow([ev.event_id, d, ev.actor,
                                             ev.attrs.get("project", ""), ev.action.split(".")[-1]])
        elif ch is Channel.VCS:
            ext_files["vcs"][1].writerow([ev.event_id, d, ev.actor, ev.attrs.get("repo", ""),
                                          ev.action.split(".")[-1], ev.attrs.get("n_files", 0)])
        elif ch is Channel.IDP:
            ext_files["idp"][1].writerow([ev.event_id, d, ev.actor, ev.attrs.get("app", ""),
                                          ev.attrs.get("elevated", False), ev.attrs.get("result", "")])
        elif ch is Channel.ENDPOINT:
            ext_files["endpoint"][1].writerow([ev.event_id, d, ev.actor, pc, ev.action])
        elif ch is Channel.HR:
            ext_files["hr"][1].writerow([ev.event_id, d, ev.actor, ev.action])

        counts[ch.value] = counts.get(ch.value, 0) + 1
        if ev.label is not Label.BENIGN:
            label_w.writerow([ev.event_id, d, ev.actor, ev.label.value,
                              "|".join(ev.technique_ids), round(ev.risk_state, 4)])

    for fh, _ in list(files.values()) + list(ext_files.values()):
        fh.close()
    label_fh.close()

    # answer key: one row per activated insider episode
    ans_path = os.path.join(outdir, "insiders.csv")
    with open(ans_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "scenario", "onset_day", "episode_end", "dataset"])
        for emp_id, traj in result.trajectories.items():
            if traj.onset_day is not None:
                w.writerow([emp_id, traj.pathway, traj.onset_day, traj.episode_end, result.config.name])

    return {"cert_export_counts": counts, "outdir": outdir, "observed": observed}
