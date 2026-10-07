"""Baseline multi-surface behaviour model (requirement R3) + benign anomalies (R4).

Produces the benign background: per employee, per workday, a realistic spread of
activity across all modern telemetry surfaces, modulated by role, personality, the
work calendar, and any active benign triggers. Benign-trigger-driven activity is
labelled :data:`Label.ANOMALOUS_BENIGN` and always comes with a *justifying* HR
context event, so a good detector can learn to exonerate it — and a lazy one will
generate false positives on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RNG, SimConfig
from .events import Channel, EventBus, Label
from .lifecycle import LifeHistory, BenignTrigger
from .org import Employee, Organization
from .psych import RiskTrajectory
from .simclock import WorkCalendar


# base mean daily event counts by channel for a "generic" role; role deltas applied
_BASE_RATES: dict[Channel, float] = {
    Channel.LOGON: 2.0,
    Channel.IDP: 6.0,
    Channel.EMAIL: 14.0,
    Channel.CHAT: 22.0,
    Channel.HTTP: 40.0,
    Channel.FILE: 25.0,
    Channel.TICKET: 3.0,
    Channel.VCS: 0.0,
    Channel.DEVICE: 0.1,
    Channel.ENDPOINT: 8.0,
}

# role_abbr -> channel multipliers (missing = 1.0)
_ROLE_DELTAS: dict[str, dict[Channel, float]] = {
    "eng": {Channel.VCS: 8.0, Channel.TICKET: 2.0, Channel.ENDPOINT: 2.0, Channel.EMAIL: 0.6},
    "sre": {Channel.VCS: 5.0, Channel.ENDPOINT: 4.0, Channel.FILE: 1.4},
    "sec": {Channel.ENDPOINT: 3.0, Channel.HTTP: 1.5, Channel.FILE: 1.2},
    "it": {Channel.ENDPOINT: 4.0, Channel.FILE: 1.5, Channel.DEVICE: 2.0},
    "ds": {Channel.FILE: 2.0, Channel.HTTP: 1.3, Channel.VCS: 2.0},
    "quant": {Channel.FILE: 2.0, Channel.VCS: 3.0},
    "sales": {Channel.EMAIL: 2.0, Channel.HTTP: 1.4, Channel.FILE: 1.2},
    "rm": {Channel.EMAIL: 2.0, Channel.FILE: 1.2},
    "trdr": {Channel.FILE: 1.6, Channel.HTTP: 1.3, Channel.EMAIL: 1.4},
    "hr": {Channel.FILE: 1.5, Channel.EMAIL: 1.5},
    "fin": {Channel.FILE: 1.6, Channel.EMAIL: 1.3},
    "bill": {Channel.FILE: 1.8},
    "nurse": {Channel.FILE: 2.2, Channel.ENDPOINT: 0.5, Channel.VCS: 0.0},
    "phys": {Channel.FILE: 2.0, Channel.EMAIL: 0.8},
    "exec": {Channel.EMAIL: 1.8, Channel.FILE: 0.8},
}

_WEB_DOMAINS = [
    "wiki.corp.example", "mail.corp.example", "jira.corp.example", "git.corp.example",
    "news.example", "search.example", "docs.example", "cloud.example",
    "social.example", "stackoverflow.example", "vendor.example",
]


# roles with a legitimate baseline for cloud upload / removable media. These make
# benign activity produce the very features a naive rule would treat as malicious,
# so no single feature separates the classes (the anti-shortcut design, R6).
_UPLOAD_BASE = {"eng": 0.18, "sre": 0.24, "ds": 0.26, "sec": 0.16, "it": 0.22,
                "sales": 0.22, "rm": 0.20, "pm": 0.12, "quant": 0.22, "trdr": 0.14,
                "bill": 0.12, "rsrch": 0.2, "ops": 0.12}
_REMOVABLE_BASE = {"it": 0.28, "sre": 0.16, "sec": 0.14, "ds": 0.10, "lab": 0.18,
                   "eng": 0.06, "rsrch": 0.12, "hw": 0.1}
_CLOUD_DOMAINS = ["cloud.example", "drive.corp.example", "backup.corp.example",
                  "s3.vendor.example", "box.corp.example"]


class BehaviorModel:
    def __init__(self, cfg: SimConfig, rng: RNG, cal: WorkCalendar) -> None:
        self.cfg = cfg
        self.rng = rng
        self.cal = cal
        self._profiles: dict[str, dict[str, float]] = {}

    def _profile(self, emp: Employee) -> dict[str, float]:
        p = self._profiles.get(emp.emp_id)
        if p is None:
            if self.cfg.legacy_cert_mode:
                # CERT regime: no legitimate upload/removable power users, so those
                # features become near-perfect markers of malice (the shortcut).
                g = self.rng.stream("profile", emp.emp_id)
                p = {"upload": 0.0, "removable": 0.0,
                     "after_hours": float(0.05 + 0.1 * (1 - emp.ocean["conscientiousness"]))}
                self._profiles[emp.emp_id] = p
                return p
            g = self.rng.stream("profile", emp.emp_id)
            # anomalous-benign employees are *heavier* legitimate power users — the
            # engineered hard negatives whose normal behaviour mimics exfiltration.
            boost = 1.8 if emp.population == "anomalous_benign" else 1.0
            up = _UPLOAD_BASE.get(emp.role_abbr, 0.07) * boost * (0.6 + 0.8 * g.random())
            rm = _REMOVABLE_BASE.get(emp.role_abbr, 0.03) * boost * (0.6 + 0.8 * g.random())
            ah = 0.05 + 0.35 * (1 - emp.ocean["conscientiousness"]) + (0.1 if boost > 1 else 0)
            p = {"upload": float(np.clip(up, 0, 0.5)),
                 "removable": float(np.clip(rm, 0, 0.45)),
                 "after_hours": float(np.clip(ah, 0, 0.6))}
            self._profiles[emp.emp_id] = p
        return p

    # -- public -----------------------------------------------------------
    def emit_day(
        self,
        bus: EventBus,
        emp: Employee,
        day: int,
        hist: LifeHistory,
        traj: RiskTrajectory,
    ) -> None:
        if self.cal.is_weekend(day):
            # light weekend presence for a minority of conscientious/eng roles
            g = self.rng.stream("wknd", emp.emp_id, day)
            if g.random() > 0.12 + 0.2 * emp.ocean["conscientiousness"]:
                return
            scale = 0.15
        else:
            scale = 1.0

        g = self.rng.stream("behavior", emp.emp_id, day)
        actives = hist.active_benign(day)
        risk = float(traj.risk[day])

        self._emit_logon(bus, emp, day, g, actives)
        for ch, rate in self._rates(emp).items():
            if ch in (Channel.LOGON,):
                continue
            mean = rate * self.cfg.activity_scale * scale
            if mean <= 0:
                continue
            count = int(g.poisson(mean))
            for _ in range(count):
                if g.random() < self.cfg.behavior_dropout:
                    continue
                self._emit_action(bus, emp, day, ch, g, actives, risk)

        # benign anomalies create signature-looking-but-explained bursts
        for trig in actives:
            self._emit_benign_anomaly(bus, emp, day, trig, g)

    # -- internals --------------------------------------------------------
    def _rates(self, emp: Employee) -> dict[Channel, float]:
        rates = dict(_BASE_RATES)
        for ch, m in _ROLE_DELTAS.get(emp.role_abbr, {}).items():
            rates[ch] = rates.get(ch, 0.0) * m
        # personality: extraversion -> more chat/email; conscientiousness -> tickets
        rates[Channel.CHAT] *= 0.6 + 0.8 * emp.ocean["extraversion"]
        rates[Channel.EMAIL] *= 0.7 + 0.6 * emp.ocean["extraversion"]
        rates[Channel.TICKET] *= 0.6 + 0.8 * emp.ocean["conscientiousness"]
        return rates

    def _work_hour(self, g: np.random.Generator, emp: Employee, after_hours_boost: float = 0.0) -> float:
        p_after = 0.05 + 0.25 * (1 - emp.ocean["conscientiousness"]) + after_hours_boost
        if g.random() < min(0.9, p_after):
            # evening/night tail
            return float(np.clip(g.normal(20.0, 2.5), 18.01, 23.9))
        lo, hi = self.cfg.workday_start_hour, self.cfg.workday_end_hour
        h = float(g.uniform(lo, hi))
        ll, lh = self.cfg.lunch_hours
        if ll <= h < lh and g.random() < 0.7:
            h = float(g.uniform(lh, hi))
        return h

    def _emit_logon(
        self, bus: EventBus, emp: Employee, day: int, g: np.random.Generator, actives: list[BenignTrigger]
    ) -> None:
        travel = any(t.kind == "business_travel" for t in actives)
        ip = emp.ip
        geo = "office"
        if travel:
            ip = f"203.0.{g.integers(1, 250)}.{g.integers(1, 250)}"
            geo = "remote"
        start_h = float(np.clip(g.normal(self.cfg.workday_start_hour + 0.5, 1.0), 5.5, 11.5))
        label = Label.ANOMALOUS_BENIGN if travel else Label.BENIGN
        bus.emit(
            self.cal.ts(day, start_h), Channel.LOGON, emp.emp_id, "logon.start",
            attrs={"host": emp.host, "ip": ip, "geo": geo, "vpn": travel},
            label=label,
        )
        end_h = float(np.clip(g.normal(self.cfg.workday_end_hour - 0.5, 1.2), start_h + 1, 23.5))
        bus.emit(
            self.cal.ts(day, end_h), Channel.LOGON, emp.emp_id, "logon.end",
            attrs={"host": emp.host, "ip": ip, "geo": geo},
            label=Label.BENIGN,
        )
        # IdP auths distributed across the day
        for _ in range(int(g.poisson(4))):
            h = self._work_hour(g, emp)
            bus.emit(
                self.cal.ts(day, h), Channel.IDP, emp.emp_id, "idp.auth",
                attrs={"app": _pick(g, ["wiki", "mail", "git", "hrportal", "cloud"]),
                       "mfa": True, "result": "success", "ip": ip},
                label=Label.BENIGN,
            )

    def _emit_action(
        self, bus: EventBus, emp: Employee, day: int, ch: Channel,
        g: np.random.Generator, actives: list[BenignTrigger], risk: float,
    ) -> None:
        crunch = any(t.kind in ("project_crunch", "oncall_rotation") for t in actives)
        h = self._work_hour(g, emp, after_hours_boost=0.2 if crunch else 0.0)
        ts = self.cal.ts(day, h)
        base_label = Label.BENIGN
        if crunch and ch in (Channel.VCS, Channel.ENDPOINT, Channel.HTTP) and h >= 18.0:
            base_label = Label.ANOMALOUS_BENIGN

        if ch is Channel.EMAIL:
            internal = g.random() < 0.8
            bus.emit(ts, ch, emp.emp_id, "email.send",
                     attrs={"to_internal": internal,
                            "n_recipients": int(g.integers(1, 5)),
                            "n_attachments": int(g.random() < 0.25),
                            "size_kb": int(g.lognormal(3.0, 1.0)),
                            "wants_render": True},
                     label=base_label)
        elif ch is Channel.CHAT:
            bus.emit(ts, ch, emp.emp_id, "chat.message",
                     attrs={"channel": f"#{emp.team}", "dm": bool(g.random() < 0.3),
                            "wants_render": bool(g.random() < 0.5)},
                     label=base_label)
        elif ch is Channel.HTTP:
            prof = self._profile(emp)
            if g.random() < prof["upload"]:
                # legitimate cloud upload (deploy artifact, share deck, back up data)
                bus.emit(ts, ch, emp.emp_id, "http.upload",
                         attrs={"domain": _pick(g, _CLOUD_DOMAINS),
                                "bytes": int(g.lognormal(12.5, 1.2)),
                                "category": "cloud"},
                         label=base_label)
            else:
                dom = _pick(g, _WEB_DOMAINS)
                bus.emit(ts, ch, emp.emp_id, "http.get",
                         attrs={"domain": dom, "bytes": int(g.lognormal(8.0, 1.5)),
                                "category": "work" if "corp" in dom else "general"},
                         label=base_label)
        elif ch is Channel.FILE:
            prof = self._profile(emp)
            op = _pick(g, ["open", "open", "open", "write", "copy", "delete"])
            asset = _pick(g, emp.sensitive_access or ["shared_docs"])
            removable = op == "copy" and g.random() < prof["removable"]
            bus.emit(ts, ch, emp.emp_id, f"file.{op}",
                     attrs={"asset": asset, "path": f"/share/{asset}/f{g.integers(9999)}",
                            "size_kb": int(g.lognormal(5.0, 1.5)),
                            "host": emp.host, "removable": removable,
                            "approved": True if removable else None},
                     label=base_label)
        elif ch is Channel.TICKET:
            act = _pick(g, ["create", "comment", "transition"])
            bus.emit(ts, ch, emp.emp_id, f"ticket.{act}",
                     attrs={"project": emp.team, "wants_render": act != "transition"},
                     label=base_label)
        elif ch is Channel.VCS:
            act = _pick(g, ["clone", "push", "push", "pr_open", "review"])
            bus.emit(ts, ch, emp.emp_id, f"vcs.{act}",
                     attrs={"repo": _pick(g, ["core", "api", "infra", "web", "data"]),
                            "n_files": int(g.integers(1, 40)),
                            "wants_render": act in ("pr_open", "review")},
                     label=base_label)
        elif ch is Channel.ENDPOINT:
            bus.emit(ts, ch, emp.emp_id, "proc.exec",
                     attrs={"proc": _pick(g, ["bash", "python", "psql", "kubectl",
                                              "code", "chrome", "excel"]),
                            "host": emp.host},
                     label=base_label)
        elif ch is Channel.DEVICE:
            bus.emit(ts, ch, emp.emp_id, "device.connect",
                     attrs={"kind": "usb_storage", "host": emp.host,
                            "approved": True},
                     label=Label.ANOMALOUS_BENIGN if g.random() < 0.3 else Label.BENIGN)

    def _emit_benign_anomaly(
        self, bus: EventBus, emp: Employee, day: int, trig: BenignTrigger, g: np.random.Generator
    ) -> None:
        """Emit the *justifying evidence* + the anomalous-but-benign signature.

        The crucial design point: every anomalous-benign burst is paired with an HR
        context event that explains it. A detector that reads context can exonerate;
        one that only sees the raw burst will false-positive.
        """
        h = self._work_hour(g, emp, after_hours_boost=0.3)
        ts = self.cal.ts(day, h)
        if trig.observable and day == trig.start_day:
            bus.emit(self.cal.ts(day, 9.0), Channel.HR, emp.emp_id, "hr.context",
                     attrs={"kind": trig.kind, "approved": True,
                            "window_days": trig.end_day - trig.start_day},
                     label=Label.BENIGN)
        if self.cfg.legacy_cert_mode:
            # CERT regime: no exfil-looking benign bursts (weak hard negatives), so
            # the positive class is cleanly separable.
            return
        jewel = _pick(g, emp.sensitive_access or ["shared_docs"])
        if trig.kind == "approved_bulk_export":
            # a large export — sometimes to removable, sometimes to cloud; approved.
            to_removable = g.random() < 0.5
            bus.emit(ts, Channel.FILE, emp.emp_id, "file.copy",
                     attrs={"asset": jewel, "n_files": int(g.integers(50, 400)),
                            "size_kb": int(g.lognormal(9.5, 0.8)),
                            "removable": to_removable, "approved": True,
                            "destination": "approved_backup"},
                     label=Label.ANOMALOUS_BENIGN)
            if not to_removable:
                bus.emit(ts + 90, Channel.HTTP, emp.emp_id, "http.upload",
                         attrs={"domain": _pick(g, _CLOUD_DOMAINS),
                                "bytes": int(g.lognormal(14.0, 0.9)), "category": "cloud"},
                         label=Label.ANOMALOUS_BENIGN)
        elif trig.kind == "oncall_rotation":
            # after-hours response work: endpoint + cloud upload (deploy/rollback)
            hh = float(np.clip(g.normal(22.0, 1.5), 18.5, 23.9))
            tt = self.cal.ts(day, hh)
            bus.emit(tt, Channel.HTTP, emp.emp_id, "http.upload",
                     attrs={"domain": _pick(g, _CLOUD_DOMAINS),
                            "bytes": int(g.lognormal(12.5, 1.1)), "category": "cloud"},
                     label=Label.ANOMALOUS_BENIGN)
            bus.emit(tt + 60, Channel.ENDPOINT, emp.emp_id, "proc.exec",
                     attrs={"proc": "kubectl", "host": emp.host}, label=Label.ANOMALOUS_BENIGN)
        elif trig.kind in ("project_crunch", "business_travel"):
            hh = float(np.clip(g.normal(20.5, 2.0), 18.1, 23.9)) if trig.kind == "project_crunch" \
                else self._work_hour(g, emp)
            tt = self.cal.ts(day, hh)
            if g.random() < 0.7:
                bus.emit(tt, Channel.HTTP, emp.emp_id, "http.upload",
                         attrs={"domain": _pick(g, _CLOUD_DOMAINS),
                                "bytes": int(g.lognormal(12.0, 1.2)), "category": "cloud"},
                         label=Label.ANOMALOUS_BENIGN)
            else:
                bus.emit(tt, Channel.EMAIL, emp.emp_id, "email.send",
                         attrs={"to_internal": trig.kind == "project_crunch",
                                "external_domain": None if trig.kind == "project_crunch" else "client.example",
                                "n_attachments": int(g.integers(1, 5)),
                                "size_kb": int(g.lognormal(9.0, 0.8)), "wants_render": True},
                         label=Label.ANOMALOUS_BENIGN)
        elif trig.kind == "team_transfer":
            other = _pick(g, ["core", "api", "finance", "clinical", "ops"])
            bus.emit(ts, Channel.FILE, emp.emp_id, "file.open",
                     attrs={"asset": "cross_team_docs", "path": f"/share/{other}/onboarding",
                            "cross_team": True},
                     label=Label.ANOMALOUS_BENIGN)


def _pick(g: np.random.Generator, items: list[Any]) -> Any:
    return items[int(g.integers(len(items)))]
