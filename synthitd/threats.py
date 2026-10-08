"""Insider pathways as *stage-based, label-level* playbooks (requirement R1/R5).

A playbook is a sequence of abstract stages. Each stage says, in detection terms,
*what observable an analyst would see* (e.g. "an unusual read of a crown-jewel asset",
"a large outbound transfer", "a privileged configuration change on a critical host")
and attaches the corresponding **MITRE ATT&CK technique id label** used only for
evaluation. Playbooks contain **no operational instructions**: they emit the same
typed events the baseline model emits, differing in target, volume, timing, and label.

Three design choices break the signature-learning shortcut that inflates CERT scores:

1. **Endogenous onset.** Stages fire only for insiders whose critical pathway
   activated (see :mod:`synthitd.psych`); the simulator never hands out a script.
2. **Stealth modulation.** A per-insider stealth level (from conscientiousness and
   the latent risk shape) spreads activity into work hours, shrinks volumes, and
   prefers low-signal channels, so there is no fixed "after-hours + USB + upload"
   fingerprint.
3. **Variant sampling.** Target asset, channel mix, and staging order are randomised
   per episode, so the positive class is a distribution, not a template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import RNG, SimConfig
from .events import Channel, EventBus, Label
from .org import Employee, Organization, CROWN_JEWELS
from .psych import RiskTrajectory
from .simclock import WorkCalendar


@dataclass
class Stage:
    """One abstract step of a pathway, expressed as a detection observable."""

    name: str
    tactic: str
    technique_ids: list[str]
    kind: str                       # dispatch key handled by ThreatEngine._emit_stage
    phase: str = "act"              # "precursor" or "act"
    weight: float = 1.0             # relative share of the stage budget


@dataclass
class Playbook:
    family: str
    description: str
    applicable_roles: list[str]     # role abbrs that can plausibly run it ([] = any)
    stages: list[Stage]
    tempo: float = 0.5              # 0=bursty (single day), 1=slow (spread over episode)

    def precursor_stages(self) -> list[Stage]:
        return [s for s in self.stages if s.phase == "precursor"]

    def act_stages(self) -> list[Stage]:
        return [s for s in self.stages if s.phase == "act"]


# --- playbook library (abstract; label-level only) --------------------------
PLAYBOOKS: dict[str, Playbook] = {
    "ip_theft": Playbook(
        family="ip_theft",
        description="Theft of intellectual property / source / designs before exit.",
        applicable_roles=["eng", "sre", "ds", "quant", "pm", "rsrch", "trdr"],
        tempo=0.6,
        stages=[
            Stage("recon_scope", "Discovery", ["T1083", "T1087"], "recon",
                  phase="precursor", weight=1.0),
            Stage("stage_collect", "Collection", ["T1005", "T1074"], "stage_collect",
                  phase="act", weight=1.5),
            Stage("exfil_transfer", "Exfiltration", ["T1052", "T1567"], "exfil",
                  phase="act", weight=1.2),
        ],
    ),
    "data_leak": Playbook(
        family="data_leak",
        description="Leak of customer/PII/PHI records to an outside party.",
        applicable_roles=["sales", "rm", "ops", "bill", "hr", "nurse", "phys", "ds"],
        tempo=0.4,
        stages=[
            Stage("recon_scope", "Discovery", ["T1087"], "recon", phase="precursor"),
            Stage("bulk_access", "Collection", ["T1005", "T1114"], "stage_collect",
                  phase="act", weight=1.5),
            Stage("exfil_email", "Exfiltration", ["T1048", "T1567.002"], "exfil",
                  phase="act", weight=1.0),
        ],
    ),
    "privilege_misuse": Playbook(
        family="privilege_misuse",
        description="Administrator/privileged-account misuse to reach data out of role.",
        applicable_roles=["it", "sec", "sre", "comp", "hw", "admin"],
        tempo=0.5,
        stages=[
            Stage("priv_probe", "Privilege Escalation", ["T1078", "T1548"], "priv_use",
                  phase="precursor"),
            Stage("cross_access", "Collection", ["T1005", "T1530"], "stage_collect",
                  phase="act", weight=1.3),
            Stage("exfil_transfer", "Exfiltration", ["T1567"], "exfil", phase="act"),
        ],
    ),
    "fraud": Playbook(
        family="fraud",
        description="Unauthorized modification of records for financial gain.",
        applicable_roles=["fin", "bill", "ops", "trdr", "hr", "admin"],
        tempo=0.7,
        stages=[
            Stage("recon_scope", "Discovery", ["T1083"], "recon", phase="precursor"),
            Stage("record_tamper", "Impact", ["T1565"], "record_tamper",
                  phase="act", weight=2.0),
        ],
    ),
    "sabotage_markers": Playbook(
        family="sabotage_markers",
        description="Pre-departure destructive-intent markers on critical systems.",
        applicable_roles=["it", "sre", "sec", "eng", "admin"],
        tempo=0.3,
        stages=[
            Stage("priv_probe", "Privilege Escalation", ["T1078"], "priv_use",
                  phase="precursor"),
            Stage("critical_change", "Impact", ["T1485", "T1489"], "sabotage_marker",
                  phase="act", weight=1.5),
        ],
    ),
}


def assign_pathways(
    cfg: SimConfig, rng: RNG, org: Organization, trajectories: dict[str, RiskTrajectory]
) -> None:
    """Pick a plausible pathway family for each activated insider (role-aware)."""
    allowed = set(cfg.enabled_playbooks) if cfg.enabled_playbooks else set(PLAYBOOKS)
    g = rng.stream("pathways")
    for emp in org.employees:
        traj = trajectories[emp.emp_id]
        if traj.onset_day is None:
            continue
        candidates = [
            name for name, pb in PLAYBOOKS.items()
            if name in allowed and (not pb.applicable_roles or emp.role_abbr in pb.applicable_roles)
        ]
        if not candidates:
            candidates = list(allowed) or list(PLAYBOOKS)
        traj.pathway = candidates[int(g.integers(len(candidates)))]
        # tempo can stretch the episode end
        pb = PLAYBOOKS[traj.pathway]
        if pb.tempo > 0.5 and traj.episode_end is not None:
            extra = int(pb.tempo * 6)
            traj.episode_end = min(cfg.horizon_days - 1, traj.episode_end + extra)


class ThreatEngine:
    def __init__(self, cfg: SimConfig, rng: RNG, cal: WorkCalendar, org: Organization) -> None:
        self.cfg = cfg
        self.rng = rng
        self.cal = cal
        self.org = org

    def stealth(self, emp: Employee, traj: RiskTrajectory) -> float:
        """0 = reckless/obvious, 1 = highly covert. Higher conscientiousness and a
        slow-building risk curve imply more covert behaviour (harder to detect)."""
        if self.cfg.legacy_cert_mode:
            return 0.0  # CERT regime: reckless, fixed obvious signature
        g = self.rng.stream("stealth", emp.emp_id)
        shape = float(np.clip(traj.risk[: (traj.onset_day or 1)].std() * 2, 0, 0.4))
        return float(np.clip(0.3 + 0.5 * emp.ocean["conscientiousness"] - shape + 0.1 * g.random(), 0, 1))

    def emit_day(
        self, bus: EventBus, emp: Employee, day: int, traj: RiskTrajectory
    ) -> None:
        if not traj.pathway:
            return
        state = traj.state_on(day)
        if state == "baseline":
            return
        pb = PLAYBOOKS[traj.pathway]
        st = self.stealth(emp, traj)
        g = self.rng.stream("threat", emp.emp_id, day)
        episode_id = f"ep-{emp.emp_id}-{traj.onset_day}"

        stages = pb.precursor_stages() if state == "precursor" else pb.act_stages()
        if not stages:
            return
        # covert insiders fire fewer stage-events per day
        n_actions = max(1, int(round(g.poisson(2.2 * (1 - 0.6 * st)) )))
        risk = float(traj.risk[day])
        for _ in range(n_actions):
            stage = _weighted_pick(g, stages)
            label = Label.PRECURSOR if state == "precursor" else Label.MALICIOUS
            self._emit_stage(bus, emp, day, g, stage, st, label, episode_id, risk)

    # -- stage dispatch (each emits label-level observables) --------------
    def _emit_stage(
        self, bus: EventBus, emp: Employee, day: int, g: np.random.Generator,
        stage: Stage, stealth: float, label: Label, episode_id: str, risk: float,
    ) -> None:
        # covert actors prefer work hours; reckless ones drift after-hours
        if g.random() < (0.2 + 0.6 * stealth):
            h = float(g.uniform(self.cfg.workday_start_hour, self.cfg.workday_end_hour))
        else:
            h = float(np.clip(g.normal(21.0, 2.0), 18.5, 23.9))
        ts = self.cal.ts(day, h)
        jewel = _pick(g, CROWN_JEWELS.get(self.org.domain, ["sensitive"]))
        tids = stage.technique_ids
        common = dict(episode_id=episode_id, technique_ids=tids, label=label, risk_state=risk)

        if stage.kind == "recon":
            bus.emit(ts, Channel.FILE, emp.emp_id, "file.open",
                     attrs={"asset": jewel, "path": f"/share/{jewel}/index",
                            "breadth": int(g.integers(5, 40)), "out_of_pattern": True},
                     **common)
        elif stage.kind == "priv_use":
            bus.emit(ts, Channel.IDP, emp.emp_id, "idp.auth",
                     attrs={"app": "admin_console", "elevated": True,
                            "out_of_role": emp.privilege < 3, "result": "success"},
                     **common)
        elif stage.kind == "stage_collect":
            vol = int(g.integers(10, 120) * (1.2 - 0.6 * stealth))
            bus.emit(ts, Channel.FILE, emp.emp_id, "file.copy",
                     attrs={"asset": jewel, "n_files": max(1, vol),
                            "size_kb": int(g.lognormal(8.5, 1.0) * (1.4 - stealth)),
                            "host": emp.host, "staging_dir": True,
                            "removable": bool(g.random() < 0.4 * (1 - stealth))},
                     **common)
        elif stage.kind == "exfil":
            # Covert insiders blend into the legitimate cloud-upload / small-email
            # population; reckless ones use the obvious removable / filehost route.
            # There is therefore no single exfil channel that marks the positive
            # class — detection needs the joint, per-user-relative pattern (R6).
            if not self.cfg.legacy_cert_mode and g.random() < (0.3 + 0.6 * stealth):
                roll = g.random()
                if roll < 0.55:
                    bus.emit(ts, Channel.HTTP, emp.emp_id, "http.upload",
                             attrs={"domain": _pick(g, ["cloud.example", "drive.corp.example",
                                                        "box.corp.example"]),
                                    "bytes": int(g.lognormal(12.5, 1.0)), "asset": jewel,
                                    "category": "cloud"},
                             **common)
                else:
                    bus.emit(ts, Channel.EMAIL, emp.emp_id, "email.send",
                             attrs={"to_internal": False,
                                    "external_domain": _pick(g, ["gmail.personal.example",
                                                                  "outlook.personal.example"]),
                                    "n_attachments": int(g.integers(1, 3)),
                                    "size_kb": int(g.lognormal(8.0, 0.7)), "asset": jewel,
                                    "wants_render": True},
                             **common)
            else:
                roll = g.random()
                if roll < 0.5:
                    bus.emit(ts, Channel.HTTP, emp.emp_id, "http.upload",
                             attrs={"domain": _pick(g, ["paste.example", "transfer.example",
                                                        "drive.personal.example"]),
                                    "bytes": int(g.lognormal(14.0, 1.0)), "asset": jewel,
                                    "category": "filehost"},
                             **common)
                else:
                    bus.emit(ts, Channel.DEVICE, emp.emp_id, "device.connect",
                             attrs={"kind": "usb_storage", "host": emp.host, "approved": False},
                             **common)
                    bus.emit(ts + 120, Channel.FILE, emp.emp_id, "file.copy",
                             attrs={"asset": jewel, "removable": True,
                                    "size_kb": int(g.lognormal(9.0, 1.0)),
                                    "n_files": int(g.integers(5, 80))},
                             **common)
        elif stage.kind == "record_tamper":
            bus.emit(ts, Channel.FILE, emp.emp_id, "file.write",
                     attrs={"asset": jewel, "record_modification": True,
                            "fields_changed": int(g.integers(1, 8)), "out_of_pattern": True},
                     **common)
        elif stage.kind == "sabotage_marker":
            # a pre-departure destructive-intent *marker* on a critical host — an
            # observable configuration/deletion event, not an instruction.
            bus.emit(ts, Channel.ENDPOINT, emp.emp_id, "config.change",
                     attrs={"host": _pick(g, ["PROD-CORE-01", "PROD-DB-01", "BACKUP-01"]),
                            "critical_asset": True, "reversible": bool(g.random() < 0.5),
                            "out_of_change_window": True},
                     **common)

    # expose the full technique set for the dataset card / benchmark
    @staticmethod
    def all_technique_ids() -> list[str]:
        tids: set[str] = set()
        for pb in PLAYBOOKS.values():
            for s in pb.stages:
                tids.update(s.technique_ids)
        return sorted(tids)


def _weighted_pick(g: np.random.Generator, stages: list[Stage]) -> Stage:
    w = np.array([s.weight for s in stages], dtype=float)
    w /= w.sum()
    return stages[int(g.choice(len(stages), p=w))]


def _pick(g: np.random.Generator, items: list[Any]) -> Any:
    return items[int(g.integers(len(items)))]
