"""Canonical event schema and the deterministic event bus (design requirement R2).

Every observable produced by the simulator is an :class:`Event`. The event bus —
not the LLM — owns identities, timestamps, causal links, and the graded ground-truth
:class:`Label`. Free-text surfaces (subject/body/message/comment) are rendered
*afterwards* by a renderer that is never shown the label, so prose cannot leak the
answer (see ``synthitd/render``).

The schema is deliberately modern and multi-surface (requirement R3): it covers the
pre-LLM CERT channels (logon, device, file, email, http) *and* the surfaces where
contemporary insider activity actually leaves traces (chat, ticketing, version
control, identity provider, DLP, endpoint).
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator


class Channel(str, enum.Enum):
    """Telemetry surface an event belongs to."""

    LOGON = "logon"            # workstation / VPN logon-logoff
    IDP = "idp"               # SSO / identity provider auth + MFA
    DEVICE = "device"         # removable media connect/disconnect
    FILE = "file"             # file open/copy/write/delete on hosts & shares
    EMAIL = "email"           # corporate email (to/from, attachments, size)
    HTTP = "http"             # web / cloud-service access
    CHAT = "chat"             # team chat (channels + DMs)
    TICKET = "ticket"         # issue tracker (create/comment/transition)
    VCS = "vcs"               # version control (clone/push/PR/review)
    DLP = "dlp"               # data-loss-prevention alerts
    ENDPOINT = "endpoint"     # process / command execution on an endpoint
    HR = "hr"                 # HR / lifecycle events (hire, review, sanction, exit)


class Label(str, enum.Enum):
    """Graded, multi-resolution ground-truth label (requirement R5).

    The ordering encodes increasing concern. ``ANOMALOUS_BENIGN`` is the crucial
    hard-negative class (requirement R4): behaviour that *looks* suspicious but is
    legitimately explained. ``PRECURSOR`` marks concerning-but-not-yet-harmful
    activity on the critical pathway (enables early-warning evaluation).
    """

    BENIGN = "benign"
    ANOMALOUS_BENIGN = "anomalous_benign"
    PRECURSOR = "precursor"
    MALICIOUS = "malicious"

    @property
    def rank(self) -> int:
        return {
            Label.BENIGN: 0,
            Label.ANOMALOUS_BENIGN: 1,
            Label.PRECURSOR: 2,
            Label.MALICIOUS: 3,
        }[self]

    @property
    def is_positive(self) -> bool:
        """Binary collapse used by classical ITD baselines (malicious vs rest)."""
        return self is Label.MALICIOUS


@dataclass
class Event:
    """A single observable on one channel at one instant.

    Attributes
    ----------
    event_id:
        Stable unique id assigned by the bus.
    ts:
        Unix timestamp (seconds, float) in simulation wall-clock.
    channel:
        The :class:`Channel` this observable belongs to.
    actor:
        Employee id that generated the event (the subject of detection).
    action:
        Short machine action verb, e.g. ``"file.copy"``, ``"logon.start"``.
    attrs:
        Structured, fully-determined attributes (sizes, counts, paths, hosts,
        recipients, booleans). These are label-free facts.
    label:
        Graded ground-truth label owned by the bus.
    episode_id:
        Groups events belonging to the same insider episode (or ``""``).
    technique_ids:
        MITRE ATT&CK technique id *labels* attached for evaluation (never shown
        to the renderer). Empty for benign events.
    risk_state:
        Latent critical-pathway risk score of the actor at event time (0..1),
        the regression target for early-warning tasks.
    render:
        Optional free-text payload spec filled in by the renderer. Keys are
        surface field names (``subject``, ``body``, ``message`` ...). ``None``
        until a renderer runs; the simulator core leaves it unset.
    causal_parent:
        event_id of the event that directly caused this one (for provenance /
        verifiability), or ``""``.
    """

    event_id: str
    ts: float
    channel: Channel
    actor: str
    action: str
    attrs: dict[str, Any] = field(default_factory=dict)
    label: Label = Label.BENIGN
    episode_id: str = ""
    technique_ids: list[str] = field(default_factory=list)
    risk_state: float = 0.0
    render: dict[str, str] | None = None
    causal_parent: str = ""

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-serialisable record (what gets written to JSONL)."""
        rec = asdict(self)
        rec["channel"] = self.channel.value
        rec["label"] = self.label.value
        return rec

    @staticmethod
    def from_record(rec: dict[str, Any]) -> "Event":
        return Event(
            event_id=rec["event_id"],
            ts=rec["ts"],
            channel=Channel(rec["channel"]),
            actor=rec["actor"],
            action=rec["action"],
            attrs=rec.get("attrs", {}),
            label=Label(rec.get("label", "benign")),
            episode_id=rec.get("episode_id", ""),
            technique_ids=rec.get("technique_ids", []),
            risk_state=rec.get("risk_state", 0.0),
            render=rec.get("render"),
            causal_parent=rec.get("causal_parent", ""),
        )


class EventBus:
    """Deterministic, append-only sink that owns ids and ordering (R2).

    The bus is the single source of truth. Emitters hand it fully-specified facts
    plus a label; it stamps a monotonic id and stores the event. Nothing in the
    pipeline mutates labels after the bus records them except the labelling pass
    that runs *before* rendering.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._counter: int = 0

    def emit(
        self,
        ts: float,
        channel: Channel,
        actor: str,
        action: str,
        *,
        attrs: dict[str, Any] | None = None,
        label: Label = Label.BENIGN,
        episode_id: str = "",
        technique_ids: Iterable[str] | None = None,
        risk_state: float = 0.0,
        causal_parent: str = "",
    ) -> Event:
        self._counter += 1
        ev = Event(
            event_id=f"e{self._counter:09d}",
            ts=float(ts),
            channel=channel,
            actor=actor,
            action=action,
            attrs=dict(attrs or {}),
            label=label,
            episode_id=episode_id,
            technique_ids=list(technique_ids or []),
            risk_state=risk_state,
            causal_parent=causal_parent,
        )
        self._events.append(ev)
        return ev

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    @property
    def events(self) -> list[Event]:
        return self._events

    def finalize(self) -> list[Event]:
        """Return events in canonical (timestamp, id) order."""
        self._events.sort(key=lambda e: (e.ts, e.event_id))
        return self._events

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_record(), sort_keys=True) for e in self.finalize())
