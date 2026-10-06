"""Rendering interface and the label-blind :class:`ContentPlan` (requirement R2).

Free-text surfaces (email bodies, chat messages, ticket descriptions, PR summaries)
are produced by a *separate* pass that is shown only a :class:`ContentPlan` — neutral,
fully-benign business facts derived from an event's action and attributes. The plan
**never** contains the event's label, technique ids, risk score, or episode id.

Consequence: the renderer produces ordinary-looking workplace prose for *every*
event, including on-pathway ones. That is deliberate. A detector must therefore rely
on behavioural and structural signal, not on keyword-spotting the word "exfiltrate"
in a body — the exact shortcut that makes NLP-on-CERT look better than it is.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..events import Channel, Event
from ..org import Organization


# which (channel, action-prefix) pairs carry renderable free text, and which fields
_RENDERABLE: dict[tuple[str, str], list[str]] = {
    ("email", "email.send"): ["subject", "body"],
    ("chat", "chat.message"): ["message"],
    ("ticket", "ticket.create"): ["title", "body"],
    ("ticket", "ticket.comment"): ["body"],
    ("vcs", "vcs.pr_open"): ["title", "description"],
    ("vcs", "vcs.review"): ["body"],
}


@dataclass
class ContentPlan:
    """A label-blind description of the prose to generate for one event."""

    event_id: str
    channel: str
    surface: str              # "email" | "chat" | "ticket" | "vcs"
    fields: list[str]         # which text fields to produce
    role_title: str
    team: str
    # neutral business context only:
    topic: str                # generic subject matter, e.g. "release", "pii-records"
    intent_hint: str          # "share_files" | "status_update" | "question" | "report"
    to_internal: bool = True
    formality: str = "neutral"
    n_recipients: int = 1

    def to_prompt_payload(self) -> dict[str, Any]:
        """Exactly what a renderer may see. Audited to contain no label fields."""
        return {
            "surface": self.surface,
            "fields": self.fields,
            "role": self.role_title,
            "team": self.team,
            "topic": self.topic,
            "intent": self.intent_hint,
            "audience": "internal" if self.to_internal else "external",
            "formality": self.formality,
            "recipients": self.n_recipients,
        }


# attributes that must never reach a renderer (defence in depth for R2)
_FORBIDDEN_PAYLOAD_KEYS = {
    "label", "technique_ids", "risk_state", "episode_id", "out_of_pattern",
    "out_of_role", "elevated", "staging_dir", "record_modification",
    "critical_asset", "out_of_change_window",
}


def _topic_for(ev: Event) -> str:
    a = ev.attrs
    for key in ("asset", "repo", "project", "domain", "app"):
        if key in a and isinstance(a[key], str):
            return _generic_topic(a[key])
    return "general"


def _generic_topic(raw: str) -> str:
    # collapse to a neutral topic word; strip anything evocative of sensitivity
    mapping = {
        "source_code": "codebase", "ci_secrets": "build-config", "pii": "records",
        "phi": "records", "client_pii": "records", "customer_list": "accounts",
        "trade_book": "positions", "ehr": "charts", "financials": "finance",
        "payroll": "payroll", "models": "models", "roadmap": "planning",
        "mna": "corp-dev", "trial_data": "study-data",
    }
    return mapping.get(raw, raw.replace("_", "-"))


def _intent_for(ev: Event) -> str:
    act = ev.action
    if act.startswith("email"):
        return "share_files" if ev.attrs.get("n_attachments") else "status_update"
    if act.startswith("chat"):
        return "question" if ev.attrs.get("dm") else "status_update"
    if act.startswith("ticket.create"):
        return "report"
    if act.startswith("ticket"):
        return "status_update"
    if act.startswith("vcs"):
        return "report"
    return "status_update"


def collect_plans(events: list[Event], org: Organization) -> list[ContentPlan]:
    """Scan the stream and build label-blind plans for renderable events."""
    plans: list[ContentPlan] = []
    for ev in events:
        if not ev.attrs.get("wants_render"):
            continue
        fields = _RENDERABLE.get((ev.channel.value, ev.action))
        if not fields:
            # exfil emails reuse the email.send renderer; still label-blind
            if ev.channel is Channel.EMAIL and ev.action == "email.send":
                fields = ["subject", "body"]
            else:
                continue
        try:
            emp = org.by_id(ev.actor)
            role, team = emp.role_title, emp.team
        except KeyError:
            role, team = "Employee", "general"
        plans.append(
            ContentPlan(
                event_id=ev.event_id,
                channel=ev.channel.value,
                surface=ev.channel.value,
                fields=list(fields),
                role_title=role,
                team=team,
                topic=_topic_for(ev),
                intent_hint=_intent_for(ev),
                to_internal=bool(ev.attrs.get("to_internal", True)),
                n_recipients=int(ev.attrs.get("n_recipients", 1)),
            )
        )
    return plans


class Renderer(abc.ABC):
    """A pluggable text backend. Implementations receive only ContentPlans."""

    name: str = "base"

    @abc.abstractmethod
    def render(self, plans: list[ContentPlan]) -> dict[str, dict[str, str]]:
        """Return ``{event_id: {field: text}}`` for the given plans."""
        raise NotImplementedError
