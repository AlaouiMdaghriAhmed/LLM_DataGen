"""Deterministic template renderer — zero API cost (requirement R7).

Produces plausible, label-blind workplace prose from a :class:`ContentPlan` with a
seeded RNG, so the entire pipeline runs reproducibly and for free. The Anthropic
backend is a drop-in quality upgrade; this is the always-available default and the
one the test-suite uses.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import ContentPlan, Renderer


_GREET = ["Hi team,", "Hi all,", "Hello,", "Hey,", "Morning all,"]
_SIGN = ["Thanks,", "Best,", "Cheers,", "Regards,", "Thanks!"]

_EMAIL_BODY = [
    "Please find the latest {topic} materials attached for your review.",
    "Sharing the {topic} files ahead of our sync — let me know if anything looks off.",
    "Here is the {topic} update you asked for. Happy to walk through it.",
    "Attaching the {topic} package. Flag any questions before end of day.",
    "Quick note on {topic}: the numbers are refreshed and ready.",
]
_EMAIL_SUBJ = ["{topic} update", "{topic} files", "re: {topic}", "{topic} — for review",
               "{topic} package"]
_CHAT = [
    "can someone take a look at the {topic} items when free?",
    "pushed the {topic} changes, review appreciated",
    "heads up — {topic} is updated on the share",
    "quick q on {topic}, got a sec?",
    "{topic} looks good on my end, moving on",
]
_TICKET_TITLE = ["{topic}: follow-up needed", "Investigate {topic} issue",
                 "{topic} cleanup", "Update {topic} docs", "{topic} request"]
_TICKET_BODY = [
    "Tracking the {topic} work item. Scope is small; assigning to the team.",
    "Noticed the {topic} task is outstanding — adding details and next steps.",
    "Please action the {topic} request by end of sprint.",
]
_PR_TITLE = ["{topic}: refactor", "Fix {topic} edge case", "Add {topic} tests",
             "{topic} performance pass", "Update {topic} config"]
_PR_DESC = [
    "Small change to {topic}. Tests pass locally. Requesting review.",
    "Refactors {topic} for clarity; no behavioural change expected.",
    "Addresses the {topic} ticket; please review the diff.",
]


class TemplateRenderer(Renderer):
    name = "template"

    def render(self, plans: list[ContentPlan]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for p in plans:
            g = _gen(p.event_id)
            topic = p.topic
            fields: dict[str, str] = {}
            for f in p.fields:
                fields[f] = self._field(f, p.surface, topic, g)
            out[p.event_id] = fields
        return out

    def _field(self, field: str, surface: str, topic: str, g: np.random.Generator) -> str:
        def pick(items):
            return items[int(g.integers(len(items)))].format(topic=topic)

        if surface == "email":
            if field == "subject":
                return pick(_EMAIL_SUBJ)
            return f"{pick(_GREET)}\n\n{pick(_EMAIL_BODY)}\n\n{pick(_SIGN)}"
        if surface == "chat":
            return pick(_CHAT)
        if surface == "ticket":
            return pick(_TICKET_TITLE) if field == "title" else pick(_TICKET_BODY)
        if surface == "vcs":
            return pick(_PR_TITLE) if field == "title" else pick(_PR_DESC)
        return pick(_EMAIL_BODY)


def _gen(event_id: str) -> np.random.Generator:
    h = hashlib.blake2b(event_id.encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))
