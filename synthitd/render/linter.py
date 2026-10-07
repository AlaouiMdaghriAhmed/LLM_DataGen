"""Leakage linter — enforces R2 even against a misbehaving renderer.

No matter what produced the prose (template or LLM), rendered text is scanned for
tokens that could leak the ground-truth label: label words, intent/affect words a
blind renderer should never know, MITRE technique ids, and the literal asset names
the plan deliberately hid behind generic topics. Offending text is flagged and, by
default, redacted so the published corpus cannot be shortcut by keyword search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# words that would betray the label if they appeared in benign-looking prose
_LABEL_TERMS = [
    "malicious", "insider threat", "insider", "exfiltrat", "exfil", "steal", "stole",
    "stolen", "sabotage", "fraud", "attack", "attacker", "breach", "illegal",
    "unauthoriz", "leak confidential", "sell data", "competitor deal", "revenge",
    "disgruntl", "precursor", "anomalous", "anomaly", "suspicious", "covert",
]
_TECHNIQUE_RE = re.compile(r"\bT1\d{3}(?:\.\d{3})?\b", re.IGNORECASE)


@dataclass
class LintReport:
    scanned: int = 0
    flagged: int = 0
    redactions: int = 0
    terms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scanned_fields": self.scanned,
            "flagged_fields": self.flagged,
            "redactions": self.redactions,
            "leak_rate": round(self.flagged / max(1, self.scanned), 5),
            "term_hits": dict(sorted(self.terms.items(), key=lambda kv: -kv[1])),
        }


def _hits(text: str) -> list[str]:
    low = text.lower()
    found = [t for t in _LABEL_TERMS if t in low]
    if _TECHNIQUE_RE.search(text):
        found.append("technique_id")
    return found


def lint_and_clean(
    rendered: dict[str, dict[str, str]], redact: bool = True
) -> tuple[dict[str, dict[str, str]], LintReport]:
    report = LintReport()
    out: dict[str, dict[str, str]] = {}
    for eid, fields in rendered.items():
        clean_fields: dict[str, str] = {}
        for name, text in fields.items():
            report.scanned += 1
            found = _hits(text or "")
            if found:
                report.flagged += 1
                for t in found:
                    report.terms[t] = report.terms.get(t, 0) + 1
                if redact:
                    report.redactions += 1
                    text = "[redacted: see structured fields]"
            clean_fields[name] = text
        out[eid] = clean_fields
    return out, report
