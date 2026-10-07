"""Label-blind rendering pass (requirement R2).

``render_events`` scans a finished event stream for renderable free-text surfaces,
builds label-blind :class:`ContentPlan`s, runs the chosen backend, lints the output
for label leakage, and writes the cleaned text back onto each event's ``render``
field. The core simulator never calls this — rendering is strictly downstream of the
labelled stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import SimConfig
from ..events import Event
from ..org import Organization
from .base import ContentPlan, Renderer, collect_plans
from .linter import lint_and_clean, LintReport
from .template import TemplateRenderer

__all__ = [
    "Renderer", "ContentPlan", "collect_plans", "TemplateRenderer",
    "render_events", "RenderResult", "make_renderer",
]


@dataclass
class RenderResult:
    backend: str
    n_plans: int
    n_rendered: int
    lint: dict[str, Any]
    backend_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "plans": self.n_plans,
            "rendered": self.n_rendered,
            "lint": self.lint,
            "backend_stats": self.backend_stats,
        }


def make_renderer(cfg: SimConfig) -> Renderer:
    if cfg.renderer == "anthropic":
        from .anthropic_backend import AnthropicRenderer

        return AnthropicRenderer(model=cfg.model)
    return TemplateRenderer()


def render_events(
    events: list[Event],
    org: Organization,
    cfg: SimConfig,
    renderer: Renderer | None = None,
    redact_leaks: bool = True,
) -> RenderResult:
    renderer = renderer or make_renderer(cfg)
    plans = collect_plans(events, org)

    # subsample / cap for cost control (requirement R7)
    if cfg.render_fraction < 1.0 and plans:
        import numpy as np

        g = np.random.default_rng(cfg.seed)
        keep = g.random(len(plans)) < cfg.render_fraction
        plans = [p for p, k in zip(plans, keep) if k]
    if cfg.render_max_events is not None:
        plans = plans[: cfg.render_max_events]

    rendered = renderer.render(plans) if plans else {}
    rendered, lint = lint_and_clean(rendered, redact=redact_leaks)

    by_id = {e.event_id: e for e in events}
    n_rendered = 0
    for eid, fields in rendered.items():
        ev = by_id.get(eid)
        if ev is not None:
            ev.render = fields
            n_rendered += 1

    backend_stats = getattr(renderer, "stats", None)
    return RenderResult(
        backend=renderer.name,
        n_plans=len(plans),
        n_rendered=n_rendered,
        lint=lint.to_dict(),
        backend_stats=backend_stats.to_dict() if backend_stats else {},
    )
