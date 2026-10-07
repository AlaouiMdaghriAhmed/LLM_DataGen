"""Claude Messages API renderer — a label-blind, cost-bounded quality upgrade.

Design notes (mapped to the ``claude-api`` guidance):

* **Label-blind.** The model is handed only :meth:`ContentPlan.to_prompt_payload`.
* **Structured output.** Responses are constrained with ``output_config.format``
  (JSON schema), so we get machine-parseable items back and never scrape prose.
* **Prompt caching.** The instruction block is a byte-stable, cached system prompt;
  only the per-batch item list varies, so repeated batches hit the cache.
* **Batching.** Plans are chunked (default 20/request) to amortise latency/cost; a
  :meth:`render_via_batch_api` path uses the 50%-cheaper Message Batches API for
  large offline jobs.
* **Retries & graceful degradation.** The SDK retries 429/5xx; on persistent failure
  (or if the SDK/credentials are missing) a chunk falls back to the template renderer
  so the pipeline never hard-fails.
* **Model.** Defaults to ``claude-opus-4-8`` with low effort and no extended thinking
  (rendering is a simple transformation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .base import ContentPlan, Renderer
from .template import TemplateRenderer

_SYSTEM = (
    "You generate short, realistic, strictly neutral workplace text for a synthetic "
    "office-communications dataset. You will receive a JSON list of items; each item "
    "describes a role, team, generic topic, intent, audience, and the text fields to "
    "produce. Write ordinary, mundane business prose as that role plausibly would.\n\n"
    "Hard rules:\n"
    "- Produce ONLY the requested fields for each item.\n"
    "- Keep each field short (subject <= 8 words; body/message/description <= 60 words).\n"
    "- Never imply wrongdoing, risk, urgency beyond the routine, security, monitoring, "
    "or anything unusual. This is ordinary day-to-day correspondence.\n"
    "- Do not invent real names, emails, secrets, credentials, or specific numbers.\n"
    "- Do not mention that this is synthetic or that you are an AI.\n"
    "Return a JSON object: {\"items\": [{\"event_id\": str, \"fields\": {field: text}}]}"
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "fields": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["event_id", "fields"],
            },
        }
    },
    "required": ["items"],
}


@dataclass
class RenderStats:
    backend: str = "anthropic"
    requests: int = 0
    items: int = 0
    fallback_items: int = 0
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "requests": self.requests,
            "items_rendered": self.items,
            "fallback_items": self.fallback_items,
            "cache_read_tokens": self.cache_read_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class AnthropicRenderer(Renderer):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        chunk_size: int = 20,
        max_tokens: int = 4096,
        effort: str = "low",
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.effort = effort
        self._fallback = TemplateRenderer()
        self.stats = RenderStats(backend="anthropic")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so the dep is optional

            self._client = anthropic.Anthropic()
        return self._client

    # -- Renderer API -----------------------------------------------------
    def render(self, plans: list[ContentPlan]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for i in range(0, len(plans), self.chunk_size):
            chunk = plans[i : i + self.chunk_size]
            try:
                out.update(self._render_chunk(chunk))
            except Exception as exc:  # degrade gracefully, never hard-fail
                self.stats.fallback_items += len(chunk)
                out.update(self._fallback.render(chunk))
                print(f"[render] chunk fell back to template: {type(exc).__name__}: {exc}")
        return out

    def _payload(self, chunk: list[ContentPlan]) -> str:
        items = []
        for p in chunk:
            d = p.to_prompt_payload()
            d["event_id"] = p.event_id  # id is not a label; needed to map results
            items.append(d)
        return json.dumps({"items": items}, sort_keys=True)

    def _render_chunk(self, chunk: list[ContentPlan]) -> dict[str, dict[str, str]]:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": _SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": self._payload(chunk)}],
        )
        self.stats.requests += 1
        u = resp.usage
        self.stats.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.stats.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.stats.output_tokens += getattr(u, "output_tokens", 0) or 0

        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        data = json.loads(text)
        result: dict[str, dict[str, str]] = {}
        wanted = {p.event_id: set(p.fields) for p in chunk}
        for item in data.get("items", []):
            eid = item.get("event_id")
            if eid in wanted:
                fields = {k: str(v) for k, v in item.get("fields", {}).items() if k in wanted[eid]}
                result[eid] = fields
                self.stats.items += 1
        # fill any the model omitted, from the template renderer
        missing = [p for p in chunk if p.event_id not in result]
        if missing:
            self.stats.fallback_items += len(missing)
            result.update(self._fallback.render(missing))
        return result

    # -- offline bulk path ------------------------------------------------
    def render_via_batch_api(self, plans: list[ContentPlan], poll_seconds: int = 30):
        """Render using the Message Batches API (50% cheaper; up to 24h latency).

        Returns ``{event_id: {field: text}}``. Falls back to the synchronous path on
        any error. Intended for large, non-interactive dataset builds.
        """
        import time
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = self._get_client()
        reqs = []
        chunk_map: dict[str, list[ContentPlan]] = {}
        for i in range(0, len(plans), self.chunk_size):
            chunk = plans[i : i + self.chunk_size]
            cid = f"chunk-{i // self.chunk_size:05d}"
            chunk_map[cid] = chunk
            reqs.append(
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=[{"type": "text", "text": _SYSTEM,
                                 "cache_control": {"type": "ephemeral"}}],
                        output_config={"effort": self.effort,
                                       "format": {"type": "json_schema", "schema": _SCHEMA}},
                        messages=[{"role": "user", "content": self._payload(chunk)}],
                    ),
                )
            )
        batch = client.messages.batches.create(requests=reqs)
        while True:
            b = client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(poll_seconds)

        out: dict[str, dict[str, str]] = {}
        for res in client.messages.batches.results(batch.id):
            chunk = chunk_map.get(res.custom_id, [])
            if res.result.type != "succeeded":
                out.update(self._fallback.render(chunk))
                continue
            msg = res.result.message
            text = next((blk.text for blk in msg.content if getattr(blk, "type", "") == "text"), "")
            try:
                data = json.loads(text)
            except Exception:
                out.update(self._fallback.render(chunk))
                continue
            wanted = {p.event_id: set(p.fields) for p in chunk}
            seen = set()
            for item in data.get("items", []):
                eid = item.get("event_id")
                if eid in wanted:
                    out[eid] = {k: str(v) for k, v in item.get("fields", {}).items() if k in wanted[eid]}
                    seen.add(eid)
            out.update(self._fallback.render([p for p in chunk if p.event_id not in seen]))
        return out
