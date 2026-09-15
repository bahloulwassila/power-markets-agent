"""Claude-powered daily brief generator.

Takes the deterministic ReportData produced in Phase 2 and asks Claude to
synthesize it into a structured brief. The output is a JSON dict that
matches BRIEF_SCHEMA — enforced via Anthropic's tool-use mechanism, plus
a client-side validation pass.

Design choices:
    * Tool-use for structured output (rather than "just parse the JSON in
      the response") — the SDK guarantees the shape at the API level.
    * One retry on validation failure, with an explicit error message
      injected so Claude can self-correct.
    * The generator does not know about markdown formatting — it only
      produces the structured brief. Rendering is a separate concern.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

from anthropic import Anthropic

from src.agent.prompts import SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt
from src.agent.schema import BRIEF_SCHEMA, validate_brief_payload
from src.analysis.report_data import ReportData
from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Brief:
    """A fully-formed brief ready for rendering."""
    report_date: str
    prompt_version: str
    model: str
    payload: dict          # matches BRIEF_SCHEMA
    input_tokens: int      # useful for cost tracking
    output_tokens: int

    def to_dict(self) -> dict:
        return {
            "report_date": self.report_date,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "payload": self.payload,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class BriefGenerationError(RuntimeError):
    """Raised when Claude cannot produce a valid brief after retries."""


class BriefGenerator:
    """Wraps the Anthropic API to produce structured daily briefs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = CLAUDE_MODEL,
        max_tokens: int = CLAUDE_MAX_TOKENS,
        max_retries: int = 1,
        _client: Anthropic | None = None,
    ) -> None:
        # _client is a test hook — production code should not set it.
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        if _client is not None:
            self._client = _client
        else:
            key = api_key or ANTHROPIC_API_KEY
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is missing. Add it to your .env "
                    "or pass api_key explicitly."
                )
            self._client = Anthropic(api_key=key)

    def generate(self, report: ReportData) -> Brief:
        """Ask Claude to synthesize the report into a structured brief."""
        report_date = report.report_date
        payload_json = json.dumps(report.to_dict(), indent=2, default=str)
        user_prompt = build_user_prompt(report_date, payload_json)

        messages: list[dict] = [
            {"role": "user", "content": user_prompt},
        ]

        last_errors: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for attempt in range(self.max_retries + 1):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=[BRIEF_SCHEMA],
                tool_choice={"type": "tool", "name": BRIEF_SCHEMA["name"]},
                messages=messages,
            )

            input_tokens += _safe_token_count(response, "input_tokens")
            output_tokens += _safe_token_count(response, "output_tokens")

            tool_use_block = _extract_tool_use(response)
            if tool_use_block is None:
                last_errors = ["Claude did not use the brief tool"]
            else:
                payload = tool_use_block.get("input") or {}
                errors = validate_brief_payload(payload)
                if not errors:
                    return Brief(
                        report_date=report_date,
                        prompt_version=PROMPT_VERSION,
                        model=self.model,
                        payload=payload,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                last_errors = errors

            # If we get here, we need to retry. Inject the errors into the
            # conversation so Claude can self-correct rather than making
            # the same mistake.
            if attempt < self.max_retries:
                logger.warning(
                    "Brief validation failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, last_errors,
                )
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response failed validation:\n"
                        + "\n".join(f"- {e}" for e in last_errors)
                        + "\n\nRegenerate the brief, using the tool, "
                        "fixing every listed error. Do not repeat mistakes."
                    ),
                })

        raise BriefGenerationError(
            f"Claude failed to produce a valid brief after "
            f"{self.max_retries + 1} attempts. Last errors: {last_errors}"
        )


# ---------- Response parsing helpers ----------
def _extract_tool_use(response) -> dict | None:
    """Pull the first tool_use content block out of the response.

    Anthropic returns a list of typed content blocks; when tool_choice is
    forced, one of them will be of type 'tool_use'. We normalize it to a
    plain dict so tests can build fake responses easily.
    """
    for block in response.content:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type == "tool_use":
            if isinstance(block, dict):
                return block
            # SDK object → to_dict-ish access
            return {
                "type": "tool_use",
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", None),
            }
    return None


def _safe_token_count(response, key: str) -> int:
    """Extract usage counts robustly across SDK versions."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    val = getattr(usage, key, None)
    if val is None and isinstance(usage, dict):
        val = usage.get(key)
    return int(val or 0)


def generate_brief(
    report: ReportData,
    *,
    api_key: str | None = None,
) -> Brief:
    """Convenience wrapper — construct a generator and call it once."""
    gen = BriefGenerator(api_key=api_key)
    return gen.generate(report)
