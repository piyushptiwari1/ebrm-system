"""LLM reader: given retrieved chat turns + question, produce an answer."""

from __future__ import annotations

import os

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.temporal.dates import parse_lme_date

_READER_SYSTEM = (
    "You are a precise long-term memory assistant. Use ONLY the provided "
    "chat history excerpts to answer the user's question. If the answer "
    "is not present in the excerpts, reply exactly: I don't know. "
    "Be concise (one or two short sentences). Preserve dates, numbers and "
    "names exactly as they appear in the history.\n"
    "If the question asks for a count, duration, or how-many-days/weeks/"
    "months, FIRST identify the two relevant dates from the excerpts (or the "
    "excerpt date and 'today's date' below), then compute the difference "
    "step by step, then state the final number. Do not say 'I don't know' "
    "if the dates are visible in the excerpts \u2014 do the arithmetic.\n"
    "If the question asks 'which happened first / most recently / before / "
    "after', compare the session dates of the relevant excerpts directly "
    "rather than relying on phrases like 'three weeks ago'."
)

_READER_USER_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Relevant chat history excerpts (sorted oldest \u2192 newest):\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _format_turn(turn: OfficialTurn) -> str:
    return f"[{turn.session_date}] [{turn.role}] {turn.content.strip()}"


def _chronological(turns: list[OfficialTurn]) -> list[OfficialTurn]:
    """Sort turns by session_date ascending; unparseable dates go last."""

    def key(t: OfficialTurn) -> tuple[int, float]:
        dt = parse_lme_date(t.session_date)
        if dt is None:
            return (1, 0.0)
        return (0, dt.timestamp())

    return sorted(turns, key=key)


class AzureOpenAIReader:
    """Compose an answer from retrieved turns using an Azure deployment."""

    def __init__(
        self,
        *,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AzureOpenAIReader requires `openai`. "
                "Install with: pip install 'ebrm-system[embedders]'"
            ) from exc

        self._deployment = deployment or os.environ["AZURE_DEPLOYMENT_NAME"]
        self._client = AzureOpenAI(
            azure_endpoint=endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version or os.environ["AZURE_API_VERSION"],
        )
        self.name = f"azure-reader-{self._deployment}"
        self._max_tokens = max_tokens
        self._temperature = temperature

    def read(
        self,
        episode: OfficialEpisode,
        retrieved_turns: list[OfficialTurn],
    ) -> str:
        ordered = _chronological(retrieved_turns)
        context = "\n".join(_format_turn(t) for t in ordered)
        try:
            rsp = self._client.chat.completions.create(
                model=self._deployment,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _READER_SYSTEM},
                    {
                        "role": "user",
                        "content": _READER_USER_TEMPLATE.format(
                            today=episode.question_date,
                            context=context or "(no excerpts retrieved)",
                            question=episode.question,
                        ),
                    },
                ],
            )
        except Exception:
            # Treat as abstention so downstream judge handles it correctly.
            return "I don't know."
        return (rsp.choices[0].message.content or "").strip()


__all__ = ["AzureOpenAIReader"]
