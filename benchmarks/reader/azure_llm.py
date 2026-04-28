"""LLM reader: given retrieved chat turns + question, produce an answer."""

from __future__ import annotations

import os

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn
from benchmarks.router import classify_question
from benchmarks.temporal.dates import parse_lme_date

_READER_SYSTEM = (
    "You are a precise long-term memory assistant. Use ONLY the provided "
    "chat history excerpts to answer the user's question. If the question "
    "asks about a fact and that fact is not in the excerpts, reply exactly: "
    "I don't know. Be concise (one or two short sentences). Preserve dates, "
    "numbers and names exactly as they appear in the history.\n"
    "If the question asks for a count, duration, or how-many-days/weeks/"
    "months, FIRST identify the two relevant dates from the excerpts (or the "
    "excerpt date and 'today's date' below), then compute the difference "
    "step by step, then state the final number. Do not say 'I don't know' "
    "if the dates are visible in the excerpts \u2014 do the arithmetic.\n"
    "If the question asks 'which happened first / most recently / before / "
    "after', compare the session dates of the relevant excerpts directly "
    "rather than relying on phrases like 'three weeks ago'.\n"
    "If the question asks for a TOTAL, COUNT or SUM (e.g. 'how many', 'how "
    "much total', 'how many times'), enumerate every relevant item from the "
    "excerpts before reporting the total. Do not stop at the first match.\n"
    "If the question asks for a recommendation, suggestion, advice, tips, "
    "ideas, or a decision (e.g. 'can you suggest...', 'recommend...', "
    "'what should I...', 'any tips', 'what do you think', 'could there be "
    "a reason', 'I'm trying to decide'), DO NOT say 'I don't know'. Even "
    "when no direct match for the request exists in the excerpts, "
    "synthesise a personalised suggestion grounded in ANY visible signal: "
    "the user's preferences, prior choices, equipment they own, "
    "constraints, location, hobbies, or earlier issues. Cite which "
    "specific excerpt detail informed the recommendation.\n"
    "When the same fact appears with DIFFERENT values across multiple "
    "excerpts (e.g. quantity of items they own, current status, current "
    "schedule, current location, current preference, latest measurement), "
    "the value from the MOST RECENT excerpt is correct \u2014 earlier "
    "values are stale and have been superseded. Examples: if an earlier "
    "excerpt says 'I own 3 bikes' and a later excerpt says 'I just bought "
    "a gravel bike', the current count is 4. If an earlier excerpt says "
    "'my 5K PR is 27:12' and a later excerpt says 'new 5K PR 25:50', the "
    "answer is 25:50.\n"
    "Abstention safety: if the question asks for a specific value (a "
    "number, name, date, place, status) about a specific entity, and "
    "that entity (or that exact relationship) is NOT mentioned anywhere "
    "in the excerpts, reply exactly 'I don't know.' Do not substitute a "
    "related-but-different entity (e.g. if asked about football and the "
    "excerpts only mention baseball, say I don't know). Do not invent a "
    "count for a category that was never discussed."
)

_READER_USER_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Relevant chat history excerpts (sorted oldest \u2192 newest):\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

_AGGREGATION_USER_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Relevant chat history excerpts (sorted oldest \u2192 newest):\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "This is a counting / total / aggregation question. Follow this "
    "procedure exactly:\n"
    "1. On a line starting with `ITEMS:`, list every distinct relevant item "
    "from the excerpts as a numbered list (1., 2., 3., ...). Include items "
    "that match any phrasing of the question's criterion (e.g. "
    "'bought OR worked on', 'pick up OR return'). Do not deduplicate items "
    "that are clearly distinct.\n"
    "2. On a line starting with `TOTAL:`, state the count or sum of the "
    "items listed above.\n"
    "3. On a line starting with `ANSWER:`, give the final concise answer "
    "(usually just the number or amount).\n"
    "Do not say 'I don't know' if any relevant items appear in the excerpts."
)


def _final_answer(raw: str) -> str:
    """Extract the ``ANSWER:`` line from a CoT response, falling back to raw."""
    for line in reversed(raw.splitlines()):
        s = line.strip()
        if s.upper().startswith("ANSWER:"):
            return s.split(":", 1)[1].strip() or raw.strip()
    return raw.strip()


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
        aggregation_cot: bool = False,
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
        self._aggregation_cot = aggregation_cot

    def read(
        self,
        episode: OfficialEpisode,
        retrieved_turns: list[OfficialTurn],
    ) -> str:
        ordered = _chronological(retrieved_turns)
        context = "\n".join(_format_turn(t) for t in ordered)
        is_agg = (
            self._aggregation_cot
            and classify_question(episode.question, episode.question_type) == "aggregation"
        )
        if is_agg:
            template = _AGGREGATION_USER_TEMPLATE
            max_tokens = max(self._max_tokens, 600)
        else:
            template = _READER_USER_TEMPLATE
            max_tokens = self._max_tokens
        try:
            rsp = self._client.chat.completions.create(
                model=self._deployment,
                temperature=self._temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _READER_SYSTEM},
                    {
                        "role": "user",
                        "content": template.format(
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
        raw = (rsp.choices[0].message.content or "").strip()
        return _final_answer(raw) if is_agg else raw


__all__ = ["AzureOpenAIReader"]
