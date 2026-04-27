"""LLM reader: given retrieved chat turns + question, produce an answer."""

from __future__ import annotations

import os

from benchmarks.datasets.longmemeval_official import OfficialEpisode, OfficialTurn

_READER_SYSTEM = (
    "You are a precise long-term memory assistant. Use ONLY the provided "
    "chat history excerpts to answer the user's question. If the answer "
    "is not present in the excerpts, reply exactly: I don't know. "
    "Be concise (one or two short sentences). Preserve dates, numbers and "
    "names exactly as they appear in the history."
)

_READER_USER_TEMPLATE = (
    "Today's date is {today}.\n\n"
    "Relevant chat history excerpts (most relevant first):\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _format_turn(turn: OfficialTurn) -> str:
    return f"[{turn.session_date}] [{turn.role}] {turn.content.strip()}"


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
        context = "\n".join(_format_turn(t) for t in retrieved_turns)
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
        return (rsp.choices[0].message.content or "").strip()


__all__ = ["AzureOpenAIReader"]
