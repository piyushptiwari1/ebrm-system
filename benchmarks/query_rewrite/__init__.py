"""Query rewriting / multi-query retrieval (v0.27).

Implements the "MultiQueryRetriever" pattern from LangChain / IRCoT /
Self-Ask: an LLM rewrites the user's question into N alternative
phrasings (paraphrases + sub-aspects), each is sent through the base
retriever, and the candidate sets are RRF-fused before downstream
reranking and reading.

Targeted at multi-session and aggregation questions where a single
phrasing of the query is unlikely to recall every relevant turn.
Strictly opt-in (router-gated by the runner) to avoid the v0.26 trap of
globally over-applying a useful-on-some / harmful-on-others change.
"""

from __future__ import annotations

from benchmarks.query_rewrite.azure_llm import AzureOpenAIQueryRewriter
from benchmarks.query_rewrite.base import QueryRewriter

__all__ = ["AzureOpenAIQueryRewriter", "QueryRewriter"]
