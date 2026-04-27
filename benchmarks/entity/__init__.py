"""Lightweight entity-aware reranker.

Extracts probable named entities, quoted phrases, numbers, and unit
literals from the question, then up-weights candidate turns that
mention any of them. No NLP dependency: regex-based, fast, robust on
multi-session and knowledge-update queries where the right turn is
identified by a specific name / number / quoted phrase.
"""

from benchmarks.entity.extractor import extract_entities
from benchmarks.entity.reranker import EntityReranker

__all__ = ["EntityReranker", "extract_entities"]
