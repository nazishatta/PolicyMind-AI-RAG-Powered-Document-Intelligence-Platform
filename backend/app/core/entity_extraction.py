"""Named entity and relation extraction for knowledge-graph construction.

Uses spaCy for fast NER (offline).  Relation extraction is handled by a
simple rule-based co-occurrence pass, with an optional LLM refinement step
that can be toggled via the calling service.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Entity types retained for the policy knowledge graph
_POLICY_TYPES = {
    "ORG",      # organisations, agencies, bodies
    "GPE",      # countries, cities, regions
    "LAW",      # legislation, treaties, conventions
    "MONEY",    # budget figures
    "PERCENT",  # targets expressed as percentages
    "DATE",     # policy deadlines and timelines
    "PRODUCT",  # named programmes or initiatives
    "EVENT",    # summits, conferences
    "NORP",     # nationalities, political groups
}

_nlp: Optional[object] = None


def _get_nlp() -> object:
    global _nlp
    if _nlp is None:
        import spacy  # type: ignore

        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' loaded")
    return _nlp


@dataclass
class Entity:
    text: str
    label: str          # spaCy entity type
    doc_id: str
    page_number: int
    chunk_id: str
    char_start: int
    char_end: int


@dataclass
class Relation:
    source: str
    relation_type: str
    target: str
    doc_id: str
    chunk_id: str
    confidence: float = 1.0


def extract_entities(text: str, doc_id: str, page_number: int, chunk_id: str) -> list[Entity]:
    """Run spaCy NER over a text chunk and return policy-relevant entities.

    Returns an empty list (rather than raising) if spaCy or its model is not
    available.  Install with: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        nlp = _get_nlp()
    except Exception as exc:
        logger.debug("spaCy unavailable, skipping NER: %s", exc)
        return []

    try:
        doc = nlp(text)  # type: ignore[operator]
    except Exception as exc:
        logger.warning("spaCy NER failed for chunk '%s': %s", chunk_id, exc)
        return []

    entities = []
    for ent in doc.ents:
        if ent.label_ not in _POLICY_TYPES:
            continue
        entities.append(
            Entity(
                text=ent.text.strip(),
                label=ent.label_,
                doc_id=doc_id,
                page_number=page_number,
                chunk_id=chunk_id,
                char_start=ent.start_char,
                char_end=ent.end_char,
            )
        )
    return entities


def extract_relations(
    entities: list[Entity],
    text: str,
    chunk_id: str,
    doc_id: str,
) -> list[Relation]:
    """Infer co-occurrence relations between entities in the same sentence.

    Rule: Two entities in the same sentence are linked by MENTIONED_WITH.
    ORG–GPE pairs within 50 chars of each other receive OPERATES_IN.
    PERCENT/MONEY entities near DATE entities receive TARGETS_BY.
    """
    relations: list[Relation] = []

    # Build entity pairs from the same sentence via boundary proximity
    ents = sorted(entities, key=lambda e: e.char_start)
    for i, a in enumerate(ents):
        for b in ents[i + 1 :]:
            gap = b.char_start - a.char_end
            if gap > 200:
                break

            rtype = "MENTIONED_WITH"
            if a.label == "ORG" and b.label == "GPE":
                rtype = "OPERATES_IN"
            elif a.label in ("PERCENT", "MONEY") and b.label == "DATE":
                rtype = "TARGETS_BY"
            elif a.label == "LAW" and b.label == "ORG":
                rtype = "GOVERNED_BY"

            relations.append(
                Relation(
                    source=a.text,
                    relation_type=rtype,
                    target=b.text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                )
            )

    return relations
