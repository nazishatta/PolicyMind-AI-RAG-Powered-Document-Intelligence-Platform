# One-Page Executive Summary — PolicyMind-AI Research Direction

**Classification:** Private / Personal Records
**Prepared by:** Nazish Atta | **Date:** 2026-06-08
**Full detail:** See `TECHNICAL_IMPACT_MEMO.md`, `PROPOSED_RESEARCH_DIRECTION.md`, `CODEBASE_EVIDENCE_MAP.md`

---

## Technical Direction

Design and advance trustworthy AI-powered document intelligence systems for public-interest, policy, regulatory, and research documents. The core technical approach is GraphRAG — hybrid retrieval that fuses dense vector similarity with structured knowledge graph evidence — combined with citation-backed answer generation and a deterministic trust layer that computes reliability signals from pipeline state rather than from language model self-reporting.

---

## Public-Interest Problem

Policy documents (legislation, international agreements, regulatory submissions, public-health guidelines) are among the most consequential documents in modern governance. They are too long for LLM context windows, dense with named-entity relationships that pure vector search cannot capture, and require accountability standards — source citations, calibrated uncertainty — that standard AI tools do not provide. A system that generates confident-sounding answers with no citations and no uncertainty signal is unsuitable for policy research, civic technology, or investigative journalism.

---

## Implemented Evidence (Verified in Source Code)

| Component | Key Evidence |
|---|---|
| **GraphRAG hybrid retrieval** | `retriever.py`: proportional graph boost (`Σ entity_conf / \|entity_set\|`), multi-hop expansion with `_MULTIHOP_DISCOUNT = 0.8`, per-call `graph_depth` 1–3 without instance mutation |
| **Citation-backed QA** | `citation_engine.py` + `rag_pipeline.py`: full provenance (chunk_id, page_number, section_heading, excerpt), post-generation reranking by page references named in answer |
| **Deterministic trust layer** | 5 pipeline functions: `answer_type`, `evidence_quality`, `confidence_note`, `confidence`, `limitations` — none use the LLM; reproducible across providers |
| **Trust gate** | Structural refusal before any LLM call when corpus is empty — eliminates primary hallucination pathway |
| **9-endpoint REST API** | FastAPI + Pydantic v2; OpenAPI auto-generated; typed error hierarchy; dependency injection |
| **405-test offline suite** | 11 modules; 405 passed, 1 skipped; conftest.py in-memory stack; no API key required |
| **3-job CI pipeline** | Ruff lint + Mypy; pytest on Python 3.10/3.11/3.12; live smoke test (40/40 checks passed) |
| **Provider-agnostic LLM layer** | `BaseLLMProvider` ABC; Anthropic, OpenAI, Mock — pipeline changes no code to swap models |
| **9-class evaluation framework** | `test_evaluation.py` (59 tests) + `scripts/evaluate.py`; covers retrieval, citation, faithfulness, hallucination risk, latency, reproducibility |
| **Responsible AI documentation** | `docs/responsible_ai.md`: 8 trust mechanisms, intended use, operator obligations; `.env.example`: zero secrets |

**Status:** v0.1.0 Alpha. Backend complete. Frontend planned, not implemented.

---

## Remaining Gaps (Priority Order)

1. **No real benchmark.** All evaluation uses synthetic documents. No empirical retrieval quality measurement against real policy corpora exists yet.
2. **No ablation evaluation.** Alpha (0.7) and graph depth (default 1) were set by design reasoning, not measurement. The core GraphRAG claim — that hybrid retrieval outperforms vector-only — has not been empirically tested.
3. **No cross-encoder reranking.** A well-established retrieval quality improvement; the existing `HybridRetriever` interface accommodates it without structural changes.
4. **Rule-based relation extraction (4 types only).** Proximity co-occurrence misses syntactically complex relationships. Dependency-parse extraction would improve graph edge quality.
5. **Heuristic confidence score.** `mean_top3_cosine + graph_bonus` is an ordinal proxy, not a calibrated probability. No correlation with human-assessed answer quality has been measured.

---

## Next Milestones

| Milestone | Scope |
|---|---|
| **M1 — Hosted demo** | Deploy to Railway/Render in mock mode; zero API key required for any reviewer |
| **M2 — Frontend + Docker** | Streamlit interface for non-technical users; Docker Compose for reproducible deployment |
| **M3 — Real benchmark** | 500+ annotated QA pairs from EUR-Lex or UN documents; first empirical retrieval baseline |
| **M4 — Ablation evaluation** | Vector-only vs hybrid vs depth variants on benchmark; justify or revise alpha = 0.7 |
| **M5 — Improved graph quality** | Dependency-parse relation extraction; calibrated confidence scoring on annotated data |

---

*All claims in this summary are grounded in the PolicyMind-AI source code and the three supporting documents in this directory. No capability is claimed that is not verified in the codebase.*
