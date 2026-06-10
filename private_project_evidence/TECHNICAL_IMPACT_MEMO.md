# Technical Impact Memo — PolicyMind-AI

**Classification:** Private / Personal Records
**Project:** PolicyMind-AI — GraphRAG Document Intelligence Platform
**Technology Domain:** Applied AI, Information Retrieval, Backend Engineering
**Prepared by:** Nazish Atta
**Date:** 2026-06-08
**Basis:** All claims verified against source files in the PolicyMind-AI repository.

---

## 1. Project Overview

PolicyMind-AI is an independently designed and implemented backend API platform that applies Graph-augmented Retrieval-Augmented Generation (GraphRAG) to the problem of extracting accurate, citation-backed answers from structured public-interest documents. The platform ingests plain-text or publicly accessible PDF documents, constructs a named-entity knowledge graph during ingestion, and answers natural-language questions using a hybrid retrieval algorithm that fuses dense vector similarity with graph-neighbourhood evidence. Every response includes a citation list, a numerical confidence score, a human-readable confidence note, an evidence-quality classification, and a deterministic limitations list — all computed by pipeline logic rather than by the language model.

The system is fully implemented, tested, and documented. The backend API is operational in offline mock mode without any external API keys or database infrastructure. A 405-test suite runs entirely offline and passes in full. A GitHub Actions CI pipeline with three jobs (lint, multi-version unit tests, end-to-end smoke test) and a Codespaces devcontainer enable any reviewer to evaluate the system without local setup.

**Key metrics (verified by running the test suite and smoke test):**
- 405 tests passing, 1 skipped (network-dependent test intentionally excluded from CI)
- 406 tests collected across 11 modules
- Smoke test: 40/40 checks passed against a live server in mock mode
- CI: Python 3.10, 3.11, 3.12 matrix
- Coverage gate: 60% enforced at `fail_under` in `pyproject.toml`
- Version: 0.1.0, Development Status: Alpha

---

## 2. Public-Interest Problem Addressed

Policy documents — legislation, international agreements, public-health guidelines, environmental frameworks, regulatory submissions — share a structural challenge: they are long, internally cross-referenced, contain domain-specific named entities (institutions, legal instruments, monetary targets, percentage thresholds), and are frequently too long for any language model's context window. Extracting accurate, traceable answers from these documents is a high-stakes task: errors that go uncorrected can affect policy analysis, journalism, civic research, and public-sector decision-making.

Existing retrieval-augmented generation systems fail at this class of documents in three specific ways:

**First, pure semantic search misses structured relationships.** If a user asks "which institutions are responsible for oversight of the just transition framework?", a standard vector search retrieves passages that are textually similar to the question. However, the answer may depend on graph-level relationships among named entities — e.g., that "Ministry of Environment" and "Regional Development Fund" are both identified as oversight bodies across multiple document sections, connected via typed graph edges — information that is explicit in the document structure but diffuse in raw text similarity.

**Second, most systems provide no reliability signal.** Chat-with-documents tools typically return an answer without indicating how confident the system is, how many source passages support it, what the evidence quality is, or what the system cannot answer. For high-stakes research use cases this is actively harmful.

**Third, hallucination pathways are left open.** Systems that call a language model even when the retrieved context is empty or irrelevant produce plausible-sounding but factually groundless responses. This is a fundamental failure mode that requires an architectural solution — not a prompt-engineering workaround.

PolicyMind-AI addresses all three failure modes through architectural decisions rather than model tuning: the hybrid retrieval algorithm combines vector and graph evidence, the trust layer generates reliability signals deterministically from pipeline state, and the trust gate prevents any language model call when no relevant passages are retrieved.

---

## 3. Technical Originality

The following design decisions distinguish PolicyMind-AI from standard RAG implementations. Each is implemented in and verifiable from the source code.

### 3.1 Proportional Graph Boost in Hybrid Score Fusion

**File:** `backend/app/core/retriever.py`

Standard GraphRAG implementations apply a binary graph boost — a chunk either has graph evidence or it does not. PolicyMind-AI implements a proportional graph boost:

```
graph_boost = Σ entity_confidences[e] / |entity_set|
```

where `entity_confidences[e]` is the mean confidence of all edges involving entity `e`. The fused retrieval score is:

```
fused = α × vector_score + (1 − α) × graph_boost
```

where α = 0.7 (configurable via `GRAPH_VECTOR_ALPHA` environment variable). This means a chunk whose entities have high-confidence graph connections scores proportionally higher — not just a binary presence/absence penalty. The result is a smooth ranking signal that rewards well-connected, high-confidence graph evidence.

### 3.2 Multi-Hop Graph Neighbourhood Expansion with Confidence Discounting

**File:** `backend/app/core/retriever.py` — `_expand_graph()`, constant `_MULTIHOP_DISCOUNT = 0.8`

The retriever supports graph traversal up to 3 hops from directly matched entities. Indirect evidence (entities connected to matched entities via intermediate nodes) is included but discounted at `0.8` per hop. This means evidence from a second-degree neighbour contributes `0.8²` of a direct neighbour's confidence weight. The effect is that multi-hop inference is admitted as evidence while being appropriately discounted relative to direct co-occurrence.

The `graph_depth` parameter (1–3) can be specified per API call without mutating instance state:

```python
effective_depth = graph_depth if graph_depth is not None else self._graph_depth
```

### 3.3 Post-Generation Citation Reranking

**File:** `backend/app/services/rag_pipeline.py` — `_rerank_citations_by_answer()`

After the language model generates its answer, the pipeline parses page-number references from the answer text using a regex (`\bp\.?\s*(\d+)\b`) and stable-sorts the `citations` list so that pages explicitly named in the answer appear first. This ensures the API response's citation order reflects the language model's own emphasis — a form of post-hoc citation alignment that is not present in standard RAG pipelines.

### 3.4 Deterministic Trust Layer (Not Prompt-Engineered)

**File:** `backend/app/services/rag_pipeline.py` — five trust-layer functions

All five trust fields are computed by deterministic pipeline logic, not by asking the language model to self-report:

- `_compute_confidence()` — numerical formula over retrieval scores and graph evidence count
- `_compute_confidence_note()` — human-readable breakdown of what drove the score
- `_classify_answer_type()` — `cited` / `partial` / `refused` / `no_corpus` by pattern matching and state inspection
- `_classify_evidence_quality()` — `strong` / `moderate` / `weak` / `insufficient` by chunk count and score thresholds
- `_infer_limitations()` — deterministic list of active caveats based on pipeline state

Because these functions do not use the language model, their outputs are reproducible and do not vary with model temperature or version.

### 3.5 Trust Gate: Structural Hallucination Prevention

**File:** `backend/app/services/rag_pipeline.py` — `query()` method

When the retrieval step returns zero chunks, the pipeline returns a structured `no_corpus` response immediately without making any language model call. This eliminates the primary hallucination pathway — generating content from an empty or irrelevant context. The refusal response is fully structured (same schema as a normal answer) so downstream consumers handle it programmatically rather than catching unexpected empty strings.

### 3.6 Citation-First System Prompt Engineering

**File:** `backend/app/services/rag_pipeline.py` — `_SYSTEM_PROMPT`

The system prompt instructs the language model to: (1) cite sources before making claims, (2) use inline page references in the format `[p. N]`, (3) prefix partial answers with `"Based on limited evidence:"`, (4) refuse explicitly when the context does not support an answer, and (5) make no speculative conclusions. This prompt is version-controlled and consistent across all language model providers.

### 3.7 Provider-Agnostic LLM Interface

**File:** `backend/app/services/llm_service.py`

`BaseLLMProvider` is an abstract base class defining `complete()`, `acomplete()`, `provider_name`, and `model_id`. Three concrete implementations exist: `AnthropicProvider` (Claude), `OpenAIProvider` (GPT), and `MockProvider` (deterministic offline stub). A factory function selects the active provider from environment configuration. The pipeline layer is entirely provider-agnostic and requires no modification to switch models.

---

## 4. System Architecture

The system is divided into eight cohesive layers, each with a single responsibility and a typed interface.

```
Client / API consumer
        |
FastAPI API layer (routes: root, health, ingest, query, graph)
        |
Schema / Validation layer (Pydantic v2 request/response models)
        |
        +----------------------+
        |                      |
Ingestion layer           Query layer
connectors.py             rag_pipeline.query()
document_loader.py        retriever.retrieve()
text_chunker.py           citation_engine.py
entity_extraction.py      Trust layer functions
        |                      |
        +----------------------+
        |                      |
Embedding layer           LLM Provider layer
sentence-transformers     BaseLLMProvider ABC
all-MiniLM-L6-v2          Anthropic / OpenAI / Mock
        |
        +---------------------+
        |                     |
Vector store layer       Graph layer
ChromaDB / InMemory      NetworkX DiGraph / Neo4j
        |
Configuration (pydantic-settings, env vars, lru_cache singleton)
        |
Testing layer (11 modules, 405 tests, conftest.py in-memory stack)
```

**Storage:** ChromaDB is the default persistent vector store. An `InMemoryVectorStore` (NumPy dot-product similarity) is used in tests and mock mode. The knowledge graph is stored in a NetworkX `DiGraph` by default; a Neo4j adapter satisfies the same `BaseGraphService` interface.

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional normalised float32 vectors. The model is lazy-loaded on first use and cached at the process level.

**Entity types extracted:** ORG, GPE, LAW, MONEY, PERCENT, DATE, PRODUCT, EVENT, NORP (nine types filtered from spaCy `en_core_web_sm`).

**Relation types extracted:** MENTIONED_WITH, OPERATES_IN, TARGETS_BY, GOVERNED_BY — assigned by proximity co-occurrence rules in `entity_extraction.py`.

---

## 5. Implementation Contributions

All components described in this memo were independently designed and implemented. The following is a function-level account of original contributions:

### Backend API Layer
- Designed and implemented all 9 API routes across 5 route modules (`root`, `health`, `ingest`, `query`, `graph`)
- Defined the typed error hierarchy for ingestion failures: `DocumentFetchError` → 502, `DocumentSizeError` → 413, `DocumentEmptyError`/`ParseError` → 422
- Implemented FastAPI dependency injection for pipeline instantiation (`dependencies.py`)
- Implemented all 8 Pydantic v2 schema classes: `QueryRequest`, `IngestRequest`, `AnswerResponse`, `Citation`, `RetrievedChunk`, `GraphEvidence`, `DocumentDetail`, `HealthResponse`

### Ingestion Pipeline
- Implemented `DocumentConnector` Protocol and `connector_from_request` factory
- Implemented `load_from_url()` with httpx streaming, size limit enforcement, `pdfplumber` → `PyPDF2` fallback chain
- Implemented `PolicyDocument` dataclass with SHA-256 stable `doc_id` (hash of normalised title + text)
- Implemented recursive semantic chunker (`_split()`) with 8-level separator hierarchy and hard-split fallback
- Implemented `_detect_heading()` with a 5-pattern regex covering article/section/numbered/ALL-CAPS/lettered headings
- Implemented `TextChunk` dataclass with `chunk_id` format `{doc_id}_p{page:04d}_c{idx:06d}`, `citation_label()`, and `word_count`/`char_count` properties

### Knowledge Graph Layer
- Implemented `extract_entities()` with spaCy NER and graceful fallback to empty list when spaCy absent
- Implemented `extract_relations()` with four typed proximity co-occurrence rules
- Implemented `InMemoryGraphService` using NetworkX `DiGraph` with node/edge CRUD and multi-hop neighbour traversal
- Defined `BaseGraphService` ABC satisfying the same interface for NetworkX and Neo4j backends

### Retrieval Layer
- Implemented `HybridRetriever.retrieve()` with candidate over-fetching (2× top_k), doc_id filter, `_expand_graph()`, and `_fuse_scores()`
- Implemented proportional graph boost: `graph_boost = Σ entity_confidences / |entity_set|`
- Implemented multi-hop graph expansion with `_MULTIHOP_DISCOUNT = 0.8` per indirect hop and triple deduplication
- Implemented per-call `graph_depth` override without instance state mutation

### RAG Pipeline
- Implemented `GraphRAGPipeline.ingest()` orchestrating chunking, embedding, vector upsert, entity extraction, relation extraction, and graph storage
- Implemented `GraphRAGPipeline.query()` orchestrating retrieval, trust gate, LLM call, citation reranking, and trust layer
- Implemented `_rerank_citations_by_answer()` with regex page-reference parsing and stable-sort
- Implemented `_compute_confidence()`: `min(mean_top3_cosine + len(graph_triples) × 0.05, 1.0)`, with 0.15 cap on graph contribution
- Implemented `_classify_answer_type()`: deterministic `cited` / `partial` / `refused` / `no_corpus` classification
- Implemented `_classify_evidence_quality()`: `strong` / `moderate` / `weak` / `insufficient` by chunk count and score thresholds
- Implemented `_infer_limitations()`: deterministic caveats including low confidence, entity extraction status, single-document scope, graph depth, and corpus size
- Wrote and version-controlled `_SYSTEM_PROMPT` enforcing citation-first, inline citations, refusal, and no-speculation rules

### LLM Service
- Designed and implemented `BaseLLMProvider` ABC with `complete()`, `acomplete()`, `provider_name`, `model_id`
- Implemented `AnthropicProvider` with native `anthropic.Anthropic` and `anthropic.AsyncAnthropic`
- Implemented `OpenAIProvider` with native `openai.OpenAI` and `openai.AsyncOpenAI`
- Implemented `MockProvider` with context-aware deterministic stub: parses page markers from context, references question text, returns structured fake citations
- Implemented `build_llm_provider` factory with environment-driven selection

### Vector Store
- Implemented `BaseVectorStore` ABC with `upsert`, `search`, `get_by_doc_id`, `list_documents`, `delete_by_doc_id`, `chunk_count`
- Implemented `InMemoryVectorStore` with NumPy dot-product similarity and doc-id filtering
- Implemented `ChromaVectorStore` with persistent collection, metadata-aware search, and doc-id filter

### Embeddings
- Implemented `EmbeddingModel` wrapping `SentenceTransformer` with lazy-load, process-level `functools.cache`, `normalize_embeddings=True`

### Configuration
- Implemented `Settings(BaseSettings)` with `lru_cache` singleton, all environment variables documented, mock/memory defaults for offline operation
- Ensured all API key fields exclude from `repr()` output

### Test Suite
- Designed 11 test modules covering unit, integration, API, evaluation, and smoke categories
- Wrote `conftest.py` with in-memory fixtures: `StubEmbedding` (seeded NumPy), `InMemoryVectorStore`, `InMemoryGraphService`, `MockProvider`, `api_client`
- Implemented `TestGraphDepth` class in `test_api.py` with 8 tests covering `graph_depth` 1/2/3 acceptance, 0/4 rejection, and full response-shape verification
- Implemented 9 evaluation classes in `test_evaluation.py`: retrieval relevance, citation accuracy, entity extraction quality, relation extraction quality, answer faithfulness, source coverage, hallucination risk, latency bounds, reproducibility

### DevOps and Documentation
- Authored `.github/workflows/ci.yml` with 3-job pipeline (lint, test matrix 3.10/3.11/3.12, smoke test)
- Authored `.devcontainer/devcontainer.json` with Python 3.11 Bullseye, `postCreateCommand`, `remoteEnv` mock defaults, 6 VS Code extensions, port forwarding
- Authored 7 documentation files (README, api_usage.md, architecture.md, responsible_ai.md, evaluation_framework.md, demo_walkthrough.md, ingestion_guide.md)
- Authored `scripts/smoke_test.py` (40 checks) and `scripts/evaluate.py` (standalone benchmark)
- Configured `pyproject.toml` as single source of truth: hatchling build, ruff, mypy, pytest settings, coverage gate

---

## 6. Evidence of Engineering Rigor

### Code Quality
- All source modules use `from __future__ import annotations` for forward-compatible type hints
- Abstract base classes (`BaseVectorStore`, `BaseGraphService`, `BaseLLMProvider`) enforce interface contracts across providers
- Pydantic v2 validates all input at API boundaries; internal data structures use Python dataclasses
- `structlog` with `stdlib.LoggerFactory()` provides structured, parseable log output across all modules
- `lru_cache` on `get_settings()` ensures configuration is validated once at startup and never re-parsed
- All embedding operations use `normalize_embeddings=True`; cosine similarity is therefore equivalent to dot product — no normalisation applied twice

### Configuration Discipline
- Zero secrets committed: `.env` is gitignored; `.env.example` has empty values for all API keys
- All configurable values (alpha, chunk_size, graph_depth default, low-confidence threshold, model names) are environment variables with documented defaults in `Settings(BaseSettings)`
- Mock mode is the default: `llm_provider=mock`, `vector_store_provider=chroma` (falling back to memory when chromadb absent), `graph_provider=memory`

### Dependency Management
- `pyproject.toml` declares all dependencies with version constraints; dev extras (`[dev]`) separate test and tooling dependencies from runtime
- Optional dependencies (`spaCy`, `ChromaDB`, `Neo4j`) are handled gracefully: code checks for their presence and falls back without crashing
- `requirements.txt` and `requirements-dev.txt` provide pinned lists for reproducible environments

### API Contract Discipline
- All endpoints return typed Pydantic v2 response models; FastAPI generates OpenAPI 3.1 schema automatically
- Error responses use structured JSON bodies with `detail` fields, not raw strings
- `doc_id` is a deterministic SHA-256 hash of normalised title + text; re-ingesting the same document returns the same ID

### Async Architecture
- FastAPI route handlers are `async def`; pipeline methods expose both `complete()` and `acomplete()` via `BaseLLMProvider`
- `pytest-asyncio` with `asyncio_mode = auto` enables async test functions without boilerplate decorators
- `httpx.AsyncClient` is used for URL ingestion; the sync `TestClient` wraps the async app for test isolation

---

## 7. Testing and Validation Summary

**Total test count (verified):** 406 collected, 405 passed, 1 skipped.
**Smoke test (verified):** 40/40 checks passed against a live server in mock mode.

### Test Module Breakdown

| Module | Test Count | Primary Coverage |
|---|---|---|
| `test_api.py` | 81 | All 9 endpoints; input validation; `graph_depth` 1–3 acceptance and 0/4 rejection; full `AnswerResponse` shape verification |
| `test_trust_layer.py` | 61 | All 5 trust-layer functions; confidence computation; answer-type classification; evidence-quality classification; limitations inference; no-corpus refusal; partial-answer detection |
| `test_ingestion.py` | 70 | End-to-end ingestion pipeline; URL connector stubbing; chunking correctness; `doc_id` idempotency; all `IngestionError` subclass HTTP codes |
| `test_evaluation.py` | 59 | 9 evaluation classes: retrieval relevance, citation accuracy, entity extraction quality, relation extraction quality, answer faithfulness, source coverage, hallucination risk, latency bounds, reproducibility |
| `test_graph_service.py` | 25 | Node/edge CRUD; multi-hop neighbour traversal at depth 1–3; entity and relation counts; isolated-node handling |
| `test_rag_pipeline.py` | 30 | Full pipeline orchestration; trust-layer field completeness; no-corpus refusal; document CRUD; citation reranking; empty-text rejection |
| `test_retriever.py` | 19 | Vector-only path; proportional boost (multi-entity > single-entity); empty entity set; `_expand_graph` 3-tuple return; deduplication; multi-hop evidence; confidence discounting |
| `test_citation_engine.py` | 19 | `CitationEngine.record()`; `to_schema()`; APA format; relevance score clamping; `clear()` |
| `test_connectors.py` | 19 | `DocumentConnector` protocol; text connector; URL connector with httpx mock |
| `test_document_loader.py` | 12 | `PolicyDocument` construction; `doc_id` stability; page count; text normalisation |
| `test_text_chunker.py` | 11 | `min_chars` gate; overlap; section heading detection; multi-page documents |

### CI Pipeline (`.github/workflows/ci.yml`)

Three jobs run on every push and pull request:

1. **Lint:** `ruff check backend/` (fast linting + import sorting) and `mypy backend/` (static type analysis). Fails fast on style or type errors.

2. **Unit tests:** pytest on Python 3.10, 3.11, 3.12 in parallel. Coverage report generated via `pytest-cov`; XML uploaded to Codecov on the 3.11 runner. `fail_under = 60` enforced in `pyproject.toml`.

3. **Smoke test:** Starts a live Uvicorn server (`LLM_PROVIDER=mock`, `VECTOR_STORE_PROVIDER=memory`, `GRAPH_PROVIDER=memory`), polls `/health` until ready, then runs `scripts/smoke_test.py`. Verifies 40 observable system properties against the running API. Exit code 0 on full pass.

All three jobs run entirely offline; no API keys, network access, or database infrastructure are required.

### Evaluation Framework

Beyond unit and integration testing, `test_evaluation.py` implements a structured evaluation framework across nine dimensions. `scripts/evaluate.py` provides a standalone command-line benchmark that can be run against any live server instance. The evaluation measures:

- Retrieval recall against synthetic queries with known relevant pages
- Citation accuracy: whether citations returned correspond to passages that support the answer
- Entity extraction recall and precision on synthetic documents with known entities
- Relation extraction F1 on known entity pairs
- Answer faithfulness: absence of unsupported factual claims
- Source coverage: fraction of relevant source passages represented in citations
- Hallucination risk: presence of answer content with no corresponding citation
- Latency bounds: p50 and p95 end-to-end query latency under mock mode
- Reproducibility: same query returns same answer type and citations across repeated calls

---

## 8. Responsible AI and Trust Mechanisms

The following trust mechanisms are implemented in source code and documented in `docs/responsible_ai.md`:

**Trust gate against hallucination.** `rag_pipeline.query()` returns a structured `no_corpus` refusal before calling the language model when zero chunks are retrieved. The language model is never given an empty context.

**Citation-first system prompt.** `_SYSTEM_PROMPT` instructs the model to: cite before claiming, use inline `[p. N]` references, prefix partial answers with `"Based on limited evidence:"`, refuse explicitly when context is insufficient, and make no speculative inferences. This prompt is version-controlled and identical across all providers.

**Deterministic `answer_type` classification.** `cited` / `partial` / `refused` / `no_corpus` are assigned by pattern-matching on the answer text and inspecting pipeline state — not by the language model. This means the classification is reproducible across model versions.

**Deterministic evidence-quality classification.** `strong` / `moderate` / `weak` / `insufficient` are assigned by inspecting chunk counts and score distributions — not by the language model.

**Deterministic confidence score.** `confidence = min(mean_top3_cosine + min(n_triples × 0.05, 0.15), 1.0)`. The formula and its parameters are version-controlled in `rag_pipeline.py`. The score is `None` (not zero) when no corpus exists — honest representation of missing information.

**Low-confidence preamble injection.** When `confidence < 0.35`, the retriever note injected into the user prompt instructs the language model to qualify its response. This is a pipeline-level constraint, not a post-hoc disclaimer.

**Deterministic limitations list.** `_infer_limitations()` produces a list of active caveats from pipeline state: low confidence, entity extraction skipped (spaCy absent), single-document scope, shallow graph depth, small corpus size. These caveats are machine-readable fields, not prose paragraphs.

**Provider transparency.** Every `AnswerResponse` includes `provider` and `model` fields identifying exactly which language model generated the answer.

**Mock provider for safe evaluation.** `MockProvider` produces structured responses with no API calls, no external data transmission, and no costs. It is the default provider in CI and Codespaces.

**No secrets in version control.** `.env` is gitignored. `.env.example` has empty values for all API keys. `Settings` fields for API keys are excluded from `repr()` output.

---

## 9. Potential Beneficiaries

Based on the system's capabilities as implemented, the following practitioner groups stand to benefit from this class of technology:

**Policy researchers and analysts.** The citation-backed answer format, with structured `citations` carrying `page_number`, `section_heading`, `excerpt`, and `relevance_score`, enables rapid location of source passages in long policy frameworks. The trust layer's `evidence_quality` and `limitations` fields allow analysts to triage high-confidence findings from uncertain ones before committing to conclusions.

**Civic technologists and legal aid organisations.** The offline-capable, API-key-optional architecture (via mock mode and in-memory storage) means the platform can be deployed in resource-constrained environments or demonstrated without cloud infrastructure. The MIT license permits unrestricted use and modification.

**Public interest journalists.** The `answer_type: refused` response — returned when context does not support a claim — provides an explicit signal when a document does not address a question, reducing the risk of publishing unsupported inferences as fact.

**Academic and interdisciplinary researchers.** The evaluation framework (`test_evaluation.py`, `scripts/evaluate.py`) provides a structured template for measuring retrieval quality, citation accuracy, and hallucination risk — metrics directly applicable to information-retrieval and NLP research.

**AI/ML engineers and educators.** The codebase demonstrates a complete, testable GraphRAG pipeline from first principles: embedding, vector store, knowledge graph construction, hybrid score fusion, and trust layer — all in approximately 2,500 lines of production-quality Python. It serves as a reference implementation for each architectural component.

**Government and public-sector digital teams.** The structured, deterministic trust layer — where confidence, evidence quality, and limitations are computed from pipeline state rather than generated by the language model — aligns with responsible AI requirements that many public-sector frameworks are beginning to mandate.

---

## 10. Future Impact Roadmap

Practical extensions that would materially expand the system's capabilities, ordered by implementation feasibility:

**Near-term (weeks):**

1. **Hosted public demo.** Deploy to Railway, Render, or Hugging Face Spaces with `LLM_PROVIDER=mock`. No API key required; any reviewer accesses a live working system via URL.

2. **Frontend dashboard.** A Streamlit application would provide document upload, query input, citation display, and confidence visualisation. The FastAPI backend is fully ready; no backend changes required.

3. **Docker Compose.** A `Dockerfile` and `docker-compose.yml` would enable reproducible deployment across all platforms and cloud targets.

**Medium-term (months):**

4. **Real benchmark dataset.** Using publicly licensed policy documents from EUR-Lex, the UN Document Portal, or the World Bank Open Knowledge Repository, with human-annotated question-answer pairs, would replace synthetic evaluation with measurable retrieval quality against real documents.

5. **Cross-encoder reranking.** Adding a cross-encoder reranking step after initial retrieval would improve precision. The existing `BaseVectorStore` and `HybridRetriever` interfaces accommodate this without architectural changes.

6. **Dependency-parse-based relation extraction.** Replacing the proximity co-occurrence rules with a spaCy dependency-parse relation extractor would improve the quality and coverage of graph edges — particularly for complex sentences where entities are syntactically related but not adjacent.

7. **Streaming API responses.** FastAPI supports `StreamingResponse`; adding streaming for the answer generation step would improve perceived latency for long answers and enable progressive rendering in a frontend.

**Longer-term:**

8. **Calibrated confidence scoring.** Replacing the heuristic confidence formula with a calibrated retrieval scorer (trained on annotated data) would make the confidence signal interpretable as a probability rather than an ordinal ranking.

9. **Multi-document cross-referencing.** The current knowledge graph stores entity-relation triples across documents but does not yet resolve co-references across documents (e.g., "the Commission" in document A and "European Commission" in document B as the same entity). Cross-document entity resolution would significantly improve multi-document query quality.

10. **Structured output extraction.** Adding a structured extraction mode — returning entities, figures, deadlines, and obligations as typed JSON in addition to a prose answer — would enable downstream programmatic use of policy document analysis.

---

## 11. Limitations and Next Steps

These are honest, evidence-based limitations of the system as it currently exists:

**Mock LLM produces no analytical value.** `MockProvider` returns structured stubs confirming the pipeline is functional. Generating useful answers requires a configured Anthropic or OpenAI API key.

**Relation extraction is rule-based.** The four relation types (MENTIONED_WITH, OPERATES_IN, TARGETS_BY, GOVERNED_BY) are assigned by proximity co-occurrence. The extractor does not perform dependency parsing or semantic role labelling. Complex multi-clause sentences with discontinuous entity mentions will produce lower-quality graph edges.

**Confidence formula is heuristic.** `mean_top3_cosine + graph_bonus` is a proxy for answer reliability, not a calibrated probability. It correctly orders responses by evidence strength but should not be interpreted as an absolute accuracy estimate.

**Evaluation uses synthetic documents.** The 4 evaluation documents in `test_evaluation.py` are synthetic paragraphs created for test coverage. There is no human-labelled benchmark. Real-world retrieval quality is unknown.

**No frontend implemented.** The browser interface is planned; the FastAPI backend is complete and ready. A frontend would be required for non-technical users to access the system.

**No authentication.** All endpoints are unauthenticated. API key authentication is a straightforward addition via FastAPI's dependency injection but is not present in this version.

**No Docker image.** Deployment requires manual Python dependency installation. A `Dockerfile` is a near-term addition.

**spaCy dependency is optional but impactful.** Without spaCy and `en_core_web_sm`, entity extraction and graph population are silently skipped. All graph-related features (`graph_evidence`, graph boost, multi-hop traversal) degrade to no-ops. The `limitations` field notifies callers, but the capability gap is significant.

---

## 12. Short Executive Summary for Expert Reviewers

PolicyMind-AI is an independently designed and fully implemented GraphRAG backend API for structured public-interest document analysis. It is not a tutorial, a wrapper, or a framework composition — it is a ground-up implementation of hybrid vector-and-graph retrieval with original algorithmic contributions: proportional graph boost scoring, multi-hop confidence discounting, and a five-field deterministic trust layer that makes reliability signals machine-readable without relying on the language model to self-report.

The system is production-patterned: typed interfaces throughout, abstract base classes enabling provider substitution, Pydantic v2 validation at API boundaries, structured logging, and a 405-test suite that passes fully offline. A three-job GitHub Actions pipeline (lint, multi-version unit tests, end-to-end smoke test) enforces quality on every commit. A Codespaces devcontainer enables zero-setup evaluation in a browser.

The engineering decisions — trust gate preventing LLM calls on empty context, per-call graph depth without instance state mutation, post-generation citation reranking, mock provider enabling offline CI — are the product of system design thinking rather than tutorial replication. The codebase is fully readable, verifiable against the claims in this memo, and designed to be extended.

The project is at Alpha status. The backend is complete. The frontend is planned. The path to a publicly demonstrable, research-quality document intelligence system is a matter of weeks of additional engineering.

---

*All claims in this memo are based on direct inspection of source files in the PolicyMind-AI repository. No capability is claimed that is not implemented in and verifiable from the source code.*
