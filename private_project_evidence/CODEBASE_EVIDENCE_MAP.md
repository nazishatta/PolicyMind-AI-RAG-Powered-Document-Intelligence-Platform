# Codebase Evidence Map — PolicyMind-AI

**Classification:** Private / Personal Records
**Prepared by:** Nazish Atta
**Date:** 2026-06-08
**Purpose:** Map existing PolicyMind-AI components to the proposed research direction; identify verified evidence; catalogue gaps; recommend next work.
**Basis:** All claims verified by reading source files. No capability is claimed that is not implemented in the repository.

---

## How to Read This Document

Each section below describes a research capability, identifies the specific source files and functions that implement it, lists the concrete evidence that a reviewer could verify, identifies the skill demonstrated, and notes what is absent or incomplete. A priority-ordered gap list follows at the end.

---

## Component 1: API-First Document Intelligence Platform

**Research relevance:** The proposed research direction requires an API-first architecture so that the document intelligence system is usable as a service — by frontends, by other systems, by evaluation harnesses — without coupling consumers to implementation internals. API-first design also enforces clean interface boundaries that are necessary for rigorous ablation evaluation.

### What is implemented

**Files:** `backend/app/api/routes/` (five route modules: root, health, ingest, query, graph), `backend/app/main.py`, `backend/app/schemas/` (request and response models)

**Endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Index — lists endpoints and active provider config |
| `/health` | GET | Enriched health check with chunk/entity/relation counts |
| `/api/v1/ingest` | POST | Document ingestion (text or URL) |
| `/api/v1/query` | POST | GraphRAG question answering |
| `/api/v1/documents` | GET | List all indexed documents |
| `/api/v1/documents/{doc_id}` | GET | Document detail and metadata |
| `/api/v1/documents/{doc_id}` | DELETE | Remove document and chunks |
| `/api/v1/graph/stats` | GET | Knowledge graph statistics |
| `/api/v1/graph/neighbours` | GET | Entity neighbourhood traversal |

**Schema design:** All requests and responses use Pydantic v2 models. `QueryRequest` enforces `question` length (5–2000 chars), `top_k` range (1–20), `graph_depth` range (1–3). `AnswerResponse` carries 14 typed fields. `IngestRequest` accepts either `text` or `url`. Error responses use structured JSON bodies with HTTP status codes matched to failure type (413, 415, 422, 502).

**OpenAPI generation:** FastAPI auto-generates OpenAPI 3.1 schema at `/docs` (Swagger UI) and `/redoc` (ReDoc) — interactive documentation available immediately when the server starts.

**Dependency injection:** `backend/app/api/dependencies.py` provides `get_pipeline()` — the pipeline is instantiated once and injected into route handlers via `Depends()`, enabling clean separation between route logic and service construction.

### Evidence a reviewer can verify

- Run `uvicorn backend.app.main:app --reload` → `/docs` shows all 9 endpoints with schemas
- `POST /api/v1/query` with `{"question": "x" * 2001}` → 422 (Pydantic validation at boundary)
- `GET /health` → returns `chunk_count`, `graph_entities`, `graph_relations` (operational observability)
- All response schemas in `backend/app/schemas/response.py` — 14 fields in `AnswerResponse`

### Skills demonstrated

FastAPI application design, Pydantic v2 schema engineering, REST API contract discipline, HTTP status code semantics, typed dependency injection, OpenAPI documentation generation.

### Gaps

- No authentication. All endpoints are publicly accessible. Adding API key or OAuth2 via FastAPI `Depends` is straightforward but not implemented.
- No rate limiting or request-size throttling at the API layer (file-size limit is enforced inside the loader, not at the API).
- No streaming response for the query endpoint. Answers are returned in full after generation completes.
- No versioning strategy beyond `v1` in the path — no deprecation headers or version negotiation.

---

## Component 2: GraphRAG Hybrid Retrieval

**Research relevance:** This is the primary technical contribution of the research direction. Hybrid retrieval — fusing dense vector similarity with structured knowledge graph evidence — is the mechanism by which policy-relevant entity relationships are incorporated into retrieval ranking. The specific algorithmic choices (proportional boost, multi-hop discounting, per-call depth override) are original design decisions that require empirical evaluation.

### What is implemented

**File:** `backend/app/core/retriever.py`

**Hybrid score fusion:**
```python
fused = α × vector_score + (1 − α) × graph_boost
# α = 0.7 default (GRAPH_VECTOR_ALPHA env var)
# graph_boost = Σ entity_confidences[e] / |entity_set|
```
`graph_boost` is proportional — not binary. A chunk whose entities have higher mean graph-edge confidence ranks proportionally higher. This is implemented in `_fuse_scores()`.

**Multi-hop graph expansion:**
```python
_MULTIHOP_DISCOUNT = 0.8   # applied per indirect hop
effective_depth = graph_depth if graph_depth is not None else self._graph_depth
```
`_expand_graph()` collects direct edges at hop 1, then for `depth > 1` collects neighbours' edges discounted by `0.8` per hop. Triples are deduplicated. Returns a 3-tuple: `(entity_hits, evidence, entity_confidences)`.

**Candidate over-fetching:** `top_k × 2` candidates fetched before fusion, capped at `top_k + 10` — ensures the reranking step has a meaningful candidate pool.

**Per-call depth override:** `retrieve(graph_depth=None)` accepts an optional caller-supplied depth without mutating the `HybridRetriever` instance state.

**doc_id filter:** Optional `doc_id` parameter scopes retrieval to a single document.

### Evidence a reviewer can verify

- `test_retriever.py` (19 tests): proportional boost test asserts multi-entity chunk scores higher than single-entity chunk; `_expand_graph` 3-tuple return tested; deduplication tested; multi-hop discounting tested
- `backend/app/config.py`: `GRAPH_VECTOR_ALPHA = 0.7` as default setting
- `_MULTIHOP_DISCOUNT = 0.8` defined as module-level constant in `retriever.py`
- `test_api.py` `TestGraphDepth`: `graph_depth=1/2/3` → 200; `graph_depth=0/4` → 422

### Skills demonstrated

Hybrid retrieval algorithm design, knowledge graph traversal, score fusion, configurable retrieval hyperparameters, API parameter forwarding without state mutation, multi-hop graph evidence with confidence discounting.

### Gaps

- **Alpha is fixed at 0.7 without empirical justification.** No ablation evaluation has compared retrieval quality at different alpha values against annotated relevance judgements.
- **No cross-encoder reranking.** The current pipeline does not use a cross-encoder (e.g., sentence-transformers cross-encoder models) to rescore the top-k candidates after initial retrieval. Cross-encoder reranking is a well-established retrieval quality improvement.
- **graph_depth default of 1 limits multi-hop evidence in practice.** The default favours speed over graph depth; the tradeoff against retrieval quality has not been measured.
- **No BM25 or sparse retrieval component.** The hybrid combines vector + graph but lacks a sparse keyword component. For policy documents with specific technical terminology, sparse retrieval often complements dense search effectively.
- **Entity extraction in the retriever path uses the same extractor as ingestion.** If spaCy is absent, the retriever falls back to empty entity sets — meaning graph boost is zero for all chunks, degrading to pure vector search with no warning at the retrieval level.

---

## Component 3: Citation-Backed Answer Generation

**Research relevance:** Citation discipline is the mechanism by which the system's outputs are made accountable and verifiable. This is not a cosmetic feature — it is the difference between an AI system that produces assertions and one that produces evidence-grounded claims that practitioners can act on.

### What is implemented

**Files:** `backend/app/core/citation_engine.py`, `backend/app/services/rag_pipeline.py`

**CitationEngine:** Accumulates `CitationRecord` objects during retrieval. Each record carries `chunk_id`, `doc_id`, `doc_title`, `page_number`, `section_heading`, `excerpt`, `relevance_score`. Exports as Pydantic `Citation` objects or APA-format strings.

**Citation provenance in retrieval:** Every `TextChunk` carries `page_number`, `section_heading`, `char_start`, `char_end`, `source_url` — all populated during chunking from the `PolicyDocument` page structure.

**System prompt discipline:** `_SYSTEM_PROMPT` in `rag_pipeline.py` instructs the language model to: cite before claiming, use inline `[p. N]` references, prefix partial answers with `"Based on limited evidence:"`, refuse explicitly when context is insufficient, and make no speculative inferences.

**Context block construction:** Retrieved chunks are passed to the language model with explicit page markers (`--- Page N ---`) so the model can cite specific pages.

**Post-generation citation reranking:**
```python
# Parses: "p. 12", "p12", "page 12" etc.
_PAGE_REF_RE = re.compile(r"\bp\.?\s*(\d+)\b")
```
`_rerank_citations_by_answer()` stable-sorts citations so pages explicitly named in the answer appear first. Pages not referenced by the model remain in order after referenced pages.

**Machine-readable `citations` field:** The API response `citations` list is a typed list of `Citation` Pydantic objects — not a prose bibliography. Each citation is independently parseable.

### Evidence a reviewer can verify

- `test_citation_engine.py` (19 tests): `record()`, `to_schema()`, APA format, relevance score clamping, `clear()`
- `backend/app/schemas/response.py`: `Citation` model with `chunk_id`, `doc_id`, `doc_title`, `page_number`, `section_heading`, `excerpt`, `relevance_score` — all typed fields
- `_rerank_citations_by_answer()` in `rag_pipeline.py`: regex implementation, stable sort, tested in `test_rag_pipeline.py`
- `test_api.py`: full `AnswerResponse` shape test asserts `citations` field is present

### Skills demonstrated

Citation provenance tracking, citation schema design, post-generation retrieval reranking, system prompt engineering for source attribution, structured API response design for downstream programmatic use.

### Gaps

- **No automated citation verification.** The system reranks citations but does not verify whether each generated claim is actually supported by the cited passage. Automated claim-citation alignment (e.g., using an NLI model to check whether the cited passage entails the generated claim) is absent.
- **The regex page reference parser is fragile.** `\bp\.?\s*(\d+)\b` matches `p. 12` and `p12` but may miss `page twelve`, Roman numerals, or non-standard citation formats used in some policy documents.
- **Citation deduplication is not explicit.** If the same passage appears in multiple retrieved chunks (due to overlap), it may appear as multiple citations with slightly different `chunk_id`s.
- **APA format is approximate.** The APA format method produces a reasonable approximation but does not handle all edge cases (missing author, date-less documents, organisation-as-author).

---

## Component 4: Deterministic Trust Layer

**Research relevance:** The trust layer is the mechanism that makes AI-generated answers auditable. It replaces language model self-reporting with deterministic pipeline-derived signals. This is the core responsible AI contribution of the research direction.

### What is implemented

**File:** `backend/app/services/rag_pipeline.py` — five functions and one structural gate

| Function | Output | How Computed |
|---|---|---|
| `_compute_confidence()` | `float \| None` | `min(mean_top3_cosine + min(n_triples × 0.05, 0.15), 1.0)`; `None` when no corpus |
| `_compute_confidence_note()` | `str` | Human-readable breakdown of what drove the score |
| `_classify_answer_type()` | `cited \| partial \| refused \| no_corpus` | Pattern-matching on answer text + pipeline state |
| `_classify_evidence_quality()` | `strong \| moderate \| weak \| insufficient` | Chunk count + score distribution thresholds |
| `_infer_limitations()` | `list[str]` | Active caveats from pipeline state flags |
| Trust gate | Structured refusal | Returns `no_corpus` response before LLM call when `len(chunks) == 0` |

**Low-confidence preamble injection:** When `confidence < 0.35`, a retrieval note is prepended to the user prompt instructing the language model to qualify its response. This is a pipeline-level constraint, not a post-hoc disclaimer.

**Provider and model transparency:** Every `AnswerResponse` includes `provider` (str) and `model` (str) identifying exactly which language model generated the answer.

**`limitations` field examples (from `_infer_limitations()`):**
- "Confidence below reliable threshold" (confidence < 0.35)
- "Named entity extraction was not available; graph evidence may be limited"
- "Query was scoped to a single document"
- "Graph traversal depth was 1; deeper relationships may not have been explored"
- "Corpus is small; answer may not reflect the full document set"

### Evidence a reviewer can verify

- `test_trust_layer.py` (61 tests): largest test module; tests all five functions for correct output under varied pipeline states
- `backend/app/schemas/response.py`: `AnswerResponse` — `answer_type: Literal["cited","partial","refused","no_corpus"]`, `evidence_quality: Literal["strong","moderate","weak","insufficient"]`, `confidence: Optional[float]`, `limitations: list[str]`
- `test_api.py` `TestGraphDepth.test_query_graph_depth_2_returns_full_response_shape`: asserts all trust fields present in every response
- `docs/responsible_ai.md`: trust mechanisms formally documented

### Skills demonstrated

Responsible AI system design, deterministic signal extraction from pipeline state, structured uncertainty communication, API schema design for trust, hallucination prevention through architectural constraint rather than prompt engineering.

### Gaps

- **Confidence formula is heuristic, not calibrated.** `mean_top3_cosine + graph_bonus` is an ordinal ranking signal. Its correlation with actual answer correctness on real documents has not been measured.
- **answer_type classification uses pattern-matching.** The classifier uses keyword matching on the generated answer text. It may misclassify answers that do not follow the expected linguistic patterns, especially for non-English documents or unusual formatting.
- **`_infer_limitations()` is a static list of flags.** The limitations list does not include dynamic caveats such as "the most relevant passage is from a different section than the question targets" or "retrieved chunks span more than 5 pages apart."
- **No human evaluation of trust signal accuracy.** The trust layer fields have not been compared against human assessments of answer quality on annotated data.
- **Confidence is `None` rather than 0 when no corpus.** This is the correct design choice (absence of information is different from zero confidence), but downstream consumers need to handle `None` explicitly — this is a potential integration footgun.

---

## Component 5: Knowledge Graph Construction

**Research relevance:** The knowledge graph is the data structure that enables graph-augmented retrieval. Its quality — how accurately entity types are identified and how informatively relation types are assigned — determines the ceiling on retrieval improvement from graph augmentation.

### What is implemented

**Files:** `backend/app/core/entity_extraction.py`, `backend/app/services/graph_service.py`

**Entity types (9):** ORG, GPE, LAW, MONEY, PERCENT, DATE, PRODUCT, EVENT, NORP — filtered from spaCy `en_core_web_sm` to policy-relevant types.

**Relation types (4):** MENTIONED_WITH, OPERATES_IN (ORG + GPE), TARGETS_BY (PERCENT/MONEY + DATE), GOVERNED_BY (LAW + ORG) — assigned by proximity co-occurrence rules.

**Graph storage:** NetworkX `DiGraph` where nodes are entity names and edges carry `source`, `relation`, `target`, `doc_id`, `confidence`. `BaseGraphService` ABC defines the interface; Neo4j adapter satisfies the same interface.

**Graceful fallback:** When spaCy is absent, `extract_entities()` returns an empty list and logs a debug message — ingestion continues without graph data.

**Graph API exposure:** `GET /api/v1/graph/stats` and `GET /api/v1/graph/neighbours?entity=&depth=` expose graph data via REST, enabling external inspection and potential frontend visualisation.

### Evidence a reviewer can verify

- `test_graph_service.py` (25 tests): node/edge CRUD, multi-hop traversal at depth 1–3, entity/relation counts, isolated-node handling
- `backend/app/core/entity_extraction.py`: `_POLICY_TYPES` set, `extract_entities()`, `extract_relations()` with four proximity rules
- `GET /api/v1/graph/stats` in CI smoke test — verifies graph population after ingestion
- `backend/app/services/graph_service.py`: `BaseGraphService` ABC with `add_entity`, `add_relation`, `get_neighbours`, `get_edges`

### Skills demonstrated

Named entity recognition, information extraction, knowledge graph construction, graph data structure design, abstract service interface design, graceful optional-dependency handling.

### Gaps

- **Relation extraction is rule-based and limited.** Only four relation types based on proximity co-occurrence. Complex multi-clause sentences, passive voice, and long-range dependencies are not captured. A dependency-parse-based extractor using spaCy's `dep_` attributes would substantially improve coverage.
- **No cross-document entity resolution.** "European Commission" in document A and "the Commission" in document B are stored as separate nodes. Co-reference resolution across documents is absent.
- **Confidence scores for edges are not yet fine-grained.** All proximity-based edges receive a default confidence; edges are not scored based on syntactic distance, co-occurrence frequency, or context type.
- **spaCy `en_core_web_sm` is a general-purpose model.** A policy-domain fine-tuned NER model would extract entities with higher precision, particularly for LAW (legislative instruments) and PRODUCT (policy mechanisms) types.
- **The graph is not persisted across server restarts in in-memory mode.** NetworkX `DiGraph` is in-memory only; restarting the server clears the graph unless using Neo4j.

---

## Component 6: Document Ingestion Pipeline

**Research relevance:** The quality and breadth of the ingestion pipeline determines what kinds of documents the research system can process. A production-quality ingestion pipeline — with PDF parsing, page structure preservation, section-heading detection, and citation provenance — is a prerequisite for the evaluation methodology.

### What is implemented

**Files:** `backend/app/core/document_loader.py`, `backend/app/core/text_chunker.py`, `backend/app/core/connectors.py`

**Input modes:** Raw text payload (`text` field) or publicly accessible URL (PDF or plain text). `connector_from_request()` factory selects the connector.

**PDF parsing:** `pdfplumber` (preferred, higher layout fidelity) → `PyPDF2` fallback. Page structure preserved as `doc.pages: list[tuple[int, str]]`.

**Stable `doc_id`:** SHA-256 hash of normalised title + text. Re-ingesting the same document returns the same `doc_id` — idempotent.

**Recursive semantic chunker:** 8-level separator hierarchy (`\n\n` → `\n` → `. ` → `? ` → `! ` → `; ` → `, ` → ` `) with hard-split fallback. Default chunk_size=512, overlap=64.

**Section-heading detection:** `_detect_heading()` applies a 5-pattern regex to the first 5 lines of each chunk: Article/Section/numbered/ALL-CAPS/lettered headings. Heading populates `TextChunk.section_heading`.

**File-size cap:** `MAX_DOCUMENT_SIZE_MB` environment variable limits URL-ingested file sizes.

**Error hierarchy:** `DocumentFetchError` → 502, `DocumentSizeError` → 413, `DocumentEmptyError` / `ParseError` → 422.

### Evidence a reviewer can verify

- `test_ingestion.py` (70 tests): largest test module by coverage scope; end-to-end ingestion; error codes; idempotency; PDF fallback
- `test_document_loader.py` (12 tests): `PolicyDocument` construction, `doc_id` stability, page count, text normalisation
- `test_text_chunker.py` (11 tests): `min_chars` gate, overlap, section-heading detection, multi-page documents
- `test_connectors.py` (19 tests): protocol tests, URL connector with httpx mock
- `TextChunk.chunk_id` format: `{doc_id}_p{page:04d}_c{idx:06d}` — verifiable in chunker source

### Skills demonstrated

PDF parsing and fallback chain design, recursive text splitting, citation provenance preservation at chunk level, idempotent document identification, typed error hierarchy, streaming URL ingestion with size limits.

### Gaps

- **No table extraction.** Policy documents frequently contain tables with numerical targets, compliance thresholds, and obligation matrices. pdfplumber supports table extraction; it is not currently used.
- **No image/figure handling.** Figures, charts, and maps in PDF documents are ignored during ingestion.
- **No DOCX/ODT/HTML support.** Only plain text and PDF are supported. Legislative and regulatory documents are frequently distributed as DOCX or HTML.
- **Section-heading detection is regex-based.** It handles common formats well but will miss domain-specific heading styles (e.g., UN document section numbering).
- **Chunk size is fixed at 512 characters.** Adaptive chunking — where chunk size is adjusted based on section type or content density — is not implemented.

---

## Component 7: Embedding Layer

**Research relevance:** The quality of dense embeddings determines the ceiling on vector retrieval performance. The embedding model choice, normalisation, and caching strategy all affect retrieval quality, latency, and memory footprint.

### What is implemented

**File:** `backend/app/core/embeddings.py`

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, normalised float32 vectors.
**Lazy loading:** Model is instantiated on first call, not at import time.
**Process-level caching:** `functools.cache` on the constructor — model loaded once per process, reused across requests.
**Normalisation:** `normalize_embeddings=True` in `SentenceTransformer.encode()` — cosine similarity equals dot product, no normalisation applied twice.
**Batch operations:** `embed_batch()` passes lists directly to `encode()`.

### Evidence a reviewer can verify

- `backend/app/core/embeddings.py`: 78 lines, compact and readable
- `conftest.py`: `StubEmbedding` (seeded NumPy random vectors) used in all tests — no real model required for the test suite
- `InMemoryVectorStore`: dot-product similarity with normalised vectors — numerically equivalent to cosine

### Skills demonstrated

Embedding model lifecycle management, lazy loading, process-level caching, normalisation discipline, test isolation via stub embedding.

### Gaps

- **Single embedding model.** The system uses one model for both document chunks and queries. Domain-specific models (e.g., fine-tuned on legal or policy text) might improve retrieval precision on policy corpora.
- **No late interaction or ColBERT-style embeddings.** MaxSim-style token-level similarity is absent; only CLS/mean-pooled representations are used.
- **384 dimensions may be insufficient for longer, more complex chunks.** Larger embedding models (e.g., `all-mpnet-base-v2` at 768 dims) may capture more nuanced semantic relationships.
- **No embedding cache between server restarts.** Embeddings are recomputed on re-ingestion. An embedding cache keyed by chunk text hash would avoid redundant computation for large re-ingestions.

---

## Component 8: LLM Provider Layer

**Research relevance:** Provider abstraction enables ablation evaluation across different language models without changing the pipeline. It also enables offline evaluation via the mock provider — essential for reproducible CI and cost-free benchmarking.

### What is implemented

**File:** `backend/app/services/llm_service.py`

**`BaseLLMProvider` ABC:** defines `complete()`, `acomplete()`, `provider_name`, `model_id` — all subclasses satisfy this interface.

**`AnthropicProvider`:** Sync (`anthropic.Anthropic`) and async (`anthropic.AsyncAnthropic`) — full Claude API support.

**`OpenAIProvider`:** Sync (`openai.OpenAI`) and async (`openai.AsyncOpenAI`) — full GPT API support.

**`MockProvider`:** Deterministic stub. Inspects the context block for page markers, references the question text, and returns a structured fake answer with fake citations. No API call, no cost, no external dependency. Default provider in CI and Codespaces.

**Factory:** `build_llm_provider(settings)` selects provider from `LLM_PROVIDER` env var. Pipeline is entirely provider-agnostic.

**Configuration:** `model`, `temperature`, `max_tokens` are configurable per provider via environment variables.

### Evidence a reviewer can verify

- `test_rag_pipeline.py` (30 tests): all use MockProvider via `conftest.py` — no API key required
- `backend/app/services/llm_service.py`: 218 lines — ABC, three concrete classes, factory
- CI env: `LLM_PROVIDER=mock` in `.github/workflows/ci.yml` and `.devcontainer/devcontainer.json`
- `GET /health` returns `llm_provider` and `llm_model` — confirms which provider is active

### Skills demonstrated

Abstract base class design, dependency inversion principle, async Python (native async clients for both Anthropic and OpenAI), offline testing via deterministic mock, environment-driven provider selection.

### Gaps

- **No streaming completions.** Both Anthropic and OpenAI support streaming; the current providers return full completions. Streaming is a planned feature that would improve perceived latency.
- **No retry logic or backoff.** Transient API failures (rate limits, timeouts) are not handled with exponential backoff. A single failure propagates immediately as a 500 error.
- **No token counting before call.** The pipeline does not check whether the context + prompt exceeds the model's context window before calling the API. For very large documents with many chunks, this could silently truncate input.
- **No cost tracking.** API call costs (tokens in/out, provider pricing) are not tracked or logged.

---

## Component 9: Evaluation Framework

**Research relevance:** The evaluation framework defines how the research direction's claims are measured. Nine orthogonal evaluation dimensions provide coverage of retrieval quality, citation accuracy, trust layer reliability, and system-level properties.

### What is implemented

**Files:** `backend/tests/test_evaluation.py` (59 tests), `scripts/evaluate.py` (standalone benchmark)

**Nine evaluation dimensions:**

| Class | What It Measures |
|---|---|
| `TestRetrievalRelevance` | Whether retrieved chunks are relevant to the query; recall at k |
| `TestCitationAccuracy` | Whether citations correspond to passages that support the answer |
| `TestEntityExtractionQuality` | Recall and precision of NER on synthetic documents with known entities |
| `TestRelationExtractionQuality` | F1 of relation extraction on known entity pairs |
| `TestAnswerFaithfulness` | Absence of claims not grounded in retrieved passages |
| `TestSourceCoverage` | Fraction of relevant source passages represented in citations |
| `TestHallucinationRisk` | Presence of answer content with no corresponding citation |
| `TestLatencyBounds` | p50 and p95 end-to-end query latency under mock mode |
| `TestReproducibility` | Same query → same answer_type and citations across repeated calls |

**Standalone benchmark (`scripts/evaluate.py`):** Command-line script that runs evaluation against any live server instance. Produces a formatted report with per-dimension scores.

### Evidence a reviewer can verify

- `test_evaluation.py`: 59 tests, the second-largest test module by count
- `scripts/evaluate.py`: runnable against a live server
- CI smoke test (`scripts/smoke_test.py`): 40 checks including end-to-end trust layer field verification

### Skills demonstrated

Evaluation methodology design, multi-dimensional benchmark construction, systematic measurement of retrieval, citation, trust, and latency properties, separation of evaluation from unit testing.

### Gaps

- **All evaluation uses synthetic documents.** The four evaluation documents are synthetic paragraphs constructed for test coverage. There is no real policy document benchmark.
- **No human-annotated gold standard.** Retrieval relevance, citation accuracy, and answer faithfulness are evaluated against synthetic ground truth, not human judgements.
- **Hallucination risk is measured structurally, not semantically.** The current approach checks for the structural presence of citations, not whether the answer's claims are semantically entailed by the cited passages. NLI-based entailment checking would provide a stronger faithfulness signal.
- **No statistical significance reporting.** Evaluation results do not report variance, confidence intervals, or significance tests.
- **No cross-document evaluation.** All evaluation queries target single documents. Multi-document synthesis evaluation is absent.

---

## Component 10: Testing and CI Infrastructure

**Research relevance:** A reproducible, offline test suite is a prerequisite for credible benchmarking and ablation evaluation. The CI pipeline ensures that every code change is validated before integration. Coverage gates prevent regression of test discipline.

### What is implemented

**Files:** `backend/tests/` (11 modules, 406 tests), `backend/tests/conftest.py`, `.github/workflows/ci.yml`

**Test count (verified):** 406 collected, 405 passed, 1 skipped (network-dependent URL connector test).

**Test modules:**

| Module | Tests |
|---|---|
| `test_api.py` | 81 |
| `test_trust_layer.py` | 61 |
| `test_ingestion.py` | 70 |
| `test_evaluation.py` | 59 |
| `test_graph_service.py` | 25 |
| `test_rag_pipeline.py` | 30 |
| `test_retriever.py` | 19 |
| `test_citation_engine.py` | 19 |
| `test_connectors.py` | 19 |
| `test_document_loader.py` | 12 |
| `test_text_chunker.py` | 11 |

**`conftest.py` fixtures:** `StubEmbedding` (seeded NumPy), `InMemoryVectorStore`, `InMemoryGraphService`, `MockProvider`, `api_client` (FastAPI `TestClient`) — full in-memory stack, no external dependencies.

**CI pipeline (`.github/workflows/ci.yml`):**
- Job 1: Ruff lint (`ruff check backend/`) + Mypy (`mypy backend/`)
- Job 2: pytest on Python 3.10, 3.11, 3.12 in parallel; `pytest-cov` with XML; Codecov upload on 3.11
- Job 3: Live server smoke test (start Uvicorn with mock stack, poll `/health`, run `scripts/smoke_test.py` — 40 checks)
- All jobs run offline: `LLM_PROVIDER=mock`, `VECTOR_STORE_PROVIDER=memory`, `GRAPH_PROVIDER=memory`

**Coverage gate:** `fail_under = 60` in `pyproject.toml`.

### Evidence a reviewer can verify

- `pytest backend/tests/` — 405 passing, 1 skipped
- `.github/workflows/ci.yml` — three job definitions, Python matrix, mock env vars
- `conftest.py` — fixtures readable; no API key or network access required
- Smoke test: 40/40 checks passed (verified in this session)

### Skills demonstrated

Comprehensive test design covering unit, integration, API, evaluation, and smoke categories; async test support (`asyncio_mode = auto`); CI/CD pipeline design; multi-version Python matrix; coverage measurement; offline-first test strategy.

### Gaps

- **Coverage gate is 60%, not higher.** The `fail_under = 60` gate was chosen conservatively for an Alpha release. As the system matures, raising this to 80%+ would be appropriate.
- **No mutation testing.** Mutation testing (e.g., using `mutmut`) would verify that the test suite actually catches logic errors, not just exercises code paths.
- **No performance regression tests.** Latency bounds in `TestLatencyBounds` use mock mode; real-LLM latency regression testing against a baseline is absent.
- **No end-to-end tests with a real LLM.** All CI tests use mock mode. A nightly integration test against a real API would catch provider-specific issues.
- **`test_evaluation.py` tests use synthetic documents.** As noted above — the evaluation framework is complete but not yet applied to real documents.

---

## Component 11: Configuration and Responsible AI Documentation

**Research relevance:** Responsible AI infrastructure — configuration discipline, trust documentation, usage guidelines — is part of the research contribution. A system that computes trust signals but does not document what they mean, what their limitations are, or how they should be used is incomplete as a research artefact.

### What is implemented

**Files:** `backend/app/config.py`, `docs/responsible_ai.md`, `.env.example`

**`Settings(BaseSettings)`:** All configuration from environment variables. API key fields excluded from `repr()`. Mock mode as default (`llm_provider=mock`). `lru_cache` singleton. All settings documented with defaults and types.

**`docs/responsible_ai.md`:** Documents eight trust mechanisms, intended use cases, out-of-scope use cases, and operator obligations.

**`.env.example`:** Template with all variables, empty API key values, and comments explaining each setting.

**Structured logging (`backend/app/utils/logging.py`):** `structlog` with `stdlib.LoggerFactory()` — structured, parseable log output. UTF-8 stream configuration prevents encoding errors.

### Evidence a reviewer can verify

- `backend/app/config.py`: 103 lines — all settings visible, mock defaults explicit
- `docs/responsible_ai.md`: formal documentation of trust mechanisms
- `.env.example`: no secrets, all values empty or safe
- `backend/app/utils/logging.py`: 42 lines — structlog configuration

### Skills demonstrated

12-factor app configuration, secrets hygiene, responsible AI documentation, structured logging, provider transparency in API responses.

### Gaps

- **No formal audit trail.** API calls are logged with structlog but there is no persistent audit log of queries, retrieved documents, or trust scores. An audit trail would be important for production deployments in regulated environments.
- **`docs/responsible_ai.md` is documentation, not enforcement.** The responsible use guidelines describe intended use but do not technically prevent misuse.
- **No data retention or deletion policy documentation.** What happens to ingested documents in terms of retention, access control, or deletion is not formally specified.

---

## Summary: Current Gaps in Priority Order

The following gaps are ranked by their impact on advancing the research direction. The top five are the most important near-term improvements.

### Priority 1 (Critical — needed for real evidence)
**Real benchmark dataset.** All evaluation currently uses synthetic documents. Constructing a ground-truth benchmark from publicly licensed policy documents (EUR-Lex, UN, World Bank) with human-annotated question-answer pairs is the single most important next step. Without it, retrieval quality claims cannot be empirically supported.

### Priority 2 (High — core research contribution)
**Ablation evaluation of alpha and graph depth.** The hybrid retrieval formula has two key hyperparameters (alpha = 0.7, default depth = 1) that were set by design intuition, not empirical measurement. Running ablations on the benchmark — vector-only vs hybrid vs different alpha values vs different depths — would provide the empirical evidence needed to justify the design choices.

### Priority 3 (High — retrieval quality)
**Cross-encoder reranking.** Adding a cross-encoder reranking step after initial candidate retrieval is a well-established technique with consistent retrieval quality improvements in the NLP literature. The `HybridRetriever` interface accommodates this without structural changes.

### Priority 4 (High — graph quality)
**Dependency-parse-based relation extraction.** Replacing the four proximity co-occurrence rules with spaCy dependency-tree traversal would materially improve graph edge quality, particularly for complex sentences where entity relationships are expressed through syntactic structure rather than proximity.

### Priority 5 (Medium — trust calibration)
**Calibrated confidence scoring.** The current heuristic formula (`mean_top3_cosine + graph_bonus`) is an ordinal ranking signal. Training a lightweight calibration model on annotated data to produce probabilities would make the confidence score interpretable as a reliability estimate.

### Priority 6 (Medium — system completeness)
**Frontend interface.** A Streamlit prototype would make the system accessible to non-technical reviewers and potential users. The FastAPI backend is complete; no backend changes are needed.

### Priority 7 (Medium — deployment)
**Docker Compose.** Containerised deployment would make the system portable across environments and enable consistent evaluation conditions.

### Priority 8 (Lower — citation quality)
**Automated citation verification using NLI.** Post-generation checking of whether cited passages entail generated claims would strengthen the system's citation discipline beyond prompt-level enforcement.

### Priority 9 (Lower — document scope)
**Table extraction and DOCX/HTML support.** Policy documents frequently contain tables and are distributed in DOCX format. Extending the ingestion pipeline to handle these formats would substantially expand the document types the system can analyse.

### Priority 10 (Lower — graph scale)
**Cross-document entity resolution.** Resolving "European Commission" and "the Commission" as the same node across documents would improve multi-document query quality.

---

## Recommended Next Technical Step

**Build the first real benchmark dataset.**

All other improvements — ablation evaluation, cross-encoder reranking, calibrated confidence, citation verification — require a real annotated benchmark to measure. Without a benchmark, claims about retrieval quality improvements are hypotheses rather than evidence.

**Concretely:**
1. Download 5–10 publicly licensed policy documents from EUR-Lex (`https://eur-lex.europa.eu`), the UN Document Portal, or the World Bank Open Knowledge Repository.
2. Ingest each document using `POST /api/v1/ingest`.
3. Write 50–100 question-answer pairs per document with ground-truth page references and supporting passage labels.
4. Run `scripts/evaluate.py` against the benchmark.
5. Record baseline retrieval recall, citation accuracy, and confidence calibration metrics.
6. Run the vector-only ablation (set `GRAPH_VECTOR_ALPHA=1.0`) against the same benchmark.
7. Compare results.

This produces the first empirical evidence that the GraphRAG design choice improves over pure vector retrieval on real policy documents — which is the central claim of the research direction.

---

*All current-state claims in this document were verified by reading source files before writing. Gaps describe what is absent, not what failed — the system performs correctly within its implemented scope.*
