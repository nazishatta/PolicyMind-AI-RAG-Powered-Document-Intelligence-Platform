# PolicyMind-AI — API Usage Guide

GraphRAG-powered policy document intelligence. Ingest policy PDFs, extract knowledge graphs, and ask citation-backed questions.

---

## Quick Start (local or GitHub Codespaces)

```bash
# 1 — Install dependencies
pip install -e ".[dev]"

# 2 — Create a .env (all keys optional; mock providers work out of the box)
cp .env.example .env

# 3 — Start the server
PYTHONPATH=backend uvicorn backend.app.main:app --reload --port 8000

# 4 — Open the interactive docs
#     http://localhost:8000/docs   (Swagger UI)
#     http://localhost:8000/redoc  (ReDoc)
```

No API keys are required to run in mock mode. The server starts and all endpoints are functional immediately.

---

## Base URL

| Environment       | Base URL                |
|-------------------|-------------------------|
| Local             | `http://localhost:8000` |
| GitHub Codespaces | forwarded port 8000     |

All data endpoints are prefixed: `/api/v1/`

---

## Authentication

No authentication is required in the default (mock) configuration. To use real LLM providers, set `LLM_PROVIDER` and the corresponding API key in `.env` — keys are never committed to the repository.

---

## Endpoints

### GET /

Returns the service directory: name, version, status, and a map of every available endpoint.

**Request**
```bash
curl http://localhost:8000/
```

**Response**
```json
{
  "name": "PolicyMind-AI",
  "version": "0.1.0",
  "description": "GraphRAG-powered policy document intelligence platform",
  "status": "running",
  "docs_url": "/docs",
  "redoc_url": "/redoc",
  "endpoints": {
    "root": "GET  /",
    "health": "GET  /health",
    "docs": "GET  /docs  (Swagger UI)",
    "redoc": "GET  /redoc",
    "ingest": "POST /api/v1/ingest",
    "query": "POST /api/v1/query",
    "list_documents": "GET  /api/v1/documents",
    "document_detail": "GET  /api/v1/documents/{doc_id}",
    "delete_document": "DELETE /api/v1/documents/{doc_id}",
    "graph_stats": "GET  /api/v1/graph/stats",
    "graph_neighbours": "GET  /api/v1/graph/neighbours?entity=<name>&depth=1"
  },
  "providers": {
    "llm": "mock",
    "llm_model": "mock-1.0",
    "vector_store": "memory",
    "graph": "memory"
  }
}
```

---

### GET /health

Liveness and readiness probe. Returns live statistics for the vector store, knowledge graph, and configured LLM provider.

**Request**
```bash
curl http://localhost:8000/health
```

**Response**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "llm_provider": "mock",
  "llm_model": "mock-1.0",
  "vector_store": "memory",
  "vector_store_chunks": 42,
  "graph_provider": "memory",
  "graph_enabled": false,
  "graph_entities": 18,
  "graph_relations": 7
}
```

| Field                 | Description                                            |
|-----------------------|--------------------------------------------------------|
| `status`              | Always `"ok"` when the service is healthy              |
| `vector_store_chunks` | Total chunks currently indexed                         |
| `graph_entities`      | Named entities extracted across all ingested documents |
| `graph_relations`     | Typed relations stored in the knowledge graph          |

---

### POST /api/v1/ingest

Ingest a policy document by URL or raw text. The document is chunked, embedded, and indexed in the vector store. Entities and relations are written to the knowledge graph.

**Request body**

| Field              | Type    | Required | Description                                               |
|--------------------|---------|----------|-----------------------------------------------------------|
| `url`              | string  | No*      | Publicly accessible PDF or plain-text URL                 |
| `text`             | string  | No*      | Raw text payload (used when no URL is provided)           |
| `title`            | string  | No       | Human-readable document title                             |
| `source_label`     | string  | No       | Provenance label (e.g. `"WHO"`, `"IPCC AR6"`)             |
| `max_file_size_mb` | number  | No       | Per-request size cap in MB                                |

*One of `url` or `text` must be provided.

**curl — ingest by URL**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.org/policy.pdf",
    "title": "Example Policy 2024",
    "source_label": "ExampleGov"
  }'
```

**curl — ingest raw text**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Section 1: Scope. This policy applies to all public agencies and requires annual reporting on emissions reductions under the 2030 framework.",
    "title": "Internal Policy Draft",
    "source_label": "DraftV1"
  }'
```

**Response (201 Created)**
```json
{
  "doc_id": "a3f1c9e2",
  "title": "Example Policy 2024",
  "source_url": "https://example.org/policy.pdf",
  "page_count": 12,
  "chunk_count": 34,
  "entities_extracted": 47,
  "graph_nodes_added": 12,
  "processing_status": "completed"
}
```

**Python**
```python
import httpx

resp = httpx.post("http://localhost:8000/api/v1/ingest", json={
    "url": "https://example.org/policy.pdf",
    "title": "Example Policy 2024",
})
resp.raise_for_status()
doc = resp.json()
print(doc["doc_id"], doc["chunk_count"])
```

**Error responses**

| Status | Cause                                       |
|--------|---------------------------------------------|
| 422    | No `url` or `text` provided; empty document |
| 413    | Document exceeds `max_file_size_mb`         |
| 415    | Unsupported content type                    |
| 502    | URL fetch failure                           |

---

### POST /api/v1/query

Ask a natural-language question over ingested documents. Returns a cited answer, raw retrieved passages, knowledge graph evidence, a confidence score, and known limitations.

**Request body**

| Field                  | Type    | Required | Default | Description                                                                              |
|------------------------|---------|----------|---------|------------------------------------------------------------------------------------------|
| `question`             | string  | Yes      | —       | Natural-language question (5–2000 chars)                                                 |
| `top_k`                | integer | No       | 5       | Number of passages to retrieve (1–20)                                                    |
| `doc_id`               | string  | No       | `null`  | Restrict retrieval to a single document; omit to search all ingested documents           |
| `include_graph_evidence` | boolean | No     | `true`  | Run graph neighbourhood expansion and include graph triples in scoring and response      |
| `graph_depth`          | integer | No       | `null`  | Graph traversal depth: `1` = direct edges only; `2` = also collect 1-hop neighbour edges (confidence discounted ×0.8); `3` = two levels of indirect evidence. Omit to use server default (1). |

**curl — basic query**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What emissions targets does the policy set for 2030?",
    "top_k": 5,
    "include_graph_evidence": true
  }'
```

**curl — multi-hop graph expansion**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which agencies are responsible for just transition enforcement?",
    "include_graph_evidence": true,
    "graph_depth": 2
  }'
```

**Response**
```json
{
  "query_id": "3f7a2c1d-9b4e-4a1c-8d3f-2e5b7c0a1d9e",
  "question": "What emissions targets does the policy set for 2030?",
  "answer": "[MOCK ANSWER] Based on 1 retrieved passage(s) (p.1)...",

  "answer_type": "cited",
  "evidence_quality": "moderate",
  "confidence_note": "Derived from 1 retrieved passage (mean top-3 cosine similarity: 0.82), without graph evidence. Evidence quality is moderate — verify key claims against the source.",

  "citations": [
    {
      "chunk_id": "a3f1c9e2-chunk-001",
      "doc_id": "a3f1c9e2",
      "doc_title": "Example Policy 2024",
      "page_number": 1,
      "section_heading": "Section 2: Targets",
      "excerpt": "By 2030 all signatory states shall reduce net emissions by 45%...",
      "relevance_score": 0.87
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "a3f1c9e2-chunk-001",
      "doc_id": "a3f1c9e2",
      "doc_title": "Example Policy 2024",
      "page_number": 1,
      "section_heading": "Section 2: Targets",
      "text": "By 2030 all signatory states shall reduce net emissions by 45%...",
      "relevance_score": 0.87
    }
  ],
  "graph_evidence": [
    {
      "entity": "2030 Target",
      "relation": "MENTIONED_WITH",
      "target": "Article 12",
      "source_doc_id": "a3f1c9e2",
      "confidence": 0.9
    }
  ],
  "confidence": 0.87,
  "limitations": [
    "Mock LLM provider active — responses are deterministic stubs."
  ],
  "latency_ms": 143.2,
  "provider": "mock",
  "model": "mock-model"
}
```

**Field reference**

| Field              | Description                                                                                                           |
|--------------------|-----------------------------------------------------------------------------------------------------------------------|
| `answer_type`      | `cited` / `partial` / `refused` / `no_corpus` — programmatic classification of this response                        |
| `evidence_quality` | `strong` / `moderate` / `weak` / `insufficient` — derived from confidence and passage count                          |
| `confidence_note`  | Human-readable explanation: passage count, mean cosine similarity, graph boost contribution                          |
| `answer`           | Generated answer text with inline page references                                                                     |
| `citations`        | Source references ordered by explicit page mention in the answer, then descending relevance score                     |
| `retrieved_chunks` | Raw passages returned from the vector store, with relevance scores in `[0, 1]`                                        |
| `graph_evidence`   | Knowledge-graph triples (`entity` → `relation` → `target`) that influenced retrieval scoring                         |
| `confidence`       | Mean top-3 cosine similarity + graph boost, capped at 1.0. `null` when no passages were retrieved                   |
| `limitations`      | Deterministic caveats: mock provider active, sparse graph, low confidence, etc.                                      |
| `latency_ms`       | End-to-end wall-clock time in milliseconds                                                                            |

**Python**
```python
import httpx

resp = httpx.post("http://localhost:8000/api/v1/query", json={
    "question": "What emissions targets does the policy set for 2030?",
    "top_k": 5,
    "include_graph_evidence": True,
    "graph_depth": 2,   # optional: expand to 2-hop graph neighbours
})
resp.raise_for_status()
result = resp.json()
print(result["answer_type"], result["evidence_quality"])
print(result["answer"])
for c in result["citations"]:
    print(f"  [{c['doc_title']} p.{c['page_number']}] {c['excerpt'][:80]}...")
```

---

### GET /api/v1/documents

List all documents currently indexed in the vector store.

**Request**
```bash
curl http://localhost:8000/api/v1/documents
```

**Response**
```json
{
  "documents": [
    {
      "doc_id": "a3f1c9e2",
      "doc_title": "Example Policy 2024",
      "source_url": "https://example.org/policy.pdf",
      "chunk_count": 34
    }
  ],
  "total": 1
}
```

---

### GET /api/v1/documents/{doc_id}

Return detailed metadata for a single document.

**Request**
```bash
curl http://localhost:8000/api/v1/documents/a3f1c9e2
```

**Response**
```json
{
  "doc_id": "a3f1c9e2",
  "doc_title": "Example Policy 2024",
  "source_url": "https://example.org/policy.pdf",
  "source_label": "ExampleGov",
  "chunk_count": 34,
  "page_count": 12,
  "sections": [
    "Section 1: Scope",
    "Section 2: Targets",
    "Section 3: Enforcement"
  ],
  "status": "indexed"
}
```

Returns **404** if the document ID is not found.

---

### DELETE /api/v1/documents/{doc_id}

Remove a document and all its chunks from the vector store. Graph entities are retained (they may be shared across documents).

**Request**
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/a3f1c9e2
```

**Response**
```json
{
  "doc_id": "a3f1c9e2",
  "deleted": true,
  "message": "Document 'a3f1c9e2' and all its chunks have been removed."
}
```

Returns **404** if the document ID is not found.

---

### GET /api/v1/graph/stats

Knowledge graph summary: entity count, relation count, and top entities by degree.

**Request**
```bash
curl http://localhost:8000/api/v1/graph/stats
```

**Response**
```json
{
  "total_entities": 18,
  "total_relations": 7,
  "graph_provider": "memory",
  "graph_enabled": false
}
```

---

### GET /api/v1/graph/neighbours

Retrieve the graph neighbourhood of a named entity (up to `depth` hops).

**Query parameters**

| Param    | Required | Default | Description                        |
|----------|----------|---------|------------------------------------|
| `entity` | Yes      | —       | Entity name (case-sensitive)       |
| `depth`  | No       | `1`     | Traversal depth (1–3)              |

**Request**
```bash
curl "http://localhost:8000/api/v1/graph/neighbours?entity=2030+Target&depth=1"
```

**Response**
```json
{
  "entity": "2030 Target",
  "neighbours": [
    {"name": "Example Policy 2024", "label": "DOCUMENT", "doc_ids": ["a3f1c9e2"], "properties": {}}
  ],
  "edges": [
    {
      "source": "2030 Target",
      "relation": "defined_by",
      "target": "Example Policy 2024",
      "doc_id": "a3f1c9e2",
      "confidence": 0.9
    }
  ],
  "total_entities": 18,
  "total_relations": 7
}
```

---

## Full Workflow Example (Python)

```python
import httpx

BASE = "http://localhost:8000"

# 1. Ingest a document
ingest = httpx.post(f"{BASE}/api/v1/ingest", json={
    "text": (
        "Section 1: Scope. This framework applies to all federal agencies. "
        "Section 2: Targets. Agencies must reduce emissions by 45% by 2030. "
        "Section 3: Reporting. Annual progress reports are mandatory under "
        "the National Climate Accountability Act."
    ),
    "title": "Federal Climate Framework",
    "source_label": "GovDraft",
}).json()
doc_id = ingest["doc_id"]
print(f"Ingested: {doc_id}  ({ingest['chunk_count']} chunks)")

# 2. Ask a question
answer = httpx.post(f"{BASE}/api/v1/query", json={
    "question": "What are the emission reduction targets?",
    "doc_id_filter": doc_id,
}).json()
print("\nAnswer:", answer["answer"])
print("Confidence:", answer["confidence"])
for c in answer["citations"]:
    print(" •", c["apa_citation"])

# 3. Inspect the document
detail = httpx.get(f"{BASE}/api/v1/documents/{doc_id}").json()
print("\nSections:", detail["sections"])

# 4. Clean up
httpx.delete(f"{BASE}/api/v1/documents/{doc_id}")
print("\nDeleted.")
```

---

## Environment Variables

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in only what you need.

| Variable                | Default  | Description                                                      |
|-------------------------|----------|------------------------------------------------------------------|
| `LLM_PROVIDER`          | `mock`   | `mock`, `anthropic`, or `openai`                                 |
| `ANTHROPIC_API_KEY`     | —        | Required when `LLM_PROVIDER=anthropic`                           |
| `OPENAI_API_KEY`        | —        | Required when `LLM_PROVIDER=openai`                              |
| `ANTHROPIC_MODEL`       | —        | Claude model ID (e.g. `claude-opus-4-8`)                         |
| `OPENAI_MODEL`          | —        | OpenAI model ID (e.g. `gpt-4o`)                                  |
| `VECTOR_STORE_PROVIDER` | `memory` | `memory` or `chroma`                                             |
| `GRAPH_PROVIDER`        | `memory` | `memory` or `neo4j`                                              |
| `GRAPH_VECTOR_ALPHA`    | `0.7`    | Score fusion weight: `alpha * vector + (1-alpha) * graph_boost`  |
| `MAX_DOCUMENT_SIZE_MB`  | `20`     | Per-request document size cap                                    |
| `LOG_LEVEL`             | `INFO`   | Python log level                                                 |

No secrets are ever committed to the repository.

---

## Running the Test Suite

```bash
# All tests (~248) — no API keys required
python -m pytest backend/tests/ -v

# Single module
python -m pytest backend/tests/test_api.py -v

# With coverage
python -m pytest backend/tests/ --cov=backend/app --cov-report=term-missing
```

All tests use in-memory providers and a mock LLM — zero external dependencies.

---

## Error Format

All errors follow FastAPI's default RFC 7807 structure:

```json
{
  "detail": "Document 'xyz' not found."
}
```

---

## Interactive Docs

| Interface   | URL                              |
|-------------|----------------------------------|
| Swagger UI  | `http://localhost:8000/docs`     |
| ReDoc       | `http://localhost:8000/redoc`    |

Both are generated automatically from the Pydantic schemas and FastAPI route definitions — no manual maintenance required.
