# Demo Walkthrough

A step-by-step guide to running PolicyMind-AI locally: ingest a sample policy document, ask citation-backed questions, and inspect the knowledge graph. No API key or paid service is required — the demo uses the built-in mock LLM provider.

**Time:** 5–10 minutes

---

## What This Demo Shows

| Step | What you observe |
|---|---|
| Start API | FastAPI server with interactive docs at `/docs` |
| Health check | Server state: provider, vector store, graph |
| Ingest document | Chunking, embedding, entity extraction, graph indexing |
| Ask a question | Citation-backed answer with trust-layer fields |
| Trust-layer fields | `answer_type`, `evidence_quality`, `confidence_note`, `limitations` |
| Inspect graph | Knowledge-graph entity neighbourhood |

---

## Prerequisites

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the default config (LLM_PROVIDER=mock, no API key needed)
cp .env.example .env

# Optional: install spaCy for entity extraction and graph evidence
python -m spacy download en_core_web_sm
```

> **Sentence Transformers model:** `all-MiniLM-L6-v2` is downloaded automatically on first run (~90 MB). Subsequent runs use the cache.

---

## Quick Start (Automated)

If you just want to verify the whole pipeline works in one command:

```bash
# Terminal 1 — start the server
PYTHONPATH=backend uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — run the smoke test
python scripts/smoke_test.py
```

The smoke test runs 9 steps and 25+ checks without requiring any API key.
See [scripts/smoke_test.py](../scripts/smoke_test.py) for what each step verifies.

---

## Step-by-Step Walkthrough

### Step 1 — Start the API server

```bash
PYTHONPATH=backend uvicorn backend.app.main:app --reload --port 8000
```

The server starts on `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI where you can try all endpoints in the browser.

---

### Step 2 — Health check

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "llm_provider": "mock",
  "llm_model": "mock-model",
  "vector_store": "chroma",
  "vector_store_chunks": 0,
  "graph_provider": "memory",
  "graph_enabled": false,
  "graph_entities": 0,
  "graph_relations": 0
}
```

`graph_enabled: false` is normal in default configuration (memory graph, no Neo4j). Entity extraction and graph evidence still work — the graph is stored in memory and reset on server restart.

---

### Step 3 — Ingest a sample document

The API accepts a raw text payload or a public URL. This example uses a short inline text (no download required):

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The National Climate Strategy 2030 commits all signatory states to reduce greenhouse gas emissions by 55 percent relative to 1990 levels by 31 December 2030. Article 4 mandates a minimum 30 percent renewable energy share by 2025, rising to 65 percent by 2030. A Just Transition Fund of EUR 15 billion is established for 2025 to 2030, governed by a Board of member state representatives and independent experts. The European Commission shall publish annual compliance reports by 31 March each year. Non-compliant states may face suspension of Fund access and preferential trade arrangements.",
    "title": "National Climate Strategy 2030",
    "source_label": "EU Commission"
  }' | python -m json.tool
```

Expected response:

```json
{
  "doc_id": "a3f1c9e2b8d74051",
  "title": "National Climate Strategy 2030",
  "source_url": null,
  "page_count": 1,
  "chunk_count": 1,
  "entities_extracted": 6,
  "graph_nodes_added": 4,
  "processing_status": "completed",
  "message": "Document ingested successfully."
}
```

Save the `doc_id` — you can use it to scope queries to this document specifically.

**Alternatively**, ingest the included synthetic policy document:

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d "{\"text\": $(python -c "import json, pathlib; print(json.dumps(pathlib.Path('sample_data/sample_policy.txt').read_text()))"), \"title\": \"National Climate and Sustainable Development Strategy\", \"source_label\": \"PolicyMind Demo\"}" \
  | python -m json.tool
```

This document produces 4–6 chunks, 20+ entities, and a richer knowledge graph.

---

### Step 4 — Ask a question

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What emission reduction target is committed for 2030?",
    "include_graph_evidence": true
  }' | python -m json.tool
```

Annotated response (values vary by embedding quality and confidence):

```json
{
  "query_id": "3f7a2c1d-9b4e-4a1c-8d3f-2e5b7c0a1d9e",
  "question": "What emission reduction target is committed for 2030?",

  "answer": "[MOCK ANSWER] Based on 1 retrieved passage(s) (p.1), the policy corpus contains relevant information about: 'What emission reduction target is committed for 2030?'. Key excerpt: \"The National Climate Strategy 2030 commits all signatory states to reduce greenhouse gas emissions by 55 percent relative to 1990 levels by 31 December 2030...\"  For a complete, synthesised answer, configure a real LLM provider via LLM_PROVIDER=anthropic or LLM_PROVIDER=openai.",

  "answer_type": "cited",
  "evidence_quality": "moderate",
  "confidence_note": "Derived from 1 retrieved passage (mean top-3 cosine similarity: 0.61), without graph evidence. Evidence quality is moderate — verify key claims against the source.",

  "citations": [
    {
      "chunk_id": "a3f1c9e2-p001-c000",
      "doc_id": "a3f1c9e2b8d74051",
      "doc_title": "National Climate Strategy 2030",
      "page_number": 1,
      "section_heading": null,
      "excerpt": "The National Climate Strategy 2030 commits all signatory states to reduce greenhouse gas emissions by 55 percent relative to 1990 levels...",
      "relevance_score": 0.6134
    }
  ],

  "retrieved_chunks": [
    {
      "chunk_id": "a3f1c9e2-p001-c000",
      "doc_id": "a3f1c9e2b8d74051",
      "doc_title": "National Climate Strategy 2030",
      "page_number": 1,
      "section_heading": null,
      "text": "The National Climate Strategy 2030 commits all signatory states to reduce greenhouse gas emissions by 55 percent relative to 1990 levels by 31 December 2030. Article 4 mandates a minimum 30 percent renewable energy share by 2025, rising to 65 percent by 2030...",
      "relevance_score": 0.6134
    }
  ],

  "graph_evidence": [],

  "confidence": 0.613,
  "limitations": [
    "Mock LLM provider active — responses are deterministic stubs. Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai for real answers.",
    "No graph evidence matched this query. The retrieved passages may not contain recognised named entities."
  ],

  "latency_ms": 87.4,
  "provider": "mock",
  "model": "mock-model"
}
```

#### Understanding the trust-layer fields

| Field | What it means |
|---|---|
| `answer_type` | `cited` — answer is grounded in retrieved passages at adequate confidence |
| `evidence_quality` | `moderate` — confidence ≥ 0.5 with ≥ 2 passages, or confidence ≥ 0.35 |
| `confidence_note` | Human-readable breakdown of the cosine similarity and graph boost |
| `confidence` | Heuristic score in [0, 1]: mean top-3 cosine similarity + graph bonus |
| `citations` | Traceable source references — `chunk_id`, `doc_id`, `page_number`, `relevance_score` |
| `retrieved_chunks` | Verbatim passages the LLM received as context |
| `limitations` | Deterministically computed caveats — always present, never LLM-generated |

See [docs/responsible_ai.md](responsible_ai.md) for a complete description of each trust mechanism.

---

### Step 5 — Scope a query to a single document

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What fund supports the economic transition away from fossil fuels?",
    "doc_id": "REPLACE_WITH_YOUR_DOC_ID",
    "top_k": 3,
    "include_graph_evidence": false
  }' | python -m json.tool
```

`top_k` controls how many chunks are retrieved. `include_graph_evidence: false` disables graph expansion for faster, vector-only retrieval.

---

### Step 6 — Inspect the knowledge graph

**Graph statistics:**

```bash
curl -s http://localhost:8000/api/v1/graph/stats | python -m json.tool
```

```json
{
  "total_entities": 8,
  "total_relations": 12,
  "graph_provider": "memory",
  "graph_enabled": false
}
```

> If `total_entities` is 0, spaCy is not installed. Run `python -m spacy download en_core_web_sm` and re-ingest the document.

**Entity neighbourhood:**

```bash
curl -s "http://localhost:8000/api/v1/graph/neighbours?entity=European%20Commission&depth=1" \
  | python -m json.tool
```

```json
{
  "entity": "European Commission",
  "neighbours": [
    {"id": "ent-0042", "name": "Just Transition Fund", "entity_type": "ORG", "doc_id": "a3f1c9e2b8d74051"},
    {"id": "ent-0007", "name": "2030", "entity_type": "DATE", "doc_id": "a3f1c9e2b8d74051"}
  ],
  "edges": [
    {"source": "European Commission", "target": "Just Transition Fund", "relation": "MENTIONED_WITH", "chunk_id": "a3f1c9e2-p001-c000"},
    {"source": "European Commission", "target": "2030", "relation": "MENTIONED_WITH", "chunk_id": "a3f1c9e2-p001-c000"}
  ],
  "total_entities": 8,
  "total_relations": 12
}
```

---

### Step 7 — List and delete documents

```bash
# List all ingested documents
curl -s http://localhost:8000/api/v1/documents | python -m json.tool

# Get detailed metadata for a single document
curl -s http://localhost:8000/api/v1/documents/REPLACE_WITH_DOC_ID | python -m json.tool

# Delete a document (removes chunks and embeddings; graph entities are retained)
curl -s -X DELETE http://localhost:8000/api/v1/documents/REPLACE_WITH_DOC_ID | python -m json.tool
```

---

## Variant: Ingest from a Public URL

The API accepts any publicly accessible PDF or plain-text URL. The user is responsible for ensuring the URL is accessible and the document's licence permits analysis.

```bash
curl -s -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://YOUR_PUBLIC_DOCUMENT_URL.pdf",
    "title": "My Policy Document",
    "source_label": "Source Organisation"
  }' | python -m json.tool
```

Example public sources compatible with URL ingestion:
- [EUR-Lex](https://eur-lex.europa.eu) — EU legislation (CC BY 4.0)
- [World Bank Open Knowledge Repository](https://openknowledge.worldbank.org) — development reports (CC BY 3.0)
- [UN Document Portal](https://documents.un.org) — resolutions and reports

---

## Variant: Switch to a Real LLM

Edit `.env`:

```dotenv
# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Restart the server. The same API calls and response structure remain identical — only the `answer` field changes from a mock stub to a real, synthesised response. All trust-layer fields (`answer_type`, `evidence_quality`, `confidence_note`, `citations`) are pipeline-generated and provider-agnostic.

---

## Running the Full Test Suite

```bash
pytest backend/tests/ -v --tb=short
```

| Module | Tests |
|---|---|
| `test_document_loader.py` | Document loading, validation, error types |
| `test_text_chunker.py` | Chunking, overlap, boundary handling |
| `test_connectors.py` | Text and URL connectors |
| `test_citation_engine.py` | Citation recording and deduplication |
| `test_graph_service.py` | Entity and relation graph operations |
| `test_retriever.py` | Hybrid retrieval, score fusion, filters |
| `test_ingestion.py` | End-to-end ingestion pipeline |
| `test_rag_pipeline.py` | Query flow, confidence, limitations |
| `test_evaluation.py` | Evaluation metrics across 9 dimensions |
| `test_trust_layer.py` | Trust-layer: refusal, citations, evidence quality |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sentence_transformers` | Missing dependency | `pip install -r requirements.txt` |
| `graph_entities: 0` after ingest | spaCy not installed | `python -m spacy download en_core_web_sm` |
| `answer_type: no_corpus` after ingest | No chunks retrieved | Check ingest response `chunk_count > 0`; try longer text |
| Server exits with `ANTHROPIC_API_KEY required` | Provider set without key | Use `LLM_PROVIDER=mock` in `.env` for offline mode |
| `422 Unprocessable Entity` on ingest | Missing `text` or `url` field | Provide exactly one of `text` or `url`, not both |
| Slow first query | Model downloading | Wait for `all-MiniLM-L6-v2` cache (~90 MB); subsequent queries are fast |
