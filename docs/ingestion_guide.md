# Document Ingestion Guide

> **Note to users:** PolicyMind-AI does not bundle real policy documents.
> Users must supply their own document URLs or text payloads.
> Review the licensing terms of each source document before ingestion.
> This project does not grant any rights over third-party documents.

---

## Overview

The ingestion layer transforms a policy document into indexed, searchable content:

```
Input (URL or text)
       │
       ▼
  Document Loader          ← fetch, validate, parse → PolicyDocument
       │
       ▼
  Text Chunker             ← split into overlapping TextChunks
       │
       ├──► Embedding Model   ← encode chunks → float32 vectors
       │         │
       │         └──► Vector Store (ChromaDB / InMemory)
       │
       └──► Entity Extractor  ← spaCy NER + relation rules
                 │
                 └──► Knowledge Graph (NetworkX / Neo4j)
```

No raw document bytes are stored. Only processed metadata, chunks,
embeddings, and graph nodes are persisted.

---

## Supported Input Sources

| Source type | Field | Notes |
|---|---|---|
| Public URL (PDF) | `url` | Must be publicly accessible without authentication |
| Public URL (plain text) | `url` | `.txt`, `.md`, `.html` |
| Raw text payload | `text` | Max 500 000 characters |
| Future: API connectors | — | EUR-Lex, World Bank, UN Docs (planned) |

---

## API Usage

### Ingest from URL

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.org/docs/climate_strategy_2030.pdf",
    "title": "National Climate Strategy 2030",
    "source_label": "Ministry of Environment"
  }'
```

**Response (201 Created):**
```json
{
  "doc_id": "a3f8c2d1e4b7",
  "title": "National Climate Strategy 2030",
  "source_url": "https://example.org/docs/climate_strategy_2030.pdf",
  "page_count": 48,
  "chunk_count": 312,
  "entities_extracted": 87,
  "graph_nodes_added": 34,
  "message": "Document ingested successfully."
}
```

### Ingest from text payload

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article 1. All member states shall reduce emissions by 55 percent...",
    "title": "Draft Regulation",
    "source_label": "Internal Research"
  }'
```

### List ingested documents

```bash
curl http://localhost:8000/api/v1/documents
```

### Delete a document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/a3f8c2d1e4b7
```

---

## Chunk Structure

Each document is split into overlapping `TextChunk` objects. Every chunk carries:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | `{doc_id}_p{page:04d}_c{index:06d}` |
| `doc_id` | `str` | SHA-256 prefix of the raw content |
| `text` | `str` | The chunk text |
| `page_number` | `int` | 1-based page number |
| `chunk_index` | `int` | Global sequential index |
| `char_start` | `int` | Offset within the page text |
| `char_end` | `int` | End offset within the page text |
| `source_url` | `str \| None` | Origin URL (None for text payloads) |
| `section_heading` | `str \| None` | Nearest detected section heading |
| `metadata` | `dict` | doc_title, source_label, page_number, etc. |

### Chunking parameters

| Parameter | Default | Env var | Description |
|---|---|---|---|
| `chunk_size` | 512 chars | `CHUNK_SIZE` | Target characters per chunk |
| `chunk_overlap` | 64 chars | `CHUNK_OVERLAP` | Overlap between adjacent chunks |
| `min_chars` | 50 chars | — | Discard chunks shorter than this |

---

## Error Handling

The ingestion layer raises typed exceptions that map to specific HTTP status codes:

| Exception | HTTP Code | Cause |
|---|---|---|
| `DocumentFetchError` | 502 Bad Gateway | URL unreachable, HTTP 4xx/5xx, timeout |
| `DocumentSizeError` | 413 Too Large | Document exceeds `MAX_DOCUMENT_SIZE_MB` |
| `UnsupportedContentTypeError` | 415 Unsupported | Server returned image, zip, or binary |
| `DocumentEmptyError` | 422 Unprocessable | No text after parsing (scanned PDF, blank) |
| `DocumentParseError` | 422 Unprocessable | PDF parser failed |

### Common issues and fixes

**HTTP 502 — URL not publicly accessible**
> The URL requires authentication or is behind a firewall. Extract the text
> and submit it as a text payload instead.

**HTTP 413 — Document too large**
> Raise `MAX_DOCUMENT_SIZE_MB` in your `.env` or split the document into sections.

**HTTP 422 — Empty document (scanned PDF)**
> The PDF has no embedded text layer. Run OCR first:
> ```bash
> ocrmypdf input.pdf output_with_text.pdf
> ```
> Then ingest the OCR'd version via URL.

**HTTP 415 — Unsupported content type**
> The URL returned an image, spreadsheet, or archive. Download and convert
> to plain text, then submit as a text payload.

---

## Size Limits

| Limit | Default | Override |
|---|---|---|
| URL download size | 50 MB | `MAX_DOCUMENT_SIZE_MB` in `.env` |
| Text payload | 500 000 characters | — (fixed) |
| Chunk size | 512 characters | `CHUNK_SIZE` in `.env` |

---

## Connector Protocol

To add a new source connector (e.g. EUR-Lex, World Bank API):

1. Implement the `DocumentConnector` protocol in `backend/app/core/connectors.py`:

```python
class EURLexConnector:
    def __init__(self, celex_id: str) -> None:
        self._celex = celex_id

    async def fetch(self) -> PolicyDocument:
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:{self._celex}"
        return await load_from_url(url, source_label="EUR-Lex")

    @property
    def source_description(self) -> str:
        return f"EUR-Lex CELEX:{self._celex}"
```

2. Add the connector to `connector_from_request()` factory.
3. Add a new field to `IngestRequest` schema if the connector needs a custom identifier.
4. Write tests that mock the underlying `load_from_url` call.

---

## Demo: Ingest the Sample File

The repository includes a synthetic sample policy in `sample_data/sample_policy.txt`
for local testing without network access.

```bash
# Start the server
uvicorn backend.app.main:app --reload --port 8000

# Ingest the sample (Unix/macOS)
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$(cat sample_data/sample_policy.txt | tr '\n' ' ')\", \"title\": \"Demo Policy\"}"

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What emission reduction targets are set for 2030?"}'
```

---

## Running Ingestion Tests

```bash
# Full ingestion test suite (offline, no API keys, no network)
pytest backend/tests/test_ingestion.py -v

# All tests with coverage
pytest backend/tests/ --cov=backend/app --cov-report=term-missing
```
