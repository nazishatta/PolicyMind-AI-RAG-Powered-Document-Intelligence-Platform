# System Architecture

## Overview

PolicyMind-AI is structured as a modular, layered backend service. Each layer has a single responsibility and communicates through well-defined interfaces, making it straightforward to swap implementations (e.g. ChromaDB → Pinecone, NetworkX → Neo4j, mock LLM → Anthropic) without modifying the pipeline logic.

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI REST API                            │
│   POST /api/v1/ingest   POST /api/v1/query   GET /api/v1/graph  │
│   GET  /health                                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ dependency injection
┌───────────────────────────────▼─────────────────────────────────┐
│                      GraphRAG Pipeline                           │
│                 (services/rag_pipeline.py)                        │
│                                                                   │
│   ingest(doc) ──► chunk ──► embed ──► vector upsert             │
│                         └──► entity extract ──► graph write      │
│                                                                   │
│   query(q)  ──► embed q ──► vector search ─┐                    │
│                         └──► graph traverse ┤                    │
│                                    context fusion                 │
│                                    LLM completion                 │
│                                    citation engine                │
└─────────────────────────────────────────────────────────────────┘
        │               │                │              │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐ ┌────▼──────────┐
│  Vector      │ │  Graph      │ │  LLM        │ │  Embedding    │
│  Store       │ │  Service    │ │  Provider   │ │  Model        │
│              │ │             │ │             │ │               │
│  ChromaDB    │ │  NetworkX   │ │  Anthropic  │ │  Sentence     │
│  (default)   │ │  (default)  │ │  OpenAI     │ │  Transformers │
│  InMemory    │ │  Neo4j      │ │  Mock       │ │  (local)      │
└──────────────┘ └─────────────┘ └─────────────┘ └───────────────┘
```

## Directory Structure

```
backend/
├── app/
│   ├── main.py          # FastAPI app factory + lifespan
│   ├── config.py        # Pydantic Settings (env-driven)
│   ├── api/
│   │   ├── dependencies.py   # Dependency injection factory
│   │   └── routes/
│   │       ├── health.py
│   │       ├── ingest.py
│   │       ├── query.py
│   │       └── graph.py
│   ├── core/            # Pure business logic (no I/O)
│   │   ├── document_loader.py
│   │   ├── text_chunker.py
│   │   ├── embeddings.py
│   │   ├── entity_extraction.py
│   │   └── citation_engine.py
│   ├── services/        # External service integrations
│   │   ├── llm_service.py
│   │   ├── vector_store.py
│   │   ├── graph_service.py
│   │   └── rag_pipeline.py
│   ├── schemas/         # Pydantic I/O models
│   └── utils/
└── tests/
```

## Data Flow — Ingestion

```
HTTP POST /api/v1/ingest
    │
    ▼
document_loader.py          # fetch URL or wrap text → PolicyDocument
    │
    ▼
text_chunker.py             # recursive split → list[TextChunk]
    │
    ├──► embeddings.py      # SentenceTransformer → float32 matrix
    │       │
    │       └──► vector_store.upsert()   # ChromaDB or InMemory
    │
    └──► entity_extraction.py   # spaCy NER + co-occurrence relations
                │
                └──► graph_service.add_entity/add_relation()
```

## Data Flow — Query

```
HTTP POST /api/v1/query
    │
    ▼
embeddings.embed_one(question)
    │
    ├──► vector_store.search()       # top-k semantic chunks
    │
    ├──► graph_service.get_edges()   # entity neighbourhood (optional)
    │
    ▼
context fusion (chunk text + graph triples)
    │
    ▼
llm_service.acomplete(system, user_prompt)
    │
    ▼
citation_engine.record() → AnswerResponse
```

## Provider Abstraction

Each service layer uses an abstract base class (ABC).  Concrete implementations are selected at startup via `VECTOR_STORE_PROVIDER`, `GRAPH_PROVIDER`, and `LLM_PROVIDER` environment variables.  This pattern eliminates vendor lock-in and keeps test code independent of external services.

| Env Var | Options |
|---|---|
| `LLM_PROVIDER` | `mock` (default), `anthropic`, `openai` |
| `VECTOR_STORE_PROVIDER` | `chroma` (default), `memory` |
| `GRAPH_PROVIDER` | `memory` (default), `neo4j` |

## Scalability Considerations

- The embedding model is process-local and cached; it does not scale horizontally without a dedicated embedding service.
- ChromaDB is single-node; for production deployments, replace with a hosted vector database via the `BaseVectorStore` interface.
- Neo4j supports clustering for large-scale graph workloads.
- The FastAPI application is stateless (all state lives in the stores) and can be scaled horizontally behind a load balancer.
