# PolicyMind AI

![CI](https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform/actions/workflows/ci.yml/badge.svg)

**GraphRAG-powered policy document intelligence Ã¢â‚¬â€ semantic search, entity graphs, and citation-backed answers over complex public-policy corpora.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Screenshots

### Application Interface
![PolicyMind AI Interface](docs/Screenshots/01_hero_header.jpg)

### GraphRAG Knowledge Base
![GraphRAG Toggle](docs/Screenshots/02_graphrag_toggle.jpg)

### Intelligent Query Routing
![Query Routing](docs/Screenshots/03_map_reduce_routing.jpg)

### AI Answer with Citations
![Answer](docs/Screenshots/04_answer_citations.jpg)

### Structured Analysis
![Full Answer](docs/Screenshots/05_full_answer.jpg)

### Retrieval Confidence
![Confidence](docs/Screenshots/06_confidence_bar.jpg)

### Source Evidence
![Sources](docs/Screenshots/07_strong_sources.jpg)

---

## The Problem

Public policy documents Ã¢â‚¬â€ climate agreements, regulatory frameworks, health directives Ã¢â‚¬â€ are long, cross-referential, and dense with named entities: agencies, dates, targets, obligations, and relationships between them. Researchers, analysts, and civic technologists need to navigate these corpora quickly and trace every claim back to its source.

Standard keyword search misses context. Generic chat-with-PDF tools produce confident-sounding answers with no audit trail. Neither approach surfaces the structural relationships that give policy text its meaning.

---

## Why Traditional RAG Is Not Enough

Standard Retrieval-Augmented Generation retrieves the most similar text chunks and passes them to an LLM. For policy analysis this falls short in several ways:

| Capability | Standard RAG | PolicyMind AI |
|---|---|---|
| Semantic search | Ã¢Å“â€œ | Ã¢Å“â€œ |
| Citation tracing (page Ã‚Â· excerpt Ã‚Â· score) | Partial | Ã¢Å“â€œ Structured citations with page numbers |
| Entity relationship graphs | Ã¢Å“â€” | Ã¢Å“â€œ spaCy NER + NetworkX |
| Graph-augmented score fusion | Ã¢Å“â€” | Ã¢Å“â€œ Hybrid vector + graph reranking (0.7/0.3) |
| Smart query routing | Ã¢Å“â€” | Ã¢Å“â€œ Map-Reduce Ã‚Â· GraphRAG Ã‚Â· Standard |
| Multi-document comparative analysis | Ã¢Å“â€” | Ã¢Å“â€œ Per-document Map phase + synthesis Reduce |
| Explainable confidence scores | Ã¢Å“â€” | Ã¢Å“â€œ Visual progress bar with Strong/Moderate/Weak tiers |
| Provider flexibility | Varies | Ã¢Å“â€œ Groq LLaMA 3.1 primary, OpenAI GPT-4o-mini optional |

---

## What PolicyMind AI Does

PolicyMind AI is a Streamlit-based document intelligence platform that lets users upload policy PDF documents, build a semantic knowledge base, and ask natural language questions. The system retrieves relevant document passages using hybrid vector and graph search, then generates grounded answers with page-level source citations.

The pipeline has four stages: document ingestion (PyMuPDF), vector indexing (ChromaDB with cosine similarity), knowledge graph construction (spaCy NER + NetworkX), and answer generation (Groq LLaMA 3.1).

A smart query router automatically selects the best retrieval strategy: Map-Reduce for summarization and cross-document comparison, GraphRAG for relationship and entity queries, and standard semantic search for specific factual questions.

---

## Key Features

- **Smart query routing** Ã¢â‚¬â€ keyword-based classifier automatically routes to Map-Reduce, GraphRAG, or Standard RAG based on query intent
- **GraphRAG hybrid retrieval** Ã¢â‚¬â€ vector search fused with knowledge-graph entity re-ranking (70% vector weight, 30% graph boost)
- **Map-Reduce for multi-document queries** Ã¢â‚¬â€ processes each document independently in the Map phase, then synthesizes a cross-document answer in the Reduce phase
- **Grounded answers with page citations** Ã¢â‚¬â€ every answer includes document name, page number, relevance score, and verbatim text excerpt
- **Explainable confidence scores** Ã¢â‚¬â€ visual progress bar colour-coded as Strong (Ã¢â€°Â¥65%), Moderate (35Ã¢â‚¬â€œ65%), or Weak (<35%)
- **Multi-document support** Ã¢â‚¬â€ upload multiple PDFs, all indexed into a shared knowledge base with per-document retrieval
- **Local embeddings** Ã¢â‚¬â€ `all-MiniLM-L6-v2` runs fully on CPU with no external API calls, keeping document content private by default
- **Graceful degradation** Ã¢â‚¬â€ falls back to standard vector search when spaCy or NetworkX are unavailable, with clear UI indicators

---

## Architecture

```mermaid
flowchart TD
    subgraph Upload
        A[PDF Documents] --> B[PyMuPDF Text Extraction]
        B --> C[LangChain Text Chunker\nchunk_size=1000 overlap=150]
    end
    subgraph Indexing
        C --> D[sentence-transformers\nall-MiniLM-L6-v2]
        C --> E[spaCy NER\nEntity Extraction]
        D --> F[(ChromaDB\nCosine Similarity)]
        E --> G[(NetworkX\nKnowledge Graph)]
    end
    subgraph Query
        H[User Question] --> I[Smart Query Router]
        I -->|summarize/compare| J[Map-Reduce RAG\nPer-document analysis]
        I -->|relationships/entities| K[GraphRAG\nHybrid vector + graph]
        I -->|specific questions| L[Standard RAG\nSemantic search]
        J --> M[Groq LLaMA 3.1\nAnswer Generation]
        K --> M
        L --> M
        M --> N[Grounded Answer\n+ Page Citations\n+ Confidence Score]
    end
    F --> K
    F --> L
    G --> K
```

All pipeline components are resolved at startup from environment variables. No code changes are required to switch LLM providers.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Web interface |
| PDF Parsing | PyMuPDF (fitz) | Text and page extraction |
| Chunking | LangChain | RecursiveCharacterTextSplitter (size=1000, overlap=150) |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2, local CPU |
| Vector Database | ChromaDB | Cosine similarity search |
| Knowledge Graph | NetworkX | Entity relationship graph |
| Entity Extraction | spaCy | Named entity recognition (en_core_web_sm) |
| LLM | Groq LLaMA 3.1 | Answer generation (llama-3.1-8b-instant) |
| LLM Fallback | OpenAI GPT-4o-mini | Optional alternative provider |
| Language | Python 3.10+ | Core language |

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Groq API key Ã¢â‚¬â€ free at [console.groq.com](https://console.groq.com)

### Installation

```powershell
# Clone the repository
git clone https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform
cd PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the application
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

> **Note:** spaCy is optional. Without it, GraphRAG falls back to vector-only mode and entity extraction is skipped. All other features work normally.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key from [console.groq.com](https://console.groq.com) |
| `OPENAI_API_KEY` | No | OpenAI fallback (optional) |
| `CHROMA_DB_PATH` | No | ChromaDB storage path (default: `vector_db/chroma`) |
| `EMBEDDING_MODEL_NAME` | No | Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `CHUNK_SIZE` | No | Characters per chunk (default: `1000`) |
| `CHUNK_OVERLAP` | No | Overlap between chunks (default: `150`) |
| `TOP_K_RESULTS` | No | Chunks retrieved per query (default: `5`) |

Full reference: [`.env.example`](.env.example)

No secrets are committed to this repository.

---

## How to Use

1. **Upload Documents** Ã¢â‚¬â€ Upload one or more PDF policy documents using the file uploader. Multiple files are indexed together into a shared knowledge base.
2. **Build Knowledge Base** Ã¢â‚¬â€ Click **Reset & Rebuild Knowledge Base** to index documents with vector embeddings. Optionally enable GraphRAG to also build a knowledge graph.
3. **Semantic Search** Ã¢â‚¬â€ Search for relevant passages across your documents to preview retrieval quality before asking questions.
4. **Ask Questions** Ã¢â‚¬â€ Ask natural language questions. The smart router automatically selects the best strategy:
   - *Map-Reduce* is triggered by words like "summarize", "compare", "overview", "across documents"
   - *GraphRAG* is triggered by words like "relationship", "organization", "between", "responsible for"
   - *Standard RAG* handles all other specific factual questions
5. **View Sources** Ã¢â‚¬â€ Every answer includes document name, page number, relevance score, and a verbatim excerpt for every retrieved passage.

### Example Questions

```
"What are the main policy objectives in this document?"
"Which organizations are responsible for implementation?"
"What are the key recommendations?"
"Summarize the main findings across all documents"
"Compare the policy approaches across uploaded documents"
"What is the relationship between [Agency A] and [Agency B]?"
"What evidence supports the conclusions on page 12?"
"What emission reduction targets are committed to by 2030?"
```

---

## Responsible AI and Trust Layer

PolicyMind AI is designed to behave like a careful research assistant: useful, transparent, and source-grounded. Five trust mechanisms are built directly into the Streamlit pipeline:

| Mechanism | Implementation |
|---|---|
| **Visual confidence scoring** | Every answer displays a colour-coded progress bar Ã¢â‚¬â€ green (Ã¢â€°Â¥65% Strong), yellow (35Ã¢â‚¬â€œ65% Moderate), red (<35% Weak) Ã¢â‚¬â€ computed from retrieved chunk similarity scores |
| **Page-level citations** | Every source card shows document name, page number, cosine relevance score, and a verbatim excerpt Ã¢â‚¬â€ no claims without evidence |
| **Answer type classification** | Badge on every response: Document Answer Ã‚Â· Map-Reduce Ã‚Â· GraphRAG Ã‚Â· Partial Answer Ã‚Â· Evidence Only Ã¢â‚¬â€ signals how grounded the answer is |
| **Routing transparency** | UI displays which retrieval strategy was used (Standard RAG / Map-Reduce / GraphRAG) and why, so users understand the evidence path |
| **Evidence-only fallback** | When no LLM API key is configured, the system returns raw retrieved passages instead of generating an answer Ã¢â‚¬â€ the primary hallucination pathway is blocked at the source |

**What these mechanisms do not guarantee:** hallucinations remain possible with any LLM provider; retrieval is incomplete when the corpus is incomplete; confidence scores are heuristic, not calibrated probabilities. All outputs should be verified against original source documents.

---

## Project Structure

```
PolicyMind-AI/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ app/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ streamlit_app.py         # Main Streamlit application entry point
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ components/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ chat_ui.py           # Step 4: Q&A panel, answer card, confidence bar
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ source_display.py    # Step 5: Source evidence cards
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ upload_ui.py         # Step 1: PDF upload and document preview
Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ ui_helpers.py        # Shared step header renderer
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ src/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.py                # Environment variable configuration
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ document_loader.py       # PyMuPDF PDF processing (multi-file)
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ text_splitter.py         # LangChain RecursiveCharacterTextSplitter
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ vector_store.py          # ChromaDB create / query / reset
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ retriever.py             # Semantic search + retrieval quality scoring
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ rag_chain.py             # Standard RAG pipeline + Groq/OpenAI integration
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ graph_rag.py             # GraphRAG: spaCy NER + NetworkX + hybrid search
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ map_reduce_rag.py        # Map-Reduce pipeline + smart query router
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ citation_utils.py        # Source formatting helpers
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ logger.py                # Structured logging setup
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ docs/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ Screenshots/             # Application screenshots
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ raw/                     # Uploaded PDF storage (gitignored)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ vector_db/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ chroma/                  # ChromaDB persistent storage (gitignored)
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt             # Python dependencies
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .env.example                 # Environment variable template
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ README.md
```

---

## Roadmap

- [ ] **Streaming responses** Ã¢â‚¬â€ server-sent events for long LLM completions
- [ ] **Additional LLM providers** Ã¢â‚¬â€ Anthropic Claude, Mistral, and local Ollama support
- [ ] **Fine-tuned NER** Ã¢â‚¬â€ domain-adapted entity extraction for policy terminology
- [ ] **Non-English support** Ã¢â‚¬â€ multilingual embeddings for EU and UN document corpora
- [ ] **Async ingestion** Ã¢â‚¬â€ background task processing for large PDF batches
- [ ] **Persistent graph export** Ã¢â‚¬â€ GraphML / RDF serialisation for graph portability
- [ ] **Evaluation harness** Ã¢â‚¬â€ labeled Q&A pairs over public policy corpora for retrieval benchmarking

---

## Contributing

Contributions are welcome. Before opening a large PR, please open an issue to discuss scope and approach.

High-priority areas: additional LLM providers, multilingual support, and annotated evaluation datasets.

---

## License

[MIT](LICENSE) Ã¢â‚¬â€ free for research, education, and civic technology use.
