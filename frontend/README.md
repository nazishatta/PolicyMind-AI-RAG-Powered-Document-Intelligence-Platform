# PolicyMind-AI Frontend

> **Status: planned.** The frontend is not yet implemented. This directory is a placeholder for the future user interface.

---

## Planned Interface

The frontend will provide a browser-based interface for:

- Uploading or linking policy documents for ingestion
- Asking natural language questions over the corpus
- Viewing citation-backed answers with source highlighting
- Exploring the knowledge graph interactively (force-directed graph view)
- Browsing ingested documents and their entity maps

---

## Candidate Approaches

| Option | Notes |
|---|---|
| **Streamlit** | Fastest to prototype; good for research demos |
| **React + TypeScript** | Production-quality; more effort; connects to the existing FastAPI backend |
| **Gradio** | Lightweight, HuggingFace-ecosystem friendly |

---

## Backend API

The frontend communicates with the FastAPI backend at `http://localhost:8000`.  
Full API documentation: [`/docs`](http://localhost:8000/docs) or [`docs/api_usage.md`](../docs/api_usage.md).

---

## Contributing a Frontend

If you would like to build the frontend, please open a [GitHub issue](https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform/issues) describing your proposed approach before starting work. A React + TypeScript application or a Streamlit proof-of-concept are both welcome.
