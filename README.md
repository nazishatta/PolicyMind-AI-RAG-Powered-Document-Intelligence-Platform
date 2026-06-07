# PolicyMind-AI: GraphRAG-Powered Policy Document Intelligence Platform

PolicyMind-AI is an open-source GraphRAG system that transforms long policy PDFs into searchable knowledge graphs and citation-backed answers. It combines vector retrieval, entity extraction, Neo4j graph reasoning, and explainable LLM responses to support transparent policy research.

---

## Why PolicyMind-AI?

Traditional RAG systems retrieve text chunks, but policy analysis requires more than text similarity. Analysts often need to understand relationships between countries, organizations, sectors, programs, regions, and policy goals.

PolicyMind-AI addresses this by combining:

- Semantic search for relevant document chunks
- Knowledge graph extraction for entity relationships
- GraphRAG retrieval for connected evidence
- Citation-backed answers for traceability
- Explainability layer showing sources and graph evidence

---

## Public Interest Motivation

Policy documents influence decisions in climate, education, poverty reduction, digital development, health, and economic planning. However, these documents are often long, fragmented, and difficult to compare.

This project aims to make policy knowledge more transparent, auditable, and accessible by helping researchers, civic technology teams, nonprofit organizations, and public-sector analysts ask questions over complex documents with source-backed evidence.

---

## Overview
PolicyMind-AI is an open-source GraphRAG platform for explainable question answering over policy documents...

## Visual Workflow

```mermaid
flowchart LR
    A[Policy Documents] --> B[AI Document Processing]
    B --> C[Knowledge Graph + Vector Search]
    C --> D[Ask Questions]
    D --> E[Explainable Answers]
    E --> F[Citations + Graph Evidence]
