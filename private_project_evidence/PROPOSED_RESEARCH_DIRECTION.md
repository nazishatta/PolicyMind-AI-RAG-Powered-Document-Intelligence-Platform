# Proposed Research Direction: Trustworthy AI-Powered Document Intelligence for Public-Interest Domains

**Classification:** Private / Personal Records
**Prepared by:** Nazish Atta
**Date:** 2026-06-08
**Status:** Active research direction — PolicyMind-AI constitutes the working prototype

---

## Executive Summary

This document describes a research and engineering direction focused on building trustworthy AI-powered document intelligence systems for public-interest, policy, regulatory, and research documents. The core technical focus is on Graph-augmented Retrieval-Augmented Generation (GraphRAG): a hybrid retrieval architecture that combines dense semantic vector search with structured knowledge graph evidence to produce citation-backed, confidence-aware answers from complex, structured documents.

The problem being addressed is well-defined: policy documents, legislation, regulatory submissions, international agreements, and public-health guidelines are among the most consequential documents in modern governance — and they are structurally ill-suited to existing AI question-answering tools. Standard retrieval-augmented generation misses entity-level relationships. Consumer chat-with-documents tools provide no reliability signal. Most AI systems lack architectural safeguards against fabricating plausible-sounding answers when source evidence is absent or ambiguous.

The proposed research direction advances three interconnected capabilities: (1) hybrid graph-and-vector retrieval that captures both semantic similarity and structured entity relationships in policy corpora; (2) citation-backed answer generation with deterministic post-hoc verification of source support; and (3) a programmatic trust layer that computes confidence, evidence quality, and limitations from pipeline state rather than from language model self-reporting.

A working prototype — PolicyMind-AI — has been designed, implemented, tested, and documented. It demonstrates all three capabilities in a production-patterned, fully tested API platform. This prototype forms the empirical foundation for the research direction described here.

---

## Public-Interest Problem

### The Document Intelligence Gap in Public-Sector and Research Domains

Policy documents are the substrate of governance. Climate agreements specify emission targets, enforcement mechanisms, and financial obligations. Public health guidelines define diagnostic thresholds, intervention protocols, and risk stratifications. Regulatory submissions describe technical evidence and compliance requirements. Legislative instruments create rights, obligations, and institutional structures. These documents collectively represent the authoritative record of how institutions make decisions that affect large populations.

Despite their importance, these documents present a class of information retrieval challenge that current AI tools do not adequately address:

**Length and internal cross-referencing.** Policy documents routinely span dozens to hundreds of pages, with articles cross-referencing definitions in annexes, obligations referencing schedules, and implementation mechanisms distributed across non-contiguous sections. No current general-purpose language model can hold a full policy document in its context window and reason accurately across it.

**Entity and relationship density.** A climate framework document may name dozens of institutions (ORGs), jurisdictions (GPEs), legislative instruments (LAW), specific monetary commitments (MONEY), percentage targets (PERCENT), and implementation deadlines (DATE). The relationships among these entities — which institution is responsible for which obligation, which financial mechanism applies to which sector, which percentage target applies to which date — are the substantive content of the document. Retrieval systems that treat documents as bags of semantically similar text miss this relational structure entirely.

**Accountability requirements.** In research, journalism, policy analysis, and legal review contexts, an answer is only as useful as the evidence that supports it. A response that lacks source citations, that cannot be traced to a specific passage on a specific page, is not usable in professional practice. Worse, a system that generates fluent but unsupported claims actively undermines the quality of downstream analysis.

**Absence of calibrated uncertainty.** Existing AI tools almost universally present answers with uniform confidence. When a document is ambiguous, incomplete, or silent on a question, systems that generate confident-sounding responses create more risk than systems that acknowledge their limitations. Practitioners need to know not just what the system found but how much of the question the evidence actually supports.

**Hallucination pathways in RAG systems.** Standard retrieval-augmented generation pipelines call a language model even when retrieved context is empty, sparse, or irrelevant. Language models confronted with empty context have a well-documented tendency to generate plausible but factually groundless responses — a particularly dangerous failure mode for high-stakes document analysis.

These are not abstract problems. They represent real barriers to the effective use of AI in public-sector analysis, civic technology, academic research, and public-interest journalism.

---

## Technical Research Direction

The proposed research direction advances the following interconnected technical areas:

### 1. Hybrid GraphRAG Retrieval for Structured Policy Corpora

**Core problem:** Pure vector similarity retrieval treats text as a bag of semantic associations. Policy documents contain structured entity-level information — which institutions are named in which sections, how obligations are distributed across jurisdictions, how numerical targets are tied to specific implementation bodies — that is not recoverable by cosine distance alone.

**Technical approach:** Build a dual-index retrieval system that maintains both a dense vector index (for semantic similarity) and a typed knowledge graph (for entity-relation evidence). At query time, fuse vector similarity scores with graph-neighbourhood evidence using a configurable weighted formula. The graph boost term is proportional to entity confidence rather than binary — chunks whose entities appear as high-confidence graph neighbours rank proportionally higher.

**Open research question:** What is the optimal alpha (weighting between vector and graph contributions) across different document types, corpus sizes, and query patterns? How should multi-hop graph evidence be discounted? What entity types and relation types are most informative for retrieval improvement in policy versus regulatory versus scientific documents?

### 2. Citation-Backed Answer Generation with Source Verification

**Core problem:** Language models generate fluent text regardless of evidence quality. A RAG system that does not enforce citation discipline at the architecture level — not just at the prompt level — cannot reliably produce answers that are traceable to specific source passages.

**Technical approach:** Engineer a citation pipeline that (a) structures retrieved passages with page-number markers before language model generation, (b) enforces citation-first generation via system prompt constraints, (c) captures full citation provenance during retrieval (chunk_id, doc_id, page_number, section_heading, excerpt), and (d) post-processes the generated answer to rerank citations so pages explicitly named in the answer appear first. The citation API response carries machine-readable citation objects, not prose references.

**Open research question:** How can post-generation citation verification be automated — i.e., how accurately can a system detect when a generated claim is not supported by any cited passage? What are the failure modes of regex-based page reference parsing, and can they be replaced by a more robust attribution model?

### 3. Deterministic Trust Layer for AI-Generated Document Analysis

**Core problem:** Most AI systems ask the language model to self-assess its confidence. Language model self-assessments are not calibrated, not reproducible across model versions, and not trustworthy in domains where the stakes of overconfidence are high.

**Technical approach:** Compute all reliability signals from pipeline state rather than language model output. Specifically: (a) compute numerical confidence from retrieval score statistics and graph evidence count; (b) classify answer type (cited/partial/refused/no_corpus) by pattern-matching the generated answer; (c) classify evidence quality (strong/moderate/weak/insufficient) by retrieval score distributions; (d) infer a deterministic list of limitations from pipeline state flags (spaCy available, corpus size, graph depth, confidence threshold). These signals are reproducible across model providers and temperature settings.

**Open research question:** Can the heuristic confidence formula be replaced by a calibrated model trained on annotated retrieval quality data? Can the answer-type classifier be extended to distinguish degrees of partial support quantitatively?

### 4. Responsible AI Infrastructure for High-Stakes Document Analysis

**Core problem:** Deploying AI for document analysis in policy, regulatory, or research contexts requires more than a functional system. It requires architectural safeguards — structural design choices, not just prompt-level instructions — that prevent specific failure modes.

**Technical approach:** Implement a structural trust gate (no LLM call on empty context), a provider-agnostic abstraction (pipeline does not depend on any specific model), an offline-capable mode (mock provider for evaluation without API costs), and machine-readable trust fields in every API response. Document the responsible use boundaries, out-of-scope use cases, and operator obligations in structured documentation.

**Open research question:** What constitutes a complete responsible AI audit for a document intelligence system? What formal evaluation criteria should be applied to trust layer calibration? How should systems communicate their limitations to non-technical downstream users?

### 5. Evaluation Methodology for Document Intelligence Systems

**Core problem:** Evaluation of RAG systems in the open literature focuses almost entirely on question-answering accuracy benchmarks that do not apply to the structured, citation-heavy, long-document context of policy intelligence. Existing benchmarks (SQuAD, TriviaQA, NaturalQuestions) are designed for short-passage reading comprehension, not multi-page policy document analysis.

**Technical approach:** Develop an evaluation framework covering nine orthogonal dimensions: retrieval relevance, citation accuracy, entity extraction quality, relation extraction quality, answer faithfulness, source coverage, hallucination risk, latency bounds, and reproducibility. Apply this framework against real publicly licensed policy corpora with human-annotated question-answer pairs.

**Open research question:** How should "citation accuracy" be formally defined and measured for policy documents where the same claim may be partially supported by multiple non-contiguous passages? What is an appropriate gold-standard construction methodology for policy document QA benchmarks?

---

## Why GraphRAG and Citation-Backed AI Matter

Retrieval-Augmented Generation as a general technique is now well-established. Standard RAG pipelines — embed → store → retrieve → generate — are widely implemented and commercially deployed. The specific contribution of the proposed research direction lies in three areas where standard RAG is insufficient:

**Graph augmentation addresses the relational structure of policy documents.** Policy analysis frequently requires understanding who is responsible for what, under which conditions, subject to which constraints. These relationships are explicit in document text but require entity-level representation to be retrieved reliably. A knowledge graph built during ingestion and queried during retrieval captures this structure in a form that is compatible with existing vector search infrastructure.

**Citation discipline addresses the accountability gap.** The difference between an answer that says "the European Commission is responsible for enforcement" and one that says the same thing with a structured citation to Article 12, page 47, section "Oversight Bodies" is the difference between an assertion and a verifiable claim. In professional practice, unverifiable AI-generated assertions cannot be acted upon. Citation-backed answers can be.

**Deterministic trust signals address the calibration gap.** Research on language model self-assessment consistently finds that models are not well-calibrated — they express similar levels of confidence on questions they answer correctly and questions where they confabulate. Computing trust signals from retrieval statistics and pipeline state rather than from model output produces signals that are reproducible, auditable, and interpretable without requiring access to model internals.

---

## Methodology

The research methodology is empirical and iterative, grounded in a working prototype:

**1. Prototype-first development.** Build and validate architectural decisions in a production-patterned codebase before generalising. PolicyMind-AI serves this function: each design choice (proportional graph boost, multi-hop discounting, deterministic trust layer, trust gate) is implemented, tested, and verifiable.

**2. Ablation-style evaluation.** Measure retrieval quality with and without graph augmentation, with and without citation reranking, with and without the trust gate — to isolate the contribution of each architectural component. This establishes which components produce measurable improvements and under what conditions.

**3. Domain-specific corpus construction.** Source publicly licensed policy document collections (EUR-Lex, UN Document Portal, World Bank Open Knowledge, US Federal Register) for evaluation. Annotate a representative sample with question-answer pairs and citation labels to serve as a ground-truth benchmark.

**4. Iterative trust layer calibration.** Measure the correlation between the heuristic confidence score and human-assessed answer quality across annotated samples. Use this to inform a calibrated confidence model that replaces the heuristic formula.

**5. Open publication of methodology and results.** Document evaluation protocols, benchmark construction methodology, and ablation results in a form suitable for technical review and replication.

---

## Existing Prototype: PolicyMind-AI

PolicyMind-AI is the working prototype for this research direction. It is not a demonstration or tutorial — it is a fully implemented, tested, and documented system that instantiates all of the core architectural decisions described in this document.

**What is implemented and verified:**

- Hybrid GraphRAG retrieval with proportional graph boost and multi-hop confidence discounting (`backend/app/core/retriever.py`)
- Full ingestion pipeline: text/URL ingest, PDF parsing, recursive semantic chunking with section-heading detection, named entity recognition, typed relation extraction, vector store upsert, graph construction (`backend/app/services/rag_pipeline.py`, `backend/app/core/`)
- Five-field deterministic trust layer: `answer_type`, `evidence_quality`, `confidence_note`, `confidence`, `limitations` (`backend/app/services/rag_pipeline.py`)
- Trust gate: structural refusal when corpus is empty, before any LLM call (`backend/app/services/rag_pipeline.py`)
- Citation-first system prompt with inline page references, enforced refusal, and partial-answer prefix (`_SYSTEM_PROMPT` in `rag_pipeline.py`)
- Post-generation citation reranking by page references named in the answer (`_rerank_citations_by_answer()`)
- Provider-agnostic LLM interface with Anthropic, OpenAI, and Mock implementations (`backend/app/services/llm_service.py`)
- Per-call `graph_depth` override (1–3) without instance state mutation (`backend/app/core/retriever.py`)
- Abstract base class interfaces for vector store and graph store — provider substitution without pipeline changes (`backend/app/services/vector_store.py`, `backend/app/services/graph_service.py`)
- Nine-class evaluation framework covering retrieval, citation, entity extraction, relation extraction, faithfulness, source coverage, hallucination risk, latency, reproducibility (`backend/tests/test_evaluation.py`)
- 405-test offline suite across 11 modules; smoke test with 40 observable system checks
- Three-job GitHub Actions CI pipeline with Python 3.10/3.11/3.12 matrix
- GitHub Codespaces devcontainer for zero-setup evaluation

**Version:** 0.1.0, Development Status: Alpha.
**Frontend:** Planned, not implemented. Backend API is complete.

---

## Expected Users and Beneficiaries

**Policy researchers and academics.** Document-intensive research in political science, public administration, international relations, and environmental policy requires reliable, traceable extraction of facts from long documents. The citation-backed, confidence-aware API directly addresses this use case.

**Civic technologists and NGOs.** Organisations analysing public documents for advocacy, compliance monitoring, or public accountability benefit from a system that returns machine-readable citations and can be deployed offline or in resource-constrained environments via mock mode.

**Investigative journalists.** The `answer_type: refused` response provides an explicit signal when a document does not contain evidence for a claim — a critical feature for responsible reporting that distinguishes absence of evidence from evidence of absence.

**Government digital services teams.** Public-sector agencies that maintain large regulatory or legislative document collections can use this architecture to enable internal search and analysis tools that meet emerging responsible AI requirements: explainability, source attribution, uncertainty quantification.

**AI/ML researchers.** The evaluation framework and ablation methodology are applicable beyond policy documents to any domain where citation discipline and trust calibration matter. The prototype serves as a reference implementation for GraphRAG system design.

**Software engineers and open-source contributors.** The codebase demonstrates production-quality GraphRAG implementation at a level of detail rarely available in open-source repositories: abstract interfaces, typed schemas, comprehensive tests, and working CI.

---

## Expected Technical Contributions

The following technical contributions are expected from this research direction, grounded in the existing prototype:

1. **Proportional graph boost algorithm for hybrid RAG retrieval.** A formal description and empirical evaluation of the `fused = α × vector_score + (1 − α) × graph_boost` formula where `graph_boost` is entity-confidence-weighted, not binary. Ablation comparing binary versus proportional boost on policy corpus retrieval.

2. **Multi-hop graph neighbourhood expansion with confidence discounting.** Formal characterisation of the `_MULTIHOP_DISCOUNT = 0.8` per-hop discount scheme, with evaluation of optimal depth (1–3) versus retrieval quality tradeoffs.

3. **Deterministic trust layer specification.** A formal specification of the five-field trust layer (answer_type, evidence_quality, confidence_note, confidence, limitations) computed from pipeline state, including evaluation of its correlation with human-assessed answer quality.

4. **Citation reranking by answer-text page-reference parsing.** Description and evaluation of post-generation citation reranking: does explicit page reference in the answer text correlate with higher passage relevance than citations not mentioned in the answer?

5. **Evaluation framework for policy document QA.** A nine-dimension evaluation methodology — retrieval relevance, citation accuracy, entity extraction quality, relation extraction quality, answer faithfulness, source coverage, hallucination risk, latency, reproducibility — applicable to policy and regulatory document corpora.

6. **Open-source reference implementation.** A fully tested, documented, and CI-validated GraphRAG codebase that other researchers and practitioners can inspect, reproduce, and extend.

---

## 3–5 Year Roadmap

### Year 1: Prototype Consolidation and Empirical Baseline

- Release PolicyMind-AI publicly with hosted demo (mock mode, no API key required)
- Build frontend interface: document upload, query, citation view, confidence display
- Construct first real benchmark: 500+ annotated question-answer pairs from publicly licensed EUR-Lex and UN documents
- Run ablation experiments: vector-only vs hybrid retrieval quality; graph depth 1 vs 2 vs 3 vs retrieval recall
- Publish evaluation framework and benchmark construction methodology
- Containerise with Docker Compose for reproducible deployment

### Year 2: Retrieval Quality Improvement

- Replace proximity co-occurrence relation extraction with dependency-parse-based extractor (spaCy dependency trees)
- Implement cross-document entity resolution: co-reference resolution across documents sharing the same corpus
- Extend entity types to domain-specific categories: treaty obligations, compliance thresholds, enforcement mechanisms
- Calibrate confidence scoring: train a lightweight calibration model on annotated retrieval quality data
- Evaluate retrieval quality improvements on the Year 1 benchmark with before/after comparison

### Year 3: Trust Layer and Citation Verification

- Implement automated citation verification: post-generation check whether each generated claim is supported by a retrieved passage
- Extend answer_type classification to quantitative partial support: what fraction of the question is addressed by evidence?
- Evaluate trust layer calibration: correlation between computed confidence and human-assessed correctness across 1000+ annotated instances
- Publish trust layer specification as a reusable open standard for RAG systems

### Year 4–5: Scale and Generalisation

- Extend to additional public document domains: scientific literature (arXiv, PubMed OA), regulatory submissions (SEC EDGAR, FDA), legislative proceedings
- Develop domain-adaptive entity extraction: configurable entity type sets for different document domains without retraining the NER model
- Evaluate multi-document synthesis: answering questions that require evidence from multiple ingested documents
- Publish findings across all three core technical areas (retrieval, citation, trust) in peer-reviewed venues

---

## Limitations

**Current prototype is Alpha.** PolicyMind-AI is at version 0.1.0. The backend is complete; the frontend, Docker deployment, and real benchmark are not yet implemented.

**Evaluation uses synthetic documents.** All current evaluation data is synthetic. Real-world retrieval quality against real policy documents is not yet measured.

**Confidence formula is heuristic.** The current confidence score formula is a proxy, not a calibrated probability.

**Relation extraction is rule-based.** The four relation types are assigned by proximity co-occurrence, not by semantic role labelling or dependency parsing. Complex sentences will produce lower-quality graph edges.

**Entity extraction depends on optional spaCy.** Without spaCy, graph construction is skipped. The system degrades gracefully but loses graph-augmented retrieval entirely.

**No frontend.** Non-technical users cannot interact with the system without a browser interface.

**No authentication.** The current API is unauthenticated; adding token-based auth is a near-term addition.

---

## Next Milestones

**Milestone 1 (immediate):** Public hosted demo on Railway or Render with mock mode. No API key required for evaluators.

**Milestone 2 (near-term):** Docker Compose for reproducible local deployment. Streamlit frontend for non-technical users.

**Milestone 3 (medium-term):** First annotated benchmark dataset from publicly licensed EUR-Lex or UN documents. First ablation evaluation report comparing retrieval architectures.

**Milestone 4 (medium-term):** Replace rule-based relation extractor with spaCy dependency-parse extractor. Measure graph quality improvement.

**Milestone 5 (medium-term):** Calibrated confidence scoring. Measure correlation between confidence score and human-assessed answer quality.

**Milestone 6 (longer-term):** Automated citation verification. Post-generation check of claim-to-passage support.

---

*All current-state claims in this document are grounded in the PolicyMind-AI source code, which has been read and verified before this document was written. Future milestones are projections, not commitments.*
