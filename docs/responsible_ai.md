# Responsible AI and Trust Layer

PolicyMind-AI is designed to behave like a careful research assistant: useful, transparent, and entirely source-grounded. This document describes the technical trust mechanisms built into the system, the known limitations of each, and the obligations that remain with users and operators.

---

## Intended Use

PolicyMind-AI is appropriate for:

- Research and exploratory analysis of **public policy documents**
- Civic technology applications that need document intelligence over large corpora
- Educational use by students, researchers, and journalists working with policy text
- Prototyping and demonstration of GraphRAG techniques

It is **not** appropriate for:

- Automated legal advice or formal legal research
- High-stakes regulatory compliance determinations without human review
- Processing confidential, classified, or personally identifiable documents
- Replacing professional policy analysts or legal counsel

---

## The Eight Trust Mechanisms

### 1. Citation-First Answer Generation

The system prompt instructs the LLM to begin every answer by naming the source document(s) and page(s) it is drawing from, and to include inline page references after each factual claim. This is enforced through the prompt, not by post-processing.

**System prompt rule (verbatim):**
> CITATION-FIRST: Begin every answer by stating which document(s) and page(s) you are drawing from. After each factual claim, add a brief source reference in brackets, e.g. "(p.4, Section 2)" or "(p.7)".

Every answer is accompanied by a structured `citations` list in the API response — each entry carries `chunk_id`, `doc_id`, `doc_title`, `page_number`, `section_heading`, and `relevance_score`, enabling any claim to be traced to its exact source.

---

### 2. Refusal When Sources Do Not Support an Answer

When no relevant passages are found in the indexed corpus, the pipeline **does not call the LLM**. Calling an LLM with an empty context is the primary hallucination pathway; the pipeline eliminates this risk by refusing before generation.

The refusal response:
- Contains the phrase: *"The available documents do not contain enough information to answer this question."*
- Sets `answer_type = "no_corpus"` so consuming applications can detect and handle refusals programmatically.
- Returns empty `citations`, `retrieved_chunks`, and `graph_evidence`.
- Still surfaces a `limitations` list explaining why no answer was given.

When the LLM itself judges the retrieved context insufficient (possible with real providers), the system prompt instructs it to produce the same refusal phrase, which sets `answer_type = "refused"`.

---

### 3. Confidence Notes

Every query response includes two confidence signals:

**`confidence` (float, [0, 1] or null)**
Computed as the mean cosine similarity of the top-3 retrieved chunks plus a graph-evidence bonus:
```
confidence = min(mean_top3_cosine + len(graph_triples) × 0.05, 1.0)
```
`null` when no passages were retrieved.

**`confidence_note` (string)**
A human-readable explanation of the confidence score, e.g.:
```
"Derived from 3 retrieved passages (mean top-3 cosine similarity: 0.72), with
+0.10 boost from 2 graph triples. Evidence quality is moderate — verify key
claims against the source."
```

This note always accompanies the numeric score so that users who are not familiar with cosine similarity can understand what the score means in plain language.

---

### 4. Categorical Evidence Quality

`evidence_quality` provides a single categorical label that summarises the retrieval situation:

| Label | Condition |
|---|---|
| `strong` | confidence ≥ 0.7 **and** ≥ 3 passages retrieved |
| `moderate` | confidence ≥ 0.5 **and** ≥ 2 passages retrieved |
| `weak` | confidence ≥ 0.35 |
| `insufficient` | confidence < 0.35 or no passages retrieved |

This label is intended for programmatic use — a downstream application can gate certain actions (e.g. generating a report) on `evidence_quality` being `"strong"` or `"moderate"`.

---

### 5. Answer Type Classification

`answer_type` classifies every response into one of four categories:

| Value | Meaning |
|---|---|
| `cited` | Answer is grounded in retrieved passages at adequate confidence |
| `partial` | Evidence was found but is incomplete or weak (confidence < 0.35, or the LLM used the "Based on limited evidence:" prefix) |
| `refused` | The LLM judged the retrieved context insufficient and declined to answer |
| `no_corpus` | No documents were indexed when the query was made — LLM was not called |

Applications that display answers to users should surface this classification prominently. A `partial` or `refused` answer should be presented with appropriate caveats; a `no_corpus` answer should prompt the user to ingest relevant documents.

---

### 6. Low-Evidence Preamble

When `confidence < 0.35` but passages were retrieved, the pipeline injects a preamble into the user prompt before LLM generation:

```
[RETRIEVAL NOTE: Confidence score is low (0.28). The retrieved passages may not
fully address the question. You MUST begin your answer with "Based on limited
evidence:" and clearly state which parts of the question the context cannot answer.]
```

This instructs the LLM to signal uncertainty in its answer text itself, independently of the structured `confidence` and `evidence_quality` fields.

---

### 7. Source Traceability

Every passage that contributes to an answer is exposed in the API response at two levels:

**`citations`** — processed, citation-formatted references sorted by relevance:
- `chunk_id` — unique identifier within the vector store
- `doc_id` — document identifier returned by the ingest endpoint
- `doc_title` — human-readable document title
- `page_number` — source page
- `section_heading` — detected section heading (if present)
- `relevance_score` — cosine similarity clamped to [0, 1], rounded to 4 d.p.

**`retrieved_chunks`** — raw passage text before answer generation:
- Contains up to 500 characters of the verbatim passage
- Enables users to read the exact text the LLM received

The intersection rule is enforced in the pipeline: `citation.chunk_id` is always a subset of `retrieved_chunk.chunk_id`. There are no hidden document assumptions — if a passage was used, it appears in the response.

---

### 8. Deterministic Limitations Inference

A structured `limitations` list is produced for every response — not generated by the LLM, but computed deterministically by inspecting retrieval statistics. Current conditions:

| Condition | Limitation text |
|---|---|
| `LLM_PROVIDER=mock` | "Mock LLM provider active — responses are deterministic stubs." |
| No passages retrieved | "No relevant passages were found in the corpus." |
| Exactly 1 passage retrieved | "Only one supporting passage was retrieved." |
| Knowledge graph is empty | "Knowledge graph is empty — entity extraction requires spaCy." |
| No graph evidence for this query | "No graph evidence matched this query." |
| Confidence < 0.4 | "Retrieval confidence is low (X.XX)." |

These limitations are always present when applicable, regardless of whether the LLM would mention them. They represent a systematic, auditable disclosure of known shortcomings for each query.

---

## Response Structure

A complete query response carrying all trust fields:

```json
{
  "query_id": "3f7a2c1d-...",
  "question": "What emission reduction targets are committed to by 2030?",
  "answer": "According to the National Emissions Reduction Framework 2030 (p.1, Section 1), signatory states are committed to reducing greenhouse gas emissions by 45 percent relative to 2005 baseline levels by 2030 (p.1).",
  "answer_type": "cited",
  "evidence_quality": "moderate",
  "confidence_note": "Derived from 3 retrieved passages (mean top-3 cosine similarity: 0.71), without graph evidence. Evidence quality is moderate — verify key claims against the source.",
  "citations": [
    {
      "chunk_id": "doc1-p001-c000",
      "doc_id": "a3f1c9e2",
      "doc_title": "National Emissions Reduction Framework 2030",
      "page_number": 1,
      "section_heading": "Section 1: Emission Targets",
      "relevance_score": 0.8814
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "doc1-p001-c000",
      "doc_id": "a3f1c9e2",
      "doc_title": "National Emissions Reduction Framework 2030",
      "page_number": 1,
      "section_heading": "Section 1: Emission Targets",
      "text": "All signatory states shall reduce greenhouse gas emissions by 45 percent relative to 2005 baseline levels by 2030...",
      "relevance_score": 0.8814
    }
  ],
  "graph_evidence": [],
  "confidence": 0.71,
  "limitations": [
    "Mock LLM provider active — responses are deterministic stubs.",
    "No graph evidence matched this query."
  ],
  "latency_ms": 142.3,
  "provider": "mock",
  "model": "mock-model"
}
```

---

## What the System Does Not Guarantee

**These trust mechanisms reduce risk but do not eliminate it:**

- **Hallucination**: LLMs can produce hallucinations even when grounded in retrieved context. The citation engine mitigates but does not eliminate this. All answers should be verified against the original source documents identified in `citations`.

- **Retrieval completeness**: Semantic search retrieves the most similar passages, not necessarily all relevant passages. A corpus that is incomplete or poorly chunked will produce incomplete answers without any warning beyond a low `confidence` score.

- **Entity extraction accuracy**: spaCy's `en_core_web_sm` model has moderate precision on domain-specific policy terminology. Graph evidence may be missing or inaccurate for specialised concepts.

- **Relation extraction accuracy**: Co-occurrence heuristics capture surface-level proximity, not logical or causal relationships. `MENTIONED_WITH` relations are weak signals.

- **Confidence calibration**: The confidence score is a heuristic based on cosine similarity, not a calibrated probability. A score of 0.8 does not mean there is an 80% chance the answer is correct.

- **Language coverage**: The pipeline is optimised for English. Non-English documents produce degraded NER and embedding quality.

---

## Bias and Fairness Considerations

- `all-MiniLM-L6-v2` was trained on general English text. It may not represent domain-specific policy vocabulary equitably across different geopolitical or cultural contexts.
- The system surfaces only what it has been given. Systematic gaps in the corpus (e.g. excluding documents from certain regions or organisations) produce systematic gaps in answers.
- LLM providers apply their own content policies and safety systems, which may affect answer framing on sensitive topics.

---

## Data Handling

- PolicyMind-AI does not transmit document content to any third party when operating with `LLM_PROVIDER=mock` (the default).
- When `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai`, retrieved document excerpts (up to ~2 000 characters per query) are sent to those providers' APIs, subject to their data handling policies.
- For sensitive documents, use `LLM_PROVIDER=mock` or a locally hosted model.
- The `.gitignore` excludes `data/` and `.env` from version control. Never commit document data or API keys.

---

## Human Oversight Obligations

All outputs from PolicyMind-AI should be treated as a starting point for analysis, not a final authority. Users are responsible for:

1. **Verifying citations** against original source documents — use `citations[].page_number` and `citations[].doc_title` to locate the source.
2. **Assessing completeness** — check whether the `evidence_quality` and `confidence` values suggest the corpus was sufficient.
3. **Reading the limitations** — every `limitations` entry identifies a specific caveat; take them seriously.
4. **Not acting on `partial` or `refused` answers** without additional research.
5. **Applying professional judgement** before using any output to inform policy decisions.

---

## Reporting Issues

If you observe problematic outputs — hallucinations, incorrect citations, biased responses, or unexpected refusals — please open a [GitHub issue](https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform/issues) including:

- The question asked
- The document(s) ingested (title and source URL if public)
- The full API response (remove any sensitive content)
- A description of the expected vs. observed behaviour
