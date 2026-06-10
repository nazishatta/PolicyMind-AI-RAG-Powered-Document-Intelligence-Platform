# Evaluation Framework

PolicyMind-AI organises quality measurement into nine dimensions that together cover the full GraphRAG pipeline — from document ingestion through knowledge-graph construction, hybrid retrieval, answer generation, and citation tracing.

This document defines each dimension, its metrics and rubric, the synthetic dataset used for automated tests, and instructions for researchers who want to evaluate the system on their own corpora.

---

## Evaluation Dimensions at a Glance

| # | Dimension | What is measured | Primary metric |
|---|---|---|---|
| 1 | Retrieval Relevance | Are the right passages retrieved? | Recall@k, MRR |
| 2 | Citation Accuracy | Are citations structurally complete and correctly bounded? | Field completeness |
| 3 | Entity Extraction Quality | Are named entities correctly identified? | Entity recall |
| 4 | Relation Extraction Quality | Are entity relationships correctly typed? | Relation precision |
| 5 | Answer Faithfulness | Is the answer grounded in retrieved context? | LLM-as-judge / confidence |
| 6 | Source Coverage | Do retrieved passages cover the answer's evidentiary basis? | Keyword hit-rate@5 |
| 7 | Hallucination Risk | Are known limitations transparently surfaced? | Limitation surface rate |
| 8 | Latency | Is the system fast enough for interactive use? | p50 / p95 query latency |
| 9 | Reproducibility | Do identical inputs produce identical outputs? | doc_id and chunk-count stability |

---

## 1. Retrieval Relevance

**What it measures:** Whether the vector + graph hybrid retrieval pipeline returns passages that actually address the question, ranked appropriately.

### Metrics

**Recall@k**
```
Recall@k = |{gold keywords found in top-k chunks}| / |{gold keywords}|
```

`gold keywords` are words or phrases that must appear in a relevant passage (e.g. `"45 percent"`, `"2030"`, `"Just Transition Fund"`). Recall@k measures the fraction found in the top-k retrieved passages.

**Mean Reciprocal Rank (MRR)**
```
MRR = (1/|Q|) × Σ (1 / rank_i)
```
where `rank_i` is the position of the first chunk containing any gold keyword for query `i`. MRR is 1.0 when the top chunk always contains relevant content, and approaches 0 when relevant content appears only at the bottom of the ranked list.

### Rubric

| Score | Meaning |
|---|---|
| Recall@5 ≥ 0.8 | Strong: most gold keywords found in top 5 passages |
| Recall@5 0.5–0.8 | Moderate: partial coverage; consider larger top_k |
| Recall@5 < 0.5 | Weak: retrieval may be misaligned; check embeddings |
| MRR ≥ 0.7 | The most relevant passage is consistently near the top |

### Expected baseline

With the mock stub embeddings (random unit vectors, `seed=42`), retrieval order is essentially random. Recall@5 ≈ 0.3–0.5 is expected at random for well-written synthetic docs where keywords appear in a minority of chunks. These numbers reflect embedding quality, not retrieval pipeline correctness.

Use `VECTOR_STORE_PROVIDER=chroma` with `sentence-transformers/all-MiniLM-L6-v2` (the default real embedding model) for meaningful retrieval quality measurement.

### Monotonicity invariant (always testable)

Regardless of embedding quality:
```
Recall@k₂ ≥ Recall@k₁  when k₂ > k₁
```
This invariant holds with any embedding model and is verified in `test_evaluation.py::TestRetrievalRelevance::test_recall_k5_gte_recall_k1_monotone`.

---

## 2. Citation Accuracy

**What it measures:** Whether each citation produced by the `CitationEngine` carries the full provenance required for a researcher to locate and verify the source.

### Required citation fields

| Field | Type | Constraint |
|---|---|---|
| `chunk_id` | string | Non-empty |
| `doc_id` | string | Matches the ingest response `doc_id` |
| `doc_title` | string | Non-empty |
| `page_number` | int | ≥ 0 |
| `section_heading` | string or null | — |
| `relevance_score` | float | ∈ [0.0, 1.0] |

### Metric: field completeness

```
completeness = |{citations with all required fields and valid score}| / |{total citations}|
```

A completeness of 1.0 means every cited passage is fully auditable.

### Score clamping

Relevance scores are clamped in `CitationEngine.record()`:
```python
relevance_score = round(max(0.0, min(1.0, raw_score)), 4)
```

This prevents negative cosine similarities (possible with the stub embedding model) from propagating to the API response.

### Rubric

| Score | Meaning |
|---|---|
| 1.0 | Every citation is structurally complete |
| 0.9–1.0 | Acceptable; isolated missing optional fields |
| < 0.9 | Citations may be incomplete; investigate the CitationEngine |

---

## 3. Entity Extraction Quality

**What it measures:** Whether the spaCy NER pipeline correctly identifies policy-relevant named entities — organisations, locations, legislation, financial figures, and dates.

### Entity types retained

| spaCy label | Policy-domain meaning |
|---|---|
| `ORG` | Agencies, ministries, international bodies |
| `GPE` | Countries, regions, cities |
| `LAW` | Acts, regulations, treaties, directives |
| `MONEY` | Budget figures and financial commitments |
| `PERCENT` | Targets, thresholds, rates |
| `DATE` | Policy deadlines, timelines, review periods |
| `PRODUCT` | Named programmes or initiatives |
| `EVENT` | Summits, conferences |
| `NORP` | Political groups, nationalities |

### Metric: entity recall

```
entity_recall = |{expected entities found in extracted set}| / |{expected entities}|
```

Matching is case-insensitive and substring-based (e.g. `"Environment Agency"` matches extracted text `"Environment Agency"`).

### Rubric

| Recall | Meaning |
|---|---|
| ≥ 0.8 | Strong extraction; most policy entities captured |
| 0.5–0.8 | Moderate; domain-specific terms may be missed |
| < 0.5 | Weak; consider a fine-tuned NER model |
| 0.0 | spaCy not installed — regex fallback active |

### Graceful degradation

`extract_entities()` returns `[]` rather than raising an exception when spaCy or its model is unavailable. The regex fallback in `HybridRetriever._extract_entities_from_text()` captures capitalised phrases for graph expansion, but these are not stored as typed entity objects.

To enable full entity extraction:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## 4. Relation Extraction Quality

**What it measures:** Whether the co-occurrence heuristics correctly assign typed relations between co-located entities.

### Relation types and rules

| Relation | Trigger rule |
|---|---|
| `OPERATES_IN` | ORG entity within 200 chars of a GPE entity |
| `TARGETS_BY` | PERCENT or MONEY entity within 200 chars of a DATE |
| `GOVERNED_BY` | LAW entity within 200 chars of an ORG |
| `MENTIONED_WITH` | Any two entities within 200 chars (default) |

### Metric: relation precision

Since ground-truth relation labels are not available for synthetic documents, precision is approximated as:
```
relations_per_entity = graph_relations / graph_entities
```

A ratio > 0 confirms that the extraction pipeline ran. Meaningful precision measurement requires human-annotated relation triples (see §Evaluation Datasets below).

### Rubric

| Ratio | Meaning |
|---|---|
| > 0.5 | Pipeline ran; entities are connected in the graph |
| = 0.0 | Extraction did not run (spaCy not installed) |

---

## 5. Answer Faithfulness

**What it measures:** Whether the generated answer contains only claims that can be traced to retrieved passages.

### LLM-as-judge approach (with a real provider)

For each `(answer, retrieved_chunks)` pair, prompt a second LLM:

```
You are an impartial evaluator. Given the question, the retrieved passages, and
the generated answer, rate answer faithfulness on a scale of 1–5:

  5 = all claims are directly supported by the provided passages
  4 = nearly all claims are supported; one minor extrapolation
  3 = most claims are supported; one claim cannot be verified
  2 = several claims go beyond the provided passages
  1 = the answer contains significant hallucinations

Question: {question}

Retrieved passages:
{context}

Generated answer:
{answer}

Faithfulness rating (1–5):
```

Report the fraction of answers rated 4 or 5 as the faithfulness score.

### Automated proxy (mock provider)

When the mock LLM is active, faithfulness measurement relies on the pipeline's own transparency mechanisms:

- **Confidence score**: mean top-3 cosine similarity + graph boost. Represents retrieval quality, not generation quality.
- **Limitations list**: deterministically surfaced when the mock is active, corpus is sparse, or confidence is low.
- **Mock limitation rate**: fraction of queries where the mock-provider limitation string appears. Should be 1.0 with `LLM_PROVIDER=mock`.

### Rubric

| Signal | Strong | Weak |
|---|---|---|
| LLM-as-judge (real provider) | ≥ 90% rated 4–5/5 | < 70% rated 4–5/5 |
| Confidence (mock) | > 0.6 | < 0.4 |
| Mock limitation rate | 1.0 | < 1.0 (bug) |

---

## 6. Source Coverage

**What it measures:** Whether the retrieved passages contain enough evidence to ground a complete answer.

### Metric: keyword hit-rate@k

```
coverage@k = |{gold keywords in top-k chunk texts}| / |{gold keywords}|
```

Coverage@5 = 1.0 means every gold keyword for the query appeared in at least one of the five retrieved passages. This does not verify that the LLM used each keyword correctly — only that the evidence was available to it.

### Rubric

| Coverage@5 | Meaning |
|---|---|
| ≥ 0.8 | Sufficient evidence available in retrieved passages |
| 0.5–0.8 | Partial coverage; answer may be incomplete |
| < 0.5 | Insufficient passage coverage; increase top_k or ingest more documents |

### Monotonicity property

```
coverage@k₂ ≥ coverage@k₁  when k₂ > k₁
```

Verified in `test_evaluation.py::TestSourceCoverage::test_chunk_count_weakly_increases_with_top_k`.

---

## 7. Hallucination Risk

**What it measures:** Whether the system transparently communicates when it is likely to hallucinate, rather than returning confident-sounding answers with no basis.

### Limitation conditions (deterministic)

The `_infer_limitations()` function surfaces a plain-English limitation whenever:

| Condition | Limitation text |
|---|---|
| `LLM_PROVIDER=mock` | Mock provider active — responses are stubs |
| No chunks retrieved | No relevant passages found in corpus |
| Exactly 1 chunk retrieved | Single supporting passage — may miss context |
| Knowledge graph empty | Graph unavailable — entity extraction requires spaCy |
| No graph evidence for query | Entities in retrieved passages not in graph |
| `confidence < 0.4` | Low retrieval confidence — insufficient information |

### Metric: limitation surface rate

```
limitation_surface_rate = |{queries with ≥ 1 limitation}| / |{total queries}|
```

A rate of 1.0 means the system never returns a potentially risky answer without disclosing the reason. With `LLM_PROVIDER=mock` this should always be 1.0.

### Rubric

| Rate | Meaning |
|---|---|
| 1.0 | Every response declares its limitations |
| 0.9–1.0 | Acceptable |
| < 0.9 | Some risky responses are undisclosed — investigate |

---

## 8. Latency

**What it measures:** End-to-end query latency from API call to response, and ingestion throughput.

### Metrics and targets

| Metric | Target | Notes |
|---|---|---|
| Query latency p50 | ≤ 2 000 ms | Excluding LLM provider network latency |
| Query latency p95 | ≤ 5 000 ms | End-to-end including real LLM |
| Ingest throughput | ≥ 1 doc/s | CPU, all-MiniLM-L6-v2, 512-char chunks |

### Scoring

| p50 query latency | Score |
|---|---|
| ≤ 2 000 ms | 1.0 |
| 2 001–5 000 ms | 0.5 |
| > 5 000 ms | 0.0 |

### Notes

- `latency_ms` in the query response is measured server-side from request receipt to response dispatch. It excludes HTTP round-trip time.
- The mock LLM adds ~0 ms. Real providers add 500–3 000 ms depending on model and network conditions.
- Embedding is performed locally on CPU. GPU acceleration is available by setting `EMBEDDING_DEVICE=cuda`.

---

## 9. Reproducibility

**What it measures:** Whether the system produces identical outputs for identical inputs across independent runs.

### Invariants

| Invariant | How verified |
|---|---|
| Same document text → same `doc_id` | `doc_id` is a hash of normalised content |
| Same document text → same `chunk_count` | Chunker is deterministic |
| Same query + corpus → same top chunk | Fixed-seed stub embeddings in test mode |
| Same ingest × 2 → idempotent | Vector store upserts by `chunk_id` |

### Metric

```
reproducibility = (doc_id_stable_cases + chunk_count_stable_cases) / (2 × total_cases)
```

1.0 = fully reproducible across all tested cases.

### Notes

- Reproducibility is a binary property per case: either outputs match or they do not.
- Non-determinism would indicate a bug in content hashing, chunking, or embedding.

---

## Synthetic Evaluation Dataset

The project ships a minimal inline evaluation dataset — no large files, no external downloads. It consists of three synthetic policy documents covering distinct vocabulary domains:

| ID | Title | Domain | Chunks | Queries |
|---|---|---|---|---|
| `climate_001` | National Emissions Reduction Framework 2030 | Climate policy | ~2 | 3 |
| `health_001` | National Vaccination Programme 2024 | Health policy | ~2 | 2 |
| `trade_001` | EU-SADC Trade Agreement | Trade policy | ~2 | 2 |

Each query carries `gold_keywords` — strings that must appear in a relevant passage — and the evaluation scripts use these to compute keyword-based Recall@k and coverage metrics without requiring human annotation.

The dataset is defined in `backend/tests/test_evaluation.py` (used by the automated test suite) and `scripts/evaluate.py` (used by the standalone evaluation script).

---

## Running the Automated Evaluation Tests

The test suite covers all nine dimensions as unit and integration tests. No API keys, no running server, no external downloads are required.

```bash
# All evaluation tests (offline, ~38 tests)
python -m pytest backend/tests/test_evaluation.py -v

# Individual dimension
python -m pytest backend/tests/test_evaluation.py::TestRetrievalRelevance -v

# With coverage
python -m pytest backend/tests/test_evaluation.py --cov=backend/app --cov-report=term-missing
```

---

## Running the Standalone Evaluation Script

The evaluation script computes all nine dimensions against a live server and prints a formatted report.

```bash
# Start the server
uvicorn backend.app.main:app --reload --port 8000

# Run evaluation (mock LLM, in-memory stores)
python scripts/evaluate.py

# With real LLM — set credentials in .env first
LLM_PROVIDER=anthropic python scripts/evaluate.py

# Save JSON report
python scripts/evaluate.py --report outputs/eval_report.json
```

Example report output (mock provider):
```
======================================================================
  PolicyMind-AI Evaluation Report
  LLM provider: mock  |  Vector store: memory
======================================================================

   1. Retrieval Relevance
      Score  [████░░░░░░░░░░░░░░░░] 0.214
             Recall@1                   0.071
             Recall@3                   0.179
             Recall@5                   0.214
             MRR                        0.095
      note: Scores near 0 with mock embeddings (random vectors) are expected.

   2. Citation Accuracy
      Score  [████████████████████] 1.000
             field_completeness         1.0

   ...
```

---

## Evaluating Your Own Documents

To evaluate PolicyMind-AI on a custom corpus:

### Step 1 — Create a labeled query file

Create a JSONL file where each line describes one query and its gold keywords:

```jsonl
{"question": "What emission reduction targets are committed to by 2030?", "gold_keywords": ["45 percent", "2030"], "doc_title": "My Policy Document"}
{"question": "Which agency is responsible for enforcement?", "gold_keywords": ["Environment Agency", "enforcement"], "doc_title": "My Policy Document"}
```

### Step 2 — Ingest your documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.org/policy.pdf", "title": "My Policy Document"}'
```

### Step 3 — Run queries and collect results

```python
import httpx, json

BASE = "http://localhost:8000"
client = httpx.Client(base_url=BASE)

with open("my_queries.jsonl") as f:
    queries = [json.loads(line) for line in f]

results = []
for q in queries:
    resp = client.post("/api/v1/query", json={"question": q["question"], "top_k": 5})
    body = resp.json()
    chunks = body["retrieved_chunks"]
    gold = q["gold_keywords"]
    combined = " ".join(c["text"] for c in chunks).lower()
    hits = sum(1 for kw in gold if kw.lower() in combined)
    results.append({
        "question": q["question"],
        "recall_at_5": hits / len(gold),
        "confidence": body["confidence"],
        "citation_count": len(body["citations"]),
        "limitations": body["limitations"],
    })

for r in results:
    print(f"Recall@5={r['recall_at_5']:.2f}  conf={r['confidence']}  {r['question'][:60]}")
```

### Step 4 — LLM-as-judge faithfulness (optional)

For faithfulness measurement with a real provider, extend the script with a judge prompt:

```python
JUDGE_PROMPT = """
Rate the faithfulness of this answer on a scale of 1–5.
5 = all claims supported by the provided passages.
1 = significant hallucinations.

Context: {context}
Answer: {answer}
Rating (1–5):
"""

for r in query_results:
    rating = llm.complete(JUDGE_PROMPT.format(
        context=r["context"], answer=r["answer"]
    ))
    r["faithfulness_score"] = int(rating.strip()[0])
```

---

## Public Evaluation Corpora

The project does not ship proprietary document datasets. Researchers are encouraged to use publicly licensed policy document collections:

| Source | Coverage | License |
|---|---|---|
| [EUR-Lex](https://eur-lex.europa.eu) | EU legislation and directives | CC BY 4.0 |
| [UK Government Publications](https://www.gov.uk/government/publications) | UK policy and guidance | OGL v3.0 |
| [World Bank Open Knowledge Repository](https://openknowledge.worldbank.org) | Development policy reports | CC BY 3.0 |
| [UNDP Policy Documents](https://www.undp.org) | UN development programmes | Varies by document |

All documents must be verified as publicly licensed before ingestion. Users are responsible for complying with source document terms.

---

## Known Evaluation Limitations

- **Keyword-based Recall@k** is a proxy for retrieval relevance. It does not distinguish between a chunk that contains the keyword incidentally and one that answers the question.
- **LLM-as-judge faithfulness** is itself subject to the quality of the judge model and the specificity of the judge prompt. It should not be used as a sole quality gate.
- **Co-occurrence relation extraction** captures surface-level proximity patterns. Causal, conditional, or temporal relationships in policy text may not be captured correctly without a trained relation extraction model.
- **Evaluation with the mock provider** tests pipeline structure, not generation quality. All quality claims about answer faithfulness require evaluation with a real LLM provider.
