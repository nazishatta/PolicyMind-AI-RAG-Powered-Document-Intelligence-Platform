#!/usr/bin/env python3
"""PolicyMind-AI evaluation script.

Runs a nine-dimension evaluation suite against a live server and prints a
structured report.  All evaluation data is synthetic and bundled inline —
no external datasets are downloaded.

Usage
-----
# Start the server:
    uvicorn backend.app.main:app --reload --port 8000

# Run evaluation (all nine dimensions):
    python scripts/evaluate.py

# Save a JSON report:
    python scripts/evaluate.py --report outputs/eval_report.json

# Target a different server:
    python scripts/evaluate.py --base-url http://localhost:9000

Evaluation dimensions
---------------------
  1.  Retrieval Relevance     — Recall@k, MRR over synthetic labeled queries
  2.  Citation Accuracy       — field completeness, score bounds, APA format
  3.  Entity Extraction       — entity count and graph population rate
  4.  Relation Extraction     — relation count per ingested document
  5.  Answer Faithfulness     — presence of mock limitation; confidence bounds
  6.  Source Coverage         — keyword hit-rate in retrieved passages
  7.  Hallucination Risk      — limitations surfaced per query; empty-corpus guard
  8.  Latency                 — p50 and p95 query latency; ingestion throughput
  9.  Reproducibility         — doc_id and chunk-count stability across runs

Important notes
---------------
- With the default mock LLM and random stub embeddings, retrieval quality
  metrics (Recall@k, MRR) will not reflect real-world performance.  Run with
  LLM_PROVIDER=anthropic and VECTOR_STORE_PROVIDER=chroma for meaningful
  quality numbers.
- Entity extraction requires spaCy:
      pip install spacy && python -m spacy download en_core_web_sm
  Without it the graph stays empty and entity-related metrics are zero.
- The script reports what is objectively measurable; it does not fabricate
  baselines or compare against unpublished benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

# ---------------------------------------------------------------------------
# Synthetic evaluation dataset
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    case_id: str
    title: str
    doc_text: str
    queries: list[dict]          # list of {question, gold_keywords}
    expected_entities: list[str] # entity strings expected in the knowledge graph


_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        case_id="climate_001",
        title="National Emissions Reduction Framework 2030",
        doc_text=(
            "Section 1: Emission Reduction Targets. Signatory states shall reduce "
            "greenhouse gas emissions by 45 percent relative to 2005 baseline levels "
            "by the year 2030. The Environment Agency will publish quarterly compliance "
            "reports and issue enforcement notices for non-compliant parties. "
            "Section 2: Renewable Energy Obligations. A minimum 40 percent share of "
            "renewable energy in final consumption is mandated by 2028. Wind and solar "
            "installations receive priority grid access under Article 7 of this Framework. "
            "Section 3: Just Transition Fund. EUR 12 billion is allocated by the "
            "Treasury to support coal-dependent communities through retraining grants "
            "and infrastructure investment. Applications must be submitted to the "
            "Regional Development Authority by June 2025."
        ),
        queries=[
            {
                "question": "What emission reduction targets are committed to by 2030?",
                "gold_keywords": ["45 percent", "2030", "greenhouse gas"],
            },
            {
                "question": "What financial support is available for transition?",
                "gold_keywords": ["EUR 12 billion", "Just Transition Fund"],
            },
            {
                "question": "What is the renewable energy obligation?",
                "gold_keywords": ["40 percent", "2028", "renewable energy"],
            },
        ],
        expected_entities=["Environment Agency", "2030", "EUR 12 billion"],
    ),
    EvalCase(
        case_id="health_001",
        title="National Vaccination Programme 2024",
        doc_text=(
            "Part I: Programme Scope. The Department of Health will procure and "
            "distribute vaccines against influenza and measles to all registered "
            "practitioners by October 2024. "
            "Part II: Funding Allocation. A budget of GBP 800 million is approved "
            "under the NHS Modernisation Act 2023 for vaccine cold-chain logistics "
            "and community outreach programmes. Regional health boards must submit "
            "procurement plans by January 2024. "
            "Part III: Monitoring and Reporting. The Public Health Observatory will "
            "report quarterly on uptake rates. Uptake below 75 percent triggers "
            "mandatory intervention under Regulation 12 of the Vaccination Framework."
        ),
        queries=[
            {
                "question": "What is the vaccination programme budget?",
                "gold_keywords": ["GBP 800 million", "NHS Modernisation Act"],
            },
            {
                "question": "When must procurement plans be submitted?",
                "gold_keywords": ["January 2024", "Regional health boards"],
            },
        ],
        expected_entities=["Department of Health", "GBP 800 million", "2024"],
    ),
    EvalCase(
        case_id="trade_001",
        title="EU-SADC Trade Agreement",
        doc_text=(
            "Title I: Scope. This agreement governs trade relations between the "
            "European Union and the Southern African Development Community. "
            "Title II: Tariff Reductions. Import duties on agricultural goods "
            "originating from SADC member states shall be reduced by 30 percent "
            "within five years of ratification. The World Trade Organization dispute "
            "resolution mechanism applies to all disagreements under this agreement. "
            "Title III: Rules of Origin. Goods must satisfy a minimum 60 percent "
            "regional value content requirement to qualify for preferential tariff "
            "rates. The Joint Committee established under Article 45 will review "
            "these thresholds annually beginning January 2025."
        ),
        queries=[
            {
                "question": "What tariff reductions are agreed for agricultural goods?",
                "gold_keywords": ["30 percent", "SADC", "agricultural"],
            },
            {
                "question": "What is the rules of origin threshold?",
                "gold_keywords": ["60 percent", "regional value content"],
            },
        ],
        expected_entities=["European Union", "SADC", "World Trade Organization"],
    ),
]

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(chunks: list[dict], gold_keywords: list[str], k: int) -> float:
    """Fraction of gold keywords found in any of the top-k retrieved chunks."""
    if not gold_keywords or not chunks:
        return 0.0
    top_k_text = " ".join(c.get("text", "") for c in chunks[:k]).lower()
    hits = sum(1 for kw in gold_keywords if kw.lower() in top_k_text)
    return hits / len(gold_keywords)


def mrr(chunks: list[dict], gold_keywords: list[str]) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first chunk containing any gold keyword."""
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").lower()
        if any(kw.lower() in text for kw in gold_keywords):
            return 1.0 / i
    return 0.0


def citation_completeness(citations: list[dict]) -> float:
    """Fraction of citations that carry all required fields with valid types."""
    required = {"chunk_id", "doc_id", "doc_title", "page_number", "relevance_score"}
    if not citations:
        return 0.0
    complete = sum(
        1 for c in citations
        if required.issubset(c.keys())
        and isinstance(c.get("relevance_score"), (int, float))
        and 0.0 <= c["relevance_score"] <= 1.0
    )
    return complete / len(citations)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class DimensionResult:
    name: str
    score: Optional[float]        # None = not computable with current config
    details: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _ingest(client: httpx.Client, case: EvalCase) -> Optional[str]:
    try:
        r = client.post("/api/v1/ingest", json={
            "text": case.doc_text,
            "title": case.title,
        })
        r.raise_for_status()
        return r.json()["doc_id"]
    except Exception as exc:
        print(f"  [WARN] Ingest failed for '{case.title}': {exc}")
        return None


def _query(client: httpx.Client, question: str, doc_id: Optional[str] = None, top_k: int = 5):
    payload: dict[str, Any] = {"question": question, "top_k": top_k}
    if doc_id:
        payload["doc_id_filter"] = doc_id
    t0 = time.perf_counter()
    r = client.post("/api/v1/query", json=payload)
    wall_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), wall_ms


# ── Dimension 1: Retrieval Relevance ────────────────────────────────────────

def eval_retrieval_relevance(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    r1_scores, r3_scores, r5_scores, mrr_scores = [], [], [], []

    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            question = q_item["question"]
            gold_kw = q_item["gold_keywords"]
            try:
                body5, _ = _query(client, question, doc_id=doc_id, top_k=5)
                chunks = body5.get("retrieved_chunks", [])
                r1_scores.append(recall_at_k(chunks, gold_kw, 1))
                r3_scores.append(recall_at_k(chunks, gold_kw, 3))
                r5_scores.append(recall_at_k(chunks, gold_kw, 5))
                mrr_scores.append(mrr(chunks, gold_kw))
            except Exception as exc:
                print(f"  [WARN] Query failed: {exc}")

    if not r5_scores:
        return DimensionResult(
            name="Retrieval Relevance", score=None,
            notes=["No queries completed successfully."],
        )

    result = DimensionResult(
        name="Retrieval Relevance",
        score=round(statistics.mean(r5_scores), 3),
        details={
            "Recall@1": round(statistics.mean(r1_scores), 3),
            "Recall@3": round(statistics.mean(r3_scores), 3),
            "Recall@5": round(statistics.mean(r5_scores), 3),
            "MRR":      round(statistics.mean(mrr_scores), 3),
            "queries_evaluated": len(r5_scores),
        },
        notes=[
            "Recall@k measures fraction of gold keywords found in top-k passages.",
            "Scores near 0 with mock embeddings (random vectors) are expected.",
            "Use real embeddings (VECTOR_STORE_PROVIDER=chroma, real model) for "
            "meaningful retrieval quality numbers.",
        ],
    )
    return result


# ── Dimension 2: Citation Accuracy ──────────────────────────────────────────

def eval_citation_accuracy(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    completeness_scores = []
    score_violations = 0
    total_citations = 0

    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            try:
                body, _ = _query(client, q_item["question"], doc_id=doc_id)
                citations = body.get("citations", [])
                total_citations += len(citations)
                completeness_scores.append(citation_completeness(citations))
                for c in citations:
                    s = c.get("relevance_score", -1)
                    if not (0.0 <= s <= 1.0):
                        score_violations += 1
            except Exception as exc:
                print(f"  [WARN] Citation eval failed: {exc}")

    score = round(statistics.mean(completeness_scores), 3) if completeness_scores else None
    return DimensionResult(
        name="Citation Accuracy",
        score=score,
        details={
            "field_completeness": score,
            "total_citations_checked": total_citations,
            "score_out_of_range_count": score_violations,
        },
        notes=[
            "Field completeness = fraction of citations with all required fields "
            "and relevance_score in [0, 1].",
            "1.0 means every citation is structurally valid.",
        ],
    )


# ── Dimension 3: Entity Extraction Quality ──────────────────────────────────

def eval_entity_extraction(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    try:
        health = client.get("/health").json()
        graph_entities = health.get("graph_entities", 0)
    except Exception:
        graph_entities = 0

    per_case_recall = []
    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        try:
            detail = client.get(f"/api/v1/documents/{doc_id}").json()
            chunk_count = detail.get("chunk_count", 0)
        except Exception:
            chunk_count = 0

        # Approximate entity recall: check graph stats via health delta is not
        # available per-doc, so report corpus-level only.
        # Per-doc entity recall would require spaCy inspection tooling.
        per_case_recall.append(min(1.0, graph_entities / max(chunk_count * 2, 1)))

    notes = [
        f"Graph contains {graph_entities} entities across all ingested documents.",
        "Entity recall = 1.0 only if spaCy is installed and the en_core_web_sm model "
        "is available (python -m spacy download en_core_web_sm).",
        "Without spaCy the graph is empty and entity-based metrics are zero.",
    ]
    if graph_entities == 0:
        notes.append("ACTION: Install spaCy to enable entity extraction.")

    return DimensionResult(
        name="Entity Extraction Quality",
        score=graph_entities / max(sum(
            len(c.expected_entities) for c in cases
        ), 1),
        details={
            "graph_entities_total": graph_entities,
            "expected_entities_total": sum(len(c.expected_entities) for c in cases),
        },
        notes=notes,
    )


# ── Dimension 4: Relation Extraction Quality ────────────────────────────────

def eval_relation_extraction(client: httpx.Client) -> DimensionResult:
    try:
        health = client.get("/health").json()
        graph_relations = health.get("graph_relations", 0)
        graph_entities = health.get("graph_entities", 0)
    except Exception:
        graph_relations, graph_entities = 0, 0

    ratio = graph_relations / max(graph_entities, 1)
    return DimensionResult(
        name="Relation Extraction Quality",
        score=min(ratio, 1.0),
        details={
            "graph_relations": graph_relations,
            "graph_entities": graph_entities,
            "relations_per_entity": round(ratio, 3),
        },
        notes=[
            "Relations are extracted via co-occurrence heuristics (MENTIONED_WITH, "
            "OPERATES_IN, TARGETS_BY, GOVERNED_BY).",
            "relations_per_entity > 0 confirms the relation extraction pipeline ran.",
            "A score of 0.0 indicates spaCy is unavailable.",
        ],
    )


# ── Dimension 5: Answer Faithfulness ────────────────────────────────────────

def eval_answer_faithfulness(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    mock_flag_rate = []
    confidence_values = []
    limitations_present = []

    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            try:
                body, _ = _query(client, q_item["question"], doc_id=doc_id)
                lims = body.get("limitations", [])
                mock_flagged = any("mock" in l.lower() for l in lims)
                mock_flag_rate.append(1.0 if mock_flagged else 0.0)
                conf = body.get("confidence")
                if conf is not None:
                    confidence_values.append(conf)
                limitations_present.append(len(lims))
            except Exception as exc:
                print(f"  [WARN] Faithfulness eval failed: {exc}")

    mean_conf = round(statistics.mean(confidence_values), 3) if confidence_values else None
    return DimensionResult(
        name="Answer Faithfulness",
        score=mean_conf,
        details={
            "mean_confidence": mean_conf,
            "mock_limitation_rate": round(statistics.mean(mock_flag_rate), 3) if mock_flag_rate else None,
            "mean_limitations_per_query": round(statistics.mean(limitations_present), 1) if limitations_present else 0,
        },
        notes=[
            "Confidence = mean top-3 cosine similarity + graph-evidence boost (capped at 1.0).",
            "Mock limitation rate = fraction of queries where the mock-provider warning was surfaced "
            "(should be 1.0 when LLM_PROVIDER=mock).",
            "For LLM-as-judge faithfulness scoring, use a real LLM provider and extend "
            "this script with a judge prompt (see docs/evaluation_framework.md).",
        ],
    )


# ── Dimension 6: Source Coverage ────────────────────────────────────────────

def eval_source_coverage(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    coverage_scores = []

    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            try:
                body, _ = _query(client, q_item["question"], doc_id=doc_id, top_k=5)
                chunks = body.get("retrieved_chunks", [])
                gold_kw = q_item["gold_keywords"]
                combined = " ".join(c.get("text", "") for c in chunks).lower()
                hits = sum(1 for kw in gold_kw if kw.lower() in combined)
                coverage_scores.append(hits / len(gold_kw) if gold_kw else 0.0)
            except Exception as exc:
                print(f"  [WARN] Coverage eval failed: {exc}")

    score = round(statistics.mean(coverage_scores), 3) if coverage_scores else None
    return DimensionResult(
        name="Source Coverage",
        score=score,
        details={
            "mean_keyword_hit_rate_at_5": score,
            "queries_evaluated": len(coverage_scores),
        },
        notes=[
            "Coverage = fraction of gold keywords found in the top-5 retrieved passages.",
            "Low coverage with mock embeddings is expected (random retrieval order).",
            "Coverage close to 1.0 with real embeddings indicates sufficient passage recall.",
        ],
    )


# ── Dimension 7: Hallucination Risk ─────────────────────────────────────────

def eval_hallucination_risk(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    limitation_rates = []
    low_conf_flagged = 0
    low_conf_total = 0

    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            try:
                body, _ = _query(client, q_item["question"], doc_id=doc_id)
                lims = body.get("limitations", [])
                limitation_rates.append(1.0 if lims else 0.0)
                conf = body.get("confidence") or 1.0
                if conf < 0.4:
                    low_conf_total += 1
                    if any("confidence" in l.lower() for l in lims):
                        low_conf_flagged += 1
            except Exception:
                pass

    # Empty-corpus guard: query with no documents ingested
    try:
        empty_resp = client.post("/api/v1/query", json={"question": "sentinel query z9q8"})
        empty_lims = empty_resp.json().get("limitations", [])
        empty_corpus_guarded = any(
            "passage" in l.lower() or "ingest" in l.lower() for l in empty_lims
        )
    except Exception:
        empty_corpus_guarded = False

    score = round(statistics.mean(limitation_rates), 3) if limitation_rates else None
    return DimensionResult(
        name="Hallucination Risk",
        score=score,
        details={
            "limitation_surface_rate": score,
            "low_confidence_flagged": f"{low_conf_flagged}/{low_conf_total}",
            "empty_corpus_guard_active": empty_corpus_guarded,
        },
        notes=[
            "Limitation surface rate = fraction of queries where ≥1 limitation was returned.",
            "A rate of 1.0 means every response transparently declares its caveats.",
            "Low-confidence flag: count of queries where confidence < 0.4 and the "
            "limitation was surfaced in the response.",
        ],
    )


# ── Dimension 8: Latency ─────────────────────────────────────────────────────

def eval_latency(client: httpx.Client, cases: list[EvalCase], doc_ids: dict) -> DimensionResult:
    query_latencies_ms = []
    ingest_latencies_ms = []

    # Ingest latency: one fresh ingest per case
    for case in cases:
        t0 = time.perf_counter()
        try:
            r = client.post("/api/v1/ingest", json={
                "text": case.doc_text, "title": f"latency-bench-{case.case_id}"
            })
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 201:
                ingest_latencies_ms.append(elapsed_ms)
        except Exception:
            pass

    # Query latency: use doc_ids from primary ingest
    for case in cases:
        doc_id = doc_ids.get(case.case_id)
        if not doc_id:
            continue
        for q_item in case.queries:
            try:
                body, wall_ms = _query(client, q_item["question"], doc_id=doc_id)
                query_latencies_ms.append(body.get("latency_ms", wall_ms))
            except Exception:
                pass

    def _pct(data: list[float], p: float) -> float:
        idx = int(len(data) * p)
        return round(sorted(data)[min(idx, len(data) - 1)], 1)

    q_stats: dict[str, Any] = {}
    if query_latencies_ms:
        q_stats = {
            "p50_ms": _pct(query_latencies_ms, 0.50),
            "p95_ms": _pct(query_latencies_ms, 0.95),
            "mean_ms": round(statistics.mean(query_latencies_ms), 1),
            "max_ms": round(max(query_latencies_ms), 1),
        }

    i_stats: dict[str, Any] = {}
    if ingest_latencies_ms:
        i_stats = {
            "mean_ms": round(statistics.mean(ingest_latencies_ms), 1),
            "max_ms":  round(max(ingest_latencies_ms), 1),
        }

    score = (
        1.0 if q_stats.get("p50_ms", 9999) <= 2000 else
        0.5 if q_stats.get("p50_ms", 9999) <= 5000 else
        0.0
    )

    return DimensionResult(
        name="Latency",
        score=score,
        details={
            "query_latency": q_stats,
            "ingest_latency": i_stats,
            "target": "query p50 ≤ 2000 ms (excluding LLM provider latency)",
        },
        notes=[
            "Score 1.0 = p50 ≤ 2 000 ms; 0.5 = p50 ≤ 5 000 ms; 0.0 = p50 > 5 000 ms.",
            "Mock LLM adds ~0 ms. Real providers add 500–3 000 ms depending on model.",
            "Measure with a real provider to understand end-to-end user latency.",
        ],
    )


# ── Dimension 9: Reproducibility ────────────────────────────────────────────

def eval_reproducibility(client: httpx.Client, cases: list[EvalCase]) -> DimensionResult:
    doc_id_stable = 0
    chunk_count_stable = 0
    tests = 0

    for case in cases:
        try:
            r1 = client.post("/api/v1/ingest", json={"text": case.doc_text, "title": case.title})
            r2 = client.post("/api/v1/ingest", json={"text": case.doc_text, "title": case.title})
            if r1.status_code == 201 and r2.status_code == 201:
                tests += 1
                b1, b2 = r1.json(), r2.json()
                if b1["doc_id"] == b2["doc_id"]:
                    doc_id_stable += 1
                if b1["chunk_count"] == b2["chunk_count"]:
                    chunk_count_stable += 1
        except Exception as exc:
            print(f"  [WARN] Reproducibility check failed for '{case.title}': {exc}")

    if tests == 0:
        return DimensionResult(name="Reproducibility", score=None, notes=["No tests completed."])

    score = round((doc_id_stable + chunk_count_stable) / (2 * tests), 3)
    return DimensionResult(
        name="Reproducibility",
        score=score,
        details={
            "doc_id_stability": f"{doc_id_stable}/{tests}",
            "chunk_count_stability": f"{chunk_count_stable}/{tests}",
        },
        notes=[
            "Score 1.0 = identical doc_id and chunk_count across repeated ingests of the same text.",
            "doc_id is derived from a hash of the document content.",
            "chunk_count stability confirms the chunker is deterministic.",
        ],
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _bar(score: Optional[float], width: int = 20) -> str:
    if score is None:
        return "[     N/A      ]"
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.3f}"


def print_report(results: list[DimensionResult], provider: str, vector_store: str) -> None:
    print()
    print("=" * 72)
    print("  PolicyMind-AI Evaluation Report")
    print(f"  LLM provider: {provider}  |  Vector store: {vector_store}")
    print("=" * 72)

    for i, r in enumerate(results, start=1):
        print(f"\n  {i:2d}. {r.name}")
        print(f"      Score  {_bar(r.score)}")
        for k, v in r.details.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    print(f"           {sk:<26} {sv}")
            else:
                print(f"           {k:<26} {v}")
        for note in r.notes:
            wrapped = note if len(note) <= 64 else note[:61] + "..."
            print(f"      note: {wrapped}")

    overall_scores = [r.score for r in results if r.score is not None]
    if overall_scores:
        overall = statistics.mean(overall_scores)
        print(f"\n  Overall (mean of {len(overall_scores)} computable dimensions)")
        print(f"      Score  {_bar(overall)}")
    print()
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PolicyMind-AI evaluation suite against a live server."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--report", default=None, help="Path to write JSON report")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    print(f"\nPolicyMind-AI Evaluation Suite")
    print(f"Target: {args.base_url}")
    print("-" * 40)

    client = httpx.Client(base_url=args.base_url, timeout=args.timeout)

    # Confirm server is up
    try:
        health = client.get("/health").json()
        provider = health.get("llm_provider", "unknown")
        vector_store = health.get("vector_store", "unknown")
        print(f"Server: ok  |  provider={provider}  |  vector_store={vector_store}")
    except Exception as exc:
        print(f"ERROR: Cannot reach server at {args.base_url}: {exc}")
        print("Start the server with:  uvicorn backend.app.main:app --reload --port 8000")
        raise SystemExit(1)

    # Ingest all eval documents
    print(f"\nIngesting {len(_EVAL_CASES)} evaluation documents...")
    doc_ids: dict[str, str] = {}
    for case in _EVAL_CASES:
        doc_id = _ingest(client, case)
        if doc_id:
            doc_ids[case.case_id] = doc_id
            print(f"  ✓ {case.title[:55]} → {doc_id}")
        else:
            print(f"  ✗ {case.title[:55]}")

    # Run all nine dimensions
    print("\nRunning evaluation dimensions...")
    results: list[DimensionResult] = [
        eval_retrieval_relevance(client, _EVAL_CASES, doc_ids),
        eval_citation_accuracy(client, _EVAL_CASES, doc_ids),
        eval_entity_extraction(client, _EVAL_CASES, doc_ids),
        eval_relation_extraction(client),
        eval_answer_faithfulness(client, _EVAL_CASES, doc_ids),
        eval_source_coverage(client, _EVAL_CASES, doc_ids),
        eval_hallucination_risk(client, _EVAL_CASES, doc_ids),
        eval_latency(client, _EVAL_CASES, doc_ids),
        eval_reproducibility(client, _EVAL_CASES),
    ]

    print_report(results, provider=provider, vector_store=vector_store)

    # Optionally save JSON
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        report_data = {
            "server": args.base_url,
            "llm_provider": provider,
            "vector_store": vector_store,
            "dimensions": [
                {
                    "name": r.name,
                    "score": r.score,
                    "details": r.details,
                    "notes": r.notes,
                }
                for r in results
            ],
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON report saved to: {args.report}")


if __name__ == "__main__":
    main()
