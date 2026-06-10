#!/usr/bin/env python3
"""Smoke test -- verifies the API is reachable and all core flows work end-to-end.

Usage:
    # Start the server first, then run:
    python scripts/smoke_test.py

    # Point at a different host:
    python scripts/smoke_test.py --base-url http://localhost:8000

    # Ingest from a public URL instead of the built-in sample text:
    python scripts/smoke_test.py --doc-url https://example.org/policy.pdf

No API key is required. The server defaults to LLM_PROVIDER=mock so the full
pipeline runs without any external service. Set LLM_PROVIDER=anthropic or
LLM_PROVIDER=openai in .env for real LLM answers.

Exit code 0 on all checks passing, 1 on the first failure.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

# Inline sample -- short enough to produce a single chunk under the default
# chunk_size=512, but rich enough to exercise entity extraction and retrieval.
_SAMPLE_TEXT = (
    "The National Climate Strategy 2030 commits all signatory states to reduce "
    "greenhouse gas emissions by 55 percent relative to 1990 levels by 31 December 2030. "
    "Article 4 mandates that the energy sector achieve a minimum 30 percent renewable "
    "energy share by 2025, rising to 65 percent by 2030. "
    "A Just Transition Fund of EUR 15 billion is established for the period 2025 to 2030, "
    "governed by a Board comprising member state representatives and independent experts. "
    "The European Commission shall publish annual compliance reports by 31 March each year. "
    "Non-compliant states may face suspension of Fund access and preferential trade arrangements."
)

_VALID_ANSWER_TYPES = {"cited", "partial", "refused", "no_corpus"}
_VALID_EVIDENCE_QUALITIES = {"strong", "moderate", "weak", "insufficient"}

_pass_count = 0
_fail_count = 0


def _check(label: str, condition: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"    [PASS] {label}")
    else:
        _fail_count += 1
        suffix = f": {detail}" if detail else ""
        print(f"    [FAIL] {label}{suffix}")
        sys.exit(1)


def _section(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * (len(title) + 2)}")


def run(base_url: str, doc_url: str | None) -> None:
    client = httpx.Client(base_url=base_url, timeout=60.0)

    print(f"\nPolicyMind-AI smoke test")
    print(f"Target : {base_url}")
    print(f"Mode   : {'url=' + doc_url if doc_url else 'inline text (no download)'}")

    # -- Step 1: Health --------------------------------------------------------
    _section("Step 1  Health check")
    r = client.get("/health")
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    body = r.json()
    _check("status = ok", body.get("status") == "ok", repr(body.get("status")))
    _check("llm_provider present", "llm_provider" in body)
    _check("vector_store present", "vector_store" in body)
    print(
        f"    provider={body.get('llm_provider')}  "
        f"vector_store={body.get('vector_store')}  "
        f"graph={body.get('graph_provider')}"
    )

    # -- Step 2: Pre-ingest trust gate -----------------------------------------
    _section("Step 2  Pre-ingest query (refusal / trust gate)")
    docs_r = client.get("/api/v1/documents")
    corpus_empty = docs_r.status_code == 200 and docs_r.json().get("total", 1) == 0

    r = client.post(
        "/api/v1/query",
        json={"question": "What are the main commitments in this policy?"},
    )
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    pre_body = r.json()
    _check(
        "answer_type in valid set",
        pre_body.get("answer_type") in _VALID_ANSWER_TYPES,
        repr(pre_body.get("answer_type")),
    )
    if corpus_empty:
        _check(
            "answer_type=no_corpus (corpus is empty)",
            pre_body.get("answer_type") == "no_corpus",
            repr(pre_body.get("answer_type")),
        )
        _check(
            "no LLM call: citations empty on no_corpus",
            pre_body.get("citations") == [],
        )
    else:
        print("    (corpus already contains documents -- skipping no_corpus assertion)")

    # -- Step 3: Ingest --------------------------------------------------------
    _section("Step 3  Document ingestion")
    payload: dict[str, Any]
    if doc_url:
        print(f"    source=url  {doc_url}")
        payload = {
            "url": doc_url,
            "title": "Smoke Test Document",
            "source_label": "Smoke Test",
        }
    else:
        print("    source=text  (inline sample -- no download required)")
        payload = {
            "text": _SAMPLE_TEXT,
            "title": "National Climate Strategy 2030 (Smoke Test)",
            "source_label": "Smoke Test",
        }
    r = client.post("/api/v1/ingest", json=payload)
    _check("HTTP 201", r.status_code == 201, str(r.status_code))
    ingest = r.json()
    _check("doc_id present", bool(ingest.get("doc_id")))
    _check("chunk_count > 0", ingest.get("chunk_count", 0) > 0, str(ingest.get("chunk_count")))
    _check("entities_extracted >= 0", ingest.get("entities_extracted", -1) >= 0)
    _check(
        "processing_status = completed",
        ingest.get("processing_status") == "completed",
        repr(ingest.get("processing_status")),
    )
    doc_id: str = ingest["doc_id"]
    print(
        f"    doc_id={doc_id}  "
        f"chunks={ingest.get('chunk_count')}  "
        f"entities={ingest.get('entities_extracted')}  "
        f"graph_nodes={ingest.get('graph_nodes_added')}"
    )

    # -- Step 4: List documents ------------------------------------------------
    _section("Step 4  List documents")
    r = client.get("/api/v1/documents")
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    list_body = r.json()
    _check("documents key present", "documents" in list_body)
    _check("total >= 1", list_body.get("total", 0) >= 1, str(list_body.get("total")))

    # -- Step 5: Question answering + trust-layer checks -----------------------
    _section("Step 5  Question answering (trust-layer checks)")
    r = client.post(
        "/api/v1/query",
        json={
            "question": "What emission reduction target is committed for 2030?",
            "include_graph_evidence": True,
        },
    )
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    q = r.json()
    _check("answer non-empty", bool(q.get("answer")))
    _check(
        "answer_type valid",
        q.get("answer_type") in _VALID_ANSWER_TYPES,
        repr(q.get("answer_type")),
    )
    _check(
        "evidence_quality valid",
        q.get("evidence_quality") in _VALID_EVIDENCE_QUALITIES,
        repr(q.get("evidence_quality")),
    )
    _check("confidence_note is string", isinstance(q.get("confidence_note"), str))
    _check("citations is list", isinstance(q.get("citations"), list))
    _check("retrieved_chunks is list", isinstance(q.get("retrieved_chunks"), list))
    _check("graph_evidence is list", isinstance(q.get("graph_evidence"), list))
    _check("limitations is list", isinstance(q.get("limitations"), list))
    _check("latency_ms > 0", q.get("latency_ms", 0) > 0, str(q.get("latency_ms")))
    _check("provider present", bool(q.get("provider")))
    _check("model present", bool(q.get("model")))
    print(
        f"    answer_type={q.get('answer_type')}  "
        f"evidence_quality={q.get('evidence_quality')}  "
        f"confidence={q.get('confidence')}  "
        f"citations={len(q.get('citations', []))}  "
        f"latency={q.get('latency_ms')}ms"
    )

    # -- Step 6: Filtered query (doc_id scope) ---------------------------------
    _section("Step 6  Filtered query (single-document scope)")
    r = client.post(
        "/api/v1/query",
        json={
            "question": "What fund supports the transition to a low-carbon economy?",
            "doc_id": doc_id,
            "top_k": 3,
            "include_graph_evidence": False,
        },
    )
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    fq = r.json()
    _check("answer non-empty", bool(fq.get("answer")))
    _check("graph_evidence is list", isinstance(fq.get("graph_evidence"), list))

    # -- Step 7: Knowledge graph -----------------------------------------------
    _section("Step 7  Knowledge graph")
    r = client.get("/api/v1/graph/stats")
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    stats = r.json()
    _check("total_entities present", "total_entities" in stats)
    _check("total_relations present", "total_relations" in stats)
    n_entities: int = stats.get("total_entities", 0)
    print(
        f"    entities={n_entities}  "
        f"relations={stats.get('total_relations')}  "
        f"graph_enabled={stats.get('graph_enabled')}"
    )

    if n_entities == 0:
        print(
            "    (graph is empty -- spaCy model not installed; "
            "run: python -m spacy download en_core_web_sm)"
        )
    else:
        r = client.get(
            "/api/v1/graph/neighbours",
            params={"entity": "European Commission", "depth": 1},
        )
        _check("neighbours HTTP 200", r.status_code == 200, str(r.status_code))
        nb = r.json()
        _check("neighbours key present", "neighbours" in nb)
        _check("edges key present", "edges" in nb)
        print(
            f"    entity='European Commission'  "
            f"neighbours={len(nb.get('neighbours', []))}  "
            f"edges={len(nb.get('edges', []))}"
        )

    # -- Step 8: Get document by id --------------------------------------------
    _section("Step 8  Document detail")
    r = client.get(f"/api/v1/documents/{doc_id}")
    _check("HTTP 200", r.status_code == 200, str(r.status_code))
    detail = r.json()
    _check("doc_id matches", detail.get("doc_id") == doc_id)

    # -- Step 9: Input validation ----------------------------------------------
    _section("Step 9  Input validation")
    r = client.post("/api/v1/ingest", json={"title": "Missing source"})
    _check("rejects missing source/text (422)", r.status_code == 422, str(r.status_code))

    r = client.post(
        "/api/v1/ingest",
        json={"text": "some text", "url": "https://example.com", "title": "Both"},
    )
    _check("rejects both url and text (422)", r.status_code == 422, str(r.status_code))

    r = client.post("/api/v1/query", json={"question": "Hi"})
    _check("rejects question < 5 chars (422)", r.status_code == 422, str(r.status_code))

    r = client.get("/api/v1/documents/nonexistent-doc-id-xyz")
    _check("unknown doc_id returns 404", r.status_code == 404, str(r.status_code))

    # -- Summary ---------------------------------------------------------------
    total = _pass_count + _fail_count
    print(f"\n  {'-' * 48}")
    print(f"  {_pass_count}/{total} checks passed", end="")
    if _fail_count == 0:
        print("  -- all smoke tests passed.")
    else:
        print(f"  -- {_fail_count} check(s) failed.")
    print(f"  {'-' * 48}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolicyMind-AI end-to-end smoke test (no API key required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/smoke_test.py\n"
            "  python scripts/smoke_test.py --base-url http://localhost:8000\n"
            "  python scripts/smoke_test.py --doc-url https://example.org/policy.pdf\n"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running PolicyMind-AI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--doc-url",
        default=None,
        metavar="URL",
        help=(
            "Optional: ingest a publicly accessible PDF or plain-text URL instead of "
            "the built-in sample text. The URL must not require authentication."
        ),
    )
    args = parser.parse_args()
    run(args.base_url, args.doc_url)


if __name__ == "__main__":
    main()
