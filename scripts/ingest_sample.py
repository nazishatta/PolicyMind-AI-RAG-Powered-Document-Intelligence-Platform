#!/usr/bin/env python3
"""Ingest the bundled sample policy document for local testing and demos.

This script reads ``sample_data/sample_policy.txt``, sends it to the running
PolicyMind-AI API as a text payload, and then runs a set of demonstration
questions against the ingested document.

The sample document is synthetic and for demonstration only.
For real policy analysis, supply publicly accessible document URLs via the
``/api/v1/ingest`` endpoint with the ``url`` field.

Usage:
    # Start the server first:
    uvicorn backend.app.main:app --reload --port 8000

    # Then run this script:
    python scripts/ingest_sample.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

SAMPLE_FILE = Path(__file__).parent.parent / "sample_data" / "sample_policy.txt"

DEMO_QUESTIONS = [
    "What are the main emission reduction targets in this policy?",
    "What is the purpose of the Just Transition Fund?",
    "What compliance reporting obligations are placed on member states?",
    "How are vulnerable communities protected?",
    "When do renewable energy targets take effect?",
]


def ingest(client: httpx.Client, text: str) -> dict:
    r = client.post(
        "/api/v1/ingest",
        json={
            "text": text,
            "title": "Sample Climate Policy (Demo)",
            "source_label": "PolicyMind-AI Demo",
        },
    )
    if r.status_code != 201:
        print(f"[ERROR] Ingestion failed ({r.status_code}): {r.text}")
        sys.exit(1)
    return r.json()


def query(client: httpx.Client, question: str) -> dict:
    r = client.post(
        "/api/v1/query",
        json={"question": question, "include_graph_evidence": True},
    )
    if r.status_code != 200:
        print(f"[WARN] Query failed ({r.status_code}): {r.text}")
        return {}
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample policy and run demo queries")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--questions-only", action="store_true",
                        help="Skip ingestion; assume document already indexed.")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=60.0)

    # ── Ingestion ──────────────────────────────────────────────────────
    if not args.questions_only:
        if not SAMPLE_FILE.exists():
            print(f"[ERROR] Sample file not found: {SAMPLE_FILE}")
            sys.exit(1)

        text = SAMPLE_FILE.read_text(encoding="utf-8")
        print(f"\nIngesting: {SAMPLE_FILE.name} ({len(text):,} chars)")
        result = ingest(client, text)
        print(f"  doc_id:             {result['doc_id']}")
        print(f"  title:              {result['title']}")
        print(f"  chunks produced:    {result['chunk_count']}")
        print(f"  entities extracted: {result['entities_extracted']}")
        print(f"  graph nodes added:  {result['graph_nodes_added']}")

    # ── Demo questions ─────────────────────────────────────────────────
    print(f"\nRunning {len(DEMO_QUESTIONS)} demo questions...\n")
    print("=" * 70)

    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        print(f"\nQ{i}: {question}")
        print("-" * 70)
        result = query(client, question)
        if not result:
            continue

        print(f"Answer:\n{result.get('answer', '')}\n")

        citations = result.get("citations", [])
        if citations:
            print(f"Citations ({len(citations)}):")
            for c in citations[:3]:   # show top 3
                heading = f" [{c.get('section_heading')}]" if c.get("section_heading") else ""
                print(f"  • p.{c['page_number']}{heading} — score {c['relevance_score']:.3f}")
                print(f"    \"{c['excerpt'][:120]}...\"")

        graph = result.get("graph_evidence", [])
        if graph:
            print(f"\nGraph evidence ({len(graph)} triples):")
            for g in graph[:2]:
                print(f"  • {g['entity']} —{g['relation']}→ {g['target']}")

        print(f"\n  [{result.get('latency_ms', 0):.0f} ms | provider: {result.get('provider')}]")
        print("=" * 70)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
