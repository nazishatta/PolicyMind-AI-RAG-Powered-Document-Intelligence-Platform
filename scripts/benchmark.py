#!/usr/bin/env python3
"""Retrieval and latency benchmark for PolicyMind-AI.

Reads a JSONL file of labeled queries (or the sample_queries.md file for
latency-only measurement), runs each through the pipeline, and reports:
  - Mean / p50 / p95 query latency
  - Chunk retrieval counts
  - Citation counts per answer

Usage:
    python scripts/benchmark.py --base-url http://localhost:8000 [--n 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import httpx

SAMPLE_QUESTIONS = [
    "What emission reduction targets are committed to by 2030?",
    "Which financial mechanisms support the green transition?",
    "What compliance monitoring obligations are placed on member states?",
    "How are vulnerable communities protected under this policy?",
    "What role does the Just Transition Fund play?",
]

INGEST_TEXT = (
    "The 2030 Climate Strategy commits signatory states to reduce greenhouse gas "
    "emissions by 55 percent relative to 1990 levels. Article 12 mandates a minimum "
    "30 percent renewable energy share by 2025. A Just Transition Fund of EUR 15 billion "
    "will support affected regions. Annual compliance reports must be submitted to the "
    "European Commission by March 31. Vulnerable communities in coal-dependent areas "
    "receive priority access to transition grants."
)


def ingest_sample(client: httpx.Client) -> str:
    r = client.post(
        "/api/v1/ingest",
        json={"text": INGEST_TEXT, "title": "Benchmark Document"},
    )
    r.raise_for_status()
    return r.json()["doc_id"]


def run_queries(
    client: httpx.Client,
    questions: list[str],
    n_repeats: int = 1,
) -> list[dict[str, Any]]:
    results = []
    for q in questions:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            r = client.post("/api/v1/query", json={"question": q})
            wall_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                body = r.json()
                results.append(
                    {
                        "question": q[:60],
                        "api_latency_ms": body.get("latency_ms", 0),
                        "wall_latency_ms": round(wall_ms, 1),
                        "citation_count": len(body.get("citations", [])),
                        "graph_evidence_count": len(body.get("graph_evidence", [])),
                        "provider": body.get("provider"),
                    }
                )
            else:
                results.append({"question": q[:60], "error": r.status_code})
    return results


def report(results: list[dict]) -> None:
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results.")
        return

    latencies = [r["api_latency_ms"] for r in valid]
    wall_lats = [r["wall_latency_ms"] for r in valid]
    citations = [r["citation_count"] for r in valid]

    print("\n" + "=" * 60)
    print("  PolicyMind-AI Benchmark Report")
    print("=" * 60)
    print(f"  Queries run:        {len(results)}")
    print(f"  Successful:         {len(valid)}")
    print(f"  Failed:             {len(results) - len(valid)}")
    print()
    print(f"  API latency (ms)")
    print(f"    mean:  {statistics.mean(latencies):.0f}")
    print(f"    p50:   {statistics.median(latencies):.0f}")
    print(f"    p95:   {sorted(latencies)[int(len(latencies) * 0.95)]:.0f}")
    print(f"    max:   {max(latencies):.0f}")
    print()
    print(f"  Wall latency incl. HTTP (ms)")
    print(f"    mean:  {statistics.mean(wall_lats):.0f}")
    print()
    print(f"  Citations per answer")
    print(f"    mean:  {statistics.mean(citations):.1f}")
    print(f"    max:   {max(citations)}")
    print()
    print("  Per-query breakdown:")
    for r in valid:
        print(
            f"    [{r['api_latency_ms']:5.0f} ms] "
            f"{r['citation_count']} citations | {r['question']}"
        )
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=1, help="Repeat each query N times")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=60.0)

    print("Ingesting benchmark document...")
    doc_id = ingest_sample(client)
    print(f"doc_id: {doc_id}")

    print(f"Running {len(SAMPLE_QUESTIONS)} questions × {args.n} repeats...")
    results = run_queries(client, SAMPLE_QUESTIONS, n_repeats=args.n)
    report(results)


if __name__ == "__main__":
    main()
