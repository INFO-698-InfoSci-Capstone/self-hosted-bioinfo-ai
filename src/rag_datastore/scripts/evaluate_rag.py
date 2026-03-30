#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from typing import Any, Dict, List

from app.rag_service import HybridRetriever, QueryConfig


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_eval(retriever: HybridRetriever, dataset: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    latencies = []
    correct = 0

    for row in dataset:
        start = time.perf_counter()
        result = retriever.query(
            row["question"],
            config=QueryConfig(top_k=top_k, vector_weight=0.6, bm25_weight=0.4),
        )
        latencies.append((time.perf_counter() - start) * 1000)

        answer = result["answer"].lower()
        targets = [t.lower() for t in row.get("expected_contains", [])]
        if all(t in answer for t in targets):
            correct += 1

    accuracy = correct / len(dataset) if dataset else 0.0
    return {
        "top_k": top_k,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 2) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latency vs accuracy for hybrid RAG")
    parser.add_argument("--eval-set", required=True, help="Path to JSONL eval set")
    parser.add_argument("--index-dir", default="src/rag_datastore/faiss_index_store")
    parser.add_argument("--topk-grid", default="3,5,8,12", help="Comma-separated top-k values")
    args = parser.parse_args()

    dataset = load_eval_set(Path(args.eval_set))
    retriever = HybridRetriever(index_dir=args.index_dir)

    print("top_k\taccuracy\tavg_latency_ms\tp95_latency_ms")
    for top_k in [int(v.strip()) for v in args.topk_grid.split(",") if v.strip()]:
        metrics = run_eval(retriever, dataset, top_k)
        print(f"{metrics['top_k']}\t{metrics['accuracy']}\t{metrics['avg_latency_ms']}\t{metrics['p95_latency_ms']}")


if __name__ == "__main__":
    main()
