"""
MMR on/off comparison.

Runs retrieval-only evaluation against the QA set with and without MMR
reranking, then prints a before/after context_recall table.

Retrieval-only isolates the retrieval contribution: we measure how many
of the expected chunks appear in the top_k, independent of what the
generator later does with them. This is deliberate — mixing retrieval
and generation metrics together hides which stage is weak.

In production I would extend this to compare:
- top-k values (3 / 5 / 10)
- MMR lambda (0.5 / 0.7 / 0.9)
- dense-only vs hybrid (BM25 + dense, reciprocal-rank fused)
- chunk sizes (400 / 800 / 1200)

Each comparison would gate on a held-out set so we do not overfit the
golden questions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import EVAL_PATH
from src.retrieve import retrieve


def context_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    hits = sum(1 for e in expected_ids if e in retrieved_ids)
    return hits / len(expected_ids)


def run(eval_path: Path) -> None:
    eval_data = json.loads(eval_path.read_text())

    rows: list[dict] = []
    for item in eval_data:
        q = item["question"]
        expected = item.get("expected_chunks", [])

        baseline = [r.chunk.chunk_id for r in retrieve(q, use_mmr=False)]
        with_mmr = [r.chunk.chunk_id for r in retrieve(q, use_mmr=True)]

        rows.append({
            "id": item.get("id", q[:30]),
            "category": item.get("category", "?"),
            "baseline_recall": context_recall(baseline, expected),
            "mmr_recall": context_recall(with_mmr, expected),
            "baseline_top": baseline[:3],
            "mmr_top": with_mmr[:3],
        })

    print(f"\n{'id':6} {'category':12} {'baseline':>10} {'with MMR':>10} {'delta':>8}")
    print("-" * 50)
    for row in rows:
        delta = row["mmr_recall"] - row["baseline_recall"]
        print(
            f"{row['id']:6} {row['category']:12} "
            f"{row['baseline_recall']:>10.2f} {row['mmr_recall']:>10.2f} "
            f"{delta:>+8.2f}"
        )

    n = len(rows)
    avg_base = sum(r["baseline_recall"] for r in rows) / n
    avg_mmr = sum(r["mmr_recall"] for r in rows) / n
    print("-" * 50)
    print(
        f"{'AVG':6} {'':12} {avg_base:>10.2f} {avg_mmr:>10.2f} "
        f"{avg_mmr - avg_base:>+8.2f}"
    )

    # Diversity sanity check: count unique chunks across the top 3 per query
    baseline_unique = sum(len(set(r["baseline_top"])) for r in rows)
    mmr_unique = sum(len(set(r["mmr_top"])) for r in rows)
    print(f"\nUnique chunks across all top-3 (proxy for diversity):")
    print(f"  baseline: {baseline_unique}, with MMR: {mmr_unique}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_PATH
    run(path)
