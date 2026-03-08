"""
Retrieval evaluation: Recall@K, MRR, nDCG@K.

Input: a JSONL file where each line is:
  {"query_issue_id": "<uuid>", "relevant_issue_ids": ["<uuid>", ...]}

Usage:
  python eval/retrieval_metrics.py --eval-file eval/relevance_set.jsonl --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_MAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    score = 0.0
    for i, rid in enumerate(retrieved[:k]):
        if rid in relevant:
            score += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1
    return score


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    actual = dcg_at_k(retrieved, relevant, k)
    ideal_retrieved = list(relevant)[:k]
    ideal = dcg_at_k(ideal_retrieved, relevant, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate(eval_entries: list[dict], k_values: list[int]) -> dict:
    """
    Run retrieval pipeline for each query and compute metrics.
    Returns {"recall@K": ..., "mrr": ..., "ndcg@K": ...} averaged over queries.
    """
    from app.core.db import get_conn
    from app.retrieval.hybrid import hybrid_search

    totals: dict[str, float] = {}
    count = 0

    for entry in eval_entries:
        query_id = entry["query_issue_id"]
        relevant = set(entry["relevant_issue_ids"])

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT text_full FROM issues WHERE id = %s::uuid", (query_id,))
                row = cur.fetchone()
        if row is None or not row[0]:
            continue

        hits = hybrid_search(row[0], final_n=max(k_values) * 2)
        retrieved = [str(h.issue_id) for h in hits]
        count += 1

        for k in k_values:
            key_r = f"recall@{k}"
            key_n = f"ndcg@{k}"
            totals[key_r] = totals.get(key_r, 0.0) + recall_at_k(retrieved, relevant, k)
            totals[key_n] = totals.get(key_n, 0.0) + ndcg_at_k(retrieved, relevant, k)
        totals["mrr"] = totals.get("mrr", 0.0) + mrr(retrieved, relevant)

    if count == 0:
        return {"error": "no valid queries found"}

    return {key: round(val / count, 4) for key, val in totals.items()} | {"num_queries": count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--eval-file", required=True, help="JSONL file with query_issue_id and relevant_issue_ids.")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20], help="K values for Recall@K, nDCG@K.")
    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        print(f"error: file not found: {args.eval_file}", file=sys.stderr)
        return 1

    entries = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("error: no entries in eval file", file=sys.stderr)
        return 1

    results = evaluate(entries, args.k)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
