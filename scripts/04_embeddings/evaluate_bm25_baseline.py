from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

NEIGHBORS_PATH = ROOT / "data/features/bm25/neighbors.json"
OUTPUT_DIR = ROOT / "data/metrics/bm25"

K_VALUES = [1, 3, 5, 10]


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_query(item: dict) -> dict:
    query_group_id = item["query_group_id"]
    neighbors = item["neighbors"]

    relevant = [
        neighbor
        for neighbor in neighbors
        if neighbor["group_id"] == query_group_id
    ]

    relevant_count = len(relevant)

    result = {
        "query_index": item["query_index"],
        "query_id": item["query_id"],
        "query_group_id": query_group_id,
    }

    for k in K_VALUES:
        top_k = neighbors[:k]
        hits = [
            neighbor
            for neighbor in top_k
            if neighbor["group_id"] == query_group_id
        ]

        result[f"precision@{k}"] = len(hits) / k
        result[f"recall@{k}"] = len(hits) / relevant_count if relevant_count else 0.0

    reciprocal_rank = 0.0

    for rank, neighbor in enumerate(neighbors, start=1):
        if neighbor["group_id"] == query_group_id:
            reciprocal_rank = 1 / rank
            break

    precisions = []
    hits_count = 0

    for rank, neighbor in enumerate(neighbors, start=1):
        if neighbor["group_id"] == query_group_id:
            hits_count += 1
            precisions.append(hits_count / rank)

    result["mrr"] = reciprocal_rank
    result["map"] = average(precisions)
    result["lrap"] = result["map"]

    return result


def build_summary(per_query_metrics: list[dict]) -> dict:
    summary = {
        "method": "bm25",
        "queries_count": len(per_query_metrics),
    }

    metric_names = []

    for k in K_VALUES:
        metric_names.append(f"precision@{k}")
        metric_names.append(f"recall@{k}")

    metric_names.extend(["mrr", "map", "lrap"])

    for metric_name in metric_names:
        summary[metric_name] = average(
            [item[metric_name] for item in per_query_metrics]
        )

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    neighbors = read_json(NEIGHBORS_PATH)
    per_query_metrics = [evaluate_query(item) for item in neighbors]
    summary = build_summary(per_query_metrics)

    config = {
        "method": "bm25",
        "neighbors_path": str(NEIGHBORS_PATH.relative_to(ROOT)),
        "k_values": K_VALUES,
    }

    save_json(OUTPUT_DIR / "config.json", config)
    save_json(OUTPUT_DIR / "per_query_metrics.json", per_query_metrics)
    save_json(OUTPUT_DIR / "summary.json", summary)

    print(f"Saved BM25 metrics to {OUTPUT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
