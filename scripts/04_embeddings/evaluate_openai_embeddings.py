from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
EMBEDDINGS_PATH = ROOT / "data/features/openai_dense/embeddings.npy"
EMBEDDING_RECORDS_PATH = ROOT / "data/features/openai_dense/records_with_embedding_index.jsonl"

FEATURES_OUTPUT_DIR = ROOT / "data/features/openai_dense"
METRICS_OUTPUT_DIR = ROOT / "data/metrics/openai_dense"

TOP_K = 10
K_VALUES = [1, 3, 5, 10]


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_record_id(record: dict, index: int) -> str:
    for key in ("task_id", "id", "problem_id", "record_id"):
        if key in record:
            return str(record[key])

    return str(index)


def get_group_id(record: dict) -> str:
    for key in ("group_id", "weak_group_id", "group", "label"):
        if key in record:
            return str(record[key])

    return ""


def build_id_to_index(records: list[dict]) -> dict[str, int]:
    return {
        get_record_id(record, index): index
        for index, record in enumerate(records)
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings / np.maximum(norms, 1e-12)


def build_neighbors(records: list[dict]) -> list[dict]:
    embeddings = normalize_embeddings(np.load(EMBEDDINGS_PATH))
    embedding_records = read_jsonl(EMBEDDING_RECORDS_PATH)
    id_to_index = build_id_to_index(records)

    similarities = embeddings @ embeddings.T
    neighbors_output = []

    for embedding_index, embedding_record in enumerate(embedding_records):
        task_id = embedding_record["task_id"]
        query_index = id_to_index[task_id]
        query_record = records[query_index]

        scores = similarities[embedding_index].copy()
        scores[embedding_index] = -np.inf
        top_indices = np.argsort(-scores)[:TOP_K]

        neighbors = []

        for rank, neighbor_embedding_index in enumerate(top_indices, start=1):
            neighbor_embedding_index = int(neighbor_embedding_index)
            neighbor_record_meta = embedding_records[neighbor_embedding_index]
            neighbor_id = neighbor_record_meta["task_id"]
            neighbor_index = id_to_index[neighbor_id]
            neighbor_record = records[neighbor_index]

            neighbors.append(
                {
                    "rank": rank,
                    "index": neighbor_index,
                    "id": neighbor_id,
                    "task_number": neighbor_record.get("task_number"),
                    "group_id": get_group_id(neighbor_record),
                    "score": float(scores[neighbor_embedding_index]),
                }
            )

        neighbors_output.append(
            {
                "query_index": query_index,
                "query_id": task_id,
                "query_task_number": query_record.get("task_number"),
                "query_group_id": get_group_id(query_record),
                "neighbors": neighbors,
            }
        )

    neighbors_output.sort(key=lambda item: item["query_index"])

    return neighbors_output


def average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_query(item: dict, group_sizes: dict[str, int]) -> dict:
    query_group_id = item["query_group_id"]
    neighbors = item["neighbors"]
    relevant_count = max(group_sizes[query_group_id] - 1, 0)

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

    hits_count = 0
    precisions = []

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
        "method": "openai_dense",
        "queries_count": len(per_query_metrics),
    }

    for k in K_VALUES:
        summary[f"precision@{k}"] = average(
            [item[f"precision@{k}"] for item in per_query_metrics]
        )
        summary[f"recall@{k}"] = average(
            [item[f"recall@{k}"] for item in per_query_metrics]
        )

    for metric in ("mrr", "map", "lrap"):
        summary[metric] = average([item[metric] for item in per_query_metrics])

    return summary


def save_neighbors_csv(path: Path, neighbors: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_index",
                "query_id",
                "query_group_id",
                "rank",
                "neighbor_index",
                "neighbor_id",
                "neighbor_group_id",
                "score",
                "same_group",
            ],
        )

        writer.writeheader()

        for item in neighbors:
            for neighbor in item["neighbors"]:
                writer.writerow(
                    {
                        "query_index": item["query_index"],
                        "query_id": item["query_id"],
                        "query_group_id": item["query_group_id"],
                        "rank": neighbor["rank"],
                        "neighbor_index": neighbor["index"],
                        "neighbor_id": neighbor["id"],
                        "neighbor_group_id": neighbor["group_id"],
                        "score": neighbor["score"],
                        "same_group": item["query_group_id"] == neighbor["group_id"],
                    }
                )


def main() -> None:
    FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(DATASET_PATH)
    group_sizes = {}

    for record in records:
        group_id = get_group_id(record)
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1

    neighbors = build_neighbors(records)
    per_query_metrics = [
        evaluate_query(item, group_sizes)
        for item in neighbors
    ]
    summary = build_summary(per_query_metrics)

    config = {
        "method": "openai_dense",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "embeddings_path": str(EMBEDDINGS_PATH.relative_to(ROOT)),
        "embedding_records_path": str(EMBEDDING_RECORDS_PATH.relative_to(ROOT)),
        "top_k": TOP_K,
        "k_values": K_VALUES,
    }

    save_json(FEATURES_OUTPUT_DIR / "neighbors.json", neighbors)
    save_jsonl(FEATURES_OUTPUT_DIR / "neighbors.jsonl", neighbors)
    save_neighbors_csv(FEATURES_OUTPUT_DIR / "neighbors.csv", neighbors)

    save_json(METRICS_OUTPUT_DIR / "config.json", config)
    save_json(METRICS_OUTPUT_DIR / "per_query_metrics.json", per_query_metrics)
    save_json(METRICS_OUTPUT_DIR / "summary.json", summary)

    print(f"Saved OpenAI dense neighbors to {FEATURES_OUTPUT_DIR}")
    print(f"Saved OpenAI dense metrics to {METRICS_OUTPUT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
