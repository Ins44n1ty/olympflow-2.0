from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"

TFIDF_PATH = ROOT / "data/features/tfidf/neighbors.json"
BM25_PATH = ROOT / "data/features/bm25/neighbors.json"
DENSE_EMBEDDINGS_PATH = ROOT / "data/features/dense/embeddings.npy"
DENSE_RECORDS_PATH = ROOT / "data/features/dense/records_with_embedding_index.jsonl"

REPORT_DIR = ROOT / "reports/retrieval"
METRICS_DIR = ROOT / "data/metrics/comparison"

K_VALUES = [1, 3, 5, 10]
TOP_K = 10
CASE_LIMIT = 40


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict]) -> None:
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


def get_text(record: dict) -> str:
    parts = []

    for key in ("title", "problem_text", "text", "statement", "condition"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if parts:
        return "\n".join(parts)

    values = [
        value.strip()
        for value in record.values()
        if isinstance(value, str) and value.strip()
    ]

    return "\n".join(values)


def short_text(text: str, limit: int = 700) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def build_id_to_index(records: list[dict]) -> dict[str, int]:
    return {
        get_record_id(record, index): index
        for index, record in enumerate(records)
    }


def normalize_tfidf(records: list[dict]) -> dict[int, list[dict]]:
    data = read_json(TFIDF_PATH)
    id_to_index = build_id_to_index(records)
    result = {}

    for item in data:
        query_id = item["task_id"]
        query_index = id_to_index[query_id]
        neighbors = []

        for neighbor in item["neighbors"][:TOP_K]:
            neighbor_id = neighbor["task_id"]
            neighbor_index = id_to_index[neighbor_id]

            neighbors.append(
                {
                    "index": neighbor_index,
                    "id": neighbor_id,
                    "group_id": neighbor["group_id"],
                    "score": neighbor["similarity"],
                }
            )

        result[query_index] = neighbors

    return result


def normalize_bm25() -> dict[int, list[dict]]:
    data = read_json(BM25_PATH)
    result = {}

    for item in data:
        result[item["query_index"]] = [
            {
                "index": neighbor["index"],
                "id": neighbor["id"],
                "group_id": neighbor["group_id"],
                "score": neighbor["score"],
            }
            for neighbor in item["neighbors"][:TOP_K]
        ]

    return result


def normalize_dense(records: list[dict]) -> dict[int, list[dict]]:
    embeddings = np.load(DENSE_EMBEDDINGS_PATH)
    embedding_records = read_jsonl(DENSE_RECORDS_PATH)

    id_to_index = build_id_to_index(records)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-12)
    similarities = normalized_embeddings @ normalized_embeddings.T

    result = {}

    for embedding_row, record in enumerate(embedding_records):
        task_id = record["task_id"]
        query_index = id_to_index[task_id]

        scores = similarities[embedding_row].copy()
        scores[embedding_row] = -np.inf

        top_embedding_indices = np.argsort(-scores)[:TOP_K]
        neighbors = []

        for neighbor_embedding_index in top_embedding_indices:
            neighbor_record = embedding_records[int(neighbor_embedding_index)]
            neighbor_id = neighbor_record["task_id"]
            neighbor_index = id_to_index[neighbor_id]

            neighbors.append(
                {
                    "index": neighbor_index,
                    "id": neighbor_id,
                    "group_id": get_group_id(records[neighbor_index]),
                    "score": float(scores[neighbor_embedding_index]),
                }
            )

        result[query_index] = neighbors

    return result


def is_hit(records: list[dict], query_index: int, neighbors: list[dict], k: int) -> bool:
    query_group_id = get_group_id(records[query_index])

    return any(
        neighbor["group_id"] == query_group_id
        for neighbor in neighbors[:k]
    )


def top_indices(neighbors: list[dict], k: int) -> set[int]:
    return {neighbor["index"] for neighbor in neighbors[:k]}


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def build_case(
    records: list[dict],
    query_index: int,
    methods: dict[str, dict[int, list[dict]]],
) -> dict:
    query = records[query_index]

    case = {
        "query_index": query_index,
        "query_id": get_record_id(query, query_index),
        "query_group_id": get_group_id(query),
        "query_text": short_text(get_text(query)),
        "methods": {},
    }

    for method, method_neighbors in methods.items():
        neighbors = method_neighbors.get(query_index, [])
        top_neighbor = neighbors[0] if neighbors else None

        if top_neighbor is None:
            case["methods"][method] = None
            continue

        neighbor_index = top_neighbor["index"]
        neighbor_record = records[neighbor_index]

        case["methods"][method] = {
            "neighbor_index": neighbor_index,
            "neighbor_id": get_record_id(neighbor_record, neighbor_index),
            "neighbor_group_id": get_group_id(neighbor_record),
            "score": top_neighbor["score"],
            "hit@1": get_group_id(query) == get_group_id(neighbor_record),
            "neighbor_text": short_text(get_text(neighbor_record)),
        }

    return case


def save_case_markdown(path: Path, title: str, cases: list[dict]) -> None:
    lines = [f"# {title}", ""]

    for i, case in enumerate(cases[:CASE_LIMIT], start=1):
        lines.append(f"## Case {i}")
        lines.append("")
        lines.append(f"Query index: `{case['query_index']}`")
        lines.append(f"Query id: `{case['query_id']}`")
        lines.append(f"Query group: `{case['query_group_id']}`")
        lines.append("")
        lines.append("Query text:")
        lines.append("")
        lines.append(case["query_text"])
        lines.append("")

        for method, result in case["methods"].items():
            lines.append(f"### {method}")

            if result is None:
                lines.append("No result.")
                lines.append("")
                continue

            lines.append(f"Neighbor index: `{result['neighbor_index']}`")
            lines.append(f"Neighbor id: `{result['neighbor_id']}`")
            lines.append(f"Neighbor group: `{result['neighbor_group_id']}`")
            lines.append(f"Score: `{result['score']}`")
            lines.append(f"Hit@1: `{result['hit@1']}`")
            lines.append("")
            lines.append(result["neighbor_text"])
            lines.append("")

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_overlap_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method_left", "method_right", "k", "mean_jaccard"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(DATASET_PATH)

    methods = {
        "tfidf": normalize_tfidf(records),
        "dense": normalize_dense(records),
        "bm25": normalize_bm25(),
    }

    common_queries = set(range(len(records)))

    for method_neighbors in methods.values():
        common_queries &= set(method_neighbors)

    common_queries = sorted(common_queries)
    method_names = list(methods)

    pairwise_overlap = []

    for i, method_left in enumerate(method_names):
        for method_right in method_names[i + 1:]:
            for k in K_VALUES:
                values = []

                for query_index in common_queries:
                    left = top_indices(methods[method_left][query_index], k)
                    right = top_indices(methods[method_right][query_index], k)
                    values.append(jaccard(left, right))

                pairwise_overlap.append(
                    {
                        "method_left": method_left,
                        "method_right": method_right,
                        "k": k,
                        "mean_jaccard": sum(values) / len(values),
                    }
                )

    hit_summary = {}

    for method, method_neighbors in methods.items():
        hit_summary[method] = {}

        for k in K_VALUES:
            values = [
                is_hit(records, query_index, method_neighbors[query_index], k)
                for query_index in common_queries
            ]

            hit_summary[method][f"hit@{k}"] = sum(values) / len(values)

    bm25_wins_over_dense = []
    dense_wins_over_bm25 = []
    tfidf_wins_over_dense = []
    dense_wins_over_tfidf = []
    bm25_wins_over_tfidf = []
    tfidf_wins_over_bm25 = []
    all_fail = []
    all_hit = []

    for query_index in common_queries:
        hits = {
            method: is_hit(records, query_index, methods[method][query_index], 1)
            for method in methods
        }

        case = build_case(records, query_index, methods)

        if hits["bm25"] and not hits["dense"]:
            bm25_wins_over_dense.append(case)

        if hits["dense"] and not hits["bm25"]:
            dense_wins_over_bm25.append(case)

        if hits["tfidf"] and not hits["dense"]:
            tfidf_wins_over_dense.append(case)

        if hits["dense"] and not hits["tfidf"]:
            dense_wins_over_tfidf.append(case)

        if hits["bm25"] and not hits["tfidf"]:
            bm25_wins_over_tfidf.append(case)

        if hits["tfidf"] and not hits["bm25"]:
            tfidf_wins_over_bm25.append(case)

        if all(not value for value in hits.values()):
            all_fail.append(case)

        if all(hits.values()):
            all_hit.append(case)

    case_counts = {
        "bm25_wins_over_dense": len(bm25_wins_over_dense),
        "dense_wins_over_bm25": len(dense_wins_over_bm25),
        "tfidf_wins_over_dense": len(tfidf_wins_over_dense),
        "dense_wins_over_tfidf": len(dense_wins_over_tfidf),
        "bm25_wins_over_tfidf": len(bm25_wins_over_tfidf),
        "tfidf_wins_over_bm25": len(tfidf_wins_over_bm25),
        "all_fail": len(all_fail),
        "all_hit": len(all_hit),
    }

    summary = {
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "methods": method_names,
        "common_queries_count": len(common_queries),
        "hit_summary": hit_summary,
        "pairwise_topk_jaccard": pairwise_overlap,
        "case_counts": case_counts,
    }

    write_json(METRICS_DIR / "retrieval_comparison_summary.json", summary)
    save_overlap_csv(METRICS_DIR / "retrieval_topk_overlap.csv", pairwise_overlap)

    write_jsonl(REPORT_DIR / "error_cases_bm25_wins_over_dense.jsonl", bm25_wins_over_dense)
    write_jsonl(REPORT_DIR / "error_cases_dense_wins_over_bm25.jsonl", dense_wins_over_bm25)
    write_jsonl(REPORT_DIR / "error_cases_tfidf_wins_over_dense.jsonl", tfidf_wins_over_dense)
    write_jsonl(REPORT_DIR / "error_cases_dense_wins_over_tfidf.jsonl", dense_wins_over_tfidf)
    write_jsonl(REPORT_DIR / "error_cases_bm25_wins_over_tfidf.jsonl", bm25_wins_over_tfidf)
    write_jsonl(REPORT_DIR / "error_cases_tfidf_wins_over_bm25.jsonl", tfidf_wins_over_bm25)
    write_jsonl(REPORT_DIR / "error_cases_all_fail.jsonl", all_fail)
    write_jsonl(REPORT_DIR / "error_cases_all_hit.jsonl", all_hit)

    save_case_markdown(
        REPORT_DIR / "error_analysis_bm25_wins_over_dense.md",
        "BM25 wins over dense",
        bm25_wins_over_dense,
    )
    save_case_markdown(
        REPORT_DIR / "error_analysis_dense_wins_over_bm25.md",
        "Dense wins over BM25",
        dense_wins_over_bm25,
    )
    save_case_markdown(
        REPORT_DIR / "error_analysis_tfidf_wins_over_dense.md",
        "TF-IDF wins over dense",
        tfidf_wins_over_dense,
    )
    save_case_markdown(
        REPORT_DIR / "error_analysis_bm25_wins_over_tfidf.md",
        "BM25 wins over TF-IDF",
        bm25_wins_over_tfidf,
    )
    save_case_markdown(
        REPORT_DIR / "error_analysis_all_fail.md",
        "All methods fail",
        all_fail,
    )

    lines = [
        "# Retrieval comparison",
        "",
        f"Common queries: {len(common_queries)}",
        "",
        "## Hit@k",
        "",
    ]

    for method, values in hit_summary.items():
        lines.append(f"### {method}")
        lines.append("")

        for key, value in values.items():
            lines.append(f"- {key}: {value:.6f}")

        lines.append("")

    lines.extend(["## Case counts", ""])

    for key, value in case_counts.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Mean top-k Jaccard overlap", ""])

    for row in pairwise_overlap:
        lines.append(
            f"- {row['method_left']} vs {row['method_right']}, "
            f"k={row['k']}: {row['mean_jaccard']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "BM25 and TF-IDF are lexical sparse methods. If they outperform dense embeddings "
                "on the current weak-label protocol, this suggests that the labels are strongly "
                "aligned with local lexical similarity: neighboring tasks inside the same weak group "
                "often share physical terms, notation, objects, and formulation style. Dense vectors "
                "may retrieve semantically plausible tasks from other weak groups, but such neighbors "
                "are penalized by the current evaluation setup."
            ),
            "",
        ]
    )

    (REPORT_DIR / "retrieval_comparison_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved comparison metrics to {METRICS_DIR}")
    print(f"Saved comparison reports to {REPORT_DIR}")
    print(json.dumps(case_counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
