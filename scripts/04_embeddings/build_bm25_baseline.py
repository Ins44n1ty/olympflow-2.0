from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
OUTPUT_DIR = ROOT / "data/features/bm25"

TOP_K = 10
K1 = 1.5
B = 0.75


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


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

    string_values = [
        value.strip()
        for value in record.values()
        if isinstance(value, str) and value.strip()
    ]

    return "\n".join(string_values)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def build_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    n_docs = len(tokenized_docs)
    document_frequencies = Counter()

    for tokens in tokenized_docs:
        document_frequencies.update(set(tokens))

    return {
        token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in document_frequencies.items()
    }


def bm25_score(
    query_tokens: list[str],
    doc_tf: Counter,
    doc_len: int,
    avg_doc_len: float,
    idf: dict[str, float],
) -> float:
    score = 0.0

    for token in set(query_tokens):
        freq = doc_tf.get(token, 0)

        if freq == 0:
            continue

        numerator = freq * (K1 + 1)
        denominator = freq + K1 * (1 - B + B * doc_len / avg_doc_len)
        score += idf.get(token, 0.0) * numerator / denominator

    return score


def build_neighbors(records: list[dict]) -> list[dict]:
    texts = [get_text(record) for record in records]
    tokenized_docs = [tokenize(text) for text in texts]

    doc_term_frequencies = [Counter(tokens) for tokens in tokenized_docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)
    idf = build_idf(tokenized_docs)

    neighbors = []

    for i, query_tokens in enumerate(tokenized_docs):
        scores = []

        for j, doc_tf in enumerate(doc_term_frequencies):
            if i == j:
                continue

            score = bm25_score(
                query_tokens=query_tokens,
                doc_tf=doc_tf,
                doc_len=doc_lengths[j],
                avg_doc_len=avg_doc_len,
                idf=idf,
            )

            scores.append((j, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        top_scores = scores[:TOP_K]

        query_record = records[i]

        neighbors.append(
            {
                "query_index": i,
                "query_id": get_record_id(query_record, i),
                "query_group_id": get_group_id(query_record),
                "neighbors": [
                    {
                        "index": j,
                        "id": get_record_id(records[j], j),
                        "group_id": get_group_id(records[j]),
                        "score": score,
                    }
                    for j, score in top_scores
                ],
            }
        )

    return neighbors


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(path: Path, neighbors: list[dict]) -> None:
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
            for rank, neighbor in enumerate(item["neighbors"], start=1):
                writer.writerow(
                    {
                        "query_index": item["query_index"],
                        "query_id": item["query_id"],
                        "query_group_id": item["query_group_id"],
                        "rank": rank,
                        "neighbor_index": neighbor["index"],
                        "neighbor_id": neighbor["id"],
                        "neighbor_group_id": neighbor["group_id"],
                        "score": neighbor["score"],
                        "same_group": item["query_group_id"] == neighbor["group_id"],
                    }
                )


def save_examples(path: Path, records: list[dict], neighbors: list[dict], limit: int = 20) -> None:
    lines = []

    for item in neighbors[:limit]:
        query_text = get_text(records[item["query_index"]]).replace("\n", " ")[:500]
        lines.append(f"Query: {item['query_id']}")
        lines.append(f"Group: {item['query_group_id']}")
        lines.append(query_text)
        lines.append("Neighbors:")

        for rank, neighbor in enumerate(item["neighbors"][:5], start=1):
            neighbor_text = get_text(records[neighbor["index"]]).replace("\n", " ")[:300]
            same_group = item["query_group_id"] == neighbor["group_id"]
            lines.append(
                f"{rank}. id={neighbor['id']} group={neighbor['group_id']} "
                f"score={neighbor['score']:.6f} same_group={same_group}"
            )
            lines.append(neighbor_text)

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_PATH)
    neighbors = build_neighbors(records)

    summary = {
        "method": "bm25",
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "records_count": len(records),
        "top_k": TOP_K,
        "k1": K1,
        "b": B,
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
    }

    save_json(OUTPUT_DIR / "neighbors.json", neighbors)
    save_jsonl(OUTPUT_DIR / "neighbors.jsonl", neighbors)
    save_csv(OUTPUT_DIR / "neighbors.csv", neighbors)
    save_json(OUTPUT_DIR / "summary.json", summary)
    save_examples(OUTPUT_DIR / "examples.txt", records, neighbors)

    print(f"Saved BM25 baseline to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
