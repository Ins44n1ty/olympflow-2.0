from pathlib import Path
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOP_K = 10
MIN_DF = 1
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\$+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[_{}^]", " ", text)
    text = re.sub(r"\d+\.\d+\.", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def build_texts(records: list[dict]) -> list[str]:
    return [normalize_text(record.get("raw_text", "")) for record in records]


def build_similarity_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix)
    return vectorizer, matrix, sim


def build_neighbors(records: list[dict], sim_matrix, top_k: int) -> list[dict]:
    results = []

    for i, record in enumerate(records):
        scores = sim_matrix[i]
        neighbor_indices = sorted(
            range(len(scores)),
            key=lambda j: scores[j],
            reverse=True,
        )

        neighbors = []

        for j in neighbor_indices:
            if j == i:
                continue

            neighbor = records[j]
            neighbors.append(
                {
                    "rank": len(neighbors) + 1,
                    "task_id": neighbor["task_id"],
                    "task_number": neighbor["task_number"],
                    "group_id": neighbor.get("group_id"),
                    "group_pos": neighbor.get("group_pos"),
                    "similarity": round(float(scores[j]), 6),
                }
            )

            if len(neighbors) >= top_k:
                break

        results.append(
            {
                "task_id": record["task_id"],
                "task_number": record["task_number"],
                "group_id": record.get("group_id"),
                "group_pos": record.get("group_pos"),
                "neighbors": neighbors,
            }
        )

    return results


def evaluate_neighbors(records: list[dict], neighbor_results: list[dict]) -> dict:
    task_id_to_group = {
        record["task_id"]: record.get("group_id")
        for record in records
    }

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    total = 0

    for item in neighbor_results:
        group_id = item.get("group_id")
        if group_id is None:
            continue

        total += 1
        neighbor_groups = [task_id_to_group[n["task_id"]] for n in item["neighbors"]]

        if len(neighbor_groups) >= 1 and group_id in neighbor_groups[:1]:
            top1_hits += 1
        if len(neighbor_groups) >= 3 and group_id in neighbor_groups[:3]:
            top3_hits += 1
        if len(neighbor_groups) >= 5 and group_id in neighbor_groups[:5]:
            top5_hits += 1

    if total == 0:
        return {
            "task_count": 0,
            "top1_group_recall": None,
            "top3_group_recall": None,
            "top5_group_recall": None,
        }

    return {
        "task_count": total,
        "top1_group_recall": round(top1_hits / total, 6),
        "top3_group_recall": round(top3_hits / total, 6),
        "top5_group_recall": round(top5_hits / total, 6),
    }


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(path: Path, neighbor_results: list[dict]) -> None:
    headers = [
        "task_id",
        "task_number",
        "group_id",
        "group_pos",
        "neighbor_rank",
        "neighbor_task_id",
        "neighbor_task_number",
        "neighbor_group_id",
        "neighbor_group_pos",
        "similarity",
    ]

    def esc(value) -> str:
        text = "" if value is None else str(value)
        text = text.replace('"', '""')
        return f'"{text}"'

    lines = [",".join(headers)]

    for item in neighbor_results:
        for neighbor in item["neighbors"]:
            row = [
                item["task_id"],
                item["task_number"],
                item.get("group_id"),
                item.get("group_pos"),
                neighbor["rank"],
                neighbor["task_id"],
                neighbor["task_number"],
                neighbor.get("group_id"),
                neighbor.get("group_pos"),
                neighbor["similarity"],
            ]
            lines.append(",".join(esc(x) for x in row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_examples(path: Path, records: list[dict], neighbor_results: list[dict], limit: int = 30) -> None:
    task_by_id = {record["task_id"]: record for record in records}
    lines = []

    for item in neighbor_results[:limit]:
        task = task_by_id[item["task_id"]]
        lines.append("=" * 100)
        lines.append(
            f"{task['task_id']} | {task['task_number']} | group={task.get('group_id')} | pos={task.get('group_pos')}"
        )
        lines.append(task["raw_text"])
        lines.append("")

        for neighbor in item["neighbors"][:5]:
            neighbor_task = task_by_id[neighbor["task_id"]]
            lines.append(
                f"  -> rank={neighbor['rank']} sim={neighbor['similarity']} "
                f"{neighbor_task['task_id']} | {neighbor_task['task_number']} "
                f"| group={neighbor_task.get('group_id')} | pos={neighbor_task.get('group_pos')}"
            )
            lines.append(f"     {neighbor_task['raw_text'][:300]}")
            lines.append("")

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "dataset_grouped" / "dataset_grouped.jsonl"
    output_dir = project_root / "data" / "features" / "tfidf"
    output_dir.mkdir(parents=True, exist_ok=True)

    neighbors_json_path = output_dir / "neighbors.json"
    neighbors_jsonl_path = output_dir / "neighbors.jsonl"
    neighbors_csv_path = output_dir / "neighbors.csv"
    summary_path = output_dir / "summary.json"
    examples_path = output_dir / "examples.txt"

    records = load_jsonl(input_path)
    texts = build_texts(records)
    vectorizer, matrix, sim_matrix = build_similarity_matrix(texts)

    neighbor_results = build_neighbors(records, sim_matrix, top_k=TOP_K)
    summary = evaluate_neighbors(records, neighbor_results)
    summary["task_count_total"] = len(records)
    summary["vocab_size"] = len(vectorizer.vocabulary_)
    summary["top_k"] = TOP_K
    summary["ngram_range"] = list(NGRAM_RANGE)

    save_json(neighbors_json_path, neighbor_results)
    save_jsonl(neighbors_jsonl_path, neighbor_results)
    save_csv(neighbors_csv_path, neighbor_results)
    save_json(summary_path, summary)
    save_examples(examples_path, records, neighbor_results)

    print(f"Processed tasks: {len(records)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Top-1 group recall: {summary['top1_group_recall']}")
    print(f"Top-3 group recall: {summary['top3_group_recall']}")
    print(f"Top-5 group recall: {summary['top5_group_recall']}")
    print(f"Saved: {neighbors_json_path}")
    print(f"Saved: {neighbors_jsonl_path}")
    print(f"Saved: {neighbors_csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
