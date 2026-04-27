from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]

CLUSTER_RECORDS_PATH = ROOT / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/records_with_clusters.jsonl"
OUTPUT_JSON_PATH = ROOT / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/cluster_interpretation.json"
OUTPUT_MD_PATH = ROOT / "reports/clustering/cluster_interpretation.md"

TOP_TERMS = 12
REPRESENTATIVE_TASKS = 5
MAX_TEXT_LEN = 700


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


def get_cluster_id(record: dict) -> int:
    return int(record["cluster_id"])


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


def short_text(text: str, limit: int = MAX_TEXT_LEN) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def build_vectorizer(texts: list[str]):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=True,
        token_pattern=None,
        max_features=30000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def group_records_by_cluster(records: list[dict]) -> dict[int, list[int]]:
    clusters = defaultdict(list)

    for index, record in enumerate(records):
        clusters[get_cluster_id(record)].append(index)

    return dict(clusters)


def get_top_terms_for_cluster(
    vectorizer: TfidfVectorizer,
    matrix,
    indices: list[int],
) -> list[dict]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_matrix = matrix[indices]
    mean_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()

    if not np.any(mean_scores):
        return []

    top_indices = np.argsort(-mean_scores)[:TOP_TERMS]

    return [
        {
            "term": str(feature_names[index]),
            "score": float(mean_scores[index]),
        }
        for index in top_indices
        if mean_scores[index] > 0
    ]


def get_representative_tasks(
    records: list[dict],
    matrix,
    indices: list[int],
) -> list[dict]:
    cluster_matrix = matrix[indices]
    centroid = np.asarray(cluster_matrix.mean(axis=0)).ravel()
    centroid_norm = np.linalg.norm(centroid)

    if centroid_norm == 0:
        selected = indices[:REPRESENTATIVE_TASKS]
        return [
            build_task_item(records[index], index, None)
            for index in selected
        ]

    scores = []

    for local_pos, index in enumerate(indices):
        row = cluster_matrix[local_pos]
        similarity = float((row @ centroid / centroid_norm).item())
        scores.append((index, similarity))

    scores.sort(key=lambda item: item[1], reverse=True)

    return [
        build_task_item(records[index], index, score)
        for index, score in scores[:REPRESENTATIVE_TASKS]
    ]


def build_task_item(record: dict, index: int, score: float | None) -> dict:
    return {
        "index": index,
        "task_id": get_record_id(record, index),
        "task_number": record.get("task_number"),
        "group_id": get_group_id(record),
        "representative_score": score,
        "text": short_text(get_text(record)),
    }


def get_group_distribution(records: list[dict], indices: list[int]) -> list[dict]:
    counter = Counter(get_group_id(records[index]) for index in indices)

    return [
        {
            "group_id": group_id,
            "count": count,
        }
        for group_id, count in counter.most_common()
    ]


def guess_topic(top_terms: list[dict]) -> str:
    terms = {item["term"] for item in top_terms}
    joined_terms = " ".join(terms)

    topic_rules = [
        ("электростатика", ["заряд", "электрическ", "потенциал", "конденсатор", "пластин"]),
        ("электрические цепи", ["ток", "сопротивление", "резистор", "цепи", "цепь", "напряжение", "эдс", "диод"]),
        ("магнетизм и электромагнитная индукция", ["магнит", "индукц", "проводник", "катушк", "ленц"]),
        ("механика: движение и силы", ["скорость", "ускорение", "трение", "сила", "плоскост", "стол"]),
        ("механика: импульс и столкновения", ["удар", "сталкивается", "пластилин", "импульс", "внутренняя"]),
        ("колебания и пружины", ["пружин", "жесткость", "колеб", "амплитуд", "период"]),
        ("гидростатика", ["давление", "жидкост", "вода", "трубк", "ртуть"]),
        ("термодинамика", ["газ", "моль", "температур", "объем", "поршень", "теплот"]),
        ("геометрическая оптика", ["линз", "фокус", "экран", "изображение", "луч", "зеркал"]),
        ("орбитальное движение и гравитация", ["земли", "луны", "марса", "спутник", "орбите", "радиус", "круговой"]),
    ]

    best_topic = "неоднозначная тема"
    best_score = 0

    for topic, markers in topic_rules:
        score = sum(1 for marker in markers if marker in joined_terms)

        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


def build_interpretation(records: list[dict]) -> list[dict]:
    texts = [get_text(record) for record in records]
    vectorizer, matrix = build_vectorizer(texts)
    clusters = group_records_by_cluster(records)

    interpretations = []

    for cluster_id, indices in sorted(clusters.items()):
        top_terms = get_top_terms_for_cluster(vectorizer, matrix, indices)
        representative_tasks = get_representative_tasks(records, matrix, indices)
        group_distribution = get_group_distribution(records, indices)

        interpretations.append(
            {
                "cluster_id": cluster_id,
                "size": len(indices),
                "guessed_topic": guess_topic(top_terms),
                "top_terms": top_terms,
                "group_distribution": group_distribution,
                "representative_tasks": representative_tasks,
            }
        )

    return interpretations


def save_markdown(path: Path, interpretations: list[dict]) -> None:
    lines = [
        "# Cluster interpretation",
        "",
        "This report gives automatic cluster descriptions for `tfidf_svd_kmeans`.",
        "",
        "For each cluster, it shows top TF-IDF terms, weak-group distribution and representative tasks closest to the cluster centroid.",
        "",
    ]

    for item in interpretations:
        terms = ", ".join(term["term"] for term in item["top_terms"])
        groups = ", ".join(
            f"{entry['group_id']} ({entry['count']})"
            for entry in item["group_distribution"][:8]
        )

        lines.append(f"## Cluster {item['cluster_id']}")
        lines.append("")
        lines.append(f"- size: `{item['size']}`")
        lines.append(f"- guessed topic: `{item['guessed_topic']}`")
        lines.append(f"- top terms: {terms}")
        lines.append(f"- weak groups: {groups}")
        lines.append("")
        lines.append("Representative tasks:")
        lines.append("")

        for task in item["representative_tasks"]:
            score = task["representative_score"]

            if score is None:
                score_text = "NA"
            else:
                score_text = f"{score:.6f}"

            lines.append(
                f"- `{task['task_id']}` | task `{task['task_number']}` | "
                f"group `{task['group_id']}` | score `{score_text}`"
            )
            lines.append("")
            lines.append(task["text"])
            lines.append("")

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = read_jsonl(CLUSTER_RECORDS_PATH)
    interpretations = build_interpretation(records)

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_json(OUTPUT_JSON_PATH, interpretations)
    save_markdown(OUTPUT_MD_PATH, interpretations)

    print(f"Saved cluster interpretation to {OUTPUT_JSON_PATH}")
    print(f"Saved readable report to {OUTPUT_MD_PATH}")


if __name__ == "__main__":
    main()
