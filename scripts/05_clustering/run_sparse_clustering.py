from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
OUTPUT_DIR = ROOT / "data/clusters/tfidf_clustering"
SUMMARY_PATH = ROOT / "data/clusters/sparse_clustering_summary.json"

N_CLUSTERS = 183
RANDOM_STATE = 42
MAX_FEATURES = 30000
SVD_COMPONENTS = 128


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def build_tfidf_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=True,
        token_pattern=None,
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def build_dense_tfidf_projection(matrix):
    n_components = min(SVD_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    projected = svd.fit_transform(matrix)

    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-12)

    return svd, projected


def remap_labels(values: list[str]) -> tuple[list[int], dict[str, int]]:
    mapping = {}

    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)

    return [mapping[value] for value in values], mapping


def get_cluster_size_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)

    return {
        str(label): count
        for label, count in sorted(counts.items())
    }


def evaluate_clustering(
    true_labels: list[int],
    predicted_labels: np.ndarray,
    features: np.ndarray,
) -> dict:
    result = {
        "adjusted_rand_index": adjusted_rand_score(true_labels, predicted_labels),
        "normalized_mutual_info": normalized_mutual_info_score(true_labels, predicted_labels),
        "homogeneity": homogeneity_score(true_labels, predicted_labels),
        "completeness": completeness_score(true_labels, predicted_labels),
        "v_measure": v_measure_score(true_labels, predicted_labels),
        "cluster_size_distribution": get_cluster_size_distribution(predicted_labels),
    }

    if len(set(predicted_labels)) > 1:
        result["silhouette_cosine"] = silhouette_score(
            features,
            predicted_labels,
            metric="cosine",
        )
    else:
        result["silhouette_cosine"] = None

    return result


def build_cluster_records(
    records: list[dict],
    predicted_labels: np.ndarray,
    method_name: str,
) -> list[dict]:
    output = []

    for index, record in enumerate(records):
        item = dict(record)
        item["cluster_method"] = method_name
        item["cluster_id"] = int(predicted_labels[index])
        output.append(item)

    return output


def build_clusters_json(
    records: list[dict],
    predicted_labels: np.ndarray,
    method_name: str,
) -> list[dict]:
    clusters = {}

    for index, label in enumerate(predicted_labels):
        label = int(label)

        if label not in clusters:
            clusters[label] = []

        record = records[index]

        clusters[label].append(
            {
                "index": index,
                "task_id": get_record_id(record, index),
                "task_number": record.get("task_number"),
                "group_id": get_group_id(record),
                "text": get_text(record),
            }
        )

    return [
        {
            "cluster_id": cluster_id,
            "method": method_name,
            "size": len(items),
            "items": items,
        }
        for cluster_id, items in sorted(clusters.items())
    ]


def save_readable_clusters(path: Path, clusters: list[dict], max_text_len: int = 600) -> None:
    lines = []

    for cluster in clusters:
        lines.append(f"Cluster {cluster['cluster_id']}")
        lines.append(f"Size: {cluster['size']}")
        lines.append("")

        for item in cluster["items"]:
            text = " ".join(item["text"].split())

            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."

            lines.append(
                f"- {item['task_id']} | {item.get('task_number')} | "
                f"group={item['group_id']}"
            )
            lines.append(text)
            lines.append("")

        lines.append("=" * 80)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_kmeans(features: np.ndarray) -> np.ndarray:
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    return model.fit_predict(features)


def run_agglomerative(features: np.ndarray) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="cosine",
        linkage="average",
    )

    return model.fit_predict(features)


def save_method_outputs(
    records: list[dict],
    labels: np.ndarray,
    method_name: str,
    metrics: dict,
) -> None:
    method_dir = OUTPUT_DIR / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    cluster_records = build_cluster_records(records, labels, method_name)
    clusters = build_clusters_json(records, labels, method_name)

    save_jsonl(method_dir / "records_with_clusters.jsonl", cluster_records)
    save_json(method_dir / "clusters.json", clusters)
    save_json(method_dir / "summary.json", metrics)
    save_readable_clusters(method_dir / "clusters_readable.txt", clusters)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_PATH)
    texts = [get_text(record) for record in records]
    group_ids = [get_group_id(record) for record in records]
    true_labels, group_mapping = remap_labels(group_ids)

    vectorizer, tfidf_matrix = build_tfidf_matrix(texts)
    svd, tfidf_projection = build_dense_tfidf_projection(tfidf_matrix)

    experiments = {}

    kmeans_labels = run_kmeans(tfidf_projection)
    kmeans_metrics = evaluate_clustering(
        true_labels=true_labels,
        predicted_labels=kmeans_labels,
        features=tfidf_projection,
    )
    kmeans_metrics["method"] = "tfidf_svd_kmeans"
    save_method_outputs(records, kmeans_labels, "tfidf_svd_kmeans", kmeans_metrics)
    experiments["tfidf_svd_kmeans"] = kmeans_metrics

    agglomerative_labels = run_agglomerative(tfidf_projection)
    agglomerative_metrics = evaluate_clustering(
        true_labels=true_labels,
        predicted_labels=agglomerative_labels,
        features=tfidf_projection,
    )
    agglomerative_metrics["method"] = "tfidf_svd_agglomerative"
    save_method_outputs(
        records,
        agglomerative_labels,
        "tfidf_svd_agglomerative",
        agglomerative_metrics,
    )
    experiments["tfidf_svd_agglomerative"] = agglomerative_metrics

    summary = {
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "records_count": len(records),
        "weak_group_count": len(group_mapping),
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "tfidf": {
            "max_features": MAX_FEATURES,
            "matrix_shape": list(tfidf_matrix.shape),
            "vocab_size": len(vectorizer.vocabulary_),
            "min_df": 2,
            "max_df": 0.9,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
        "svd": {
            "n_components": int(svd.n_components),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        },
        "experiments": experiments,
    }

    save_json(SUMMARY_PATH, summary)

    print(f"Saved TF-IDF clustering to {OUTPUT_DIR}")
    print(f"Saved sparse clustering summary to {SUMMARY_PATH}")

    for method, metrics in experiments.items():
        print(method)
        print(json.dumps({
            "adjusted_rand_index": metrics["adjusted_rand_index"],
            "normalized_mutual_info": metrics["normalized_mutual_info"],
            "homogeneity": metrics["homogeneity"],
            "completeness": metrics["completeness"],
            "v_measure": metrics["v_measure"],
            "silhouette_cosine": metrics["silhouette_cosine"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
