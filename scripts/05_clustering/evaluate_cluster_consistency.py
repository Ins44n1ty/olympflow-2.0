from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
OUTPUT_DIR = ROOT / "data/clusters/consistency"
REPORT_DIR = ROOT / "reports/clustering"

N_CLUSTERS = 183
MAX_FEATURES = 30000
SVD_COMPONENTS = 128
SEEDS = [0, 1, 2, 3, 4, 7, 11, 21, 42, 100]
SUBSAMPLE_FRACTION = 0.85
BOOTSTRAP_RUNS = 20


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


def build_features(texts: list[str]) -> np.ndarray:
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

    n_components = min(SVD_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
    )

    features = svd.fit_transform(matrix)
    norms = np.linalg.norm(features, axis=1, keepdims=True)

    return features / np.maximum(norms, 1e-12)


def fit_kmeans(features: np.ndarray, seed: int) -> np.ndarray:
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=seed,
        n_init=20,
    )

    return model.fit_predict(features)


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + labels)

        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.6f}" for value in row])


def build_pairwise_matrices(labels_by_seed: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    seeds = list(labels_by_seed)
    ari_matrix = np.zeros((len(seeds), len(seeds)))
    nmi_matrix = np.zeros((len(seeds), len(seeds)))

    for i, seed_i in enumerate(seeds):
        for j, seed_j in enumerate(seeds):
            labels_i = labels_by_seed[seed_i]
            labels_j = labels_by_seed[seed_j]

            ari_matrix[i, j] = adjusted_rand_score(labels_i, labels_j)
            nmi_matrix[i, j] = normalized_mutual_info_score(labels_i, labels_j)

    return ari_matrix, nmi_matrix


def upper_triangle_mean(matrix: np.ndarray) -> float:
    values = []

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            values.append(matrix[i, j])

    return float(np.mean(values))


def run_subsample_stability(features: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(42)
    n = len(features)
    size = int(n * SUBSAMPLE_FRACTION)
    runs = []

    for run_id in range(BOOTSTRAP_RUNS):
        indices = np.sort(rng.choice(n, size=size, replace=False))
        sub_features = features[indices]

        labels_a = fit_kmeans(sub_features, seed=run_id)
        labels_b = fit_kmeans(sub_features, seed=run_id + 1000)

        runs.append(
            {
                "run_id": run_id,
                "sample_size": int(size),
                "ari": adjusted_rand_score(labels_a, labels_b),
                "nmi": normalized_mutual_info_score(labels_a, labels_b),
            }
        )

    return runs


def plot_heatmap(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cluster_sizes(labels: np.ndarray, path: Path) -> None:
    sizes = sorted(Counter(labels).values(), reverse=True)

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sizes)), sizes)
    plt.xlabel("cluster rank")
    plt.ylabel("cluster size")
    plt.title("Cluster size distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(INPUT_PATH)
    texts = [get_text(record) for record in records]
    features = build_features(texts)

    labels_by_seed = {
        seed: fit_kmeans(features, seed)
        for seed in SEEDS
    }

    seed_names = [str(seed) for seed in SEEDS]
    ari_matrix, nmi_matrix = build_pairwise_matrices(labels_by_seed)

    save_matrix_csv(OUTPUT_DIR / "pairwise_ari_matrix.csv", ari_matrix, seed_names)
    save_matrix_csv(OUTPUT_DIR / "pairwise_nmi_matrix.csv", nmi_matrix, seed_names)

    plot_heatmap(
        ari_matrix,
        seed_names,
        REPORT_DIR / "cluster_consistency_ari_heatmap.png",
        "Pairwise ARI between KMeans runs",
    )

    plot_heatmap(
        nmi_matrix,
        seed_names,
        REPORT_DIR / "cluster_consistency_nmi_heatmap.png",
        "Pairwise NMI between KMeans runs",
    )

    reference_labels = labels_by_seed[42]
    plot_cluster_sizes(reference_labels, REPORT_DIR / "cluster_size_distribution.png")

    cluster_sizes = Counter(int(label) for label in reference_labels)
    subsample_runs = run_subsample_stability(features)

    summary = {
        "method": "tfidf_svd_kmeans",
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "records_count": len(records),
        "n_clusters": N_CLUSTERS,
        "seeds": SEEDS,
        "mean_pairwise_ari": upper_triangle_mean(ari_matrix),
        "mean_pairwise_nmi": upper_triangle_mean(nmi_matrix),
        "min_pairwise_ari": float(np.min(ari_matrix[np.triu_indices_from(ari_matrix, k=1)])),
        "min_pairwise_nmi": float(np.min(nmi_matrix[np.triu_indices_from(nmi_matrix, k=1)])),
        "max_pairwise_ari": float(np.max(ari_matrix[np.triu_indices_from(ari_matrix, k=1)])),
        "max_pairwise_nmi": float(np.max(nmi_matrix[np.triu_indices_from(nmi_matrix, k=1)])),
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "bootstrap_runs": BOOTSTRAP_RUNS,
        "mean_subsample_ari": float(np.mean([item["ari"] for item in subsample_runs])),
        "mean_subsample_nmi": float(np.mean([item["nmi"] for item in subsample_runs])),
        "cluster_size_summary_seed_42": {
            "min": int(min(cluster_sizes.values())),
            "max": int(max(cluster_sizes.values())),
            "mean": float(np.mean(list(cluster_sizes.values()))),
            "median": float(np.median(list(cluster_sizes.values()))),
        },
        "subsample_runs": subsample_runs,
    }

    save_json(OUTPUT_DIR / "consistency_summary.json", summary)

    lines = [
        "# Cluster consistency summary",
        "",
        f"Method: `{summary['method']}`",
        f"Records: `{summary['records_count']}`",
        f"Clusters: `{summary['n_clusters']}`",
        "",
        "## Full-data repeated KMeans stability",
        "",
        f"- mean pairwise ARI: `{summary['mean_pairwise_ari']:.6f}`",
        f"- mean pairwise NMI: `{summary['mean_pairwise_nmi']:.6f}`",
        f"- min pairwise ARI: `{summary['min_pairwise_ari']:.6f}`",
        f"- min pairwise NMI: `{summary['min_pairwise_nmi']:.6f}`",
        "",
        "## Subsample stability",
        "",
        f"- subsample fraction: `{summary['subsample_fraction']}`",
        f"- bootstrap runs: `{summary['bootstrap_runs']}`",
        f"- mean subsample ARI: `{summary['mean_subsample_ari']:.6f}`",
        f"- mean subsample NMI: `{summary['mean_subsample_nmi']:.6f}`",
        "",
        "## Cluster size distribution for seed 42",
        "",
        f"- min size: `{summary['cluster_size_summary_seed_42']['min']}`",
        f"- max size: `{summary['cluster_size_summary_seed_42']['max']}`",
        f"- mean size: `{summary['cluster_size_summary_seed_42']['mean']:.6f}`",
        f"- median size: `{summary['cluster_size_summary_seed_42']['median']:.6f}`",
        "",
        "## Interpretation",
        "",
        (
            "This report measures whether the selected TF-IDF + SVD + KMeans clustering "
            "is stable under different random seeds and subsampled runs. High ARI/NMI "
            "means that the clustering structure is not only a one-run artifact."
        ),
        "",
    ]

    (REPORT_DIR / "cluster_consistency_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved consistency metrics to {OUTPUT_DIR}")
    print(f"Saved consistency reports to {REPORT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
