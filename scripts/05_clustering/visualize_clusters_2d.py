from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
DENSE_EMBEDDINGS_PATH = ROOT / "data/features/dense/embeddings.npy"

CLUSTER_FILES = {
    "dense_agglomerative": ROOT / "data/clusters/best_dense_agglomerative/records_with_clusters.jsonl",
    "tfidf_svd_kmeans": ROOT / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/records_with_clusters.jsonl",
    "tfidf_svd_agglomerative": ROOT / "data/clusters/tfidf_clustering/tfidf_svd_agglomerative/records_with_clusters.jsonl",
}

OUTPUT_DIR = ROOT / "reports/clustering"

RANDOM_STATE = 42
TSNE_PERPLEXITY = 30
TFIDF_MAX_FEATURES = 30000
SVD_COMPONENTS = 100


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

    return f"record_{index:04d}"


def get_text(record: dict) -> str:
    parts = []

    for key in ("text", "problem_text", "statement", "condition", "title"):
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


def get_cluster_id(record: dict) -> int:
    for key in ("cluster_id", "cluster", "label"):
        if key in record:
            return int(record[key])

    raise KeyError("Cluster id field not found")


def get_section(record: dict) -> int:
    task_number = str(record.get("task_number", ""))

    if "." in task_number:
        first = task_number.split(".", 1)[0]

        if first.isdigit():
            return int(first)

    group_id = str(record.get("group_id", ""))

    if "_" in group_id:
        first = group_id.split("_", 1)[0]

        if first.isdigit():
            return int(first)

    return 0


def load_cluster_labels(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    records = read_jsonl(path)

    ids = []
    labels = []
    sections = []

    for index, record in enumerate(records):
        ids.append(get_record_id(record, index))
        labels.append(get_cluster_id(record))
        sections.append(get_section(record))

    return ids, np.array(labels), np.array(sections)


def build_tfidf_svd_features(records: list[dict]) -> np.ndarray:
    texts = [get_text(record) for record in records]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)
    n_components = min(SVD_COMPONENTS, matrix.shape[1] - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    features = svd.fit_transform(matrix)

    return normalize(features)


def reduce_pca(features: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(features)


def reduce_tsne(features: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(features)


def plot_scatter(
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        s=18,
        alpha=0.75,
        cmap="tab20",
        linewidths=0,
    )
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.colorbar(scatter, label="cluster id")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_section_scatter(
    points: np.ndarray,
    sections: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=sections,
        s=18,
        alpha=0.75,
        cmap="tab10",
        linewidths=0,
    )
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.colorbar(scatter, label="section")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_markdown(generated_files: list[str]) -> None:
    lines = [
        "# 2D cluster visualizations",
        "",
        "This report contains 2D projections of task clusters.",
        "",
        "The plots are exploratory visualizations. PCA gives a simple linear projection, while t-SNE tries to preserve local neighborhoods and is better for seeing small cluster clouds.",
        "",
        "Color means cluster id for cluster plots and section id for section plots.",
        "",
        "## Generated plots",
        "",
    ]

    for file in generated_files:
        lines.append(f"- `reports/clustering/{file}`")

    lines.append("")

    path = OUTPUT_DIR / "cluster_2d_visualizations.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_records = read_jsonl(DATASET_PATH)
    generated_files = []

    tfidf_features = build_tfidf_svd_features(dataset_records)
    dense_features = normalize(np.load(DENSE_EMBEDDINGS_PATH))

    feature_sources = {
        "tfidf_svd_kmeans": tfidf_features,
        "tfidf_svd_agglomerative": tfidf_features,
        "dense_agglomerative": dense_features,
    }

    for method_name, cluster_path in CLUSTER_FILES.items():
        print(f"Visualizing {method_name}")

        _, labels, sections = load_cluster_labels(cluster_path)
        features = feature_sources[method_name]

        pca_points = reduce_pca(features)
        tsne_points = reduce_tsne(features)

        pca_file = f"{method_name}_clusters_pca.png"
        tsne_file = f"{method_name}_clusters_tsne.png"
        section_file = f"{method_name}_sections_tsne.png"

        plot_scatter(
            pca_points,
            labels,
            f"{method_name}: PCA cluster projection",
            OUTPUT_DIR / pca_file,
        )
        plot_scatter(
            tsne_points,
            labels,
            f"{method_name}: t-SNE cluster projection",
            OUTPUT_DIR / tsne_file,
        )
        plot_section_scatter(
            tsne_points,
            sections,
            f"{method_name}: t-SNE section projection",
            OUTPUT_DIR / section_file,
        )

        generated_files.extend([pca_file, tsne_file, section_file])

    save_markdown(generated_files)

    print(f"Saved 2D cluster visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
