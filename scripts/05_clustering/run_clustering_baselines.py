from pathlib import Path
import json
import re

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 20


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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


def build_tfidf_features(records: list[dict]) -> tuple[np.ndarray, dict]:
    texts = [normalize_text(record.get("raw_text", "")) for record in records]
    vectorizer = TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray().astype(np.float32), {
        "vocab_size": len(vectorizer.vocabulary_),
        "min_df": TFIDF_MIN_DF,
        "max_df": TFIDF_MAX_DF,
        "ngram_range": list(TFIDF_NGRAM_RANGE),
    }


def build_dense_features(project_root: Path) -> tuple[np.ndarray, dict]:
    path = project_root / "data" / "features" / "dense" / "embeddings.npy"
    matrix = np.load(path).astype(np.float32)
    return matrix, {
        "embedding_dim": int(matrix.shape[1]),
        "source": str(path),
    }


def build_label_ids(records: list[dict]) -> tuple[list[int], dict]:
    group_ids = [record["group_id"] for record in records]
    unique_groups = sorted(set(group_ids))
    group_to_int = {group_id: idx for idx, group_id in enumerate(unique_groups)}
    labels = [group_to_int[group_id] for group_id in group_ids]
    return labels, group_to_int


def evaluate_clustering(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "ari": round(float(adjusted_rand_score(y_true, y_pred)), 6),
        "nmi": round(float(normalized_mutual_info_score(y_true, y_pred)), 6),
        "homogeneity": round(float(homogeneity_score(y_true, y_pred)), 6),
        "completeness": round(float(completeness_score(y_true, y_pred)), 6),
        "v_measure": round(float(v_measure_score(y_true, y_pred)), 6),
    }


def run_kmeans(x: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    return model.fit_predict(x)


def run_agglomerative(x: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(x)


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    dataset_path = project_root / "data" / "dataset_grouped" / "dataset_grouped.jsonl"
    output_dir = project_root / "data" / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(dataset_path)
    y_true, group_to_int = build_label_ids(records)
    n_clusters = len(group_to_int)

    tfidf_x, tfidf_meta = build_tfidf_features(records)
    dense_x, dense_meta = build_dense_features(project_root)

    experiments = []

    tfidf_kmeans = run_kmeans(tfidf_x, n_clusters)
    experiments.append(
        {
            "representation": "tfidf",
            "method": "kmeans",
            "n_clusters": n_clusters,
            "metrics": evaluate_clustering(y_true, tfidf_kmeans.tolist()),
            "meta": tfidf_meta,
        }
    )

    tfidf_agg = run_agglomerative(tfidf_x, n_clusters)
    experiments.append(
        {
            "representation": "tfidf",
            "method": "agglomerative",
            "n_clusters": n_clusters,
            "metrics": evaluate_clustering(y_true, tfidf_agg.tolist()),
            "meta": tfidf_meta,
        }
    )

    dense_kmeans = run_kmeans(dense_x, n_clusters)
    experiments.append(
        {
            "representation": "dense",
            "method": "kmeans",
            "n_clusters": n_clusters,
            "metrics": evaluate_clustering(y_true, dense_kmeans.tolist()),
            "meta": dense_meta,
        }
    )

    dense_agg = run_agglomerative(dense_x, n_clusters)
    experiments.append(
        {
            "representation": "dense",
            "method": "agglomerative",
            "n_clusters": n_clusters,
            "metrics": evaluate_clustering(y_true, dense_agg.tolist()),
            "meta": dense_meta,
        }
    )

    summary = {
        "task_count": len(records),
        "group_count": n_clusters,
        "experiments": experiments,
    }

    save_json(output_dir / "clustering_summary.json", summary)

    print(f"Tasks: {len(records)}")
    print(f"Groups: {n_clusters}")
    for exp in experiments:
        print(
            f"{exp['representation']} | {exp['method']} | "
            f"ARI={exp['metrics']['ari']} | "
            f"NMI={exp['metrics']['nmi']} | "
            f"V={exp['metrics']['v_measure']}"
        )
    print(f"Saved: {output_dir / 'clustering_summary.json'}")


if __name__ == "__main__":
    main()
