from pathlib import Path
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity


K_VALUES = [1, 3, 5, 10]
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


def compute_tfidf_similarity(texts: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
    )
    matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(matrix)
    return vectorizer, sim_matrix


def reciprocal_rank(y_true_sorted: np.ndarray) -> float:
    relevant_positions = np.flatnonzero(y_true_sorted)
    if relevant_positions.size == 0:
        return 0.0
    return 1.0 / float(relevant_positions[0] + 1)


def precision_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    k = min(k, len(y_true_sorted))
    if k == 0:
        return 0.0
    return float(np.mean(y_true_sorted[:k]))


def recall_at_k(y_true_sorted: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    k = min(k, len(y_true_sorted))
    return float(np.sum(y_true_sorted[:k]) / total_relevant)


def evaluate_ranking(records: list[dict], sim_matrix: np.ndarray) -> tuple[dict, list[dict]]:
    group_ids = [record.get("group_id") for record in records]
    n = len(records)

    precision_scores = {k: [] for k in K_VALUES}
    recall_scores = {k: [] for k in K_VALUES}
    rr_scores = []
    ap_scores = []
    y_true_rows = []
    y_score_rows = []
    per_query = []

    for i in range(n):
        query_group = group_ids[i]
        if query_group is None:
            continue

        candidate_indices = [j for j in range(n) if j != i]
        y_true = np.array([1 if group_ids[j] == query_group else 0 for j in candidate_indices], dtype=int)
        y_score = np.array([sim_matrix[i, j] for j in candidate_indices], dtype=float)

        total_relevant = int(np.sum(y_true))
        if total_relevant == 0:
            continue

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        query_metrics = {
            "task_id": records[i]["task_id"],
            "task_number": records[i]["task_number"],
            "group_id": query_group,
            "metrics": {},
        }

        for k in K_VALUES:
            p_at_k = precision_at_k(y_true_sorted, k)
            r_at_k = recall_at_k(y_true_sorted, total_relevant, k)

            precision_scores[k].append(p_at_k)
            recall_scores[k].append(r_at_k)

            query_metrics["metrics"][f"precision@{k}"] = round(p_at_k, 6)
            query_metrics["metrics"][f"recall@{k}"] = round(r_at_k, 6)

        rr = reciprocal_rank(y_true_sorted)
        ap = float(average_precision_score(y_true, y_score))

        rr_scores.append(rr)
        ap_scores.append(ap)
        y_true_rows.append(y_true)
        y_score_rows.append(y_score)

        query_metrics["metrics"]["rr"] = round(rr, 6)
        query_metrics["metrics"]["ap"] = round(ap, 6)

        per_query.append(query_metrics)

    max_len = max(len(row) for row in y_true_rows)
    y_true_matrix = np.array([np.pad(row, (0, max_len - len(row))) for row in y_true_rows], dtype=int)
    y_score_matrix = np.array([np.pad(row, (0, max_len - len(row)), constant_values=-1e9) for row in y_score_rows], dtype=float)

    summary = {
        "query_count": len(per_query),
        "precision": {
            f"precision@{k}": round(float(np.mean(precision_scores[k])), 6)
            for k in K_VALUES
        },
        "recall": {
            f"recall@{k}": round(float(np.mean(recall_scores[k])), 6)
            for k in K_VALUES
        },
        "mrr": round(float(np.mean(rr_scores)), 6),
        "map": round(float(np.mean(ap_scores)), 6),
        "lrap": round(float(label_ranking_average_precision_score(y_true_matrix, y_score_matrix)), 6),
    }

    return summary, per_query


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "dataset_grouped" / "dataset_grouped.jsonl"
    output_dir = project_root / "data" / "metrics" / "tfidf"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    per_query_path = output_dir / "per_query_metrics.json"
    config_path = output_dir / "config.json"

    records = load_jsonl(input_path)
    texts = build_texts(records)
    vectorizer, sim_matrix = compute_tfidf_similarity(texts)

    ranking_summary, per_query = evaluate_ranking(records, sim_matrix)

    summary = {
        "task_count": len(records),
        "vocab_size": len(vectorizer.vocabulary_),
        "ranking_metrics": ranking_summary,
    }

    config = {
        "input_path": str(input_path),
        "k_values": K_VALUES,
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "ngram_range": list(NGRAM_RANGE),
        "relevance_definition": "same group_id, excluding the query itself",
        "metrics": ["precision@k", "recall@k", "mrr", "map", "lrap"],
    }

    save_json(summary_path, summary)
    save_json(per_query_path, per_query)
    save_json(config_path, config)

    print(f"Processed tasks: {len(records)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"MRR: {ranking_summary['mrr']}")
    print(f"MAP: {ranking_summary['map']}")
    print(f"LRAP: {ranking_summary['lrap']}")
    for k in K_VALUES:
        print(f"Precision@{k}: {ranking_summary['precision'][f'precision@{k}']}")
        print(f"Recall@{k}: {ranking_summary['recall'][f'recall@{k}']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {per_query_path}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()
