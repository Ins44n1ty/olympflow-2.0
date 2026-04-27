from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

METHODS = {
    "tfidf": ROOT / "data/metrics/tfidf/summary.json",
    "dense": ROOT / "data/metrics/dense/summary.json",
    "bm25": ROOT / "data/metrics/bm25/summary.json",
    "openai_dense": ROOT / "data/metrics/openai_dense/summary.json",
    "cross_encoder": ROOT / "data/metrics/cross_encoder_rerank/summary.json",
}

OUTPUT_DIR = ROOT / "reports/retrieval"

METRICS = [
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr",
    "map",
    "lrap",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_summary(method: str, data: dict) -> dict:
    row = {"method": method}

    if "ranking_metrics" in data:
        ranking = data["ranking_metrics"]

        row["queries_count"] = ranking.get("query_count")

        for metric in METRICS:
            if metric.startswith("precision@"):
                row[metric] = ranking.get("precision", {}).get(metric)
            elif metric.startswith("recall@"):
                row[metric] = ranking.get("recall", {}).get(metric)
            else:
                row[metric] = ranking.get(metric)

        return row

    row["queries_count"] = data.get("queries_count")

    for metric in METRICS:
        row[metric] = data.get(metric)

    return row


def build_metrics_table() -> pd.DataFrame:
    rows = []

    for method, path in METHODS.items():
        if path.exists():
            rows.append(extract_summary(method, read_json(path)))

    return pd.DataFrame(rows)


def save_markdown_table(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.round(6).to_markdown(index=False) + "\n", encoding="utf-8")


def plot_at_k(df: pd.DataFrame, prefix: str, title: str, output_path: Path) -> None:
    k_values = [1, 3, 5, 10]
    metric_names = [f"{prefix}@{k}" for k in k_values]

    plt.figure(figsize=(8, 5))

    for _, row in df.iterrows():
        values = [row[metric] for metric in metric_names]
        plt.plot(k_values, values, marker="o", label=row["method"])

    plt.xlabel("k")
    plt.ylabel(prefix)
    plt.title(title)
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ranking_metrics(df: pd.DataFrame, output_path: Path) -> None:
    metric_names = ["mrr", "map", "lrap"]
    plot_df = df.set_index("method")[metric_names]

    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("method")
    ax.set_ylabel("score")
    ax.set_title("Ranking metrics")
    ax.legend(title="metric")
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_metrics_table()

    if df.empty:
        raise RuntimeError("No retrieval summary files found.")

    columns = ["method", "queries_count", *METRICS]
    df = df[columns]

    df.to_csv(OUTPUT_DIR / "retrieval_metrics_table.csv", index=False)
    save_markdown_table(OUTPUT_DIR / "retrieval_metrics_summary.md", df)

    plot_at_k(
        df=df,
        prefix="precision",
        title="Precision@k",
        output_path=OUTPUT_DIR / "precision_at_k.png",
    )

    plot_at_k(
        df=df,
        prefix="recall",
        title="Recall@k",
        output_path=OUTPUT_DIR / "recall_at_k.png",
    )

    plot_ranking_metrics(
        df=df,
        output_path=OUTPUT_DIR / "ranking_metrics.png",
    )

    print(f"Saved retrieval report to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
