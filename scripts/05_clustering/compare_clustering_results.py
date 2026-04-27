from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

OLD_SUMMARY_PATH = ROOT / "data/clusters/clustering_summary.json"
SPARSE_SUMMARY_PATH = ROOT / "data/clusters/sparse_clustering_summary.json"
OUTPUT_PATH = ROOT / "reports/clustering/clustering_comparison.md"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_value(value) -> str:
    if value is None:
        return "NA"

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def rows_from_old_summary(data: dict) -> list[dict]:
    rows = []

    for experiment in data.get("experiments", []):
        metrics = experiment["metrics"]

        rows.append(
            {
                "method": f"{experiment['representation']} + {experiment['method']}",
                "n_clusters": experiment.get("n_clusters"),
                "adjusted_rand_index": metrics.get("ari"),
                "normalized_mutual_info": metrics.get("nmi"),
                "homogeneity": metrics.get("homogeneity"),
                "completeness": metrics.get("completeness"),
                "v_measure": metrics.get("v_measure"),
                "silhouette_cosine": metrics.get("silhouette_cosine"),
                "source": "previous_clustering_summary",
            }
        )

    return rows


def rows_from_sparse_summary(data: dict) -> list[dict]:
    rows = []

    for method, metrics in data.get("experiments", {}).items():
        rows.append(
            {
                "method": method,
                "n_clusters": data.get("n_clusters"),
                "adjusted_rand_index": metrics.get("adjusted_rand_index"),
                "normalized_mutual_info": metrics.get("normalized_mutual_info"),
                "homogeneity": metrics.get("homogeneity"),
                "completeness": metrics.get("completeness"),
                "v_measure": metrics.get("v_measure"),
                "silhouette_cosine": metrics.get("silhouette_cosine"),
                "source": "new_sparse_clustering",
            }
        )

    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    if OLD_SUMMARY_PATH.exists():
        rows.extend(rows_from_old_summary(read_json(OLD_SUMMARY_PATH)))

    if SPARSE_SUMMARY_PATH.exists():
        rows.extend(rows_from_sparse_summary(read_json(SPARSE_SUMMARY_PATH)))

    rows.sort(
        key=lambda row: (
            row["n_clusters"] if row["n_clusters"] is not None else -1,
            row["v_measure"] if row["v_measure"] is not None else -1,
        ),
        reverse=True,
    )

    lines = [
        "# Clustering comparison",
        "",
        "| method | n_clusters | ARI | NMI | homogeneity | completeness | V-measure | silhouette cosine | source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    format_value(row["n_clusters"]),
                    format_value(row["adjusted_rand_index"]),
                    format_value(row["normalized_mutual_info"]),
                    format_value(row["homogeneity"]),
                    format_value(row["completeness"]),
                    format_value(row["v_measure"]),
                    format_value(row["silhouette_cosine"]),
                    row["source"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            (
                "The most directly comparable clustering experiments are those with "
                "`n_clusters` equal to the number of weak groups. In this dataset, "
                "the weak group count is 183."
            ),
            "",
            (
                "The metrics ARI, NMI, homogeneity, completeness and V-measure measure "
                "agreement with weak labels. They should not be interpreted as final "
                "semantic clustering quality because weak labels are local pseudo-labels, "
                "not manually validated semantic classes."
            ),
            "",
            (
                "Sparse TF-IDF clustering is useful as a lexical baseline. Dense clustering "
                "remains useful because it can capture broader semantic similarity, even when "
                "lexical sparse methods are stronger in retrieval."
            ),
            "",
        ]
    )

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved clustering comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
