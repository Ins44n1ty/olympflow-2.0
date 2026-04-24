from pathlib import Path
import json
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_txt(path: Path, clusters: list[dict]) -> None:
    lines = []

    for cluster in clusters:
        lines.append("=" * 100)
        lines.append(f"cluster_id: {cluster['cluster_id']}")
        lines.append(f"size: {cluster['size']}")
        lines.append(f"group_ids: {cluster['group_ids']}")
        lines.append("")

        for item in cluster["items"]:
            lines.append("-" * 80)
            lines.append(
                f"{item['task_id']} | {item['task_number']} | "
                f"group={item['group_id']} | pos={item['group_pos']} | "
                f"pages={item['page_start']}-{item['page_end']}"
            )
            lines.append(item["raw_text"])
            lines.append("")

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    records_path = project_root / "data" / "dataset_grouped" / "dataset_grouped.jsonl"
    embeddings_path = project_root / "data" / "features" / "dense" / "embeddings.npy"

    output_dir = project_root / "data" / "clusters" / "best_dense_agglomerative"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(records_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    group_ids = [record["group_id"] for record in records]
    n_clusters = len(set(group_ids))

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = model.fit_predict(embeddings)

    records_with_clusters = []
    clusters_map = defaultdict(list)

    for record, cluster_label in zip(records, cluster_labels):
        item = {
            "task_id": record["task_id"],
            "task_number": record["task_number"],
            "group_id": record.get("group_id"),
            "group_pos": record.get("group_pos"),
            "section_id": record.get("section_id"),
            "task_index_in_section": record.get("task_index_in_section"),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "raw_text": record.get("raw_text", ""),
            "cluster_id": int(cluster_label),
        }
        records_with_clusters.append(item)
        clusters_map[int(cluster_label)].append(item)

    clusters = []
    for cluster_id, items in clusters_map.items():
        items = sorted(
            items,
            key=lambda x: (
                x["section_id"] if x["section_id"] is not None else 10**9,
                x["task_index_in_section"] if x["task_index_in_section"] is not None else 10**9,
            ),
        )

        unique_group_ids = sorted({item["group_id"] for item in items if item["group_id"] is not None})

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": len(items),
                "group_ids": unique_group_ids,
                "items": items,
            }
        )

    clusters.sort(key=lambda x: (-x["size"], x["cluster_id"]))

    summary = {
        "model": "dense + agglomerative",
        "task_count": len(records),
        "cluster_count": len(clusters),
        "largest_cluster_size": max(cluster["size"] for cluster in clusters),
        "smallest_cluster_size": min(cluster["size"] for cluster in clusters),
    }

    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "clusters.json", clusters)
    save_jsonl(output_dir / "records_with_clusters.jsonl", records_with_clusters)
    save_txt(output_dir / "clusters_readable.txt", clusters)

    print(f"Tasks: {len(records)}")
    print(f"Clusters: {len(clusters)}")
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'clusters.json'}")
    print(f"Saved: {output_dir / 'records_with_clusters.jsonl'}")
    print(f"Saved: {output_dir / 'clusters_readable.txt'}")


if __name__ == "__main__":
    main()
