from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"

CLUSTER_SOURCES = {
    "dense_agglomerative": ROOT / "data/clusters/best_dense_agglomerative/records_with_clusters.jsonl",
    "tfidf_svd_kmeans": ROOT / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/records_with_clusters.jsonl",
    "tfidf_svd_agglomerative": ROOT / "data/clusters/tfidf_clustering/tfidf_svd_agglomerative/records_with_clusters.jsonl",
}

OUTPUT_DIR = ROOT / "data/clusters/cluster_quality"
REPORT_DIR = ROOT / "reports/clustering"

MIN_ANALYZED_CLUSTER_SIZE = 4
MAX_TEXT_LEN = 800


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_record_id(record: dict, index: int) -> str:
    for key in ("task_id", "id", "problem_id", "record_id"):
        if key in record:
            return str(record[key])

    return f"record_{index:04d}"


def get_task_number(record: dict) -> str:
    value = record.get("task_number")

    if value is not None:
        return str(value)

    return ""


def get_group_id(record: dict) -> str:
    for key in ("group_id", "weak_group_id", "group", "label"):
        if key in record:
            return str(record[key])

    return ""


def get_cluster_id(record: dict) -> str:
    for key in ("cluster_id", "cluster", "label"):
        if key in record:
            return str(record[key])

    raise KeyError("Cluster id field not found")


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


def short_text(text: str, limit: int = MAX_TEXT_LEN) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def extract_year(text: str) -> int | None:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)

    if not years:
        return None

    return int(years[-1])


def extract_section(record: dict) -> str:
    task_number = get_task_number(record)

    if "." in task_number:
        return task_number.split(".", 1)[0]

    group_id = get_group_id(record)

    if "_" in group_id:
        return group_id.split("_", 1)[0]

    return ""


def add_missing_fields(records: list[dict]) -> list[dict]:
    enriched = []

    for index, record in enumerate(records):
        item = dict(record)
        text = get_text(item)

        item["_index"] = index
        item["_task_id"] = get_record_id(item, index)
        item["_task_number"] = get_task_number(item)
        item["_group_id"] = get_group_id(item)
        item["_year"] = extract_year(text)
        item["_section"] = extract_section(item)
        item["_text"] = text

        enriched.append(item)

    return enriched


def group_by_cluster(records: list[dict]) -> dict[str, list[dict]]:
    clusters = defaultdict(list)

    for record in records:
        clusters[get_cluster_id(record)].append(record)

    return dict(clusters)


def entropy(counter: Counter) -> float:
    import math

    total = sum(counter.values())

    if total == 0:
        return 0.0

    value = 0.0

    for count in counter.values():
        p = count / total
        value -= p * math.log(p)

    return value


def analyze_cluster(cluster_id: str, records: list[dict]) -> dict:
    years = [record["_year"] for record in records if record["_year"] is not None]
    sections = [record["_section"] for record in records if record["_section"]]
    groups = [record["_group_id"] for record in records if record["_group_id"]]

    year_counter = Counter(years)
    section_counter = Counter(sections)
    group_counter = Counter(groups)

    size = len(records)
    unique_years = sorted(year_counter)
    unique_sections = sorted(section_counter)
    unique_groups = sorted(group_counter)

    flags = []

    if size == 4 and len(unique_years) > 1:
        flags.append("size_4_multiple_years")

    if size > 4 and len(unique_years) >= 4:
        flags.append("large_cluster_many_years")

    if size > 4 and len(unique_sections) >= 3:
        flags.append("large_cluster_many_sections")

    if size > 4 and len(unique_groups) >= size:
        flags.append("large_cluster_no_group_concentration")

    dominant_year_share = 0.0
    dominant_section_share = 0.0
    dominant_group_share = 0.0

    if year_counter:
        dominant_year_share = max(year_counter.values()) / size

    if section_counter:
        dominant_section_share = max(section_counter.values()) / size

    if group_counter:
        dominant_group_share = max(group_counter.values()) / size

    return {
        "cluster_id": cluster_id,
        "size": size,
        "years": unique_years,
        "year_distribution": dict(sorted(year_counter.items())),
        "sections": unique_sections,
        "section_distribution": dict(sorted(section_counter.items())),
        "group_distribution": dict(group_counter.most_common()),
        "dominant_year_share": dominant_year_share,
        "dominant_section_share": dominant_section_share,
        "dominant_group_share": dominant_group_share,
        "year_entropy": entropy(year_counter),
        "section_entropy": entropy(section_counter),
        "group_entropy": entropy(group_counter),
        "flags": flags,
        "tasks": [
            {
                "index": record["_index"],
                "task_id": record["_task_id"],
                "task_number": record["_task_number"],
                "group_id": record["_group_id"],
                "year": record["_year"],
                "section": record["_section"],
                "text": short_text(record["_text"]),
            }
            for record in sorted(records, key=lambda item: item["_task_id"])
        ],
    }


def analyze_method(method_name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    records = add_missing_fields(read_jsonl(path))
    clusters = group_by_cluster(records)

    cluster_items = []
    analyzed_clusters = []

    for cluster_id, cluster_records in sorted(clusters.items(), key=lambda item: int(item[0])):
        item = analyze_cluster(cluster_id, cluster_records)
        cluster_items.append(item)

        if item["size"] >= MIN_ANALYZED_CLUSTER_SIZE:
            analyzed_clusters.append(item)

    size_counter = Counter(item["size"] for item in cluster_items)

    flagged_clusters = [
        item
        for item in analyzed_clusters
        if item["flags"]
    ]

    return {
        "method": method_name,
        "source_path": str(path.relative_to(ROOT)),
        "records_count": len(records),
        "cluster_count": len(clusters),
        "analyzed_min_size": MIN_ANALYZED_CLUSTER_SIZE,
        "analyzed_cluster_count": len(analyzed_clusters),
        "flagged_cluster_count": len(flagged_clusters),
        "cluster_size_distribution": dict(sorted(size_counter.items())),
        "largest_cluster_size": max(size_counter) if size_counter else 0,
        "clusters_size_ge_4": analyzed_clusters,
        "flagged_clusters": flagged_clusters,
    }


def plot_cluster_size_distribution(method_name: str, method_result: dict) -> None:
    distribution = method_result["cluster_size_distribution"]

    sizes = []
    counts = []

    for size, count in distribution.items():
        sizes.append(int(size))
        counts.append(int(count))

    plt.figure(figsize=(8, 5))
    plt.bar(sizes, counts)
    plt.xlabel("Cluster size")
    plt.ylabel("Number of clusters")
    plt.title(f"Cluster size distribution: {method_name}")
    plt.tight_layout()

    path = REPORT_DIR / f"{method_name}_cluster_size_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()


def save_method_markdown(method_name: str, method_result: dict) -> None:
    lines = [
        f"# Cluster quality analysis: {method_name}",
        "",
        f"Source: `{method_result['source_path']}`",
        f"Records: `{method_result['records_count']}`",
        f"Clusters: `{method_result['cluster_count']}`",
        f"Clusters with size >= {MIN_ANALYZED_CLUSTER_SIZE}: `{method_result['analyzed_cluster_count']}`",
        f"Flagged clusters: `{method_result['flagged_cluster_count']}`",
        f"Largest cluster size: `{method_result['largest_cluster_size']}`",
        "",
        "## Cluster size distribution",
        "",
        "| cluster size | number of clusters |",
        "|---:|---:|",
    ]

    for size, count in method_result["cluster_size_distribution"].items():
        lines.append(f"| {size} | {count} |")

    lines.extend(
        [
            "",
            "## Flagged clusters",
            "",
        ]
    )

    if not method_result["flagged_clusters"]:
        lines.append("No flagged clusters.")
        lines.append("")
    else:
        for item in method_result["flagged_clusters"]:
            lines.append(
                f"- cluster `{item['cluster_id']}`, size `{item['size']}`, "
                f"flags: `{', '.join(item['flags'])}`"
            )

        lines.append("")

    lines.extend(
        [
            "## Clusters with size >= 4",
            "",
        ]
    )

    for item in method_result["clusters_size_ge_4"]:
        years = ", ".join(str(year) for year in item["years"]) if item["years"] else "unknown"
        sections = ", ".join(item["sections"]) if item["sections"] else "unknown"
        groups = ", ".join(
            f"{group} ({count})"
            for group, count in list(item["group_distribution"].items())[:10]
        )

        flags = ", ".join(item["flags"]) if item["flags"] else "none"

        lines.append(f"### Cluster {item['cluster_id']}")
        lines.append("")
        lines.append(f"- size: `{item['size']}`")
        lines.append(f"- years: `{years}`")
        lines.append(f"- sections: `{sections}`")
        lines.append(f"- dominant year share: `{item['dominant_year_share']:.3f}`")
        lines.append(f"- dominant section share: `{item['dominant_section_share']:.3f}`")
        lines.append(f"- dominant group share: `{item['dominant_group_share']:.3f}`")
        lines.append(f"- flags: `{flags}`")
        lines.append(f"- weak groups: {groups}")
        lines.append("")
        lines.append("Tasks:")
        lines.append("")

        for task in item["tasks"]:
            lines.append(
                f"- `{task['task_id']}` | task `{task['task_number']}` | "
                f"group `{task['group_id']}` | year `{task['year']}` | "
                f"section `{task['section']}`"
            )
            lines.append("")
            lines.append(task["text"])
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path = REPORT_DIR / f"{method_name}_cluster_quality_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_summary_markdown(results: dict[str, dict]) -> None:
    lines = [
        "# Cluster quality analysis",
        "",
        "This report checks clusters with size >= 4 for the main clustering methods.",
        "",
        "For clusters with size = 4, the main sanity check is whether tasks come from the same year.",
        "For clusters with size > 4, the report summarizes years, sections, weak groups and suspicious cases for manual inspection.",
        "",
        "## Summary",
        "",
        "| method | records | clusters | clusters size >= 4 | flagged clusters | largest cluster |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method_name, result in results.items():
        lines.append(
            f"| {method_name} | "
            f"{result['records_count']} | "
            f"{result['cluster_count']} | "
            f"{result['analyzed_cluster_count']} | "
            f"{result['flagged_cluster_count']} | "
            f"{result['largest_cluster_size']} |"
        )

    lines.extend(
        [
            "",
            "## Generated detailed reports",
            "",
        ]
    )

    for method_name in results:
        lines.append(f"- `reports/clustering/{method_name}_cluster_quality_analysis.md`")
        lines.append(f"- `reports/clustering/{method_name}_cluster_size_distribution.png`")

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- A flagged cluster is not automatically bad. It only means that it deserves manual inspection.",
            "- Clusters with size = 4 are expected to often match one weak group and therefore one year.",
            "- Larger clusters may be meaningful if they collect recurring problem templates across years.",
            "- If a large cluster mixes many years and many sections, it may be too broad or lexically driven.",
            "",
        ]
    )

    output_path = REPORT_DIR / "cluster_quality_analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for method_name, path in CLUSTER_SOURCES.items():
        print(f"Analyzing {method_name}")
        result = analyze_method(method_name, path)

        results[method_name] = result

        save_json(OUTPUT_DIR / f"{method_name}_cluster_quality.json", result)
        save_method_markdown(method_name, result)
        plot_cluster_size_distribution(method_name, result)

    compact_summary = {
        method_name: {
            "source_path": result["source_path"],
            "records_count": result["records_count"],
            "cluster_count": result["cluster_count"],
            "analyzed_cluster_count": result["analyzed_cluster_count"],
            "flagged_cluster_count": result["flagged_cluster_count"],
            "largest_cluster_size": result["largest_cluster_size"],
            "cluster_size_distribution": result["cluster_size_distribution"],
        }
        for method_name, result in results.items()
    }

    save_json(OUTPUT_DIR / "cluster_quality_summary.json", compact_summary)
    save_summary_markdown(results)

    print(f"Saved cluster quality data to {OUTPUT_DIR}")
    print(f"Saved cluster quality reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
