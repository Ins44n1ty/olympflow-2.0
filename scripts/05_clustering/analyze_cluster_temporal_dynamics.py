from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

CLUSTER_QUALITY_DIR = ROOT / "data/clusters/cluster_quality"
OUTPUT_DIR = ROOT / "data/clusters/temporal_dynamics"
REPORT_DIR = ROOT / "reports/clustering"

QUALITY_SOURCES = {
    "dense_agglomerative": CLUSTER_QUALITY_DIR / "dense_agglomerative_cluster_quality.json",
    "tfidf_svd_kmeans": CLUSTER_QUALITY_DIR / "tfidf_svd_kmeans_cluster_quality.json",
    "tfidf_svd_agglomerative": CLUSTER_QUALITY_DIR / "tfidf_svd_agglomerative_cluster_quality.json",
}

MIN_CLUSTER_SIZE = 5


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_cluster_topic(method_name: str, cluster_id: str) -> str:
    if method_name != "tfidf_svd_kmeans":
        return ""

    path = ROOT / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/cluster_interpretation.json"

    if not path.exists():
        return ""

    data = read_json(path)

    for item in data:
        if str(item.get("cluster_id")) == str(cluster_id):
            return item.get("guessed_topic", "")

    return ""


def classify_temporal_pattern(years: list[int]) -> str:
    if not years:
        return "unknown"

    unique_years = sorted(set(years))

    if len(unique_years) == 1:
        return "one_year"

    span = max(unique_years) - min(unique_years)

    if span <= 2:
        return "one_time_burst"

    if len(unique_years) >= 4:
        gaps = [
            right - left
            for left, right in zip(unique_years, unique_years[1:])
        ]

        if gaps and max(gaps) - min(gaps) <= 1:
            return "periodic_candidate"

        return "recurring"

    return "multi_year"


def analyze_method(method_name: str, path: Path) -> dict:
    data = read_json(path)
    clusters = data["clusters_size_ge_4"]

    large_clusters = [
        cluster
        for cluster in clusters
        if cluster["size"] >= MIN_CLUSTER_SIZE
    ]

    analyzed = []

    for cluster in large_clusters:
        years = []

        for task in cluster["tasks"]:
            year = task.get("year")

            if year is not None:
                years.append(int(year))

        year_counter = Counter(years)
        unique_years = sorted(year_counter)
        topic = get_cluster_topic(method_name, str(cluster["cluster_id"]))

        if unique_years:
            year_span = max(unique_years) - min(unique_years)
            first_year = min(unique_years)
            last_year = max(unique_years)
        else:
            year_span = None
            first_year = None
            last_year = None

        pattern = classify_temporal_pattern(years)

        analyzed.append(
            {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "topic": topic,
                "pattern": pattern,
                "years": unique_years,
                "year_distribution": dict(sorted(year_counter.items())),
                "first_year": first_year,
                "last_year": last_year,
                "year_span": year_span,
                "sections": cluster["sections"],
                "section_distribution": cluster["section_distribution"],
                "group_distribution": cluster["group_distribution"],
                "dominant_year_share": cluster["dominant_year_share"],
                "dominant_section_share": cluster["dominant_section_share"],
                "dominant_group_share": cluster["dominant_group_share"],
                "tasks": cluster["tasks"],
            }
        )

    pattern_counter = Counter(item["pattern"] for item in analyzed)
    topic_counter = Counter(
        item["topic"]
        for item in analyzed
        if item["topic"]
    )
    span_counter = Counter(
        item["year_span"]
        for item in analyzed
        if item["year_span"] is not None
    )

    return {
        "method": method_name,
        "source_path": str(path.relative_to(ROOT)),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "large_cluster_count": len(analyzed),
        "pattern_distribution": dict(pattern_counter.most_common()),
        "topic_distribution": dict(topic_counter.most_common()),
        "year_span_distribution": dict(sorted(span_counter.items())),
        "clusters": analyzed,
    }


def plot_year_span_distribution(method_name: str, result: dict) -> None:
    distribution = result["year_span_distribution"]

    if not distribution:
        return

    spans = [int(span) for span in distribution.keys()]
    counts = [int(count) for count in distribution.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(spans, counts)
    plt.xlabel("Year span inside cluster")
    plt.ylabel("Number of clusters")
    plt.title(f"Temporal span distribution: {method_name}")
    plt.tight_layout()

    path = REPORT_DIR / f"{method_name}_cluster_year_span_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()


def save_method_markdown(method_name: str, result: dict) -> None:
    lines = [
        f"# Cluster temporal dynamics: {method_name}",
        "",
        f"Source: `{result['source_path']}`",
        f"Analyzed clusters with size >= `{MIN_CLUSTER_SIZE}`",
        f"Large clusters: `{result['large_cluster_count']}`",
        "",
        "## Pattern distribution",
        "",
        "| pattern | count |",
        "|---|---:|",
    ]

    for pattern, count in result["pattern_distribution"].items():
        lines.append(f"| {pattern} | {count} |")

    lines.extend(["", "## Topic distribution", ""])

    if result["topic_distribution"]:
        lines.extend(["| topic | count |", "|---|---:|"])

        for topic, count in result["topic_distribution"].items():
            lines.append(f"| {topic} | {count} |")
    else:
        lines.append("Topic labels are available only for `tfidf_svd_kmeans`.")

    lines.extend(["", "## Large clusters", ""])

    for cluster in result["clusters"]:
        years = ", ".join(str(year) for year in cluster["years"]) if cluster["years"] else "unknown"
        topic = cluster["topic"] if cluster["topic"] else "not_available"

        lines.append(f"### Cluster {cluster['cluster_id']}")
        lines.append("")
        lines.append(f"- size: `{cluster['size']}`")
        lines.append(f"- topic: `{topic}`")
        lines.append(f"- pattern: `{cluster['pattern']}`")
        lines.append(f"- years: `{years}`")
        lines.append(f"- year span: `{cluster['year_span']}`")
        lines.append(f"- dominant year share: `{cluster['dominant_year_share']:.3f}`")
        lines.append(f"- dominant section share: `{cluster['dominant_section_share']:.3f}`")
        lines.append("")

        lines.append("Year distribution:")
        lines.append("")

        for year, count in cluster["year_distribution"].items():
            lines.append(f"- `{year}`: {count}")

        lines.append("")
        lines.append("Representative tasks:")
        lines.append("")

        for task in cluster["tasks"][:8]:
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

    path = REPORT_DIR / f"{method_name}_cluster_temporal_dynamics.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def save_summary_markdown(results: dict[str, dict]) -> None:
    lines = [
        "# Cluster temporal dynamics",
        "",
        "This report analyzes clusters with size > 4.",
        "",
        "The goal is to check whether larger clusters represent one-time repetitions, multi-year recurring templates, or broad mixed clusters.",
        "",
        "## Summary",
        "",
        "| method | large clusters | one_year | one_time_burst | multi_year | recurring | periodic_candidate | unknown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for method_name, result in results.items():
        dist = result["pattern_distribution"]

        lines.append(
            f"| {method_name} | "
            f"{result['large_cluster_count']} | "
            f"{dist.get('one_year', 0)} | "
            f"{dist.get('one_time_burst', 0)} | "
            f"{dist.get('multi_year', 0)} | "
            f"{dist.get('recurring', 0)} | "
            f"{dist.get('periodic_candidate', 0)} | "
            f"{dist.get('unknown', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `one_year`: all tasks in the cluster come from one year.",
            "- `one_time_burst`: tasks are concentrated in a narrow interval of neighboring years.",
            "- `multi_year`: tasks appear in several years, but without a strong recurring pattern.",
            "- `recurring`: tasks are spread across a wider time interval.",
            "- `periodic_candidate`: years are spread with nearly regular gaps, so the cluster may describe a periodic template.",
            "",
            "The labels are heuristic. They are meant for fast manual inspection, not as final semantic annotation.",
            "",
            "## Generated detailed reports",
            "",
        ]
    )

    for method_name in results:
        lines.append(f"- `reports/clustering/{method_name}_cluster_temporal_dynamics.md`")
        lines.append(f"- `reports/clustering/{method_name}_cluster_year_span_distribution.png`")

    lines.append("")

    path = REPORT_DIR / "cluster_temporal_dynamics.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for method_name, path in QUALITY_SOURCES.items():
        print(f"Analyzing temporal dynamics for {method_name}")
        result = analyze_method(method_name, path)
        results[method_name] = result

        save_json(OUTPUT_DIR / f"{method_name}_cluster_temporal_dynamics.json", result)
        save_method_markdown(method_name, result)
        plot_year_span_distribution(method_name, result)

    compact_summary = {
        method_name: {
            "source_path": result["source_path"],
            "min_cluster_size": result["min_cluster_size"],
            "large_cluster_count": result["large_cluster_count"],
            "pattern_distribution": result["pattern_distribution"],
            "topic_distribution": result["topic_distribution"],
            "year_span_distribution": result["year_span_distribution"],
        }
        for method_name, result in results.items()
    }

    save_json(OUTPUT_DIR / "cluster_temporal_dynamics_summary.json", compact_summary)
    save_summary_markdown(results)

    print(f"Saved temporal dynamics data to {OUTPUT_DIR}")
    print(f"Saved temporal dynamics reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
