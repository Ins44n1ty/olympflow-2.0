from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CLUSTER_INTERPRETATION_PATH = (
    ROOT
    / "data/clusters/tfidf_clustering/tfidf_svd_kmeans/cluster_interpretation.json"
)

OUTPUT_DIR = ROOT / "data/clusters/temporal_dynamics"
REPORT_DIR = ROOT / "reports/clustering"

MIN_TOPIC_TASKS = 4


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_topic(topic: str) -> str:
    topic = topic.strip()

    if not topic:
        return "неоднозначная тема"

    return topic


def extract_year(text: str) -> int | None:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)

    if not years:
        return None

    return int(years[-1])


def get_task_text(task: dict) -> str:
    for key in ("text", "task_text", "problem_text", "statement", "condition"):
        value = task.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    values = [
        value.strip()
        for value in task.values()
        if isinstance(value, str) and value.strip()
    ]

    return " ".join(values)


def short_text(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def classify_topic_pattern(years: list[int]) -> str:
    if not years:
        return "unknown"

    unique_years = sorted(set(years))
    years_count = len(unique_years)

    if years_count == 1:
        return "one_year"

    span = max(unique_years) - min(unique_years)

    if span <= 2:
        return "one_time_burst"

    if years_count >= 6 and span >= 8:
        return "long_term_recurring"

    if years_count >= 4:
        return "recurring"

    return "multi_year"


def build_topic_records(data: list[dict]) -> dict[str, list[dict]]:
    topics = defaultdict(list)

    for cluster in data:
        topic = normalize_topic(cluster.get("guessed_topic", ""))
        cluster_id = str(cluster.get("cluster_id", ""))

        for task in cluster.get("representative_tasks", []):
            text = get_task_text(task)
            year = task.get("year")

            if year is None:
                year = extract_year(text)

            item = {
                "cluster_id": cluster_id,
                "topic": topic,
                "task_id": task.get("task_id"),
                "task_number": task.get("task_number"),
                "group_id": task.get("group_id"),
                "year": year,
                "text": short_text(text),
            }

            topics[topic].append(item)

        if not cluster.get("representative_tasks"):
            for task in cluster.get("tasks", []):
                text = get_task_text(task)
                year = task.get("year")

                if year is None:
                    year = extract_year(text)

                item = {
                    "cluster_id": cluster_id,
                    "topic": topic,
                    "task_id": task.get("task_id"),
                    "task_number": task.get("task_number"),
                    "group_id": task.get("group_id"),
                    "year": year,
                    "text": short_text(text),
                }

                topics[topic].append(item)

    return dict(topics)


def analyze_topics(topics: dict[str, list[dict]]) -> list[dict]:
    results = []

    for topic, tasks in topics.items():
        if len(tasks) < MIN_TOPIC_TASKS:
            continue

        years = [
            int(task["year"])
            for task in tasks
            if task.get("year") is not None
        ]

        year_counter = Counter(years)
        unique_years = sorted(year_counter)
        cluster_ids = sorted(set(task["cluster_id"] for task in tasks))

        if unique_years:
            first_year = min(unique_years)
            last_year = max(unique_years)
            year_span = last_year - first_year
        else:
            first_year = None
            last_year = None
            year_span = None

        pattern = classify_topic_pattern(years)

        results.append(
            {
                "topic": topic,
                "tasks_count": len(tasks),
                "clusters_count": len(cluster_ids),
                "cluster_ids": cluster_ids,
                "years_count": len(unique_years),
                "years": unique_years,
                "year_distribution": dict(sorted(year_counter.items())),
                "first_year": first_year,
                "last_year": last_year,
                "year_span": year_span,
                "pattern": pattern,
                "tasks": sorted(
                    tasks,
                    key=lambda item: (
                        item["year"] if item.get("year") is not None else 9999,
                        str(item.get("task_id")),
                    ),
                ),
            }
        )

    results.sort(
        key=lambda item: (
            item["pattern"] not in {"long_term_recurring", "recurring"},
            -item["years_count"],
            -item["tasks_count"],
            item["topic"],
        )
    )

    return results


def plot_topic_year_coverage(topic_results: list[dict]) -> None:
    selected = topic_results[:20]

    if not selected:
        return

    labels = [item["topic"] for item in selected]
    values = [item["years_count"] for item in selected]

    plt.figure(figsize=(10, 7))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Number of covered years")
    plt.ylabel("Topic")
    plt.title("Topic temporal coverage")
    plt.tight_layout()

    path = REPORT_DIR / "topic_year_coverage.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_topic_year_heatmap(topic_results: list[dict]) -> None:
    selected = [
        item
        for item in topic_results
        if item["years_count"] >= 2 and item["topic"] != "неоднозначная тема"
    ][:20]

    if not selected:
        return

    all_years = sorted(
        {
            year
            for item in selected
            for year in item["years"]
        }
    )

    if not all_years:
        return

    matrix = []

    for item in selected:
        distribution = item["year_distribution"]
        row = [distribution.get(str(year), distribution.get(year, 0)) for year in all_years]
        matrix.append(row)

    matrix_array = np.array(matrix, dtype=float)

    plt.figure(figsize=(12, max(5, len(selected) * 0.45)))
    plt.imshow(matrix_array, aspect="auto")
    plt.colorbar(label="Tasks count")
    plt.xticks(range(len(all_years)), all_years, rotation=45)
    plt.yticks(range(len(selected)), [item["topic"] for item in selected])
    plt.xlabel("Year")
    plt.ylabel("Topic")
    plt.title("Topic-year heatmap")
    plt.tight_layout()

    path = REPORT_DIR / "topic_year_heatmap.png"
    plt.savefig(path, dpi=200)
    plt.close()


def save_markdown(topic_results: list[dict]) -> None:
    pattern_counter = Counter(item["pattern"] for item in topic_results)

    lines = [
        "# Topic temporal dynamics",
        "",
        "This report analyzes temporal dynamics on the topic level.",
        "",
        "Unlike cluster-level temporal analysis, this report groups tasks by automatically guessed cluster topics. This is useful because individual clusters are often small, while topics can recur across many clusters and years.",
        "",
        "## Pattern distribution",
        "",
        "| pattern | count |",
        "|---|---:|",
    ]

    for pattern, count in pattern_counter.most_common():
        lines.append(f"| {pattern} | {count} |")

    lines.extend(
        [
            "",
            "## Topic summary",
            "",
            "| topic | tasks | clusters | years | first year | last year | span | pattern |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for item in topic_results:
        lines.append(
            f"| {item['topic']} | "
            f"{item['tasks_count']} | "
            f"{item['clusters_count']} | "
            f"{item['years_count']} | "
            f"{item['first_year']} | "
            f"{item['last_year']} | "
            f"{item['year_span']} | "
            f"{item['pattern']} |"
        )

    lines.extend(
        [
            "",
            "## Main recurring topics",
            "",
        ]
    )

    recurring = [
        item
        for item in topic_results
        if item["pattern"] in {"recurring", "long_term_recurring", "periodic_candidate"}
    ]

    if not recurring:
        lines.append("No recurring topics found.")
        lines.append("")
    else:
        for item in recurring:
            years = ", ".join(str(year) for year in item["years"])

            lines.append(f"### {item['topic']}")
            lines.append("")
            lines.append(f"- pattern: `{item['pattern']}`")
            lines.append(f"- tasks: `{item['tasks_count']}`")
            lines.append(f"- clusters: `{item['clusters_count']}`")
            lines.append(f"- years: `{years}`")
            lines.append(f"- year span: `{item['year_span']}`")
            lines.append("")
            lines.append("Year distribution:")
            lines.append("")

            for year, count in item["year_distribution"].items():
                lines.append(f"- `{year}`: {count}")

            lines.append("")
            lines.append("Examples:")
            lines.append("")

            for task in item["tasks"][:8]:
                lines.append(
                    f"- `{task['task_id']}` | task `{task['task_number']}` | "
                    f"group `{task['group_id']}` | year `{task['year']}` | "
                    f"cluster `{task['cluster_id']}`"
                )
                lines.append("")
                lines.append(task["text"])
                lines.append("")

            lines.append("---")
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "At the level of individual clusters, strict periodicity is hard to detect because clusters are small. Topic-level aggregation gives a more useful view: it shows which physical themes return across different years and clusters.",
            "",
            "The labels are automatic and should be treated as a fast exploratory analysis, not as final expert annotation.",
            "",
            "Generated plots:",
            "",
            "- `reports/clustering/topic_year_coverage.png`",
            "- `reports/clustering/topic_year_heatmap.png`",
            "",
        ]
    )

    path = REPORT_DIR / "topic_temporal_dynamics.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = read_json(CLUSTER_INTERPRETATION_PATH)

    if not isinstance(data, list):
        raise TypeError("cluster_interpretation.json must contain a list")

    topics = build_topic_records(data)
    topic_results = analyze_topics(topics)

    summary = {
        "source_path": str(CLUSTER_INTERPRETATION_PATH.relative_to(ROOT)),
        "min_topic_tasks": MIN_TOPIC_TASKS,
        "topics_count": len(topic_results),
        "pattern_distribution": dict(Counter(item["pattern"] for item in topic_results).most_common()),
        "topics": topic_results,
    }

    save_json(OUTPUT_DIR / "topic_temporal_dynamics.json", summary)
    save_markdown(topic_results)
    plot_topic_year_coverage(topic_results)
    plot_topic_year_heatmap(topic_results)

    print(f"Saved topic temporal dynamics to {OUTPUT_DIR}")
    print(f"Saved topic temporal reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
