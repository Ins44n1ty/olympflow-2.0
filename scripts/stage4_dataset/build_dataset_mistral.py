from pathlib import Path
import json
import re


TASK_NUMBER_RE = re.compile(r"^(?P<section>\d+)\.(?P<index>\d+)$")


def parse_task_number(task_number: str) -> tuple[int | None, int | None]:
    match = TASK_NUMBER_RE.match(task_number.strip())
    if not match:
        return None, None

    section_id = int(match.group("section"))
    task_index = int(match.group("index"))
    return section_id, task_index


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_tasks(path: Path) -> list[dict]:
    tasks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    return tasks


def build_dataset(tasks: list[dict]) -> list[dict]:
    dataset = []

    for idx, task in enumerate(tasks, start=1):
        task_number = task.get("task_number", "").strip()
        section_id, task_index = parse_task_number(task_number)

        page_start = task.get("page_number")
        page_end = task.get("end_page", page_start)
        raw_text = normalize_text(task.get("raw_text", ""))

        record = {
            "task_id": f"task_{idx:04d}",
            "task_number": task_number,
            "section_id": section_id,
            "task_index_in_section": task_index,
            "page_start": page_start,
            "page_end": page_end,
            "raw_text": raw_text,
        }

        dataset.append(record)

    return dataset


def save_json(path: Path, data: list[dict]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(path: Path, data: list[dict]) -> None:
    headers = [
        "task_id",
        "task_number",
        "section_id",
        "task_index_in_section",
        "page_start",
        "page_end",
        "raw_text",
    ]

    def escape_csv(value: object) -> str:
        text = "" if value is None else str(value)
        text = text.replace('"', '""')
        return f'"{text}"'

    lines = [",".join(headers)]

    for record in data:
        row = [escape_csv(record.get(header)) for header in headers]
        lines.append(",".join(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(path: Path, data: list[dict]) -> None:
    sections = sorted(
        {
            record["section_id"]
            for record in data
            if record["section_id"] is not None
        }
    )

    summary = {
        "task_count": len(data),
        "section_ids": sections,
        "first_task_number": data[0]["task_number"] if data else None,
        "last_task_number": data[-1]["task_number"] if data else None,
        "source": "tasks_mistral_merged.jsonl",
    }

    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "interim"
        / "tasks_mistral_merged"
        / "tasks_mistral_merged.jsonl"
    )

    output_dir = project_root / "data" / "final_mistral"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_json_path = output_dir / "dataset.json"
    dataset_jsonl_path = output_dir / "dataset.jsonl"
    dataset_csv_path = output_dir / "dataset.csv"
    summary_path = output_dir / "dataset_summary.json"

    tasks = load_tasks(input_path)
    dataset = build_dataset(tasks)

    save_json(dataset_json_path, dataset)
    save_jsonl(dataset_jsonl_path, dataset)
    save_csv(dataset_csv_path, dataset)
    save_summary(summary_path, dataset)

    print(f"Built dataset with {len(dataset)} tasks")
    print(f"Saved: {dataset_json_path}")
    print(f"Saved: {dataset_jsonl_path}")
    print(f"Saved: {dataset_csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
