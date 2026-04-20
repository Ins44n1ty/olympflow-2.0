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


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


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
        "group_id",
        "group_pos",
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


def label_groups(records: list[dict]) -> list[dict]:
    counters_by_section: dict[int, int] = {}

    for record in records:
        task_number = record.get("task_number", "")
        section_id, task_index = parse_task_number(task_number)

        if section_id is None or task_index is None:
            record["group_id"] = None
            record["group_pos"] = None
            continue

        if section_id not in counters_by_section:
            counters_by_section[section_id] = 0

        counters_by_section[section_id] += 1
        position_in_section = counters_by_section[section_id]

        group_index = (position_in_section - 1) // 4 + 1
        group_pos = (position_in_section - 1) % 4 + 1

        record["group_id"] = f"{section_id}_{group_index:03d}"
        record["group_pos"] = group_pos

    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "final_mistral" / "dataset.jsonl"
    output_dir = project_root / "data" / "final_mistral_grouped"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json_path = output_dir / "dataset_grouped.json"
    output_jsonl_path = output_dir / "dataset_grouped.jsonl"
    output_csv_path = output_dir / "dataset_grouped.csv"

    records = load_jsonl(input_path)
    records = label_groups(records)

    save_json(output_json_path, records)
    save_jsonl(output_jsonl_path, records)
    save_csv(output_csv_path, records)

    print(f"Labeled records: {len(records)}")
    print(f"Saved: {output_json_path}")
    print(f"Saved: {output_jsonl_path}")
    print(f"Saved: {output_csv_path}")


if __name__ == "__main__":
    main()
