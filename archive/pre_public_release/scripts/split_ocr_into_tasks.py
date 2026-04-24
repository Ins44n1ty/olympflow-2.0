from pathlib import Path
import json
import re


START_PAGE = 2
END_PAGE = 177
TASK_START_RE = re.compile(r"(?m)^(?P<number>\d+\.\d+)\.\s")


def extract_page_number(path: Path) -> int | None:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def split_tasks(text: str) -> list[dict]:
    matches = list(TASK_START_RE.finditer(text))
    tasks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        task_number = match.group("number")
        task_text = text[start:end].strip()

        tasks.append(
            {
                "task_number": task_number,
                "raw_text": task_text,
            }
        )

    return tasks


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "interim" / "ocr_text" / "txt"
    txt_output_dir = project_root / "data" / "interim" / "tasks" / "txt"
    json_output_dir = project_root / "data" / "interim" / "tasks" / "json"

    txt_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("page_*.txt"))

    if not input_paths:
        raise FileNotFoundError(f"No OCR txt files found in {input_dir}")

    filtered_paths = []
    for input_path in input_paths:
        page_number = extract_page_number(input_path)
        if page_number is None:
            continue
        if START_PAGE <= page_number <= END_PAGE:
            filtered_paths.append(input_path)

    for input_path in filtered_paths:
        text = input_path.read_text(encoding="utf-8")
        tasks = split_tasks(text)

        page_number = extract_page_number(input_path)

        page_json = {
            "page_number": page_number,
            "source_file": input_path.name,
            "task_count": len(tasks),
            "tasks": tasks,
        }

        json_path = json_output_dir / f"{input_path.stem}.json"
        json_path.write_text(
            json.dumps(page_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for task in tasks:
            task_number_safe = task["task_number"].replace(".", "_")
            task_txt_path = txt_output_dir / f"{input_path.stem}__task_{task_number_safe}.txt"
            task_txt_path.write_text(task["raw_text"], encoding="utf-8")

        print(f"{input_path.name}: {len(tasks)} tasks")

    print("Done.")


if __name__ == "__main__":
    main()
