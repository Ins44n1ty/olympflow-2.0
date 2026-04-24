from pathlib import Path
import json
import re


START_PAGE = 2
END_PAGE = 177

TASK_START_RE = re.compile(r"(?m)^(?P<number>\d+\.\d+)\.\s")
FIGURE_LINE_RE = re.compile(r"(?im)^\s*!\[.*?\]\(.*?\)\s*$")
FIGURE_CAPTION_RE = re.compile(r"(?im)^\s*Рис\.\s*\d+\.\d+.*$")
PAGE_HEADER_RE = re.compile(r"(?im)^\s*\d+\s+[А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\s\-]+$")
PAGE_FOOTER_RE = re.compile(r"(?im)^\s*[А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\s\-]+\s+\d+\s*$")


def extract_page_number(path: Path) -> int | None:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def clean_mistral_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = FIGURE_LINE_RE.sub("", text)
    text = FIGURE_CAPTION_RE.sub("", text)
    text = PAGE_HEADER_RE.sub("", text)
    text = PAGE_FOOTER_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    input_dir = project_root / "data" / "interim" / "ocr_mistral" / "txt"
    txt_output_dir = project_root / "data" / "interim" / "tasks_mistral" / "txt"
    json_output_dir = project_root / "data" / "interim" / "tasks_mistral" / "json"

    txt_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("page_*.txt"))

    if not input_paths:
        raise FileNotFoundError(f"No Mistral OCR txt files found in {input_dir}")

    filtered_paths = []
    for input_path in input_paths:
        page_number = extract_page_number(input_path)
        if page_number is None:
            continue
        if START_PAGE <= page_number <= END_PAGE:
            filtered_paths.append(input_path)

    if not filtered_paths:
        raise FileNotFoundError("No valid Mistral OCR pages in target range")

    for input_path in filtered_paths:
        page_number = extract_page_number(input_path)
        if page_number is None:
            continue

        text = input_path.read_text(encoding="utf-8")
        cleaned_text = clean_mistral_text(text)
        tasks = split_tasks(cleaned_text)

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
