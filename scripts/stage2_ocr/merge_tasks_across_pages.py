from pathlib import Path
import json
import re


TASK_NUMBER_RE = re.compile(r"^\d+\.\d+\.")
END_STRONG_RE = re.compile(r"(Билет\s+\d+,\s*\d{4}\)|\?\s*$|\.\s*$)")
END_WEAK_BAD_RE = re.compile(r"(рис\.\s*$|рисунок\s*$|=\s*$|,\s*$|-\s*$|:\s*$)")
START_BAD_RE = re.compile(r"^(Рис\.|Рисунок|[a-zA-Zа-яА-Я]\)|\d+\)|[=\-+/*]+)")
OCR_FALSE_TASK_RE = re.compile(r"^\d+\.\d+\.[^\s]")


def load_page_tasks(input_dir: Path) -> list[dict]:
    pages = []

    for path in sorted(input_dir.glob("page_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pages.append(data)

    return pages


def looks_finished(text: str) -> bool:
    tail = text.strip().splitlines()
    if not tail:
        return False

    last = tail[-1].strip()

    if END_STRONG_RE.search(last):
        return True

    if END_WEAK_BAD_RE.search(last):
        return False

    if len(last) < 20:
        return False

    return last.endswith(".")


def clean_join(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()

    if not left:
        return right
    if not right:
        return left

    if left.endswith("-"):
        return left[:-1] + right

    return left + "\n" + right


def is_real_task_start(text: str, expected_prefix: str | None = None) -> bool:
    text = text.strip()

    if not TASK_NUMBER_RE.match(text):
        return False

    if OCR_FALSE_TASK_RE.match(text):
        return False

    if expected_prefix is not None and not text.startswith(expected_prefix):
        return False

    return True


def split_head_continuation(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    continuation = []
    remainder = []
    found_task_start = False

    for line in lines:
        stripped = line.strip()

        if not found_task_start and TASK_NUMBER_RE.match(stripped):
            found_task_start = True
            remainder.append(line)
            continue

        if found_task_start:
            remainder.append(line)
        else:
            continuation.append(line)

    return "\n".join(continuation).strip(), "\n".join(remainder).strip()


def merge_tasks(pages: list[dict]) -> list[dict]:
    flat_tasks = []

    for page in pages:
        page_number = page["page_number"]
        for task in page["tasks"]:
            flat_tasks.append(
                {
                    "page_number": page_number,
                    "task_number": task["task_number"],
                    "raw_text": task["raw_text"].strip(),
                }
            )

    if not flat_tasks:
        return []

    merged = [flat_tasks[0].copy()]

    for current in flat_tasks[1:]:
        prev = merged[-1]

        prev_page = prev["page_number"]
        curr_page = current["page_number"]

        same_task_number = prev["task_number"] == current["task_number"]
        adjacent_pages = curr_page in {prev_page, prev_page + 1}

        if same_task_number and adjacent_pages:
            prev["raw_text"] = clean_join(prev["raw_text"], current["raw_text"])
            prev["end_page"] = curr_page
            continue

        prev_end_ok = looks_finished(prev["raw_text"])
        continuation_head, remainder = split_head_continuation(current["raw_text"])

        if not prev_end_ok and adjacent_pages and continuation_head:
            prev["raw_text"] = clean_join(prev["raw_text"], continuation_head)
            prev["end_page"] = curr_page

            if remainder:
                new_task_number = current["task_number"]
                first_line = remainder.splitlines()[0].strip()
                if TASK_NUMBER_RE.match(first_line):
                    new_task_number = first_line.split()[0][:-1]

                merged.append(
                    {
                        "page_number": curr_page,
                        "end_page": curr_page,
                        "task_number": new_task_number,
                        "raw_text": remainder,
                    }
                )
            continue

        if not prev_end_ok and adjacent_pages and START_BAD_RE.match(current["raw_text"].strip()):
            prev["raw_text"] = clean_join(prev["raw_text"], current["raw_text"])
            prev["end_page"] = curr_page
            continue

        merged.append(
            {
                "page_number": curr_page,
                "end_page": curr_page,
                "task_number": current["task_number"],
                "raw_text": current["raw_text"],
            }
        )

    for item in merged:
        if "end_page" not in item:
            item["end_page"] = item["page_number"]

    return merged


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "interim" / "tasks" / "json"
    output_dir = project_root / "data" / "interim" / "tasks_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_page_tasks(input_dir)
    merged_tasks = merge_tasks(pages)

    json_path = output_dir / "tasks_merged.json"
    jsonl_path = output_dir / "tasks_merged.jsonl"
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(merged_tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for task in merged_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    for task in merged_tasks:
        task_number_safe = task["task_number"].replace(".", "_")
        start_page = task["page_number"]
        end_page = task["end_page"]
        txt_name = f"task_{task_number_safe}__pages_{start_page:03d}_{end_page:03d}.txt"
        (txt_dir / txt_name).write_text(task["raw_text"], encoding="utf-8")

    print(f"Merged tasks: {len(merged_tasks)}")
    print(f"Saved: {json_path}")
    print(f"Saved: {jsonl_path}")
    print(f"Saved txt dir: {txt_dir}")


if __name__ == "__main__":
    main()
