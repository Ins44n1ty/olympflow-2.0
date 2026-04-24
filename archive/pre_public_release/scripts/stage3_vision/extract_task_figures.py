from pathlib import Path
import json
import re

import fitz


MIN_WIDTH = 120
MIN_HEIGHT = 120


def load_sample_pages(path: Path) -> list[int]:
    pages = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pages.append(int(line))

    return pages


def load_page_tasks(path: Path) -> dict[int, list[dict]]:
    result = {}

    for json_path in sorted(path.glob("page_*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        page_number = data["page_number"]
        result[page_number] = data.get("tasks", [])

    return result


def guess_task_for_figure(fig_y0: float, tasks: list[dict]) -> str | None:
    if not tasks:
        return None

    candidates = []

    for task in tasks:
        raw_text = task.get("raw_text", "")
        match = re.search(rf"{re.escape(task['task_number'])}\..*?(Рис\.\s*\d+\.\d+)", raw_text, re.DOTALL)
        if match:
            candidates.append(task["task_number"])

    if len(candidates) == 1:
        return candidates[0]

    return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    pdf_path = project_root / "data" / "raw" / "pdf" / "phys_book.pdf"
    sample_pages_path = (
        project_root / "data" / "processed" / "annotations" / "sample_pages.txt"
    )
    tasks_dir = project_root / "data" / "interim" / "tasks" / "json"

    figures_dir = project_root / "data" / "interim" / "figures" / "images"
    meta_dir = project_root / "data" / "interim" / "figures" / "meta"

    figures_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    selected_pages = load_sample_pages(sample_pages_path)
    page_tasks = load_page_tasks(tasks_dir)

    doc = fitz.open(pdf_path)
    all_meta = []

    for page_number in selected_pages:
        page = doc.load_page(page_number - 1)
        image_infos = page.get_image_info(xrefs=True)

        figure_index = 0

        for info in image_infos:
            bbox = fitz.Rect(info["bbox"])
            width = int(bbox.width)
            height = int(bbox.height)

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            figure_index += 1

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox)
            image_name = f"page_{page_number:03d}_figure_{figure_index:02d}.png"
            image_path = figures_dir / image_name
            pix.save(image_path)

            task_number = guess_task_for_figure(bbox.y0, page_tasks.get(page_number, []))

            meta = {
                "page_number": page_number,
                "figure_index": figure_index,
                "image_file": image_name,
                "bbox": {
                    "x0": round(bbox.x0, 2),
                    "y0": round(bbox.y0, 2),
                    "x1": round(bbox.x1, 2),
                    "y1": round(bbox.y1, 2),
                },
                "width": width,
                "height": height,
                "guessed_task_number": task_number,
            }

            meta_path = meta_dir / f"page_{page_number:03d}_figure_{figure_index:02d}.json"
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            all_meta.append(meta)
            print(f"Saved: {image_path}")

    summary_path = meta_dir / "figures_summary.json"
    summary_path.write_text(
        json.dumps(all_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
