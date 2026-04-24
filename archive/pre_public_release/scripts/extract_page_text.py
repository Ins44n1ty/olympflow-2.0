from pathlib import Path
import json

import fitz


def read_page_numbers(path: Path) -> list[int]:
    pages = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pages.append(int(line))

    return pages


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    pdf_path = project_root / "data" / "raw" / "pdf" / "phys_book.pdf"
    sample_pages_path = (
        project_root / "data" / "processed" / "annotations" / "sample_pages.txt"
    )

    text_output_dir = project_root / "data" / "interim" / "page_text" / "txt"
    meta_output_dir = project_root / "data" / "interim" / "page_text" / "meta"

    text_output_dir.mkdir(parents=True, exist_ok=True)
    meta_output_dir.mkdir(parents=True, exist_ok=True)

    page_numbers = read_page_numbers(sample_pages_path)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"PDF: {pdf_path.name}")
    print(f"Total pages: {total_pages}")
    print(f"Selected pages: {page_numbers}")

    for page_number in page_numbers:
        if not 1 <= page_number <= total_pages:
            print(f"Skip invalid page number: {page_number}")
            continue

        page = doc.load_page(page_number - 1)
        text = page.get_text("text")

        txt_path = text_output_dir / f"page_{page_number:03d}.txt"
        meta_path = meta_output_dir / f"page_{page_number:03d}.json"

        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "page_number": page_number,
            "pdf_file": pdf_path.name,
            "text_length": len(text),
            "word_count_approx": len(text.split()),
            "is_empty": len(text.strip()) == 0,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Saved text: {txt_path}")
        print(f"Saved meta: {meta_path}")

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
