from pathlib import Path

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
    output_dir = project_root / "data" / "interim" / "pages" / "selected"

    output_dir.mkdir(parents=True, exist_ok=True)

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
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        output_path = output_dir / f"page_{page_number:03d}.png"
        pix.save(output_path)
        print(f"Saved: {output_path}")

    doc.close()
    print(f"Done. Extracted pages saved to: {output_dir}")


if __name__ == "__main__":
    main()
