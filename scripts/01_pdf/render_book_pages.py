from pathlib import Path

import fitz


START_PAGE = 2
END_PAGE = 177
RENDER_SCALE = 2.5


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "data" / "raw" / "pdf" / "phys_book.pdf"
    output_dir = project_root / "data" / "interim" / "pages" / "selected"

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    start_page = max(1, START_PAGE)
    end_page = min(END_PAGE, total_pages)

    print(f"PDF: {pdf_path.name}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"Processing pages: {start_page}..{end_page}")

    for page_number in range(start_page, end_page + 1):
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
        output_path = output_dir / f"page_{page_number:03d}.png"
        pix.save(output_path)
        print(f"Saved: {output_path}")

    doc.close()
    print(f"Done. Extracted pages saved to: {output_dir}")


if __name__ == "__main__":
    main()
