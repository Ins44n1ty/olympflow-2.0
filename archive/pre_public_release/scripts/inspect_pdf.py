from pathlib import Path

import fitz


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "data" / "raw" / "pdf" / "phys_book.pdf"
    output_dir = project_root / "data" / "interim" / "pages" / "preview"

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(doc)}")

    preview_pages = min(10, len(doc))

    for page_num in range(preview_pages):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        output_path = output_dir / f"page_{page_num + 1:03d}.png"
        pix.save(output_path)
        print(f"Saved: {output_path}")

    doc.close()
    print(f"Done. Preview saved to: {output_dir}")


if __name__ == "__main__":
    main()
