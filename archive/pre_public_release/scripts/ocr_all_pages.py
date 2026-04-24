from pathlib import Path
import json
import re

from PIL import Image
import pytesseract


START_PAGE = 2
END_PAGE = 177
TESSERACT_LANG = "rus+eng"
TESSERACT_CONFIG = "--psm 6"


def extract_page_number(path: Path) -> int | None:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "interim" / "pages" / "selected"
    text_output_dir = project_root / "data" / "interim" / "ocr_text" / "txt"
    meta_output_dir = project_root / "data" / "interim" / "ocr_text" / "meta"

    text_output_dir.mkdir(parents=True, exist_ok=True)
    meta_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("page_*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    filtered_paths = []
    for image_path in image_paths:
        page_number = extract_page_number(image_path)
        if page_number is None:
            continue
        if START_PAGE <= page_number <= END_PAGE:
            filtered_paths.append(image_path)

    if not filtered_paths:
        raise FileNotFoundError("No valid page images in the target range")

    print(f"Found {len(filtered_paths)} page images")

    for image_path in filtered_paths:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
        )

        txt_path = text_output_dir / f"{image_path.stem}.txt"
        meta_path = meta_output_dir / f"{image_path.stem}.json"

        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "image_file": image_path.name,
            "text_file": txt_path.name,
            "page_number": extract_page_number(image_path),
            "text_length": len(text),
            "word_count_approx": len(text.split()),
            "is_empty": len(text.strip()) == 0,
            "tesseract_lang": TESSERACT_LANG,
            "tesseract_config": TESSERACT_CONFIG,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Saved text: {txt_path}")
        print(f"Saved meta: {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
