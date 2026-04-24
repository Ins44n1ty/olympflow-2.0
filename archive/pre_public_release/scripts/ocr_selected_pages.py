from pathlib import Path
import json

from PIL import Image
import pytesseract


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "interim" / "pages" / "selected"
    text_output_dir = project_root / "data" / "interim" / "ocr_text" / "txt"
    meta_output_dir = project_root / "data" / "interim" / "ocr_text" / "meta"

    text_output_dir.mkdir(parents=True, exist_ok=True)
    meta_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    for image_path in image_paths:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            lang="rus+eng",
            config="--psm 6",
        )

        txt_path = text_output_dir / f"{image_path.stem}.txt"
        meta_path = meta_output_dir / f"{image_path.stem}.json"

        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "image_file": image_path.name,
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

    print("Done.")


if __name__ == "__main__":
    main()
