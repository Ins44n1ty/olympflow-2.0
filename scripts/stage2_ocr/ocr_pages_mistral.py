from pathlib import Path
import base64
import json
import re
import time

from dotenv import load_dotenv
from mistralai.client import Mistral
import os


START_PAGE = 2
END_PAGE = 177
MODEL_NAME = "mistral-ocr-latest"
SLEEP_SECONDS = 0.2


def extract_page_number(path: Path) -> int | None:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def image_path_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in .env")

    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "interim" / "pages" / "selected"
    text_output_dir = project_root / "data" / "interim" / "ocr_mistral" / "txt"
    meta_output_dir = project_root / "data" / "interim" / "ocr_mistral" / "meta"
    raw_output_dir = project_root / "data" / "interim" / "ocr_mistral" / "raw"

    text_output_dir.mkdir(parents=True, exist_ok=True)
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("page_*.png"))
    image_paths = [
        path
        for path in image_paths
        if (page := extract_page_number(path)) is not None and START_PAGE <= page <= END_PAGE
    ]

    if not image_paths:
        raise FileNotFoundError(f"No page images found in {input_dir}")

    client = Mistral(api_key=api_key)

    print(f"Found {len(image_paths)} page images")

    for image_path in image_paths:
        page_number = extract_page_number(image_path)
        if page_number is None:
            continue

        txt_path = text_output_dir / f"page_{page_number:03d}.txt"
        meta_path = meta_output_dir / f"page_{page_number:03d}.json"
        raw_path = raw_output_dir / f"page_{page_number:03d}.json"

        if txt_path.exists() and meta_path.exists() and raw_path.exists():
            print(f"Skip page {page_number}: already processed")
            continue

        print(f"Processing page {page_number}")

        data_url = image_path_to_data_url(image_path)

        response = client.ocr.process(
            model=MODEL_NAME,
            document={
                "type": "image_url",
                "image_url": data_url,
            },
            confidence_scores_granularity="page",
        )

        response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        save_json(raw_path, response_dict)

        pages = response_dict.get("pages", [])
        markdown = "\n\n".join(page.get("markdown", "") for page in pages).strip()
        save_text(txt_path, markdown)

        meta = {
            "page_number": page_number,
            "image_file": image_path.name,
            "text_file": txt_path.name,
            "model": MODEL_NAME,
            "char_count": len(markdown),
            "page_count_in_response": len(pages),
            "page_confidence_scores": [
                {
                    "index": page.get("index"),
                    "confidence_scores": page.get("confidence_scores"),
                }
                for page in pages
            ],
        }
        save_json(meta_path, meta)

        print(f"Saved text: {txt_path}")
        print(f"Saved meta: {meta_path}")
        print(f"Saved raw: {raw_path}")

        time.sleep(SLEEP_SECONDS)

    print("Done.")


if __name__ == "__main__":
    main()
