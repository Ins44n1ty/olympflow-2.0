from pathlib import Path
import base64
import json
import re
import shutil
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral
import os


START_PAGE = 57
END_PAGE = 60


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


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in .env")

    project_root = Path(__file__).resolve().parents[2]
    pages_dir = project_root / "data" / "interim" / "pages" / "selected"
    local_ocr_dir = project_root / "data" / "interim" / "ocr_text" / "txt"

    output_dir = project_root / "data" / "pilot_ocr_compare"
    local_out_dir = output_dir / "local"
    mistral_out_dir = output_dir / "mistral"
    raw_json_dir = output_dir / "raw_json"
    images_out_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    local_out_dir.mkdir(parents=True, exist_ok=True)
    mistral_out_dir.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    client = Mistral(api_key=api_key)

    manifest = []

    image_paths = sorted(pages_dir.glob("page_*.png"))
    image_paths = [
        path
        for path in image_paths
        if (page := extract_page_number(path)) is not None and START_PAGE <= page <= END_PAGE
    ]

    if not image_paths:
        raise FileNotFoundError("No page images found in target range")

    for image_path in image_paths:
        page_number = extract_page_number(image_path)
        if page_number is None:
            continue

        print(f"Processing page {page_number}")

        local_txt_path = local_ocr_dir / f"page_{page_number:03d}.txt"
        if local_txt_path.exists():
            shutil.copy2(local_txt_path, local_out_dir / local_txt_path.name)
            local_text = local_txt_path.read_text(encoding="utf-8")
        else:
            local_text = ""

        shutil.copy2(image_path, images_out_dir / image_path.name)

        data_url = image_path_to_data_url(image_path)

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": data_url,
            },
            confidence_scores_granularity="page",
        )

        response_dict = response.model_dump() if hasattr(response, "model_dump") else response.dict()
        save_json(raw_json_dir / f"page_{page_number:03d}.json", response_dict)

        pages = response_dict.get("pages", [])
        markdown = "\n\n".join(page.get("markdown", "") for page in pages).strip()
        save_text(mistral_out_dir / f"page_{page_number:03d}.md", markdown)

        manifest.append(
            {
                "page_number": page_number,
                "local_char_count": len(local_text),
                "mistral_char_count": len(markdown),
                "local_text_file": str((local_out_dir / f"page_{page_number:03d}.txt").relative_to(project_root)),
                "mistral_text_file": str((mistral_out_dir / f"page_{page_number:03d}.md").relative_to(project_root)),
                "raw_json_file": str((raw_json_dir / f"page_{page_number:03d}.json").relative_to(project_root)),
            }
        )

    save_json(output_dir / "manifest.json", manifest)
    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
