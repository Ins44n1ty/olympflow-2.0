from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
from pathlib import Path
from time import time

import httpx
from dotenv import load_dotenv


START_PAGE = 2
END_PAGE = 177

MODEL_NAME = "mistral-ocr-latest"
MAX_CONCURRENT_REQUESTS = 2
MAX_RETRIES = 12
RETRY_BASE_DELAY = 5.0

API_URL = "https://api.mistral.ai/v1/ocr"


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


async def call_mistral_ocr(
    client: httpx.AsyncClient,
    api_key: str,
    image_path: Path,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "document": {
            "type": "image_url",
            "image_url": image_path_to_data_url(image_path),
        },
        "confidence_scores_granularity": "page",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=180.0,
            )

            if response.status_code == 429:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                print(
                    f"Retry {attempt + 1}/{MAX_RETRIES} for {image_path.name}: "
                    f"HTTP 429, sleep {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code in {500, 502, 503, 504}:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
                print(
                    f"Retry {attempt + 1}/{MAX_RETRIES} for {image_path.name}: "
                    f"HTTP {response.status_code}, sleep {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()

            return response.json()

        except (httpx.TimeoutException, httpx.TransportError) as error:
            if attempt + 1 == MAX_RETRIES:
                raise RuntimeError(f"Failed OCR for {image_path.name}: {error}") from error

            delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(
                f"Retry {attempt + 1}/{MAX_RETRIES} for {image_path.name}: "
                f"{error}, sleep {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Failed OCR for {image_path.name}")


async def process_page(
    client: httpx.AsyncClient,
    api_key: str,
    image_path: Path,
    text_output_dir: Path,
    meta_output_dir: Path,
    raw_output_dir: Path,
) -> bool:
    page_number = extract_page_number(image_path)

    if page_number is None:
        return True

    txt_path = text_output_dir / f"page_{page_number:03d}.txt"
    meta_path = meta_output_dir / f"page_{page_number:03d}.json"
    raw_path = raw_output_dir / f"page_{page_number:03d}.json"

    if txt_path.exists() and meta_path.exists() and raw_path.exists():
        print(f"Skip page {page_number}: already processed")
        return True

    print(f"Processing page {page_number}")

    started_at = time()

    try:
        response_dict = await call_mistral_ocr(
            client=client,
            api_key=api_key,
            image_path=image_path,
        )
    except Exception as error:
        print(f"Failed page {page_number}: {error}")
        return False

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
        "elapsed_seconds": round(time() - started_at, 3),
        "page_confidence_scores": [
            {
                "index": page.get("index"),
                "confidence_scores": page.get("confidence_scores"),
            }
            for page in pages
        ],
    }

    save_json(meta_path, meta)

    print(f"Saved page {page_number}: {txt_path}")

    return True


async def worker(
    worker_id: int,
    queue: asyncio.Queue[Path],
    client: httpx.AsyncClient,
    api_key: str,
    text_output_dir: Path,
    meta_output_dir: Path,
    raw_output_dir: Path,
    failed_pages: list[str],
) -> None:
    while True:
        image_path = await queue.get()

        try:
            ok = await process_page(
                client=client,
                api_key=api_key,
                image_path=image_path,
                text_output_dir=text_output_dir,
                meta_output_dir=meta_output_dir,
                raw_output_dir=raw_output_dir,
            )

            if not ok:
                failed_pages.append(image_path.name)

        finally:
            queue.task_done()


async def main_async() -> None:
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

    queue: asyncio.Queue[Path] = asyncio.Queue()

    for image_path in image_paths:
        queue.put_nowait(image_path)

    failed_pages: list[str] = []

    print(f"Found {len(image_paths)} page images")
    print(f"Workers: {MAX_CONCURRENT_REQUESTS}")

    async with httpx.AsyncClient() as client:
        workers = [
            asyncio.create_task(
                worker(
                    worker_id=i,
                    queue=queue,
                    client=client,
                    api_key=api_key,
                    text_output_dir=text_output_dir,
                    meta_output_dir=meta_output_dir,
                    raw_output_dir=raw_output_dir,
                    failed_pages=failed_pages,
                )
            )
            for i in range(MAX_CONCURRENT_REQUESTS)
        ]

        await queue.join()

        for task in workers:
            task.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    if failed_pages:
        failed_path = raw_output_dir.parent / "failed_pages.json"
        save_json(failed_path, {"failed_pages": sorted(failed_pages)})
        print(f"Failed pages saved to {failed_path}")
        raise RuntimeError(f"Failed pages: {sorted(failed_pages)}")

    print("Done.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
