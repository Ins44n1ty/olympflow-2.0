from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from time import time

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/dataset_grouped/dataset_grouped.jsonl"
OUTPUT_DIR = ROOT / "data/features/openai_dense"

API_URL = "https://api.openai.com/v1/embeddings"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 64
MAX_CONCURRENT_REQUESTS = 4
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


def load_env(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_record_id(record: dict, index: int) -> str:
    for key in ("task_id", "id", "problem_id", "record_id"):
        if key in record:
            return str(record[key])

    return str(index)


def get_group_id(record: dict) -> str:
    for key in ("group_id", "weak_group_id", "group", "label"):
        if key in record:
            return str(record[key])

    return ""


def get_text(record: dict) -> str:
    parts = []

    for key in ("title", "problem_text", "text", "statement", "condition"):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if parts:
        return "\n".join(parts)

    values = [
        value.strip()
        for value in record.values()
        if isinstance(value, str) and value.strip()
    ]

    return "\n".join(values)


def make_batches(records: list[dict]) -> list[tuple[int, list[dict]]]:
    batches = []

    for start in range(0, len(records), BATCH_SIZE):
        batches.append((start, records[start:start + BATCH_SIZE]))

    return batches


async def fetch_embedding_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    texts: list[str],
) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue

                response.raise_for_status()
                data = response.json()["data"]
                data.sort(key=lambda item: item["index"])

                return [item["embedding"] for item in data]

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt + 1 == MAX_RETRIES:
                    raise

                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    raise RuntimeError("Failed to fetch embedding batch.")


async def build_embeddings(records: list[dict], api_key: str, use_proxy: bool) -> np.ndarray:
    batches = make_batches(records)
    results: list[list[list[float]] | None] = [None] * len(batches)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    proxy = os.environ.get("HTTPS_PROXY") if use_proxy else None

    async with httpx.AsyncClient(proxy=proxy) as client:
        async def run_batch(batch_id: int, start: int, batch: list[dict]) -> None:
            texts = [get_text(record) for record in batch]
            embeddings = await fetch_embedding_batch(
                client=client,
                semaphore=semaphore,
                api_key=api_key,
                texts=texts,
            )

            results[batch_id] = embeddings
            print(f"batch {batch_id + 1}/{len(batches)} saved, start={start}")

        tasks = [
            run_batch(batch_id, start, batch)
            for batch_id, (start, batch) in enumerate(batches)
        ]

        await asyncio.gather(*tasks)

    embeddings = []

    for batch_result in results:
        if batch_result is None:
            raise RuntimeError("Missing batch result.")

        embeddings.extend(batch_result)

    return np.array(embeddings, dtype=np.float32)


def build_records_with_embedding_index(records: list[dict]) -> list[dict]:
    output = []

    for index, record in enumerate(records):
        output.append(
            {
                "embedding_index": index,
                "task_id": get_record_id(record, index),
                "task_number": record.get("task_number"),
                "group_id": get_group_id(record),
            }
        )

    return output


async def main_async() -> None:
    load_env(ROOT / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
    use_proxy = os.environ.get("OPENAI_USE_PROXY", "").lower() in {"1", "true", "yes"}

    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    global API_URL
    API_URL = f"{api_base}/v1/embeddings"

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time()
    records = read_jsonl(INPUT_PATH)
    embeddings = await build_embeddings(records, api_key, use_proxy)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    save_jsonl(
        OUTPUT_DIR / "records_with_embedding_index.jsonl",
        build_records_with_embedding_index(records),
    )

    meta = {
        "method": "openai_dense",
        "model": MODEL,
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "records_count": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "elapsed_seconds": round(time() - started_at, 3),
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
    }

    save_json(OUTPUT_DIR / "meta.json", meta)

    print(f"Saved OpenAI embeddings to {OUTPUT_DIR}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
