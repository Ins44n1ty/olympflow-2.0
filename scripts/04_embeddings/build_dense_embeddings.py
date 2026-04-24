from pathlib import Path
import asyncio
import json
import os
import re

import httpx
import numpy as np
from dotenv import load_dotenv


MODEL_NAME = "mistral-embed"
BATCH_SIZE = 32
MAX_CONCURRENT_REQUESTS = 8
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60.0


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_texts(records: list[dict]) -> list[str]:
    return [normalize_text(record.get("raw_text", "")) for record in records]


def chunk_list(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def post_embeddings(
    client: httpx.AsyncClient,
    api_key: str,
    texts: list[str],
) -> list[list[float]]:
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "input": texts,
    }

    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    return [item["embedding"] for item in data["data"]]


async def embed_batch(
    client: httpx.AsyncClient,
    api_key: str,
    batch_index: int,
    batch_texts: list[str],
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[list[float]]]:
    async with semaphore:
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                embeddings = await post_embeddings(client, api_key, batch_texts)
                print(f"Batch {batch_index}: ok ({len(batch_texts)} texts)")
                return batch_index, embeddings
            except Exception as exc:
                last_error = exc
                print(f"Batch {batch_index}: fail attempt {attempt}/{MAX_RETRIES} -> {exc}")
                await asyncio.sleep(attempt)

        raise last_error


async def main_async() -> None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in .env")

    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "dataset_grouped" / "dataset_grouped.jsonl"
    output_dir = project_root / "data" / "features" / "dense"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_npy_path = output_dir / "embeddings.npy"
    records_jsonl_path = output_dir / "records_with_embedding_index.jsonl"
    meta_path = output_dir / "meta.json"

    records = load_jsonl(input_path)
    texts = build_texts(records)
    batches = chunk_list(texts, BATCH_SIZE)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = httpx.Timeout(TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            embed_batch(
                client=client,
                api_key=api_key,
                batch_index=i,
                batch_texts=batch,
                semaphore=semaphore,
            )
            for i, batch in enumerate(batches)
        ]

        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])

    all_embeddings = []
    for _, batch_embeddings in results:
        all_embeddings.extend(batch_embeddings)

    if len(all_embeddings) != len(records):
        raise RuntimeError(
            f"Embeddings count mismatch: {len(all_embeddings)} != {len(records)}"
        )

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    np.save(embeddings_npy_path, embeddings_array)

    with records_jsonl_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            out = {
                "embedding_index": idx,
                "task_id": record["task_id"],
                "task_number": record["task_number"],
                "group_id": record.get("group_id"),
                "group_pos": record.get("group_pos"),
                "page_start": record.get("page_start"),
                "page_end": record.get("page_end"),
                "raw_text": record.get("raw_text"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "task_count": len(records),
        "embedding_dim": int(embeddings_array.shape[1]),
        "batch_size": BATCH_SIZE,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "input_path": str(input_path),
        "embeddings_path": str(embeddings_npy_path),
        "records_path": str(records_jsonl_path),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved: {embeddings_npy_path}")
    print(f"Saved: {records_jsonl_path}")
    print(f"Saved: {meta_path}")
    print(f"Task count: {len(records)}")
    print(f"Embedding dim: {embeddings_array.shape[1]}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
