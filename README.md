# olympflow-2.0

Пайплайн для извлечения, структурирования и первичного анализа олимпиадных задач по физике из PDF-сборника.

## О проекте

Проект решает задачу построения воспроизводимого датасета олимпиадных задач из неструктурированного PDF-источника. Основной сценарий работы:

1. рендеринг страниц исходного сборника;
2. OCR страниц с помощью Mistral OCR;
3. разбиение OCR-текста на отдельные задачи;
4. склейка задач, продолжающихся на следующей странице;
5. сборка финального task-level датасета;
6. построение weak labels по локальным группам задач;
7. вычисление текстовых и dense-признаков;
8. оценка retrieval-baselines;
9. кластеризация задач по смысловой близости.

Итоговый репозиторий приведён к виду публичной финальной версии: в рабочей части оставлены только используемые скрипты, финальные артефакты и содержательные результаты, а ранние эксперименты и промежуточные материалы вынесены в архив.

## Итоговые артефакты

Основной task-level датасет:
- `data/dataset/dataset.jsonl`
- `data/dataset/dataset.json`
- `data/dataset/dataset.csv`
- `data/dataset/dataset_summary.json`

Weak-labeled grouped датасет:
- `data/dataset_grouped/dataset_grouped.jsonl`
- `data/dataset_grouped/dataset_grouped.json`
- `data/dataset_grouped/dataset_grouped.csv`

TF-IDF признаки и retrieval baseline:
- `data/features/tfidf/summary.json`
- `data/features/tfidf/neighbors.json`
- `data/features/tfidf/neighbors.jsonl`
- `data/features/tfidf/neighbors.csv`
- `data/features/tfidf/examples.txt`

Dense embeddings:
- `data/features/dense/embeddings.npy`
- `data/features/dense/meta.json`
- `data/features/dense/records_with_embedding_index.jsonl`

Retrieval-метрики:
- `data/metrics/tfidf/summary.json`
- `data/metrics/tfidf/per_query_metrics.json`
- `data/metrics/dense/summary.json`
- `data/metrics/dense/per_query_metrics.json`

Кластеризация:
- `data/clusters/clustering_summary.json`
- `data/clusters/best_dense_agglomerative/clusters.json`
- `data/clusters/best_dense_agglomerative/records_with_clusters.jsonl`
- `data/clusters/best_dense_agglomerative/clusters_readable.txt`
- `data/clusters/best_dense_agglomerative/summary.json`

## Текущее состояние проекта

Сейчас проект покрывает полный пайплайн от PDF до финальных кластеров задач.

Что сделано:
- извлечены страницы с задачами из `phys_book.pdf` в диапазоне `2..177`;
- выполнен OCR корпуса страниц через `Mistral OCR`;
- OCR-текст разрезан на отдельные задачи;
- реализована склейка задач, продолжающихся между соседними страницами;
- собран финальный датасет задач;
- построена weak-разметка по последовательным четвёркам внутри разделов;
- построен TF-IDF baseline для поиска похожих задач;
- построены dense embeddings;
- посчитаны retrieval-метрики для TF-IDF и dense-представлений;
- запущены baseline-эксперименты по кластеризации;
- экспортирован лучший вариант кластеров.

Что не входит в финальный рабочий пайплайн:
- локальный OCR на Tesseract;
- извлечение и обработка рисунков из условий;
- ранние exploratory-скрипты и промежуточные артефакты.

Они сохранены в `archive/pre_public_release/`.

## Структура репозитория

```text
.
├── data
│   ├── raw/                     # исходный PDF
│   ├── dataset/                 # финальный task-level датасет
│   ├── dataset_grouped/         # grouped weak-labeled датасет
│   ├── features/                # признаки и retrieval-артефакты
│   │   ├── dense/
│   │   └── tfidf/
│   ├── metrics/                 # метрики retrieval
│   │   ├── dense/
│   │   └── tfidf/
│   └── clusters/                # результаты кластеризации
├── scripts
│   ├── 01_pdf/                  # рендеринг страниц из PDF
│   ├── 02_ocr/                  # OCR, split, merge across pages
│   ├── 03_dataset/              # сборка датасета и grouped-версии
│   ├── 04_embeddings/           # dense и TF-IDF retrieval
│   └── 05_clustering/           # baseline-кластеризация и экспорт
├── archive/pre_public_release/  # архив старых и экспериментальных материалов
└── src/olympflow/               # пакетная структура проекта
```

## Скрипты пайплайна

### 1. PDF
- `scripts/01_pdf/render_book_pages.py` — рендеринг страниц книги в изображения для OCR.

### 2. OCR
- `scripts/02_ocr/run_mistral_ocr.py` — OCR страниц через Mistral OCR.
- `scripts/02_ocr/split_pages_into_tasks.py` — разрезание page-level OCR текста на задачи.
- `scripts/02_ocr/merge_cross_page_tasks.py` — склейка задач, продолжающихся на следующей странице.

### 3. Dataset
- `scripts/03_dataset/build_final_dataset.py` — сборка финального task-level датасета.
- `scripts/03_dataset/build_grouped_dataset.py` — построение grouped weak-labeled версии датасета.

### 4. Embeddings and retrieval
- `scripts/04_embeddings/build_tfidf_baseline.py` — построение TF-IDF baseline.
- `scripts/04_embeddings/evaluate_tfidf_baseline.py` — оценка TF-IDF retrieval.
- `scripts/04_embeddings/build_dense_embeddings.py` — построение dense embeddings с асинхронными batched API-запросами.
- `scripts/04_embeddings/evaluate_dense_embeddings.py` — оценка dense retrieval.

### 5. Clustering
- `scripts/05_clustering/run_clustering_baselines.py` — запуск baseline-экспериментов по кластеризации.
- `scripts/05_clustering/export_best_clusters.py` — экспорт лучшего варианта кластеров в удобном виде.

## Базовые результаты

TF-IDF baseline на weak labels по последовательным четвёркам показывает, что даже простое текстовое представление уже хорошо восстанавливает локальную тематическую близость задач.

Dense embeddings дают более сильное представление для downstream-задач retrieval и кластеризации и используются в лучшем сохранённом варианте кластеров:
- `data/clusters/best_dense_agglomerative/`

Точные численные метрики вынесены в:
- `data/metrics/tfidf/summary.json`
- `data/metrics/dense/summary.json`
- `data/clusters/clustering_summary.json`

## Запуск

Установка зависимостей:

```bash
uv sync
```

Примеры запуска основных этапов:

```bash
python scripts/01_pdf/render_book_pages.py
python scripts/02_ocr/run_mistral_ocr.py
python scripts/02_ocr/split_pages_into_tasks.py
python scripts/02_ocr/merge_cross_page_tasks.py
python scripts/03_dataset/build_final_dataset.py
python scripts/03_dataset/build_grouped_dataset.py
python scripts/04_embeddings/build_tfidf_baseline.py
python scripts/04_embeddings/evaluate_tfidf_baseline.py
python scripts/04_embeddings/build_dense_embeddings.py
python scripts/04_embeddings/evaluate_dense_embeddings.py
python scripts/05_clustering/run_clustering_baselines.py
python scripts/05_clustering/export_best_clusters.py
```

## Технологии

- Python 3.12
- uv
- PyMuPDF
- Mistral OCR API
- asyncio
- httpx
- NumPy
- scikit-learn
- Polars
- Matplotlib

## Архив

Папка `archive/pre_public_release/` содержит:
- ранние версии скриптов;
- локальный OCR и его сравнение с Mistral;
- обработку рисунков;
- промежуточные датасеты;
- вспомогательные аннотации и эксперименты, не вошедшие в финальный публичный пайплайн.

Эти материалы сохранены для воспроизводимости истории проекта, но не являются частью основной рабочей версии репозитория.
