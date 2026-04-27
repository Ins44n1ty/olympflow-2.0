# olympflow-2.0

`olympflow-2.0` — воспроизводимый пайплайн для извлечения, структурирования и первичного анализа олимпиадных задач по физике из PDF-сборника.

Проект берёт неструктурированный PDF-источник, извлекает из него текст задач, собирает датасет и строит базовые инструменты для поиска похожих задач и кластеризации. Основной фокус проекта — не только получить набор задач, но и сделать аккуратную исследовательскую инфраструктуру: данные, промежуточные артефакты, метрики, отчёты и скрипты должны быть разложены по понятным этапам.

Основной источник: `data/raw/pdf/phys_book.pdf`, страницы `2..177`.

## Что делает проект

Пайплайн состоит из нескольких последовательных этапов:

1. рендеринг страниц PDF в изображения;
2. OCR страниц через Mistral OCR;
3. разбиение OCR-текста на отдельные задачи;
4. склейка задач, которые продолжаются на соседней странице;
5. сборка финального task-level датасета;
6. построение grouped weak-labeled датасета;
7. построение retrieval baselines;
8. построение dense embeddings;
9. оценка качества retrieval;
10. кластеризация задач;
11. анализ устойчивости кластеров;
12. автоматическая интерпретация кластеров.

Идея проекта: построить основу для анализа корпуса олимпиадных задач по физике. После сборки датасета задачи можно сравнивать по текстовой похожести, искать близкие формулировки, группировать задачи в кластеры и анализировать повторяющиеся темы.

## Текущее состояние

Сейчас проект находится в состоянии аккуратной публичной версии.

Сделано:

- исходный PDF помещён в рабочую структуру проекта;
- страницы с задачами обработаны через OCR;
- OCR-текст разрезан на отдельные задачи;
- задачи, продолжающиеся между соседними страницами, склеены;
- собран финальный task-level датасет;
- собран grouped weak-labeled датасет;
- реализованы retrieval baselines;
- реализованы dense embeddings через асинхронные API-запросы;
- реализованы скрипты оценки retrieval;
- реализованы скрипты кластеризации;
- добавлены отчёты по сравнению retrieval-методов;
- добавлены отчёты по кластеризации, устойчивости и интерпретации кластеров;
- старые и экспериментальные материалы вынесены в архив.

Не входит в основной рабочий пайплайн:

- локальный OCR на Tesseract;
- извлечение и обработка рисунков из условий;
- старые exploratory-скрипты;
- промежуточные экспериментальные датасеты.

Эти материалы сохранены в `archive/pre_public_release/`.

## Структура репозитория

```text
.
├── README.md
├── EXPERIMENTS.md
├── archive/pre_public_release/
├── configs/
├── data/
│   ├── raw/pdf/
│   ├── dataset/
│   ├── dataset_grouped/
│   ├── features/
│   │   ├── tfidf/
│   │   ├── bm25/
│   │   └── dense/
│   ├── metrics/
│   │   ├── tfidf/
│   │   ├── bm25/
│   │   ├── dense/
│   │   └── comparison/
│   └── clusters/
├── docs/
├── notebooks/
├── reports/
│   ├── retrieval/
│   └── clustering/
├── scripts/
│   ├── 01_pdf/
│   ├── 02_ocr/
│   ├── 03_dataset/
│   ├── 04_embeddings/
│   ├── 05_clustering/
│   └── 06_reports/
├── src/olympflow/
├── tests/
├── pyproject.toml
└── uv.lock
```

## Основные каталоги

`data/raw/pdf/` — исходный PDF-сборник.

`data/dataset/` — финальный task-level датасет в нескольких форматах.

`data/dataset_grouped/` — grouped weak-labeled датасет для retrieval и clustering evaluation.

`data/features/` — признаки и retrieval-артефакты.

`data/metrics/` — метрики retrieval и сравнения методов.

`data/clusters/` — результаты кластеризации, экспортированные кластеры и служебные файлы анализа.

`reports/retrieval/` — таблицы, графики и qualitative error analysis для retrieval.

`reports/clustering/` — сравнение кластеризаций, устойчивость кластеров и автоматическая интерпретация.

`archive/pre_public_release/` — старые версии скриптов, ранние эксперименты, локальный OCR, обработка рисунков и промежуточные материалы.

## Скрипты пайплайна

### 1. PDF

- `scripts/01_pdf/render_book_pages.py` — рендеринг страниц книги в изображения для OCR.

### 2. OCR

- `scripts/02_ocr/run_mistral_ocr.py` — OCR страниц через Mistral OCR.
- `scripts/02_ocr/split_pages_into_tasks.py` — разрезание page-level OCR текста на отдельные задачи.
- `scripts/02_ocr/merge_cross_page_tasks.py` — склейка задач, продолжающихся на следующей странице.

### 3. Dataset

- `scripts/03_dataset/build_final_dataset.py` — сборка финального task-level датасета.
- `scripts/03_dataset/build_grouped_dataset.py` — построение grouped weak-labeled версии датасета.

### 4. Retrieval and embeddings

- `scripts/04_embeddings/build_tfidf_baseline.py` — построение TF-IDF retrieval baseline.
- `scripts/04_embeddings/evaluate_tfidf_baseline.py` — оценка TF-IDF retrieval.
- `scripts/04_embeddings/build_bm25_baseline.py` — построение BM25 retrieval baseline.
- `scripts/04_embeddings/evaluate_bm25_baseline.py` — оценка BM25 retrieval.
- `scripts/04_embeddings/build_dense_embeddings.py` — построение dense embeddings через async batched API-запросы.
- `scripts/04_embeddings/evaluate_dense_embeddings.py` — оценка dense retrieval.
- `scripts/04_embeddings/compare_retrieval_methods.py` — сравнение retrieval-методов и сохранение error analysis.

### 5. Clustering

- `scripts/05_clustering/run_clustering_baselines.py` — базовые эксперименты по TF-IDF и dense clustering.
- `scripts/05_clustering/export_best_clusters.py` — экспорт выбранного варианта кластеров.
- `scripts/05_clustering/run_sparse_clustering.py` — sparse clustering на TF-IDF + SVD.
- `scripts/05_clustering/compare_clustering_results.py` — сравнение clustering baseline.
- `scripts/05_clustering/evaluate_cluster_consistency.py` — проверка устойчивости кластеризации.
- `scripts/05_clustering/interpret_clusters.py` — автоматическая интерпретация кластеров.

### 6. Reports

- `scripts/06_reports/plot_retrieval_metrics.py` — построение таблиц и графиков retrieval-метрик.

## Основные артефакты

Task-level датасет:

- `data/dataset/dataset.jsonl`
- `data/dataset/dataset.json`
- `data/dataset/dataset.csv`
- `data/dataset/dataset_summary.json`

Grouped weak-labeled датасет:

- `data/dataset_grouped/dataset_grouped.jsonl`
- `data/dataset_grouped/dataset_grouped.json`
- `data/dataset_grouped/dataset_grouped.csv`

Retrieval features and metrics:

- `data/features/tfidf/`
- `data/features/bm25/`
- `data/features/dense/`
- `data/metrics/tfidf/`
- `data/metrics/bm25/`
- `data/metrics/dense/`
- `data/metrics/comparison/`

Clustering:

- `data/clusters/clustering_summary.json`
- `data/clusters/best_dense_agglomerative/`
- `data/clusters/tfidf_clustering/`
- `data/clusters/consistency/`

Reports:

- `reports/retrieval/`
- `reports/clustering/`

Подробные численные результаты и выводы по экспериментам вынесены в `EXPERIMENTS.md`.

## Запуск

Установка зависимостей:

```bash
uv sync
```

Полный запуск пайплайна:

```bash
python scripts/01_pdf/render_book_pages.py
python scripts/02_ocr/run_mistral_ocr.py
python scripts/02_ocr/split_pages_into_tasks.py
python scripts/02_ocr/merge_cross_page_tasks.py

python scripts/03_dataset/build_final_dataset.py
python scripts/03_dataset/build_grouped_dataset.py

python scripts/04_embeddings/build_tfidf_baseline.py
python scripts/04_embeddings/evaluate_tfidf_baseline.py

python scripts/04_embeddings/build_bm25_baseline.py
python scripts/04_embeddings/evaluate_bm25_baseline.py

python scripts/04_embeddings/build_dense_embeddings.py
python scripts/04_embeddings/evaluate_dense_embeddings.py

python scripts/04_embeddings/compare_retrieval_methods.py

python scripts/06_reports/plot_retrieval_metrics.py

python scripts/05_clustering/run_clustering_baselines.py
python scripts/05_clustering/export_best_clusters.py

python scripts/05_clustering/run_sparse_clustering.py
python scripts/05_clustering/compare_clustering_results.py
python scripts/05_clustering/evaluate_cluster_consistency.py
python scripts/05_clustering/interpret_clusters.py
```

Быстрый пересчёт аналитической части после уже готового датасета:

```bash
python scripts/04_embeddings/build_bm25_baseline.py
python scripts/04_embeddings/evaluate_bm25_baseline.py
python scripts/04_embeddings/compare_retrieval_methods.py
python scripts/06_reports/plot_retrieval_metrics.py

python scripts/05_clustering/run_sparse_clustering.py
python scripts/05_clustering/compare_clustering_results.py
python scripts/05_clustering/evaluate_cluster_consistency.py
python scripts/05_clustering/interpret_clusters.py
```

## Async API usage

Асинхронность используется в построении dense embeddings:

- `scripts/04_embeddings/build_dense_embeddings.py`

Скрипт использует:

- `asyncio`;
- `httpx`;
- batched API requests;
- ограничение числа одновременных запросов.

Это нужно для более быстрого построения dense embeddings и более аккуратной работы с внешним API.

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
- pandas

## Архив

Папка `archive/pre_public_release/` содержит:

- ранние версии скриптов;
- локальный OCR на Tesseract;
- сравнение локального OCR с Mistral OCR;
- обработку рисунков;
- pilot OCR compare;
- processed annotations;
- промежуточные датасеты;
- вспомогательные материалы, не вошедшие в финальный рабочий пайплайн.

Эти материалы сохранены для истории проекта и воспроизводимости старых экспериментов, но не являются частью основной рабочей версии.
