# olympflow-2.0 experiments

Этот файл содержит численные результаты и основные выводы по retrieval и clustering экспериментам.

README.md описывает архитектуру проекта и общий пайплайн. Здесь собраны именно экспериментальные результаты, их интерпретация и ссылки на соответствующие артефакты.

## Evaluation setup

Основная оценка проводится на grouped weak-labeled датасете:

- `data/dataset_grouped/dataset_grouped.jsonl`
- `data/dataset_grouped/dataset_grouped.json`
- `data/dataset_grouped/dataset_grouped.csv`

Weak labels построены по локальным группам задач. Это не ручная semantic gold-разметка, а pseudo-labels, которые отражают локальную тематико-лексическую близость задач внутри исходного сборника.

Поэтому все метрики нужно трактовать как согласование с текущей weak-label постановкой, а не как абсолютное качество смыслового поиска или смысловой кластеризации.

## Retrieval results

Сравнивались четыре retrieval-подхода:

- TF-IDF;
- BM25;
- Mistral dense embeddings;
- OpenAI dense embeddings.

Итоговая таблица:

| method | precision@1 | precision@3 | precision@5 | precision@10 | recall@1 | recall@3 | recall@5 | recall@10 | MRR | MAP | LRAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.701513 | 0.586887 | 0.375241 | 0.198487 | 0.234296 | 0.588262 | 0.626777 | 0.662999 | 0.789453 | 0.595507 | 0.595507 |
| Mistral dense | 0.693260 | 0.565337 | 0.370289 | 0.203301 | 0.231774 | 0.566713 | 0.618524 | 0.679046 | 0.783012 | 0.588709 | 0.588709 |
| BM25 | 0.726648 | 0.623626 | 0.398077 | 0.211126 | 0.332418 | 0.834478 | 0.891712 | 0.950549 | 0.815008 | 0.777160 | 0.777160 |
| OpenAI dense | 0.663462 | 0.511905 | 0.332143 | 0.183379 | 0.221841 | 0.513278 | 0.554945 | 0.612637 | 0.740361 | 0.697736 | 0.697736 |

Основной результат: BM25 остаётся лучшим retrieval baseline на текущей weak-label постановке.

Особенно заметен прирост BM25 по `recall@10`:

- BM25: `0.950549`;
- TF-IDF: `0.662999`;
- Mistral dense: `0.679046`;
- OpenAI dense: `0.612637`.

OpenAI dense embeddings на модели `text-embedding-3-small` не улучшили retrieval-качество относительно Mistral dense и sparse baselines. При этом OpenAI dense показал более высокий `MAP/LRAP`, чем Mistral dense, но уступил по `precision@k`, `recall@k` и `MRR`.

Это усиливает главный вывод: текущая weak-label постановка особенно хорошо согласована с lexical similarity, поэтому sparse-подходы оказываются сильнее dense-представлений.

Основные файлы:

- `data/metrics/tfidf/summary.json`
- `data/metrics/dense/summary.json`
- `data/metrics/bm25/summary.json`
- `data/metrics/openai_dense/summary.json`
- `reports/retrieval/retrieval_metrics_summary.md`
- `reports/retrieval/retrieval_metrics_table.csv`
- `reports/retrieval/precision_at_k.png`
- `reports/retrieval/recall_at_k.png`
- `reports/retrieval/ranking_metrics.png`

## OpenAI dense baseline

OpenAI dense embeddings были построены через:

- `scripts/04_embeddings/build_openai_embeddings.py`;
- `scripts/04_embeddings/evaluate_openai_embeddings.py`.

Использованная модель:

- `text-embedding-3-small`.

Артефакты:

- `data/features/openai_dense/embeddings.npy`
- `data/features/openai_dense/meta.json`
- `data/features/openai_dense/records_with_embedding_index.jsonl`
- `data/features/openai_dense/neighbors.json`
- `data/features/openai_dense/neighbors.jsonl`
- `data/features/openai_dense/neighbors.csv`
- `data/metrics/openai_dense/config.json`
- `data/metrics/openai_dense/per_query_metrics.json`
- `data/metrics/openai_dense/summary.json`

Технические детали:

- embeddings строятся асинхронно;
- используется `asyncio`;
- используется `httpx`;
- запросы отправляются батчами;
- число одновременных запросов ограничивается;
- ключ и endpoint подтягиваются из `.env`.

## Retrieval comparison and error analysis

Был добавлен отдельный сравнительный анализ retrieval-методов:

- `scripts/04_embeddings/compare_retrieval_methods.py`

Он считает:

- top-k hit rates;
- top-k Jaccard overlap между методами;
- случаи, где один метод выигрывает у другого;
- случаи, где все методы ошибаются;
- markdown-отчёты с конкретными примерами задач.

### Top-1 case counts

| case type | count |
|---|---:|
| BM25 wins over dense | 37 |
| dense wins over BM25 | 12 |
| TF-IDF wins over dense | 29 |
| dense wins over TF-IDF | 23 |
| BM25 wins over TF-IDF | 25 |
| TF-IDF wins over BM25 | 6 |
| all methods hit | 480 |
| all methods fail | 182 |

### Top-k Jaccard overlap

| pair | Jaccard@1 | Jaccard@3 | Jaccard@5 | Jaccard@10 |
|---|---:|---:|---:|---:|
| TF-IDF vs Mistral dense | 0.756868 | 0.676511 | 0.463681 | 0.377718 |
| TF-IDF vs BM25 | 0.875000 | 0.781044 | 0.609688 | 0.555851 |
| Mistral dense vs BM25 | 0.777473 | 0.694643 | 0.492560 | 0.421381 |

Интерпретация:

- TF-IDF и BM25 ближе друг к другу, чем Mistral dense к sparse-методам.
- BM25 и TF-IDF хорошо работают, когда внутри weak-группы есть почти повторяющиеся формулировки, одинаковые объекты, обозначения и физические ситуации.
- Dense embeddings иногда возвращают физически похожие задачи из другой weak-группы. В текущем протоколе это считается ошибкой, хотя содержательно такой сосед может быть релевантным.
- Случаи `all fail` часто показывают ограничение weak labels: найденная задача может быть очень похожей, но находиться в другой локальной группе.

Основные файлы:

- `data/metrics/comparison/retrieval_comparison_summary.json`
- `data/metrics/comparison/retrieval_topk_overlap.csv`
- `reports/retrieval/retrieval_comparison_summary.md`
- `reports/retrieval/retrieval_analysis_notes.md`
- `reports/retrieval/error_analysis_bm25_wins_over_dense.md`
- `reports/retrieval/error_analysis_dense_wins_over_bm25.md`
- `reports/retrieval/error_analysis_tfidf_wins_over_dense.md`
- `reports/retrieval/error_analysis_all_fail.md`

## Clustering results

Сравнивались sparse и dense представления для кластеризации.

Новый лучший sparse baseline:

- representation: TF-IDF;
- dimensionality reduction: TruncatedSVD;
- clustering: KMeans;
- number of clusters: `183`, по числу weak groups.

Итоговое сравнение:

| method | n_clusters | ARI | NMI | homogeneity | completeness | V-measure | silhouette cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf_svd_kmeans | 183 | 0.623575 | 0.922664 | 0.918593 | 0.926772 | 0.922664 | 0.721749 |
| tfidf_svd_agglomerative | 183 | 0.605400 | 0.922042 | 0.915607 | 0.928567 | 0.922042 | 0.720988 |
| tfidf + agglomerative | 183 | 0.507817 | 0.907352 | 0.896951 | 0.917997 | 0.907352 | NA |
| dense_mistral + agglomerative | 183 | 0.539734 | 0.903987 | 0.898006 | 0.910048 | 0.903987 | NA |
| dense_mistral + kmeans | 183 | 0.486780 | 0.889802 | 0.881233 | 0.898540 | 0.889802 | NA |
| tfidf + kmeans | 183 | 0.485106 | 0.887302 | 0.880353 | 0.894362 | 0.887302 | NA |

Основной результат: `tfidf_svd_kmeans` стал лучшим clustering baseline по согласованию с weak labels.

Это согласуется с retrieval-экспериментами: текущая weak-label постановка сильно согласована с sparse лексическими признаками.

Основные файлы:

- `data/clusters/clustering_summary.json`
- `data/clusters/sparse_clustering_summary.json`
- `data/clusters/tfidf_clustering/tfidf_svd_kmeans/`
- `data/clusters/tfidf_clustering/tfidf_svd_agglomerative/`
- `reports/clustering/clustering_comparison.md`

## Cluster consistency

Для лучшего варианта `tfidf_svd_kmeans` была проверена устойчивость к разным random seeds и подвыборкам.

Результаты:

| metric | value |
|---|---:|
| mean pairwise ARI | 0.873194 |
| mean pairwise NMI | 0.978900 |
| min pairwise ARI | 0.837930 |
| min pairwise NMI | 0.973036 |
| mean subsample ARI | 0.880546 |
| mean subsample NMI | 0.980707 |

Размеры кластеров для seed `42`:

| statistic | value |
|---|---:|
| min size | 1 |
| max size | 10 |
| mean size | 3.978142 |
| median size | 4 |

Интерпретация:

- кластеризация стабильна при разных random seeds;
- стабильность сохраняется на подвыборках;
- медианный размер кластера равен `4`, что естественно для текущей weak-label постановки по локальным четвёркам задач;
- найденная структура не выглядит случайным артефактом одного запуска.

Основные файлы:

- `data/clusters/consistency/consistency_summary.json`
- `data/clusters/consistency/pairwise_ari_matrix.csv`
- `data/clusters/consistency/pairwise_nmi_matrix.csv`
- `reports/clustering/cluster_consistency_summary.md`
- `reports/clustering/cluster_consistency_ari_heatmap.png`
- `reports/clustering/cluster_consistency_nmi_heatmap.png`
- `reports/clustering/cluster_size_distribution.png`

## Cluster quality analysis

Был добавлен отдельный анализ кластеров размера `>= 4`.

Цель этого анализа — проверить, насколько крупные кластеры выглядят адекватно с точки зрения weak labels, годов и разделов. Для кластеров размера `4` основная sanity check-идея такая: если кластер действительно соответствует локальной группе задач, то задачи часто должны быть из одного года. Для кластеров размера `> 4` важнее понять, является ли кластер содержательным повторяющимся шаблоном или слишком широкой смесью разных тем.

Анализ проводился для трёх основных clustering variants:

| method | records | clusters | clusters size >= 4 | flagged clusters | largest cluster |
|---|---:|---:|---:|---:|---:|
| dense_agglomerative | 728 | 183 | 135 | 16 | 9 |
| tfidf_svd_kmeans | 728 | 183 | 144 | 5 | 10 |
| tfidf_svd_agglomerative | 728 | 183 | 140 | 6 | 10 |

Главный вывод: `tfidf_svd_kmeans` выглядит наиболее аккуратно среди сравниваемых методов по числу flagged clusters. У него больше кластеров размера `>= 4`, но при этом заметно меньше подозрительных кластеров, чем у dense agglomerative.

Важно, что flagged cluster не означает автоматически плохой кластер. Это только сигнал, что кластер стоит проверить вручную. Например, большой кластер может быть хорошим, если он собирает повторяющийся физический шаблон из разных лет.

Основные файлы:

- `reports/clustering/cluster_quality_analysis.md`
- `reports/clustering/dense_agglomerative_cluster_quality_analysis.md`
- `reports/clustering/tfidf_svd_kmeans_cluster_quality_analysis.md`
- `reports/clustering/tfidf_svd_agglomerative_cluster_quality_analysis.md`
- `reports/clustering/dense_agglomerative_cluster_size_distribution.png`
- `reports/clustering/tfidf_svd_kmeans_cluster_size_distribution.png`
- `reports/clustering/tfidf_svd_agglomerative_cluster_size_distribution.png`
- `data/clusters/cluster_quality/cluster_quality_summary.json`

## Cluster temporal dynamics

Для кластеров размера `> 4` был добавлен отдельный анализ временной динамики.

Цель — понять, как повторяются похожие задачи во времени:

- `one_year`: все задачи в кластере из одного года;
- `one_time_burst`: задачи сосредоточены в узком интервале соседних лет;
- `multi_year`: задачи встречаются в нескольких годах без явной периодичности;
- `recurring`: задачи распределены по более широкому временному интервалу;
- `periodic_candidate`: годы идут почти с регулярными промежутками.

Результаты:

| method | large clusters | one_year | one_time_burst | multi_year | recurring | periodic_candidate | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_agglomerative | 36 | 3 | 6 | 25 | 1 | 0 | 1 |
| tfidf_svd_kmeans | 27 | 6 | 7 | 13 | 0 | 1 | 0 |
| tfidf_svd_agglomerative | 30 | 4 | 10 | 15 | 1 | 0 | 0 |

Главный вывод: большая часть крупных кластеров не является строго периодической. Чаще встречается либо одноразовая концентрация похожих задач в одном или соседних годах, либо более широкое multi-year повторение похожего шаблона.

`tfidf_svd_kmeans` снова выглядит наиболее аккуратно: у него меньше крупных кластеров, больше `one_year` случаев и нет `unknown`. Это согласуется с предыдущими clustering-метриками и quality analysis.

Основные файлы:

- `reports/clustering/cluster_temporal_dynamics.md`
- `reports/clustering/dense_agglomerative_cluster_temporal_dynamics.md`
- `reports/clustering/tfidf_svd_kmeans_cluster_temporal_dynamics.md`
- `reports/clustering/tfidf_svd_agglomerative_cluster_temporal_dynamics.md`
- `reports/clustering/dense_agglomerative_cluster_year_span_distribution.png`
- `reports/clustering/tfidf_svd_kmeans_cluster_year_span_distribution.png`
- `reports/clustering/tfidf_svd_agglomerative_cluster_year_span_distribution.png`
- `data/clusters/temporal_dynamics/cluster_temporal_dynamics_summary.json`

### Topic-level temporal dynamics

Так как отдельные кластеры обычно небольшие, строгую периодичность внутри одного кластера найти трудно. Поэтому дополнительно был проведён topic-level temporal analysis: задачи агрегируются не по отдельным кластерам, а по автоматически интерпретированным темам из `tfidf_svd_kmeans`.

На этом уровне временная динамика становится заметнее: многие физические темы возвращаются в разные годы и покрывают длинный временной интервал.

| topic | tasks | clusters | years | first year | last year | span | pattern |
|---|---:|---:|---:|---:|---:|---:|---|
| геометрическая оптика | 104 | 26 | 13 | 1991 | 2004 | 13 | long_term_recurring |
| электростатика | 95 | 24 | 13 | 1992 | 2004 | 12 | long_term_recurring |
| термодинамика | 89 | 22 | 12 | 1991 | 2003 | 12 | long_term_recurring |
| механика: движение и силы | 68 | 17 | 11 | 1991 | 2004 | 13 | long_term_recurring |
| электрические цепи | 54 | 15 | 11 | 1991 | 2003 | 12 | long_term_recurring |
| колебания и пружины | 44 | 11 | 10 | 1992 | 2004 | 12 | long_term_recurring |
| орбитальное движение и гравитация | 31 | 9 | 6 | 1991 | 2003 | 12 | long_term_recurring |
| гидростатика | 25 | 7 | 6 | 1992 | 2004 | 12 | long_term_recurring |
| магнетизм и электромагнитная индукция | 16 | 5 | 5 | 1993 | 2004 | 11 | recurring |
| механика: импульс и столкновения | 14 | 4 | 3 | 1993 | 2004 | 11 | multi_year |

Главный вывод: строгой периодичности на уровне отдельных кластеров почти не видно, потому что кластеры маленькие. Однако на уровне автоматически интерпретированных тем повторяемость проявляется хорошо: оптика, электростатика, термодинамика, механика, электрические цепи, колебания и гравитационные задачи возвращаются на протяжении большого промежутка лет.

Это означает, что корпус имеет выраженную долгосрочную тематическую структуру: конкретные формулировки и локальные шаблоны меняются, но крупные физические темы регулярно возвращаются.

Основные файлы:

- `reports/clustering/topic_temporal_dynamics.md`
- `reports/clustering/topic_year_coverage.png`
- `reports/clustering/topic_year_heatmap.png`
- `data/clusters/temporal_dynamics/topic_temporal_dynamics.json`

## Cluster interpretation

Для лучшего варианта `tfidf_svd_kmeans` была добавлена автоматическая интерпретация кластеров.

Для каждого кластера сохраняются:

- top TF-IDF terms;
- guessed topic;
- weak-group distribution;
- representative tasks closest to the cluster centroid.

Примеры автоматически найденных тем:

- электрические цепи;
- геометрическая оптика;
- термодинамика;
- орбитальное движение и гравитация;
- электростатика;
- колебания и пружины;
- гидростатика.

Основные файлы:

- `reports/clustering/cluster_interpretation.md`
- `data/clusters/tfidf_clustering/tfidf_svd_kmeans/cluster_interpretation.json`

## Main conclusions

1. BM25 является лучшим retrieval baseline на текущей weak-label постановке.
2. TF-IDF остаётся сильным sparse baseline и хорошо согласуется с локальной природой weak labels.
3. Mistral dense embeddings не превосходят sparse retrieval, но дают альтернативную семантическую структуру соседства.
4. OpenAI dense embeddings на `text-embedding-3-small` также не превосходят sparse retrieval и не улучшают результат относительно Mistral dense по большинству top-k метрик.
5. Лучший clustering baseline сейчас — `tfidf_svd_kmeans`.
6. Кластеризация устойчива к random seed и подвыборкам.
7. Автоматическая интерпретация кластеров показывает, что многие кластеры имеют понятную физическую тематику.
8. Метрики относительно weak labels нужно трактовать осторожно: weak labels являются локальными pseudo-labels, а не ручной semantic gold standard.
9. На уровне отдельных кластеров строгая периодичность почти не проявляется, но topic-level temporal analysis показывает долгосрочную повторяемость крупных физических тем: оптики, электростатики, термодинамики, механики, электрических цепей, колебаний и гравитационных задач.

## What these results mean

Главный содержательный вывод проекта на текущем этапе: корпус задач хорошо структурируется через лексико-тематическую близость.

Это объясняет, почему sparse retrieval и sparse clustering оказались настолько сильными. Внутри локальных групп часто повторяются физические объекты, обозначения, параметры, формулировочные шаблоны и близкие постановки задач. BM25 особенно хорошо использует эту структуру.

Dense embeddings остаются полезными как альтернативное семантическое представление, но текущие weak labels не всегда справедливо оценивают dense retrieval. Dense-модель может найти содержательно похожую задачу из другой локальной группы, и такая пара будет считаться ошибкой, хотя физически она может быть близкой.

Эксперимент с OpenAI dense embeddings показывает, что проблема не сводится только к конкретной модели Mistral embeddings. Даже другая современная embedding-модель не дала выигрыша над sparse baseline в этой постановке.

Поэтому результаты лучше трактовать не как доказательство того, что sparse-представления абсолютно лучше dense-представлений, а как доказательство того, что текущая weak-label постановка особенно хорошо согласована с lexical similarity.
