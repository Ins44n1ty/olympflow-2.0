#!/usr/bin/env bash

set -e

MODE="${1:-analysis}"

run() {
    echo
    echo "==> $1"
    uv run python "$1"
}

if [ "$MODE" = "full" ]; then
    echo "Running full olympflow pipeline"

    uv sync

    run scripts/01_pdf/render_book_pages.py

    run scripts/02_ocr/run_mistral_ocr.py
    run scripts/02_ocr/split_pages_into_tasks.py
    run scripts/02_ocr/merge_cross_page_tasks.py

    run scripts/03_dataset/build_final_dataset.py
    run scripts/03_dataset/build_grouped_dataset.py

    run scripts/04_embeddings/build_tfidf_baseline.py
    run scripts/04_embeddings/evaluate_tfidf_baseline.py

    run scripts/04_embeddings/build_bm25_baseline.py
    run scripts/04_embeddings/evaluate_bm25_baseline.py

    run scripts/04_embeddings/build_dense_embeddings.py
    run scripts/04_embeddings/evaluate_dense_embeddings.py

    run scripts/04_embeddings/build_openai_embeddings.py
    run scripts/04_embeddings/evaluate_openai_embeddings.py

    run scripts/04_embeddings/compare_retrieval_methods.py
    run scripts/06_reports/plot_retrieval_metrics.py

    run scripts/05_clustering/run_clustering_baselines.py
    run scripts/05_clustering/export_best_clusters.py

    run scripts/05_clustering/run_sparse_clustering.py
    run scripts/05_clustering/compare_clustering_results.py
    run scripts/05_clustering/evaluate_cluster_consistency.py
    run scripts/05_clustering/interpret_clusters.py

    run scripts/05_clustering/analyze_cluster_quality.py
    run scripts/05_clustering/analyze_cluster_temporal_dynamics.py
    run scripts/05_clustering/analyze_topic_temporal_dynamics.py

elif [ "$MODE" = "analysis" ]; then
    echo "Running analysis-only olympflow pipeline"

    run scripts/04_embeddings/build_tfidf_baseline.py
    run scripts/04_embeddings/evaluate_tfidf_baseline.py

    run scripts/04_embeddings/build_bm25_baseline.py
    run scripts/04_embeddings/evaluate_bm25_baseline.py

    run scripts/04_embeddings/build_openai_embeddings.py
    run scripts/04_embeddings/evaluate_openai_embeddings.py

    run scripts/04_embeddings/compare_retrieval_methods.py
    run scripts/06_reports/plot_retrieval_metrics.py

    run scripts/05_clustering/run_sparse_clustering.py
    run scripts/05_clustering/compare_clustering_results.py
    run scripts/05_clustering/evaluate_cluster_consistency.py
    run scripts/05_clustering/interpret_clusters.py

    run scripts/05_clustering/analyze_cluster_quality.py
    run scripts/05_clustering/analyze_cluster_temporal_dynamics.py
    run scripts/05_clustering/analyze_topic_temporal_dynamics.py

else
    echo "Unknown mode: $MODE"
    echo "Usage:"
    echo "  ./run_pipeline.sh analysis"
    echo "  ./run_pipeline.sh full"
    exit 1
fi

echo
echo "Done."
