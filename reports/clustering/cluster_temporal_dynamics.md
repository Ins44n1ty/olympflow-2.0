# Cluster temporal dynamics

This report analyzes clusters with size > 4.

The goal is to check whether larger clusters represent one-time repetitions, multi-year recurring templates, or broad mixed clusters.

## Summary

| method | large clusters | one_year | one_time_burst | multi_year | recurring | periodic_candidate | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_agglomerative | 36 | 3 | 6 | 25 | 1 | 0 | 1 |
| tfidf_svd_kmeans | 27 | 6 | 7 | 13 | 0 | 1 | 0 |
| tfidf_svd_agglomerative | 30 | 4 | 10 | 15 | 1 | 0 | 0 |

## Interpretation

- `one_year`: all tasks in the cluster come from one year.
- `one_time_burst`: tasks are concentrated in a narrow interval of neighboring years.
- `multi_year`: tasks appear in several years, but without a strong recurring pattern.
- `recurring`: tasks are spread across a wider time interval.
- `periodic_candidate`: years are spread with nearly regular gaps, so the cluster may describe a periodic template.

The labels are heuristic. They are meant for fast manual inspection, not as final semantic annotation.

## Generated detailed reports

- `reports/clustering/dense_agglomerative_cluster_temporal_dynamics.md`
- `reports/clustering/dense_agglomerative_cluster_year_span_distribution.png`
- `reports/clustering/tfidf_svd_kmeans_cluster_temporal_dynamics.md`
- `reports/clustering/tfidf_svd_kmeans_cluster_year_span_distribution.png`
- `reports/clustering/tfidf_svd_agglomerative_cluster_temporal_dynamics.md`
- `reports/clustering/tfidf_svd_agglomerative_cluster_year_span_distribution.png`
