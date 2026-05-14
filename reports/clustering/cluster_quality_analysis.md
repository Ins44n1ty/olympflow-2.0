# Cluster quality analysis

This report checks clusters with size >= 4 for the main clustering methods.

For clusters with size = 4, the main sanity check is whether tasks come from the same year.
For clusters with size > 4, the report summarizes years, sections, weak groups and suspicious cases for manual inspection.

## Summary

| method | records | clusters | clusters size >= 4 | flagged clusters | largest cluster |
|---|---:|---:|---:|---:|---:|
| dense_agglomerative | 728 | 183 | 135 | 16 | 9 |
| tfidf_svd_kmeans | 728 | 183 | 144 | 5 | 10 |
| tfidf_svd_agglomerative | 728 | 183 | 140 | 6 | 10 |

## Generated detailed reports

- `reports/clustering/dense_agglomerative_cluster_quality_analysis.md`
- `reports/clustering/dense_agglomerative_cluster_size_distribution.png`
- `reports/clustering/tfidf_svd_kmeans_cluster_quality_analysis.md`
- `reports/clustering/tfidf_svd_kmeans_cluster_size_distribution.png`
- `reports/clustering/tfidf_svd_agglomerative_cluster_quality_analysis.md`
- `reports/clustering/tfidf_svd_agglomerative_cluster_size_distribution.png`

## Interpretation notes

- A flagged cluster is not automatically bad. It only means that it deserves manual inspection.
- Clusters with size = 4 are expected to often match one weak group and therefore one year.
- Larger clusters may be meaningful if they collect recurring problem templates across years.
- If a large cluster mixes many years and many sections, it may be too broad or lexically driven.
