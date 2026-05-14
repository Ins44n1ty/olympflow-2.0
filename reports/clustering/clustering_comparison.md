# Clustering comparison

| method | n_clusters | ARI | NMI | homogeneity | completeness | V-measure | silhouette cosine | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tfidf_svd_kmeans | 183 | 0.631697 | 0.925553 | 0.921622 | 0.929518 | 0.925553 | 0.725231 | new_sparse_clustering |
| tfidf_svd_agglomerative | 183 | 0.605400 | 0.922042 | 0.915607 | 0.928567 | 0.922042 | 0.720969 | new_sparse_clustering |
| tfidf + agglomerative | 183 | 0.507817 | 0.907352 | 0.896951 | 0.917997 | 0.907352 | NA | previous_clustering_summary |
| dense + agglomerative | 183 | 0.539734 | 0.903987 | 0.898006 | 0.910048 | 0.903987 | NA | previous_clustering_summary |
| tfidf + kmeans | 183 | 0.522415 | 0.895465 | 0.891091 | 0.899882 | 0.895465 | NA | previous_clustering_summary |
| dense + kmeans | 183 | 0.479189 | 0.888946 | 0.880980 | 0.897056 | 0.888946 | NA | previous_clustering_summary |

## Notes

The most directly comparable clustering experiments are those with `n_clusters` equal to the number of weak groups. In this dataset, the weak group count is 183.

The metrics ARI, NMI, homogeneity, completeness and V-measure measure agreement with weak labels. They should not be interpreted as final semantic clustering quality because weak labels are local pseudo-labels, not manually validated semantic classes.

Sparse TF-IDF clustering is useful as a lexical baseline. Dense clustering remains useful because it can capture broader semantic similarity, even when lexical sparse methods are stronger in retrieval.
