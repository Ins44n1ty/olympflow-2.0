# Cluster consistency summary

Method: `tfidf_svd_kmeans`
Records: `728`
Clusters: `183`

## Full-data repeated KMeans stability

- mean pairwise ARI: `0.868962`
- mean pairwise NMI: `0.977468`
- min pairwise ARI: `0.827025`
- min pairwise NMI: `0.970549`

## Subsample stability

- subsample fraction: `0.85`
- bootstrap runs: `20`
- mean subsample ARI: `0.881813`
- mean subsample NMI: `0.981101`

## Cluster size distribution for seed 42

- min size: `2`
- max size: `10`
- mean size: `3.978142`
- median size: `4.000000`

## Interpretation

This report measures whether the selected TF-IDF + SVD + KMeans clustering is stable under different random seeds and subsampled runs. High ARI/NMI means that the clustering structure is not only a one-run artifact.
