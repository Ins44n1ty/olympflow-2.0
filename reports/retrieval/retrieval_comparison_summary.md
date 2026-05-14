# Retrieval comparison

Common queries: 728

## Hit@k

### tfidf

- hit@1: 0.700549
- hit@3: 0.879121
- hit@5: 0.898352
- hit@10: 0.927198

### dense

- hit@1: 0.690934
- hit@3: 0.870879
- hit@5: 0.894231
- hit@10: 0.932692

### bm25

- hit@1: 0.726648
- hit@3: 0.905220
- hit@5: 0.931319
- hit@10: 0.950549

## Case counts

- bm25_wins_over_dense: 37
- dense_wins_over_bm25: 11
- tfidf_wins_over_dense: 29
- dense_wins_over_tfidf: 22
- bm25_wins_over_tfidf: 25
- tfidf_wins_over_bm25: 6
- all_fail: 183
- all_hit: 480

## Mean top-k Jaccard overlap

- tfidf vs dense, k=1: 0.758242
- tfidf vs dense, k=3: 0.676511
- tfidf vs dense, k=5: 0.463359
- tfidf vs dense, k=10: 0.378111
- tfidf vs bm25, k=1: 0.875000
- tfidf vs bm25, k=3: 0.781044
- tfidf vs bm25, k=5: 0.610343
- tfidf vs bm25, k=10: 0.555674
- dense vs bm25, k=1: 0.778846
- dense vs bm25, k=3: 0.694231
- dense vs bm25, k=5: 0.492271
- dense vs bm25, k=10: 0.421370

## Interpretation

BM25 and TF-IDF are lexical sparse methods. If they outperform dense embeddings on the current weak-label protocol, this suggests that the labels are strongly aligned with local lexical similarity: neighboring tasks inside the same weak group often share physical terms, notation, objects, and formulation style. Dense vectors may retrieve semantically plausible tasks from other weak groups, but such neighbors are penalized by the current evaluation setup.
