# Retrieval analysis notes

## Main result

BM25 is the strongest retrieval baseline on the current weak-label setup.

| method | precision@1 | recall@10 | mrr | map |
|---|---:|---:|---:|---:|
| TF-IDF | 0.701513 | 0.662999 | 0.789453 | 0.595507 |
| Dense | 0.693260 | 0.679046 | 0.783012 | 0.588709 |
| BM25 | 0.726648 | 0.950549 | 0.815008 | 0.777160 |

BM25 improves over both TF-IDF and dense embeddings. The largest gain is in recall@10: BM25 reaches 0.950549, while TF-IDF has 0.662999 and dense has 0.679046.

## Why sparse methods work well here

The current weak labels are built from local consecutive groups of tasks inside sections. Such groups often share:

- the same physical objects;
- the same notation;
- the same formulation style;
- similar numbers and constants;
- repeated phrases;
- nearby source context.

Because of this, lexical sparse methods are very competitive. BM25 is especially strong because it rewards important term matches while being more robust to document length than raw TF-IDF.

## Why dense embeddings do not dominate

Dense embeddings often retrieve semantically plausible tasks from other weak groups. Under the current protocol, this is counted as an error even when the retrieved task is physically very close to the query.

This means that dense retrieval is not necessarily bad. Rather, the evaluation labels are coarse and local. They are not a manually validated semantic gold standard.

## Error analysis summary

BM25 wins over dense in 37 top-1 cases, while dense wins over BM25 in 12 cases.

BM25 wins mostly when the correct weak-group neighbor is nearly lexical: similar wording, repeated physical setting, same objects, same variables, or close statement template.

Dense wins when sparse methods overfit to repeated words and retrieve a lexically similar but weak-label-wrong task. In these cases, dense sometimes better captures the broader local meaning.

There are 182 cases where all methods fail at top-1. Many of these failures are not necessarily bad retrieval examples: the returned neighbor is often physically very similar but belongs to another weak group. This reveals a limitation of the weak-label evaluation protocol.

## Top-k overlap

TF-IDF and BM25 are the closest pair of methods:

| pair | Jaccard@1 | Jaccard@3 | Jaccard@5 | Jaccard@10 |
|---|---:|---:|---:|---:|
| TF-IDF vs dense | 0.756868 | 0.676511 | 0.463681 | 0.377718 |
| TF-IDF vs BM25 | 0.875000 | 0.781044 | 0.609688 | 0.555851 |
| Dense vs BM25 | 0.777473 | 0.694643 | 0.492560 | 0.421381 |

This supports the interpretation that BM25 and TF-IDF exploit similar lexical evidence, while dense embeddings form a somewhat different neighborhood structure.

## Conclusion

For the current weak-labeled grouped retrieval task, BM25 should be considered the strongest retrieval baseline. TF-IDF remains a strong and simple baseline. Dense embeddings are still useful, especially for semantic similarity and clustering, but the current weak labels favor lexical local similarity.
