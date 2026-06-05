# Rerank Report — Module 8 Thursday Stretch

## Setup

- Hybrid `k_in`: **50**
- Re-ranked `k_out`: **5**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hardware (CPU model, RAM, OS): **Windows 10, 16GB RAM, MINGW64 (Git Bash/Docker Desktop)**

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (lab baseline) | 0.850 | 0.698 | 15 ms (Stage 1) |
| Hybrid + cross-encoder rerank | **0.916** | **0.775** | **185 ms (15ms + 170ms)** |

## When Does Re-Ranking Pay Off?

Re-ranking pays off most significantly for complex technical queries where the Bi-Encoder (Stage 1) retrieves the correct document but ranks it low (e.g., position 10-20). For example, a query like **"cleaning up messy functions"** might be ranked lower by hybrid search because it lacks exact keywords like "refactoring." The Cross-Encoder, processing the query and document text jointly, recognizes the semantic alignment and promotes it to the top 5, surfacing the gold document when hybrid alone did not.

## Latency Overhead

The cross-encoder adds approximately **170 ms** per query. This overhead scales linearly with `k_in` (Stage 2 latency $\approx$ 3.4 ms per candidate pair), meaning doubling `k_in` to 100 would double the Stage 2 latency to ~340ms. Importantly, this overhead is independent of the corpus size; while the hybrid retrieve (Stage 1) may slow down as the corpus grows from thousands to millions of documents, the cross-encoder only ever "sees" the fixed `k_in` candidates.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

The cross-encoder becomes the bottleneck at a query volume exceeding **5-10 Queries Per Second (QPS)** on a single CPU core, as the 170ms sequential processing time limits throughput. In terms of corpus size, once the dataset exceeds **1 million documents**, Stage 1 latency often rises significantly, and the 200ms total budget becomes harder to justify. At this scale, or for high-traffic production environments, an aggressive caching layer or a learned re-ranker (distilled model) would be required to maintain performance without the sequential scoring bottleneck.