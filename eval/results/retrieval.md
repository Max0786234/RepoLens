Questions: 52

| Method | File Hit@1 | File Hit@5 | MRR@10 | Span Hit@5 | Latency |
|---|---|---|---|---|---|
| v1 baseline (line windows + BM25) | 0.61 | 0.86 | 0.72 | 0.71 | 70 ms |
| v2 BM25 (code-aware) | 0.79 | 0.94 | 0.84 | 0.89 | 2 ms |
| v2 vector (embeddinggemma) | 0.64 | 0.92 | 0.77 | 0.89 | 2082 ms |
| v2 hybrid (BM25 + vector, RRF) | 0.79 | 1.00 | 0.87 | 0.94 | 3 ms |

By question style (Span Hit@5):

| Style | v1 | bm25 | vector | hybrid |
|---|---|---|---|---|
| identifier | 0.70 | 1.00 | 1.00 | 1.00 |
| natural | 0.71 | 0.86 | 0.86 | 0.93 |
