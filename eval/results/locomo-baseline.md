# LoCoMo text-only evidence retrieval baseline (2026-10-29)

1527 valid-evidence questions, 30 synthetic chitchat negatives, 10 independent conversations. Upstream LoCoMo commit 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 (CC BY-NC 4.0); data SHA256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2@e8f8c211226b894fcb81acc59f3b34ba3efd5f42 (CPU, normalized, 256 tokens). Tokenizer: english-lower-alnum-v1. Cutoff is synthetic last-session + 1 day, not original QA chronology. Default parameters, no score-driven tuning. RRF is k=60; fragments skipped because there are no authored impressions with verified evidence mappings.

| mode | R@5 | R@10 | MRR@10 | neg false | P50 ms | P95 ms | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.4262 | 0.4978 | 0.3486 | 50.00% | 4.67 | 6.83 | 0.00% |
| vector | 0.2376 | 0.3301 | 0.1822 | 100.00% | 87.24 | 97.59 | 0.00% |
| rrf | 0.3727 | 0.4580 | 0.2774 | 100.00% | 86.60 | 107.77 | 0.00% |
| rrf+time | 0.3666 | 0.4522 | 0.2647 | 100.00% | 86.48 | 95.06 | 0.00% |
| rrf+k=0 | 0.4184 | 0.5086 | 0.3134 | 100.00% | 86.94 | 96.68 | 0.00% |
| rrf+k=10 | 0.4181 | 0.5153 | 0.2987 | 100.00% | 87.55 | 96.97 | 0.00% |
| rrf+k=120 | 0.3691 | 0.4587 | 0.2767 | 100.00% | 87.38 | 96.45 | 0.00% |

Recall is mean evidence fraction, before Pack budget; MRR uses first gold in top10. All negative Pack scores use the mode's fused rank score, not a calibrated similarity. Vector/RRF always surface something: retrieval recall alone is insufficient for deployment. This is an adapted retrieval benchmark, not official LoCoMo answer accuracy.

CPU: Ryzen 5 7640HS, 2 torch threads. Exact environment: ../requirements-cpu-lock.txt. Raw per-question CSV and JSON remain outside the repository; see [the public evaluation protocol](../../docs/evaluation/README.md).
