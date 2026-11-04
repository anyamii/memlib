# Evaluation protocol and aggregate evidence

These are previously recorded aggregate results, not a rerun during repository
preparation. No source chats, per-query logs, answers, private dictionaries,
models or vectors are redistributed. See the root README for all P6–P9 aggregates
and their limits. The public baseline table is in
[LoCoMo P2](../../eval/results/locomo-baseline.md).

## Public P2/P3

LoCoMo upstream commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`,
`data/locomo10.json` SHA256
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
Ten conversations are indexed independently. Exclude category 5, missing evidence,
and evidence IDs absent from that conversation: 1527 eligible questions remain.
Only conversation text is indexed, never QA, answers, summaries or observations.
Cutoff is the last session plus one day, a synthetic full-history condition.
Thirty fabricated chitchat negatives are distributed across conversations.

R@5/R@10 are mean gold-evidence coverage before the final Pack budget, not
any-gold hit rate. MRR@10 uses the first gold rank. Negative false emission means
a nonempty final Pack. Use the frozen tokenizer/model revisions and RRF settings
in the root README; Pack defaults are five entries, 1200 Unicode characters,
two entries per parent. P3 and the private comparisons disable time weighting.

| Public experiment | R@5 | Scope |
|---|---:|---|
| P2 BM25 | 42.62% | lexical baseline |
| P2 MiniLM | 23.76% | normalized dense, 256 tokens |
| P2 BM25 + MiniLM | 37.27% | RRF k=60, 50 per lane |
| P3 bge-m3 | 14.33% | CLS, normalized, 512 tokens |

The included `eval/run.py` runs P2 and optional BGE retrieval. `eval/phase3.py`
and `scripts/run_phase3.sh` implement resumable P3, with model/data/output paths
outside this repository. Stage A measures query encoding; Stage B replays cached
vectors, so its latency is not online latency. Model weights are never committed.

## P4 and private experiments

P4 recorded 1427 cross-conversation negatives, excluding 100 name-collision pairs
from 1527 candidates. At 152/1527 positive rejections (9.95%), BM25 normalized
score gating reduced cross-conversation emissions from 1427/1427 to 516/1427
(36.16%). Synthetic chitchat still emitted 15/30. These are same-set scans.
The public candidate generation chain is missing: `eval/phase4.py` is excluded.
These aggregate claims do not imply a reproducible P4 command is provided.

Private Chinese R@5 means any-gold hit in the first five; Pack recall is measured
after final gating and budgeting. It is not comparable directly to public mean
coverage. The 200 new chitchat cases exposed 57/200 emissions (28.5%); the earlier
60-case result was optimistic. The P9 intent approach missed 28/120 positives and
failed the 7% calibration target. It is not a selected deployment configuration.

Private query-group holdouts share a corpus, do not establish fact-independent
generalization, and cannot be independently reproduced from these aggregates.
All numbers are retrieval/gating metrics, not generated-answer correctness.
