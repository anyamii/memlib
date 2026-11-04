# Evaluation protocol and aggregate evidence

These are previously recorded aggregate results, not a rerun during repository
preparation. No source chats, per-query logs, answers, private dictionaries,
models or vectors are redistributed. See the root README for all P6–P11 and shadow-run aggregates
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

## P10: local intent classification on synthetic queries

A frozen 480-query synthetic set trained a lightweight classifier over fixed
bge-m3 embeddings. Full-set out-of-fold diagnostic intent misses were 8/120
(6.67%); new chitchat false positives were 1/200 (0.50%). The full pipeline had
85% positive Pack R@5 and 59/100 emissions on nonexistent events. Thresholds
were selected on the full set, so these are not independent test results.
One direction of the independent holdout missed 6/63 positives (9.52%), failing
to consistently meet the 7% target. Shared corpus and synthetic author style
remain limitations; this is not a deployment configuration.

## Frozen-P10 shadow run on real messages

Of 1500 real messages, 16 received a nonempty Pack (1.07%). All 16 were reviewed:
3 directly relevant (18.75%), 4 partly relevant, 9 irrelevant. Among 200 randomly
sampled closed messages, 7 clearly needed recall and 5 were uncertain. The 7/200
is a closed-side demand proportion, not conditional recall miss rate; demand
included same-day context. Sampling was approximately uniform by month rather
than natural traffic, and only 16 open cases limit precision estimates. Pack
precision and synthetic gold-ID recall have different denominators and cannot
be subtracted as a measured performance drop. Real-message evidence did not
support the same effectiveness suggested by the synthetic diagnostic.

## P11: real-message labels, training and replay

The 1000 labels combine 216 legacy reviews, 392 new simple-random messages and
392 enriched messages; pooled prevalence does not estimate natural traffic.
In the random sample, long-term demand was 3/392 (0.77%) and same-day context
110/392 (28.06%). Expanding the random sample over the remaining population,
with legacy counts included, estimated 0.80% and 28.27%, respectively; only
three random long-term positives make the estimates unstable.

Only long-term demand was positive; same-day and no-recall messages were
negative, and uncertain labels were excluded. Date-grouped outer five-fold
validation selected thresholds within training folds. The real-trained model
missed 2/26 positives (7.69%) and falsely admitted 353/752 negatives (46.94%),
failing the 7% miss target. At the frozen old threshold, the synthetic baseline
missed 80.77% with 3.19% false admissions; mixed training missed 15.38% with
38.03% false admissions. Different calibration and label semantics prevent
attributing all differences to training data. A later development-only diagnostic
calibrating the synthetic model on training-side real labels still missed
11.54% with 63.30% false admissions; it was added after viewing the temporal test
and is not a new blind test.

On the locked final-two-week temporal test, real training missed 0/8 and falsely
admitted 66/114 (57.89%); synthetic training at its old threshold missed 7/8 with
2/114 false admissions; mixed training missed 2/8 with 50/114 false admissions.
Eight positives cannot establish a miss rate below 7%: the one-sided 95%
binomial reference upper bound for 0/8 is about 31.23%.

A fresh simple-random replay excluded all 1000 labeled IDs. Of 1500 messages,
775 passed intent and 744 received Packs (49.60%). The 744 open cases contained
29 long-term, 211 same-day, 481 no-recall and 23 uncertain demands; the 692
clearly non-long-term cases (93.01%) describe open-case composition, not the
classifier false-positive rate. All 29 long-term Packs were reviewed: 4 directly
relevant (13.79%), 8 partly relevant, 17 irrelevant. Another random 50 from the
other 715 open cases were reviewed; only 79/744 relevance reviews are complete,
with 665 pending. The differently sampled strata cannot be pooled into overall
precision. Closed-side random review found 0 long-term demands in 200, which
neither proves zero misses nor estimates conditional long-term miss rate.

Both replays filtered memory by document date strictly before the message's
Tokyo calendar date, before retrieval and term statistics; reported independent
future-date violations were zero. Document dates do not establish fragment event
time or reproduce the historical database version. The two replays used different
sampling, so their open rates are not a controlled improvement comparison.

Real messages publish aggregate numbers only: no original or paraphrased chats,
memory content, labels or per-message results. Labels came from a single AI
annotator (Codex), without independent blind review and with user-side context
only. Date grouping does not remove cross-day topic dependence. These results
support testing recent-dialogue windows for same-day context, on-demand calls
for long-term memory, and separate retrieval improvements; they do not justify
chat integration. Private experiments and their per-message evidence are outside
the public reproduction scope. No model evaluations were rerun for this update.
