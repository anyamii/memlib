# Synthetic keyterm example

All text, names and questions in this directory are invented. This small example
checks the generic workflow; its metrics are not empirical model results.
`group` partitions holdout queries; retrieval uses the shared synthetic corpus.
Positive queries supply `gold` IDs and `domain`; negative queries supply
`negative_type` (`unseen` or `chat`). The manifest freezes the three JSON inputs.

```sh
.venv/bin/python scripts/evaluate_keyterm.py --input examples/keyterm \
  --seeds examples/keyterm/seeds.txt --stopwords examples/keyterm/stopwords.txt \
  --output ../memlib-artifacts/keyterm-synthetic
```

Outputs include mined vocabulary, query features, retrieval IDs and calibration
metrics. Treat them as private when inputs are private. Output permissions are
restricted, but permissions do not replace access control or consent. An optional
`--baseline calibration.json` verifies prior calibration rows; no private baseline,
fixed private dataset size or production thresholds are required.
