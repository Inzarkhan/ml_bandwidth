# Paper Results Summary (`slo14`, `30s`, broader SeBS train set)

## Setup

- Training suite: SeBS single-function workloads
- Known/train workloads:
  - `compression_cpu1`
  - `video-processing_cpu2`
  - `graph-bfs_mem2`
  - `graph-mst_mem3`
  - `dynamic-html_mix2`
  - `crud-api_mix4`
  - `uploader_mix5`
  - `dna-visualisation_mix6`
- Unseen/test workloads:
  - `graph-pagerank_mem1`
  - `thumbnailer_mix3`
  - `image-recognition_cpu3`
- Action space:
  - memory: `512MB`, `1024MB`
  - CPU: `0.75`, `1.0`
- Runtime target: `30s`
- Repeat aggregation: `median`
- Decision SLO multiplier: `1.4`

## Trained Model

- Model directory: `models_known_train_1h_slo14`
- Decision-time energy MAE / RMSE:
  - `41.70 / 64.36`
- Beneficial-action classifier:
  - accuracy: `0.9167`
  - precision: `0.80`
  - recall: `1.00`
  - F1: `0.8889`
- Beneficial labels:
  - train: `11`
  - test: `4`

## Known-Set Result

These numbers are SLO-valid against the measured runs.

| Metric | Value |
| --- | ---: |
| Default energy | `10311.78 J` |
| Model energy | `9957.44 J` |
| Model savings | `354.35 J` |
| Model improvement | `3.44%` |
| SLO-safe oracle savings | `636.35 J` |
| SLO-safe oracle improvement | `6.17%` |
| Oracle capture | `55.68%` |

Chosen actions:

- `compression_cpu1` -> stay at `1024MB / 1.0`
- `crud-api_mix4` -> stay at `1024MB / 1.0`
- `dna-visualisation_mix6` -> `512MB / 0.75`
- `dynamic-html_mix2` -> `1024MB / 0.75`
- `graph-bfs_mem2` -> `512MB / 0.75`
- `graph-mst_mem3` -> `512MB / 0.75`
- `uploader_mix5` -> `1024MB / 0.75`
- `video-processing_cpu2` -> stay at `1024MB / 1.0`

## Unseen-Triad Result

### Nominal evaluator output

This is the direct output of `8. validate_unseen.py` with `SEBS_SLO_MULTIPLIER=1.4`.

| Metric | Value |
| --- | ---: |
| Default energy | `1182.89 J` |
| Nominal model energy | `967.89 J` |
| Nominal model savings | `215.00 J` |
| Nominal model improvement | `18.18%` |
| Energy-only oracle savings | `228.15 J` |
| Energy-only oracle improvement | `19.29%` |
| Energy-only oracle capture | `94.24%` |

Nominal chosen actions:

- `graph-pagerank_mem1` -> `512MB / 0.75`
- `image-recognition_cpu3` -> `1024MB / 0.75`
- `thumbnailer_mix3` -> `512MB / 0.75`

### Strict measured-SLO validation

When the chosen actions are checked against the **actual measured service times** in the raw aggregate runs, the unseen result is much smaller.

| Metric | Value |
| --- | ---: |
| Actual-SLO-valid model energy | `1164.27 J` |
| Actual-SLO-valid model savings | `18.63 J` |
| Actual-SLO-valid model improvement | `1.57%` |
| SLO-safe oracle savings | `33.93 J` |
| SLO-safe oracle improvement | `2.87%` |
| SLO-safe oracle capture | `54.93%` |

Per-workload strict check:

- `graph-pagerank_mem1`
  - nominal choice: `512MB / 0.75`
  - energy: `207.51 J`
  - actual service time: `16393.01 ms`
  - allowed service time: `5268.08 ms`
  - result: **unsafe**, fallback to baseline
- `image-recognition_cpu3`
  - nominal choice: `1024MB / 0.75`
  - energy: `241.95 J`
  - actual service time: `2728.89 ms`
  - allowed service time: `4412.92 ms`
  - result: **safe**
- `thumbnailer_mix3`
  - nominal choice: `512MB / 0.75`
  - energy: `518.43 J`
  - actual service time: `41.32 ms`
  - allowed service time: `41.30 ms`
  - result: **unsafe by ~0.02 ms**, fallback to baseline

## Alternative Unseen-Partition Result

This alternative unseen set uses the same benchmark families as training, but from the unseen partition:

- `graph-mst_mem3`
- `uploader_mix5`
- `video-processing_cpu2`

### Nominal evaluator output

This is the direct output of `8. validate_unseen.py` on the complete two-batch alt-trio collection.

| Metric | Value |
| --- | ---: |
| Default energy | `1630.45 J` |
| Nominal model energy | `1262.90 J` |
| Nominal model savings | `367.55 J` |
| Nominal model improvement | `22.54%` |
| Energy-only oracle savings | `370.67 J` |
| Energy-only oracle improvement | `22.73%` |
| Energy-only oracle capture | `99.16%` |

Nominal chosen actions:

- `graph-mst_mem3` -> `1024MB / 0.75`
- `uploader_mix5` -> `1024MB / 0.75`
- `video-processing_cpu2` -> `1024MB / 0.75`

### Strict measured-SLO validation

When the alt-trio choices are checked against actual measured service times, all three selected actions violate the `1.4x` latency threshold.

| Metric | Value |
| --- | ---: |
| Actual-SLO-valid model energy | `1630.45 J` |
| Actual-SLO-valid model savings | `0.00 J` |
| Actual-SLO-valid model improvement | `0.00%` |
| SLO-safe oracle savings | `34.48 J` |
| SLO-safe oracle improvement | `2.11%` |
| SLO-safe oracle capture | `0.00%` |

Per-workload strict check:

- `graph-mst_mem3`
  - nominal choice: `1024MB / 0.75`
  - energy: `458.10 J`
  - actual service time: `316.75 ms`
  - allowed service time: `314.88 ms`
  - result: **unsafe by ~1.87 ms**, fallback to baseline
- `uploader_mix5`
  - nominal choice: `1024MB / 0.75`
  - energy: `472.81 J`
  - actual service time: `134.36 ms`
  - allowed service time: `133.40 ms`
  - result: **unsafe by ~0.96 ms**, fallback to baseline
- `video-processing_cpu2`
  - nominal choice: `1024MB / 0.75`
  - energy: `331.99 J`
  - actual service time: `470.36 ms`
  - allowed service time: `436.91 ms`
  - result: **unsafe**, fallback to baseline

## Combined Strict Result

These are the most conservative numbers to report if the paper claim must respect measured latency SLOs end to end.

| Metric | Value |
| --- | ---: |
| Combined default energy | `11494.67 J` |
| Combined actual-SLO-valid model energy | `11121.71 J` |
| Combined actual-SLO-valid savings | `372.98 J` |
| Combined actual-SLO-valid improvement | `3.24%` |
| Combined SLO-safe oracle savings | `670.28 J` |
| Combined SLO-safe oracle improvement | `5.83%` |
| Combined SLO-safe oracle capture | `55.64%` |

## Reporting Guidance

Recommended paper-safe wording:

> Under a `1.4x` latency SLO, a joint CPU-memory HGBDT policy trained on eight SeBS workloads achieved `3.44%` energy savings on known workloads and `1.57%` savings on a stricter unseen trio after validating the selected actions against measured service times. This corresponds to about `55%` of the SLO-safe oracle on both known and unseen sets.

Important caveat:

- The higher unseen number, `18.18%`, is a **nominal policy output**, not a strict measured-SLO-valid result.
- It should only be reported if clearly labeled as an energy-only or projection-level result.
- The alternative unseen-partition result, `22.54%`, is also a **nominal policy output**.
- Under strict measured-SLO validation, that alternative unseen-partition result falls to `0.00%`.

## Files Used

- `raw_sebs_known_train_1h.jsonl`
- `prepared_known_train_1h.csv`
- `models_known_train_1h_slo14/energy_hgbdt_decision_meta.json`
- `raw_sebs_unseen_slo14_all.jsonl`
- `prepared_unseen_slo14.csv`
- `raw_sebs_unseen_alttrio_all.jsonl`
- `prepared_unseen_alttrio_all.csv`
