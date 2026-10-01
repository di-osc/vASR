# M4 Pro Qwen3-ASR CER investigation — 2026-09-29

## Outcome

The intermittent extreme-CER outputs on the Metal server no longer reproduced after serializing shared model-forward work. Four full-dataset evaluations produced identical weighted CER and identical per-record results: 99/99 records passed the quality checks, with no long repeated output or `max_new_tokens` cap hit. The first two were consecutive runs on the same loaded server. Runs 3 and 4 used successively extended installed builds; run 4 includes the final ordinary continuous-scheduler gate.

The evidence points to overlapping Metal model forwards across the audio encoder, prefill, and decode pipeline stages. Those stages used to run on separate worker threads against the same model/runtime without a shared execution gate. The new per-runtime gate serializes offline Metal forwards and their completion while CPU-side preparation and scheduling remain concurrent. This is a fix validated on the current Metal service and dataset; the separate streaming API was not changed or validated.

The 17.85×–21.99× audio throughput below is the observed range during quality validation; later fresh-server runs were faster. There is no trustworthy pre-fix quality baseline, so this is not a throughput improvement claim and does not establish the planned 20% acceptance target.

## Machine and workload

- Machine: Apple M4 Pro, 48 GB unified memory; backend Metal; ASR dtype BF16.
- Model: `Qwen/Qwen3-ASR-0.6B` (`vasr-transcribe serve --model Qwen3-ASR-0.6B`).
- Dataset: `datasets/call-100`, 99 records and 5,025.0118 audio seconds. All records declare one 8 kHz channel; evaluation used mono.
- Source base: `f09b07c` with the working-tree changes in this report.
- The service used its automatically selected capacity; no manual concurrency or batch-capacity override was supplied.
- Evaluation results are in `/tmp/vasr-cer-locktest.json`, `/tmp/vasr-cer-locktest-2.json`, `/tmp/vasr-cer-final-1.json`, and `/tmp/vasr-cer-final-4.json`. They contain transcript text and remain outside the repository.

## Diagnosis

Earlier full-dataset runs varied substantially even with the same settings. Several runs produced extreme hallucinations on short chunks and hit the 256-token generation limit; the same recordings decoded normally when sent alone. A full run with continuous batching disabled still produced anomalies, so changing only the request preparation/batching mode was insufficient.

The unstable outputs occurred while the Metal server could perform encoder, paged prefill, and decode model work on different threads. Serializing those shared-device forward sections removed the observed failure mode in repeated full-dataset evaluation. The target records that previously exposed anomalies (dataset rows 52, 83, and 92) were normal in all four fixed runs:

| Dataset row | Reference chars | CER in all fixed runs |
| --- | ---: | ---: |
| 52 | 82 | 0.256098 |
| 83 | 78 | 0.410256 |
| 92 | 187 | 0.106952 |

These records still have ordinary recognition errors; the fix removes the catastrophic repeated/hallucinated output rather than making every transcript error-free.

## Changes

- Added `ModelExecutionLock` to `PagedCacheRuntime`, shared by clones of that runtime.
- On Metal, audio encoder forward and completion in both staged and ordinary/timed offline paths, shared paged prefill/decode, the legacy static paged micro-batch run, and the ordinary continuous offline scheduler now use the same gate. The runtime/cache mutex is released before waiting on the gate.
- Non-Metal execution is unchanged. The fix does not change the model weights, VAD segmentation, language decoding, or token limit.
- Streaming inference keeps its existing execution behavior and is outside this offline CER fix.
- Existing quality reporting continues to reject missing, empty, repeated-pattern, extreme-CER, or failed outputs from quality-passed throughput.

## Validation

All four full evaluations completed all 99 records with no failures or quality-gate rejections:

| Run | Build/process | Audio seconds | Wall seconds | Throughput | Weighted CER | Mean record CER | Quality-passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Initial lock build, same server | 5,025.0118 | 281.4546 | 17.8537× | 0.167497 | 0.223759 | 99/99 |
| 2 | Initial lock build, same server | 5,025.0118 | 281.0633 | 17.8786× | 0.167497 | 0.223759 | 99/99 |
| 3 | Offline encoder/static gate, fresh server | 5,025.0118 | 231.6462 | 21.6926× | 0.167497 | 0.223759 | 99/99 |
| 4 | Final offline scheduler gate, fresh server | 5,025.0118 | 228.5476 | 21.9867× | 0.167497 | 0.223759 | 99/99 |

All four runs gave identical aggregate CER and identical CER for all three previously anomalous records. None logged a chunk reaching `max_new_tokens`; diagnostic output collection found no segment output longer than 250 characters. Service load and warmup are excluded from the evaluation wall time. A separate first-two-record smoke test against a server started with `VASR_DISABLE_CONTINUOUS_BATCH=1` passed 2/2 quality checks; it is not a full-dataset fallback-path acceptance run.

Checks completed:

- `cargo test -p vasr-paged-attn --lib`: 13 passed, including the shared execution-lock serialization test.
- `cargo test -p vasr --features metal-paged-attn`: 112 passed.
- `cargo test -p vasr-transcribe --features metal`: 88 library tests and 3 binary tests passed.
- M4 Pro Metal numerical tests for grouped encoder attention against the per-window reference passed for F32 and BF16.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- `cargo install --path vasr-transcribe --force --features metal` succeeded after removing temporary diagnostic logging.

The planned three-run throughput comparison per optimization candidate, concurrency/batch sweep, and 30-minute soak remain outstanding. The CER fix establishes a repeatable quality result for this workload, but a pre-fix valid CER baseline is still unavailable; use the final-build quality-checked run as the starting point for the throughput phase.

## Reproduction

From the repository root:

```bash
cargo install --path vasr-transcribe --force --features metal
vasr-transcribe serve --model Qwen3-ASR-0.6B
```

In another terminal, run the full-dataset quality evaluation:

```bash
vasr-transcribe eval -i datasets/call-100 --timeline mono --normalizer bench
```
