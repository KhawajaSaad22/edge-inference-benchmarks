# Benchmark improvements

Reviewed 13 November 2026. The three existing tests passed. Proposed work below does
not establish model performance and is not implemented by this documentation patch.

## Priorities

1. **Comparable experiment metadata.** Record CPU/GPU identifiers, thread counts,
   precision, runtime provider, inference threshold, dependency versions and timing
   scope. Comparison currently checks input/backend/model/warmup/repetition identity,
   but does not flag differing dependency stacks or effective runtime settings.
2. **Multiple images and repetitions.** Add manifests of representative inputs and
   repeated whole trials, with per-input results, variation summaries and saved
   experiment configuration. Do not present a single-image percentile as a workload-wide guarantee.
3. **Separate measurements.** Distinguish model startup, decoding/preprocessing,
   model execution, postprocessing and end-to-end processing. Preserve GPU
   synchronization. Distinguish steady-state from warmup memory peaks.
4. **Failure-aware runs.** Persist running/completed/failed state, configuration and
   partial diagnostics, including device errors and write failures. Validate
   repetitions and input settings before expensive model initialization.
5. **Comparison reports.** Add CSV/HTML summaries and charts, compatibility warnings
   and an explicit override for intentionally different environments. Include raw
   measurements so a chart remains auditable.

## Optional backend expansion

[ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
are a suitable next adapter for comparing exported models on supported hardware.
Use the same exported model/input and check output parity against the source
framework before reporting speed differences. Precision changes require an accuracy
or numeric-parity check; provider availability does not prove acceleration.

Keep ONNX/GPU dependencies optional. First improve measurement metadata and repeated
CPU trials using existing libraries. Hardware power/thermal sampling is a later,
device-specific feature, with sampling limits and permissions recorded explicitly.

## Completion checks

Test mismatched scopes/precisions/providers, empty or mixed-size input sets, failed
trials, deterministic inputs, warmup isolation and GPU synchronization. Benchmark
the actual exported workbench pipeline as an integration case. Never infer detector
accuracy or video storage savings from latency or one compressed output sample.
