# Edge inference benchmarks

See [ROADMAP.md](ROADMAP.md) for measurement, comparison and optional-backend improvements.

Measurement update: each run writes running/completed/failed state and records
provider, precision policy, threshold, thread settings and startup/input-load time.
Comparisons reject differing runtime environments unless explicitly allowed:

```sh
python compare.py first/benchmark.json second/benchmark.json --allow-environment-differences --output comparison.html
```

The override records environment differences; it does not allow mismatched tasks,
precisions or timing scopes. JSON/CSV/HTML outputs refuse existing paths. RSS and
latency are still sampled per the scopes below; no accuracy claim is added.

Measure repeated single-image workload latency, sampled process RSS, optional PyTorch
GPU allocator peaks and one serialized output size. The initial baseline performs
actual CPU grayscale/blur/Canny processing on seeded synthetic pixels. It is not an
AI defect detector. RF-DETR benchmarking is optional and requires a checkpoint.

## CPU demo

Create/activate a virtual environment, install `requirements.txt`, then run:

```sh
python benchmark.py --output outputs/cpu-512 --size 512 --warmup 3 --repeats 20
python benchmark.py --output outputs/cpu-512-repeat --size 512 --warmup 3 --repeats 20
python compare.py outputs/cpu-512/benchmark.json outputs/cpu-512-repeat/benchmark.json
python -m unittest discover -v
```

Reports contain the input hash, seed, dependencies, operating system, raw timings,
p50/p95 latency, iterations/second and memory sampling scope. Fewer repetitions
produce less stable percentile estimates. Use idle hardware, the same workload,
input, checkpoint and measurement settings when comparing results.

CPU RSS includes model/process overhead and is sampled every 5 ms after warmup.
Brief peaks may be missed. GPU allocator peaks include the model and warmup and
exclude other processes and non-PyTorch allocations. These are distinct measures.
Timing excludes decoding, model startup, warmup and output serialization. One
compressed NPZ sample is saved; its size is not a full-video storage benchmark.

## RF-DETR and GPU

Install `requirements-model.txt` in a compatible environment:

```sh
python benchmark.py --backend rfdetr --weights models/defects.pth --device cuda --model-size nano --image sample.png --output outputs/model-gpu
```

The checkpoint must match Nano/Small segmentation. Class accuracy and defect recall
are not measured, so a fast run is not evidence of acceptable detection quality.
CUDA synchronization is included in timed inference. No weights are downloaded
by the CPU demo. Dependency versions are recorded but are not a validated lockfile.

## Linux containers

```sh
docker build --target base -t edge-benchmarks .
docker run --rm edge-benchmarks --output /tmp/demo --repeats 20
```

Use target `model` to install RF-DETR. Mount my checkpoint and input image, request
`--gpus all`, and pass `--device cuda` for GPU tests. Mount an outputs directory to
retain results, ensuring container UID 10001 can write to it. GPU driver/runtime
compatibility must be checked on each machine; Jetson needs a separate compatible
image. The optional model path is not validated by the baseline tests.

## GPU update — 13 November 2026

The workbench backend measures an explicit CV2_Image_to_Script recipe on CPU or CUDA. See [GPU_WORKBENCH.md](GPU_WORKBENCH.md) for commands, parity checks and measured scope.

## Numerical output checks — 13 November 2026

Add `--check-outputs` to comparison commands to compare each saved sample against the first report. Optional `--atol` and `--rtol` default to exact equality. Reports include array shapes, dtypes, exact agreement and maximum absolute differences; class IDs and masks always require exact agreement. A numeric mismatch writes the requested report and returns exit code 1. Incompatible shapes/keys, nonfinite values and changed hashed output archives fail explicitly.

```sh
python compare.py outputs/workbench-cpu/benchmark.json outputs/workbench-cuda/benchmark.json --allow-environment-differences --check-outputs --atol 1 --output outputs/checked-comparison.html
```

New benchmark runs record the sample archive SHA-256. Older reports without that hash remain readable, with schema/numeric checks only. The existing CPU/CUDA workbench samples passed exact equality. This checks one final sample per run, not a dataset or model accuracy. Detection array order must match; the tool does not perform semantic box matching. Multi-image trials and stage timings remain future work.
