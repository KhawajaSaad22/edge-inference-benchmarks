# Edge inference benchmarks

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
