# Workbench CPU/CUDA measurements

Use an explicit checkout of CV2_Image_to_Script containing engine.py and gpu_backend.py. Both projects' dependencies must be installed in the selected environment; CUDA needs optional CUDA-enabled PyTorch and a visible NVIDIA GPU.

```sh
python benchmark.py --backend workbench --workbench-root ../CV2_Image_to_Script --recipe ../CV2_Image_to_Script/examples/gpu_preprocessing.json --size 2048 --device cpu --warmup 3 --repeats 20 --output outputs/workbench-cpu
python benchmark.py --backend workbench --workbench-root ../CV2_Image_to_Script --recipe ../CV2_Image_to_Script/examples/gpu_preprocessing.json --size 2048 --device cuda --warmup 3 --repeats 20 --output outputs/workbench-cuda
python compare.py outputs/workbench-cpu/benchmark.json outputs/workbench-cuda/benchmark.json --allow-environment-differences --output outputs/workbench-comparison.html
```

Output directories must be new. CUDA uses strict execution: unsupported operations fail rather than silently timing CPU fallback. Recipe and engine/backend source hashes must match for comparison. The environment override permits CPU/CUDA differences but does not override different inputs or recipes.

Measurements include CPU/GPU transfers and final output transfer, exclude startup, file decoding, intermediate preview copies and serialization, and process one image at a time. A compressed final sample is saved for numerical comparison; latency alone does not establish output parity. For the example recipe compare image arrays with NumPy; allow at most one uint8 level for floating-point rounding across backends. Different preprocessing operations may need their own tolerances.

## Local result — 13 November 2026

My RTX 4060, Linux Docker/WSL2, PyTorch 2.15.0+cu132 / CUDA 13.2, OpenCV 4.14.0 and NumPy 2.4.6. Generated 2048² RGB input, seed 42; nearest resize, flip, brightness/contrast and gamma; 3 warmups, 20 measured iterations.

| Backend | Median | 95th percentile |
|---|---:|---:|
| CPU/OpenCV | 37.46 ms | 40.74 ms |
| CUDA/PyTorch | 9.49 ms | 12.87 ms |

Final arrays matched exactly (maximum absolute pixel difference 0). Median ratio: 3.95× for this recipe. GPU peak allocated memory was 125,829,632 bytes and reserved memory 241,172,480 bytes, measured by the PyTorch allocator. These exclude other processes and non-PyTorch allocations. CPU used 16 OpenCV threads.

This is a synthetic preprocessing measurement, not model accuracy, end-to-end GUI performance, real drone throughput, or a general acceleration guarantee. RAG, CLIP and DINO GPU validation scripts live in their respective repositories; they exercise real models but do not measure quality or speed.
