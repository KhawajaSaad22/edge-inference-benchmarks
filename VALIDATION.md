# Initial validation — 12 November 2026

Three automated checks passed locally and in a Python 3.11 Linux container:

- Warmup/repetition counts, synchronization boundaries and memory-report fields.
- Reproducible synthetic input, saved output and comparison compatibility guards.
- Rejection of a GPU label for the CPU preprocessing baseline.

The Linux base image builds and its CPU demo runs. The Linux check used NumPy
2.4.6, OpenCV 4.14.0.94 and psutil 7.2.2. Windows development checks also ran in an
existing environment; Linux container checks validate the declared dependency bounds.
The optional RF-DETR/checkpoint/GPU benchmark is not validated here. No accuracy,
recall, real-time drone capability or device ranking is established by these tests.
Timings from a short smoke run are not stable hardware-performance claims.
GitHub CI is configured; its first run is checked after publishing.
