# Benchmark Methodology

How Mini-OpenCV measures and reports performance.

## Scope

The benchmark suite measures **GPU-only absolute latency**. It does not measure CPU implementations and does not compute speedup ratios against OpenCV or any other library. No fixed hardware environment is documented here on purpose: numbers depend on your GPU, driver, and system state, and the only results we stand behind are the ones you reproduce yourself. The benchmark executable prints the CUDA device it detected at the start of every run.

## Measurement Methodology

### Warm-up

Before each measurement (`benchmarks/benchmark_main.cpp`):
1. Run the operation 10 times (untimed) to warm up the GPU
2. Call `cudaDeviceSynchronize()`

### Timing

A hand-written `std::chrono` timer wraps the timed loop (the Google Benchmark library is linked by the build but not used by the current harness):

```cpp
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < iterations; i++) {   // iterations = 100
    operation();
}
cudaDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
auto avg_time = (end - start) / iterations;
```

### Metrics

- **Latency**: average milliseconds per operation — the only metric the harness reports

## Image Sizes

The harness sweeps five square sizes:

| Dimensions | Pixels |
|------------|--------|
| 256×256 | 66K |
| 512×512 | 262K |
| 1024×1024 | 1.0M |
| 2048×2048 | 4.2M |
| 4096×4096 | 16.8M |

## Reproducibility

The benchmark target is `gpu_image_benchmark` (benchmarks are disabled by default):

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/bin/gpu_image_benchmark
```

The executable prints the detected CUDA device, one latency table per image size, and a pipeline section comparing sequential processing of a 10-image batch against 1 / 2 / 4 / 8 CUDA streams.

## Notes

- There are no CPU benchmarks in this repository; no CPU/GPU comparison is implied anywhere in these docs
- Per-operation timings measure kernel execution on data already resident on the device; host↔device transfer costs appear as the separate "Upload (H2D)" / "Download (D2H)" entries
- Results vary with GPU model, temperature, clock speed, and driver version
