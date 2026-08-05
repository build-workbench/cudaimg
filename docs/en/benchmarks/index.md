# Benchmarks

The benchmark suite measures the **absolute GPU latency** of Mini-OpenCV's own operators. It does not compare against CPU OpenCV or any other library, and this page deliberately publishes no speedup numbers: results depend on your hardware, driver, and system state, so the only figures worth quoting are the ones you reproduce yourself.

## What Is Measured

A single executable, `gpu_image_benchmark` (source: `benchmarks/benchmark_main.cpp`), times each operator at five square image sizes — 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096 — and prints the average time per call in milliseconds.

Covered operations:

| Category | Operations |
|----------|-----------|
| Pixel operations | Invert, grayscale, brightness adjustment |
| Convolution | Gaussian blur 3×3 / 5×5, Sobel edge detection |
| Histogram | Calculation, equalization |
| Geometric | Bilinear resize 2× / 0.5× |
| Morphology | Erode 3×3, dilate 3×3 |
| Threshold | Fixed threshold, Otsu binarization |
| Color space | RGB→HSV, RGB→YUV |
| Data transfer | Host→Device upload, Device→Host download |

A pipeline section then processes a batch of 10 images sequentially and with 1 / 2 / 4 / 8 CUDA streams via `PipelineProcessor`, reporting total batch time for each configuration.

## Timing Method

The harness uses a hand-written `std::chrono` timer (the Google Benchmark library is linked by the build but not used by the current harness):

- 10 untimed warm-up calls
- 100 timed iterations
- `cudaDeviceSynchronize()` before and after the timed loop
- Reported value = total time / iterations

Per-operation timings measure kernel execution on data already resident on the device; transfer costs appear as the separate Upload/Download entries.

## Optimization Techniques Exercised

These techniques are implemented in the kernels the benchmark measures:

### 1. Shared Memory Tiling

The convolution kernels cache the input tile plus a halo region in shared memory (`src/operators/convolution_engine.cu`), so each pixel is loaded from global memory once and reused across overlapping kernel positions.

### 2. Atomic Histogram

Histogram calculation accumulates bins with `atomicAdd` (`src/operators/histogram_calculator.cu`), letting all threads update the histogram concurrently without a separate reduction pass.

### 3. uchar4 Vectorization

Pixel-wise operators load and store four bytes per thread via `uchar4` (`src/operators/pixel_operator.cu`), turning four byte accesses into one vectorized access.

Texture memory, warp-level primitives (`__shfl`/`__reduce`), and pinned/zero-copy memory are **not** used in the current codebase.

## Reproducing Benchmarks

Benchmarks are disabled by default (`BUILD_BENCHMARKS=OFF`):

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/bin/gpu_image_benchmark
```

The executable prints the detected CUDA device first, then one latency table per image size, followed by the pipeline results.

## Methodology

See [Methodology](./methodology) for details on how times are measured and how to interpret the output.
