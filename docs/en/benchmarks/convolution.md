# Convolution Performance

What the benchmark measures for convolution operators, and the implementation constraints behind the numbers you will see.

## What the Benchmark Covers

`gpu_image_benchmark` times the following convolution-based operations at each image size (256×256 through 4096×4096):

- Gaussian blur 3×3 (sigma 1.0)
- Gaussian blur 5×5 (sigma 1.5)
- Sobel edge detection

All reported values are absolute GPU latencies in milliseconds, averaged over 100 iterations after a 10-call warm-up. The benchmark contains no CPU or OpenCV reference implementation, so no speedup ratios are published here — run it on your own hardware to get numbers for your environment.

## Kernel Size Constraints

Convolution kernels accept odd kernel sizes **up to 7×7 only**. `ConvolutionEngine::convolve`, `gaussianBlur`, and `separableConvolve` throw `std::invalid_argument` for larger or even sizes (`src/operators/convolution_engine.cu`). Consequently the benchmark tests 3×3 and 5×5 Gaussian kernels; larger sizes are not possible without changing the kernel code.

Related filters in `src/operators/filters.cu` have their own limits: median filter up to 7×7, box filter and sharpen up to 31×31.

## Optimization Notes

- All convolution paths use shared memory tiling with a halo region
- `separableConvolve` provides a two-pass (row + column) path for separable kernels
- Kernel sizes are capped at 7×7; larger kernels are not currently supported

## Back to Benchmarks

[Benchmark Overview](./)
