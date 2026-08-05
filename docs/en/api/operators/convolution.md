# Convolution Operators

GPU-accelerated convolution operations (static class `ConvolutionEngine`).

All operations write into a caller-provided output image and accept an
optional CUDA stream.

## gaussianBlur

```cpp
static void gaussianBlur(const GpuImage& input, GpuImage& output,
                         int kernelSize, float sigma,
                         cudaStream_t stream = nullptr);
```

Applies Gaussian blur with specified kernel size and sigma.

**Parameters:**
- `input`: Input image
- `output`: Output image (resized to match input if needed)
- `kernelSize`: Kernel size (must be odd, 1–7; throws `std::invalid_argument` otherwise)
- `sigma`: Standard deviation (>0)

**Optimization:** Uses shared memory tiling for efficient memory access.

## sobelEdgeDetection

```cpp
static void sobelEdgeDetection(const GpuImage& input, GpuImage& output,
                               cudaStream_t stream = nullptr);
```

Detects edges using Sobel operator. The result is the gradient magnitude image.

## convolve

```cpp
static void convolve(const GpuImage& input, GpuImage& output,
                     const float* kernel, int kernelSize,
                     BorderMode borderMode = BorderMode::Zero,
                     cudaStream_t stream = nullptr);
```

Applies custom convolution kernel.

**Parameters:**
- `kernel`: Flattened kernel weights (kernelSize × kernelSize elements)
- `kernelSize`: Kernel dimension (must be odd, 1–7)
- `borderMode`: Border handling — `BorderMode::Zero`, `Mirror`, or `Replicate`

## separableConvolve

```cpp
static void separableConvolve(const GpuImage& input, GpuImage& output,
                              const float* rowKernel, const float* colKernel,
                              int kernelSize, cudaStream_t stream = nullptr);
```

Two-pass separable convolution (more efficient for separable kernels).

## Kernel Generation

```cpp
static std::vector<float> generateGaussianKernel(int size, float sigma);    // 2D (size × size)
static std::vector<float> generateGaussianKernel1D(int size, float sigma);  // 1D
```

## Notes

- Kernel size is limited to 7×7: `kernelSize > 7` or even values throw
  `std::invalid_argument`.
- The [ImageProcessor](../core/image-processor) facade wraps these operations
  (`gaussianBlur`, `sobelEdgeDetection`, `convolve`) and returns a new `GpuImage`.
- Absolute GPU latency can be measured with the `benchmarks/` harness
  (`-DBUILD_BENCHMARKS=ON`); the project does not publish CPU comparison figures.

[Back to API](../)
