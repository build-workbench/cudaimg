# Filter Operators

Image filtering operations (static class `Filters`).

All operations write into a caller-provided output image and accept an
optional CUDA stream.

## medianFilter

```cpp
static void medianFilter(const GpuImage& input, GpuImage& output,
                         int kernelSize = 3, cudaStream_t stream = nullptr);
```

Applies median filter for noise reduction while preserving edges.

## bilateralFilter

```cpp
static void bilateralFilter(const GpuImage& input, GpuImage& output,
                            int kernelSize = 5, float sigmaSpace = 10.0f,
                            float sigmaColor = 50.0f,
                            cudaStream_t stream = nullptr);
```

Edge-preserving smoothing filter.

**Parameters:**
- `kernelSize`: Window size
- `sigmaSpace`: Spatial extent
- `sigmaColor`: Color similarity threshold

## boxFilter

```cpp
static void boxFilter(const GpuImage& input, GpuImage& output,
                      int kernelSize = 3, cudaStream_t stream = nullptr);
```

Applies box (mean) filter.

## sharpen

```cpp
static void sharpen(const GpuImage& input, GpuImage& output,
                    float strength = 1.0f, cudaStream_t stream = nullptr);
```

Sharpens image using a 3×3 sharpening kernel.

## laplacian

```cpp
static void laplacian(const GpuImage& input, GpuImage& output,
                      cudaStream_t stream = nullptr);
```

Laplacian filter (edge enhancement).

## Image Arithmetic

The same header defines static class `ImageArithmetic` for per-pixel arithmetic
(same signature style: output parameter + optional stream):

```cpp
static void add(const GpuImage& src1, const GpuImage& src2, GpuImage& output,
                cudaStream_t stream = nullptr);
static void subtract(const GpuImage& src1, const GpuImage& src2,
                     GpuImage& output, cudaStream_t stream = nullptr);
static void multiply(const GpuImage& src1, const GpuImage& src2,
                     GpuImage& output, float scale = 1.0f,
                     cudaStream_t stream = nullptr);
static void blend(const GpuImage& src1, const GpuImage& src2,
                  GpuImage& output, float alpha, cudaStream_t stream = nullptr);
static void addWeighted(const GpuImage& src1, float alpha,
                        const GpuImage& src2, float beta, GpuImage& output,
                        float gamma = 0.0f, cudaStream_t stream = nullptr);
static void absDiff(const GpuImage& src1, const GpuImage& src2,
                    GpuImage& output, cudaStream_t stream = nullptr);
static void addScalar(const GpuImage& input, GpuImage& output,
                      unsigned char value, cudaStream_t stream = nullptr);
static void multiplyScalar(const GpuImage& input, GpuImage& output,
                           float scale, cudaStream_t stream = nullptr);
```

## Notes

- Absolute GPU latency can be measured with the `benchmarks/` harness
  (`-DBUILD_BENCHMARKS=ON`); the project does not publish CPU comparison figures.

[Back to API](../)
