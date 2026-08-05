# Morphology Operators

Mathematical morphology operations (static class `Morphology`).

All operations share the same signature shape: they write into a
caller-provided output image and accept an optional CUDA stream.

```cpp
static void op(const GpuImage& input, GpuImage& output, int kernelSize = 3,
               StructuringElement element = StructuringElement::Rectangle,
               cudaStream_t stream = nullptr);
```

`StructuringElement`: `Rectangle`, `Cross`, or `Ellipse`.

## erode

```cpp
static void erode(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle,
                  cudaStream_t stream = nullptr);
```

Applies erosion (shrinks white regions).

## dilate

```cpp
static void dilate(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                   StructuringElement element = StructuringElement::Rectangle,
                   cudaStream_t stream = nullptr);
```

Applies dilation (expands white regions).

## open

```cpp
static void open(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                 StructuringElement element = StructuringElement::Rectangle,
                 cudaStream_t stream = nullptr);
```

Opening: erosion followed by dilation. Removes small bright spots.

## close

```cpp
static void close(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle,
                  cudaStream_t stream = nullptr);
```

Closing: dilation followed by erosion. Fills small dark holes.

## gradient

```cpp
static void gradient(const GpuImage& input, GpuImage& output,
                     int kernelSize = 3,
                     StructuringElement element = StructuringElement::Rectangle,
                     cudaStream_t stream = nullptr);
```

Morphological gradient: difference between dilation and erosion.

## topHat / blackHat

```cpp
static void topHat(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                   StructuringElement element = StructuringElement::Rectangle,
                   cudaStream_t stream = nullptr);

static void blackHat(const GpuImage& input, GpuImage& output,
                     int kernelSize = 3,
                     StructuringElement element = StructuringElement::Rectangle,
                     cudaStream_t stream = nullptr);
```

Top-hat: original minus opening. Black-hat: closing minus original.

## Notes

- Absolute GPU latency can be measured with the `benchmarks/` harness
  (`-DBUILD_BENCHMARKS=ON`); the project does not publish CPU comparison figures.

[Back to API](../)
