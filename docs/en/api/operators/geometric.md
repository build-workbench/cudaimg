# Geometric Operators

Geometric transformation operations. All operations write into a
caller-provided output image and accept an optional CUDA stream
(static classes `Geometric` and `ImageResizer`).

## resize

```cpp
static void resize(const GpuImage& input, GpuImage& output,
                   int newWidth, int newHeight,
                   InterpolationMode mode = InterpolationMode::Bilinear,
                   cudaStream_t stream = nullptr);

static void resizeByScale(const GpuImage& input, GpuImage& output,
                          float scaleX, float scaleY,
                          InterpolationMode mode = InterpolationMode::Bilinear,
                          cudaStream_t stream = nullptr);
```

Resizes image. `InterpolationMode`: `NearestNeighbor` or `Bilinear`
(`Bicubic` is declared but not yet implemented).

## rotate

```cpp
static void rotate(const GpuImage& input, GpuImage& output,
                   float angleDegrees, cudaStream_t stream = nullptr);
```

Rotates image by specified angle in degrees (clockwise).

```cpp
static void rotate90(const GpuImage& input, GpuImage& output,
                     int times = 1, cudaStream_t stream = nullptr);
```

Rotates by multiples of 90° (`times`: 1 = 90°, 2 = 180°, 3 = 270°) —
more efficient than arbitrary-angle rotation.

## flip

```cpp
static void flip(const GpuImage& input, GpuImage& output,
                 FlipDirection direction, cudaStream_t stream = nullptr);
```

Flips image. `direction`: `FlipDirection::Horizontal`, `Vertical`, or `Both`.

## affineTransform

```cpp
static void affineTransform(const GpuImage& input, GpuImage& output,
                            const float* matrix, int outputWidth,
                            int outputHeight, cudaStream_t stream = nullptr);
```

Applies affine transformation.

**Parameters:**
- `matrix`: 2×3 transformation matrix `[a, b, tx, c, d, ty]` (row-major)
- `outputWidth`, `outputHeight`: Output dimensions

## perspectiveTransform

```cpp
static void perspectiveTransform(const GpuImage& input, GpuImage& output,
                                 const float* matrix, int outputWidth,
                                 int outputHeight,
                                 cudaStream_t stream = nullptr);
```

Applies perspective transformation.

**Parameters:**
- `matrix`: 3×3 transformation matrix (row-major)
- `outputWidth`, `outputHeight`: Output dimensions

## crop / pad

```cpp
static void crop(const GpuImage& input, GpuImage& output,
                 int x, int y, int width, int height,
                 cudaStream_t stream = nullptr);

static void pad(const GpuImage& input, GpuImage& output,
                int top, int bottom, int left, int right,
                unsigned char padValue = 0, cudaStream_t stream = nullptr);
```

## Notes

- The [ImageProcessor](../core/image-processor) facade wraps resize
  (`resize`, `resizeByScale`) and returns a new `GpuImage`.
- Absolute GPU latency can be measured with the `benchmarks/` harness
  (`-DBUILD_BENCHMARKS=ON`); the project does not publish CPU comparison figures.

[Back to API](../)
