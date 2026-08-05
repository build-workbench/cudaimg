# 几何算子

几何变换操作。所有操作写入调用方提供的输出图像，并接受可选的 CUDA 流
（静态类 `Geometric` 与 `ImageResizer`）。

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

缩放图像。`InterpolationMode`：`NearestNeighbor` 或 `Bilinear`
（`Bicubic` 已声明但尚未实现）。

## rotate

```cpp
static void rotate(const GpuImage& input, GpuImage& output,
                   float angleDegrees, cudaStream_t stream = nullptr);
```

按指定角度（度，顺时针）旋转图像。

```cpp
static void rotate90(const GpuImage& input, GpuImage& output,
                     int times = 1, cudaStream_t stream = nullptr);
```

按 90° 的倍数旋转（`times`：1 = 90°，2 = 180°，3 = 270°）——
比任意角度旋转更高效。

## flip

```cpp
static void flip(const GpuImage& input, GpuImage& output,
                 FlipDirection direction, cudaStream_t stream = nullptr);
```

翻转图像。`direction`：`FlipDirection::Horizontal`、`Vertical` 或 `Both`。

## affineTransform

```cpp
static void affineTransform(const GpuImage& input, GpuImage& output,
                            const float* matrix, int outputWidth,
                            int outputHeight, cudaStream_t stream = nullptr);
```

应用仿射变换。

**参数：**
- `matrix`: 2×3 变换矩阵 `[a, b, tx, c, d, ty]`（行优先）
- `outputWidth`、`outputHeight`: 输出尺寸

## perspectiveTransform

```cpp
static void perspectiveTransform(const GpuImage& input, GpuImage& output,
                                 const float* matrix, int outputWidth,
                                 int outputHeight,
                                 cudaStream_t stream = nullptr);
```

应用透视变换。

**参数：**
- `matrix`: 3×3 变换矩阵（行优先）
- `outputWidth`、`outputHeight`: 输出尺寸

## crop / pad

```cpp
static void crop(const GpuImage& input, GpuImage& output,
                 int x, int y, int width, int height,
                 cudaStream_t stream = nullptr);

static void pad(const GpuImage& input, GpuImage& output,
                int top, int bottom, int left, int right,
                unsigned char padValue = 0, cudaStream_t stream = nullptr);
```

## 说明

- [ImageProcessor](../core/image-processor) 门面封装了缩放
  （`resize`、`resizeByScale`），返回新的 `GpuImage`。
- 绝对 GPU 延迟可用 `benchmarks/` 测试程序（`-DBUILD_BENCHMARKS=ON`）测量；
  项目不提供与 CPU 的对比数据。

[返回 API](../)
