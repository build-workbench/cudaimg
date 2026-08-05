# 滤波算子

图像滤波操作（静态类 `Filters`）。

所有操作写入调用方提供的输出图像，并接受可选的 CUDA 流。

## medianFilter

```cpp
static void medianFilter(const GpuImage& input, GpuImage& output,
                         int kernelSize = 3, cudaStream_t stream = nullptr);
```

应用中值滤波进行降噪，同时保持边缘。

## bilateralFilter

```cpp
static void bilateralFilter(const GpuImage& input, GpuImage& output,
                            int kernelSize = 5, float sigmaSpace = 10.0f,
                            float sigmaColor = 50.0f,
                            cudaStream_t stream = nullptr);
```

边缘保持平滑滤波器。

**参数：**
- `kernelSize`: 窗口大小
- `sigmaSpace`: 空间范围
- `sigmaColor`: 颜色相似度阈值

## boxFilter

```cpp
static void boxFilter(const GpuImage& input, GpuImage& output,
                      int kernelSize = 3, cudaStream_t stream = nullptr);
```

应用盒式（均值）滤波。

## sharpen

```cpp
static void sharpen(const GpuImage& input, GpuImage& output,
                    float strength = 1.0f, cudaStream_t stream = nullptr);
```

使用 3×3 锐化核增强图像清晰度。

## laplacian

```cpp
static void laplacian(const GpuImage& input, GpuImage& output,
                      cudaStream_t stream = nullptr);
```

拉普拉斯滤波（边缘增强）。

## 图像算术

同一头文件定义了静态类 `ImageArithmetic`，提供逐像素算术运算
（签名风格相同：输出参数 + 可选流）：

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

## 说明

- 绝对 GPU 延迟可用 `benchmarks/` 测试程序（`-DBUILD_BENCHMARKS=ON`）测量；
  项目不提供与 CPU 的对比数据。

[返回 API](../)
