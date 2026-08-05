# 卷积算子

GPU 加速的卷积操作（静态类 `ConvolutionEngine`）。

所有操作写入调用方提供的输出图像，并接受可选的 CUDA 流。

## gaussianBlur

```cpp
static void gaussianBlur(const GpuImage& input, GpuImage& output,
                         int kernelSize, float sigma,
                         cudaStream_t stream = nullptr);
```

应用指定内核大小和 sigma 的高斯模糊。

**参数：**
- `input`: 输入图像
- `output`: 输出图像（必要时自动调整为与输入相同尺寸）
- `kernelSize`: 内核大小（必须为奇数，1–7；否则抛出 `std::invalid_argument`）
- `sigma`: 标准差（>0）

**优化：** 使用共享内存分块实现高效内存访问。

## sobelEdgeDetection

```cpp
static void sobelEdgeDetection(const GpuImage& input, GpuImage& output,
                               cudaStream_t stream = nullptr);
```

使用 Sobel 算子检测边缘。结果为梯度幅值图像。

## convolve

```cpp
static void convolve(const GpuImage& input, GpuImage& output,
                     const float* kernel, int kernelSize,
                     BorderMode borderMode = BorderMode::Zero,
                     cudaStream_t stream = nullptr);
```

应用自定义卷积内核。

**参数：**
- `kernel`: 展平的内核权重（kernelSize × kernelSize 个元素）
- `kernelSize`: 内核维度（必须为奇数，1–7）
- `borderMode`: 边界处理方式——`BorderMode::Zero`、`Mirror` 或 `Replicate`

## separableConvolve

```cpp
static void separableConvolve(const GpuImage& input, GpuImage& output,
                              const float* rowKernel, const float* colKernel,
                              int kernelSize, cudaStream_t stream = nullptr);
```

两趟可分离卷积（对可分离内核更高效）。

## 内核生成

```cpp
static std::vector<float> generateGaussianKernel(int size, float sigma);    // 2D（size × size）
static std::vector<float> generateGaussianKernel1D(int size, float sigma);  // 1D
```

## 说明

- 内核大小上限为 7×7：`kernelSize > 7` 或偶数值会抛出
  `std::invalid_argument`。
- [ImageProcessor](../core/image-processor) 门面封装了这些操作
  （`gaussianBlur`、`sobelEdgeDetection`、`convolve`），返回新的 `GpuImage`。
- 绝对 GPU 延迟可用 `benchmarks/` 测试程序（`-DBUILD_BENCHMARKS=ON`）测量；
  项目不提供与 CPU 的对比数据。

[返回 API](../)
