# 形态学算子

数学形态学操作（静态类 `Morphology`）。

所有操作签名形式相同：写入调用方提供的输出图像，并接受可选的 CUDA 流。

```cpp
static void op(const GpuImage& input, GpuImage& output, int kernelSize = 3,
               StructuringElement element = StructuringElement::Rectangle,
               cudaStream_t stream = nullptr);
```

`StructuringElement`：`Rectangle`、`Cross` 或 `Ellipse`。

## erode

```cpp
static void erode(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle,
                  cudaStream_t stream = nullptr);
```

应用腐蚀操作（缩小白色区域）。

## dilate

```cpp
static void dilate(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                   StructuringElement element = StructuringElement::Rectangle,
                   cudaStream_t stream = nullptr);
```

应用膨胀操作（扩大白色区域）。

## open

```cpp
static void open(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                 StructuringElement element = StructuringElement::Rectangle,
                 cudaStream_t stream = nullptr);
```

开运算：先腐蚀后膨胀。去除小的亮点。

## close

```cpp
static void close(const GpuImage& input, GpuImage& output, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle,
                  cudaStream_t stream = nullptr);
```

闭运算：先膨胀后腐蚀。填充小的暗洞。

## gradient

```cpp
static void gradient(const GpuImage& input, GpuImage& output,
                     int kernelSize = 3,
                     StructuringElement element = StructuringElement::Rectangle,
                     cudaStream_t stream = nullptr);
```

形态学梯度：膨胀与腐蚀的差值。

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

顶帽变换：原图减开运算。黑帽变换：闭运算减原图。

## 说明

- 绝对 GPU 延迟可用 `benchmarks/` 测试程序（`-DBUILD_BENCHMARKS=ON`）测量；
  项目不提供与 CPU 的对比数据。

[返回 API](../)
