# ImageProcessor

GPU 图像操作的主入口。

## 构造函数

```cpp
ImageProcessor();
explicit ImageProcessor(Mode mode);
explicit ImageProcessor(ExecutionPolicy policy);
```

- `ImageProcessor()` 创建默认 `Sync` 模式的处理器。
- `Mode` 是 `ExecutionPolicy::Mode` 的别名：`Sync`、`Async`、`Batch`。
- 传入 `ExecutionPolicy` 可完整控制（模式 + 流）。

## 配置

```cpp
void setMemoryPooling(bool enabled);
bool isMemoryPoolingEnabled() const;
void setMode(Mode mode);
Mode mode() const;
```

## 图像传输

### loadFromMemory

```cpp
GpuImage loadFromMemory(const unsigned char* data, int width, int height,
                        int channels);
```

将原始像素数据上传到 GPU 内存。

### loadFromHost

```cpp
GpuImage loadFromHost(const HostImage& hostImage);
```

将图像数据从主机上传到 GPU 内存。

**参数：**
- `hostImage`: 主机图像数据

**返回：** 数据在 GPU 上的 `GpuImage`

### download

```cpp
HostImage download(const GpuImage& image);
```

将图像数据从 GPU 下载到主机内存。

**参数：**
- `image`: GPU 图像

**返回：** 数据在 CPU 上的 `HostImage`

### downloadToBuffer

```cpp
void downloadToBuffer(const GpuImage& image, unsigned char* buffer,
                      size_t bufferSize);
```

将图像数据下载到预分配的主机缓冲区。

## 像素操作

### invert

```cpp
GpuImage invert(const GpuImage& input);
```

反转图像颜色。

### toGrayscale

```cpp
GpuImage toGrayscale(const GpuImage& input);
```

将 RGB 图像转换为灰度图。

### adjustBrightness

```cpp
GpuImage adjustBrightness(const GpuImage& input, int offset);
```

将每个通道值加上 `offset`（截断到 0–255）来调整亮度。
参数是整数偏移量，不是比例因子。

### 原地版本

```cpp
void invertInPlace(GpuImage& image);
void adjustBrightnessInPlace(GpuImage& image, int offset);
```

直接修改原图像，而不是返回新图像。

## 卷积

### gaussianBlur

```cpp
GpuImage gaussianBlur(const GpuImage& input, int kernelSize = 5,
                      float sigma = 1.0f);
```

应用高斯模糊。

**参数：**
- `kernelSize`: 必须为奇数且 ≤ 7（否则抛出 `std::invalid_argument`）
- `sigma`: 标准差，>0

### sobelEdgeDetection

```cpp
GpuImage sobelEdgeDetection(const GpuImage& input);
```

使用 Sobel 算子检测边缘。

### convolve

```cpp
GpuImage convolve(const GpuImage& input, const float* kernel, int kernelSize);
```

应用自定义卷积核（`kernelSize × kernelSize` 个权重，奇数且 ≤ 7）。

## 直方图

### histogram

```cpp
std::array<int, 256> histogram(const GpuImage& input);
```

计算灰度直方图（256 个区间）。

### histogramRGB

```cpp
std::array<std::array<int, 256>, 3> histogramRGB(const GpuImage& input);
```

计算逐通道 RGB 直方图。

### histogramEqualize

```cpp
GpuImage histogramEqualize(const GpuImage& input);
```

应用直方图均衡化。

## 几何变换

### resize

```cpp
GpuImage resize(const GpuImage& input, int newWidth, int newHeight);
```

使用双线性插值缩放图像。

### resizeByScale

```cpp
GpuImage resizeByScale(const GpuImage& input, float scaleX, float scaleY);
```

按比例因子缩放图像。

## 同步

```cpp
void synchronize();        // 阻塞直到异步/批处理操作完成
bool isComplete() const;   // 非阻塞完成检查
```

## 示例

```cpp
ImageProcessor processor;

HostImage input = ImageIO::loadFromFile("input.jpg");
GpuImage gpu = processor.loadFromHost(input);

GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(blurred);
GpuImage gray = processor.toGrayscale(gpu);

HostImage output = processor.download(edges);
ImageIO::saveToFile(output, "output.jpg");
```
