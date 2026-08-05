# 示例代码

常见用例的代码示例。

## 基本操作

### 高斯模糊

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

int main() {
    ImageProcessor processor;
    
    HostImage input = ImageIO::loadFromFile("input.jpg");
    GpuImage gpu = processor.loadFromHost(input);
    
    // 应用 5x5 高斯模糊，sigma=1.5
    GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
    
    HostImage output = processor.download(blurred);
    ImageIO::saveToFile(output, "blurred.jpg");
}
```

### 边缘检测

```cpp
// Sobel 边缘检测
GpuImage edges = processor.sobelEdgeDetection(gpu);

// 先转灰度效果更好
GpuImage gray = processor.toGrayscale(gpu);
GpuImage grayEdges = processor.sobelEdgeDetection(gray);
```

### 图像缩放

```cpp
// 缩放到指定尺寸
GpuImage resized = processor.resize(gpu, 1920, 1080);

// 按比例缩放
GpuImage doubled = processor.resizeByScale(gpu, 2.0f, 2.0f);
```

## 图像处理流水线

```cpp
// 多步骤处理
GpuImage gray = processor.toGrayscale(gpu);
GpuImage blurred = processor.gaussianBlur(gray, 5, 1.0f);
GpuImage edges = processor.sobelEdgeDetection(blurred);
```

## 批量处理

```cpp
// 处理多张图像
std::vector<std::string> files = {"img1.jpg", "img2.jpg", "img3.jpg"};

for (const auto& file : files) {
    HostImage input = ImageIO::loadFromFile(file);
    GpuImage gpu = processor.loadFromHost(input);
    GpuImage processed = processor.gaussianBlur(gpu, 5, 1.5f);
    HostImage output = processor.download(processed);
    
    ImageIO::saveToFile(output, "processed_" + file);
}
```

## 异步 / 多流处理

```cpp
// PipelineProcessor 持有一个 CUDA 流池
PipelineProcessor pipeline(4);  // 4 条流（默认 3 条）

// 处理步骤接收图像及分配给它的流
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
});

// 图像轮流分配到各条流；返回前完成同步
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

## 形态学操作

形态学位于算子层（不在 `ImageProcessor` 门面上）：

```cpp
GpuImage eroded, dilated, opened, closed;
Morphology::erode(gpu, eroded, 5);    // 5x5 矩形结构元素
Morphology::dilate(gpu, dilated, 5);

// 开运算（先腐蚀后膨胀）
Morphology::open(gpu, opened, 5);

// 闭运算（先膨胀后腐蚀）
Morphology::close(gpu, closed, 5);
```

## 色彩空间转换

色彩转换同样是算子层 API：

```cpp
GpuImage hsv, yuv;
ColorSpace::rgbToHsv(gpu, hsv);
ColorSpace::rgbToYuv(gpu, yuv);
```

## 直方图操作

```cpp
// 计算直方图（256 个区间，定长数组）
std::array<int, 256> hist = processor.histogram(gpu);

// 直方图均衡化
GpuImage equalized = processor.histogramEqualize(gpu);
```

## 更多示例

参见仓库中的 `examples/` 目录：

- `basic_example.cpp` - 通过 `ImageProcessor` 演示像素操作、卷积、直方图和缩放
- `pipeline_example.cpp` - 使用 `PipelineProcessor` 的多流批处理
