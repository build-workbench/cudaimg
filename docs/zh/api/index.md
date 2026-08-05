# API 参考

Mini-OpenCV 完整 API 文档。

## 核心类

| 类 | 描述 |
|---|------|
| [ImageProcessor](./core/image-processor) | 图像操作的主入口 |
| [GpuImage](./core/gpu-image) | GPU 内存图像容器 |
| [DeviceBuffer](./core/device-buffer) | RAII GPU 内存管理 |

## 算子

| 类别 | 操作 |
|------|------|
| [卷积](./operators/convolution) | 高斯模糊、Sobel、自定义内核 |
| [滤波器](./operators/filters) | 中值、双边、盒式、锐化 |
| [几何变换](./operators/geometric) | 缩放、旋转、翻转、仿射 |
| [形态学](./operators/morphology) | 腐蚀、膨胀、开闭运算 |

## 图像 I/O

`ImageIO` 的所有函数均为静态函数。

| 函数 | 描述 |
|------|------|
| `ImageIO::loadFromFile(filepath)` | 从文件加载图像，返回 `HostImage` |
| `ImageIO::saveToFile(image, filepath)` | 保存图像到文件（图像在前、路径在后），返回 `bool` |
| `ImageIO::loadFromMemory(data, size)` | 从内存缓冲区解码图像 |
| `ImageIO::encodeToMemory(image, format)` | 将图像编码到内存缓冲区 |
| `ImageIO::getSupportedFormats()` | 列出可读格式 |
| `ImageIO::getWritableFormats()` | 列出可写格式 |
| `ImageIO::isFormatSupported(filepath)` | 检查文件格式是否可读 |

支持格式：

- 可读：JPEG、PNG、BMP、TGA、PSD、GIF、HDR、PNM
- 可写：JPEG、PNG、BMP、TGA

## 快速参考

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// 创建处理器
ImageProcessor processor;

// 加载并上传
HostImage host = ImageIO::loadFromFile("image.jpg");
GpuImage gpu = processor.loadFromHost(host);

// 操作
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);
GpuImage resized = processor.resize(gpu, 1920, 1080);

// 下载并保存
HostImage result = processor.download(gpu);
ImageIO::saveToFile(result, "output.jpg");
```