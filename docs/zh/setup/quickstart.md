# 快速入门

5 分钟上手 Mini-OpenCV。

## 前置条件

- 已安装 CUDA 11.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU，计算能力 ≥ 7.5

## 构建

```bash
# 克隆仓库
git clone https://github.com/AICL-Lab/mini-opencv.git
cd mini-opencv

# 配置
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 构建
cmake --build build -j$(nproc)

# 运行测试
ctest --test-dir build --output-on-failure
```

## 基本用法

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

int main() {
    // 创建 ImageProcessor
    ImageProcessor processor;

    // 从文件加载图像
    HostImage hostImage = ImageIO::loadFromFile("input.jpg");

    // 上传到 GPU
    GpuImage gpu = processor.loadFromHost(hostImage);

    // 应用操作（全部 GPU 加速）
    GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
    GpuImage edges = processor.sobelEdgeDetection(gpu);
    GpuImage gray = processor.toGrayscale(gpu);

    // 下载结果到主机
    HostImage result = processor.download(blurred);

    // 保存结果（成功返回 true）
    ImageIO::saveToFile(result, "output.jpg");

    return 0;
}
```

## 可用操作

| 操作 | 方法 |
|------|------|
| 高斯模糊 | `gaussianBlur(image, kernelSize, sigma)` |
| Sobel 边缘检测 | `sobelEdgeDetection(image)` |
| 自定义卷积 | `convolve(image, kernel, kernelSize)` |
| 灰度转换 | `toGrayscale(image)` |
| 反转颜色 | `invert(image)` |
| 亮度调整 | `adjustBrightness(image, offset)` |
| 缩放 | `resize(image, width, height)` |
| 按比例缩放 | `resizeByScale(image, scaleX, scaleY)` |
| 直方图 | `histogram(image)` / `histogramRGB(image)` |
| 直方图均衡化 | `histogramEqualize(image)` |

## 下一步

- [安装指南](./installation) - 详细安装说明
- [配置说明](./configuration) - CMake 选项
- [示例代码](../tutorials/examples) - 更多代码示例