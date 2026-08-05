# Mini-OpenCV — GPU 图像处理库

[![CI](https://github.com/AICL-Lab/mini-opencv/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/mini-opencv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://aicl-lab.github.io/mini-opencv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![Version](https://img.shields.io/badge/Version-3.0.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-informational.svg)

[English](README.md) | 简体中文

基于 CUDA 的高性能图像处理库，为计算机视觉应用提供 GPU 加速算子。

> **⚡ GPU 加速**：算子以 CUDA 内核实现（shared memory tiling、原子直方图、`uchar4` 向量化），并行处理图像。
>
> *[benchmarks/](benchmarks/) 基准测试（以 `-DBUILD_BENCHMARKS=ON` 构建）测量 GPU 绝对延迟。本仓库目前不提供 CPU/OpenCV 对比，因此此处不声明任何跨库加速倍数。*

---

## 📚 快速链接

| 资源 | 描述 |
|------|------|
| [安装指南](https://aicl-lab.github.io/mini-opencv/zh/setup/installation) | 完整的安装配置指南 |
| [快速入门](https://aicl-lab.github.io/mini-opencv/zh/setup/quickstart) | 5 分钟快速上手 |
| [架构概览](https://aicl-lab.github.io/mini-opencv/zh/architecture/overview) | 三层架构与模块边界 |
| [API 参考](https://aicl-lab.github.io/mini-opencv/zh/api/) | 完整 API 文档 |
| [示例代码](https://aicl-lab.github.io/mini-opencv/zh/tutorials/examples) | 代码示例和教程 |

**完整文档：** https://aicl-lab.github.io/mini-opencv/

---

## ✨ 功能特性

| 类别 | 算子 | 亮点 |
|------|------|------|
| **像素操作** | 反色、灰度化、亮度调整 | 逐像素并行 |
| **卷积** | 高斯模糊、Sobel 边缘检测 | Shared Memory Tiling |
| **直方图** | 计算、均衡化 | 原子操作 + 规约 |
| **几何变换** | 旋转、缩放、翻转、仿射 | 双线性插值 |
| **形态学** | 腐蚀、膨胀、开/闭运算 | 可自定义结构元素 |
| **阈值** | 全局、自适应、Otsu | 直方图驱动 |
| **色彩空间** | RGB/HSV/YUV 转换 | 批量转换 |
| **滤波** | 中值、双边、锐化 | 保边去噪 |
| **流水线** | 多步异步处理 | 多流并发 |

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层                                   │
│         图像处理器  ·  流水线处理器                            │
├─────────────────────────────────────────────────────────────┤
│                   算子层 (CUDA Kernels)                       │
│  Pixel算子    │  卷积引擎        │  几何变换                  │
│  形态学        │  色彩空间        │  滤波器                    │
│  阈值处理      │  直方图计算      │  图像缩放                  │
├─────────────────────────────────────────────────────────────┤
│                      基础设施层                               │
│  显存缓冲区  ·  GPU/Host 图像容器  ·  错误处理                 │
│  图像 I/O    ·  内存池  ·  执行上下文                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

```bash
# 克隆并构建
git clone https://github.com/AICL-Lab/mini-opencv.git
cd mini-opencv
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)

# 运行测试
ctest --test-dir build --output-on-failure

# 运行示例
./build/bin/basic_example
```

### 基础用法

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// 第1步：创建图像处理器
ImageProcessor processor;

// 第2步：创建/加载主机端图像
HostImage hostImage = ImageUtils::createHostImage(1920, 1080, 3);
// 填充 hostImage.data 数据（RGB，8位每通道）

// 第3步：上传到GPU并处理
GpuImage gpu = processor.loadFromHost(hostImage);
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);

// 第4步：下载结果回主机
HostImage result = processor.download(edges);
```

---

## 📋 系统要求

### 硬件/软件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CUDA | 11.0 | 12.x |
| CMake | 3.18 | 3.24+ |
| C++ | C++17 | C++17 |
| GPU | CC 7.5+ (Turing) | RTX 30/40 系列 |

### 支持的图像格式

| 格式 | 读取 | 写入 |
|------|------|------|
| JPEG/JPG | ✓ | ✓ |
| PNG | ✓ | ✓ |
| BMP | ✓ | ✓ |
| TGA | ✓ | ✓ |

**注意：** 所有格式均支持 8 位每通道（灰度、RGB、RGBA）

---

## 📖 文档

完整文档请访问 [GitHub Pages](https://aicl-lab.github.io/mini-opencv/)：

- [安装指南](https://aicl-lab.github.io/mini-opencv/zh/setup/installation)
- [快速入门](https://aicl-lab.github.io/mini-opencv/zh/setup/quickstart)
- [架构概览](https://aicl-lab.github.io/mini-opencv/zh/architecture/overview)
- [性能基准](https://aicl-lab.github.io/mini-opencv/zh/benchmarks/)
- [API 参考](https://aicl-lab.github.io/mini-opencv/zh/api/)
- [常见问题](https://aicl-lab.github.io/mini-opencv/zh/tutorials/faq)

---

## 🤝 贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

---

## 📝 更新日志

请参阅 [CHANGELOG.md](CHANGELOG.md) 了解版本历史。

---

## 🛠️ 常见问题

**Q: CMake 找不到 CUDA / `nvcc` 未找到**
```bash
# 显式设置 CUDA 路径
export CUDAToolkit_ROOT=/usr/local/cuda
# 或在配置时指定：
cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda
```

**Q: 运行示例显示 "CUDA is not available"**
- 确保你有计算能力 7.5+ 的 NVIDIA GPU
- 从 [NVIDIA](https://developer.nvidia.com/cuda-downloads) 安装 CUDA Toolkit 11.0+
- 验证：`nvidia-smi` 应显示你的 GPU

**Q: 测试执行时失败或崩溃**
- 检查 GPU 显存：大图像可能需要更多显存
- 尝试较小的测试图像或减少 batch size
- 使用详细输出运行：`ctest --test-dir build --output-on-failure -V`

**Q: 如何验证安装成功？**
```bash
cd build && ctest --output-on-failure
# 所有测试应该通过
```

更多问题请查看 [GitHub Discussions](https://github.com/AICL-Lab/mini-opencv/discussions)。

---

## 📄 许可证

MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

**⭐ 如果本项目对你有帮助，请给个 Star！**

如需支持，请提交 [Issue](https://github.com/AICL-Lab/mini-opencv/issues) 或发起 [讨论](https://github.com/AICL-Lab/mini-opencv/discussions)。
