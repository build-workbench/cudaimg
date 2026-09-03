# cudaimg

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

`cudaimg` 是一个个人业余练习与探索 CUDA 编程的轻量级 GPU 图像处理实验场（Personal Playground）。

以常见的图像处理算子为载体，基于现代 C++17 与 CUDA 从零实现一系列 GPU kernel。项目不追求替代 OpenCV `cv::cuda` 等成熟工业级库，核心目的在于**亲自动手实践 GPU 核心编程机制、对照纯 CPU 参考实现确保计算严谨正确，并记录开发过程中的工程折衷与踩坑经验**。

---

## 项目特点

- **CPU 黄金参考与逐像素验证**：每个 GPU kernel 都手写了对应的纯 C++ CPU 参考实现，并通过 GoogleTest 对比逐像素/逐通道结果，确保算法逻辑与边界处理完全对齐。
- **由浅入深的技术演进**：从最简单的 2D 线程映射，逐步深入到 `uchar4` 向量化访存、Shared Memory Tiling（含 Halo 单元协作加载）、可分离卷积优化、两级直方图原子规约，以及多 CUDA Stream 异步流水线。
- **现代化 C++ 工程结构**：基于 RAII 管理显存与 Stream 资源（`DeviceBuffer`、`ExecutionContext`），无复杂第三方依赖（可选 header-only `stb_image` 仅用于文件读写）。
- **真实的踩坑与设计反思**：在 [docs/pitfalls.md](docs/pitfalls.md) 中持续记录显存并发竞争、Kernel 参数传递限制、分支发散等实际遇到的技术细节与反思。

---

## 算子实现与实践演进

项目按技术复杂度由浅入深组织，核心练习路线与涵盖的技术点如下：

### 核心实践路线

| 阶段 | 核心技术点 | 对应源码入口 |
|------|------------|--------------|
| **Lv1 标量点运算** | 2D grid/block 配置、线程坐标映射、边界检查 | `pixel_operator.cu` `invertKernelScalar` |
| **Lv2 向量化访存** | `uchar4` 向量化对齐读写、1D vs 2D 映射、降级处理 | `pixel_operator.cu` `invertKernelVec4` |
| **Lv3 共享内存卷积** | Shared memory tiling、Halo 边界协作加载、`__syncthreads()` | `convolution_engine.cu` `convolveKernelShared` |
| **Lv4 可分离卷积** | 算法复杂度优化 $O(K^2) \to O(2K)$、双 pass 与中间缓冲 | `convolution_engine.cu` `separableConvolve` |
| **Lv5 直方图统计** | Block 局部直方图、`atomicAdd` 两级规约合并 | `histogram_calculator.cu` `histogramKernelShared` |
| **Lv6 几何缩放** | 浮点坐标逆映射、双线性插值（Bilinear Interpolation） | `image_resizer.cu` `resizeBilinearKernel` |
| **Lv7 异步流水线** | `cudaStream_t` 多流并发、异步排队提交与批次同步 | `pipeline_processor.cu` + `execution_context.hpp` |

### 扩展算子模块

在核心主线的基础上，进一步扩展实现的常见图像处理算子：

| 模块 | 源码入口 | 涉及概念与实现挑战 |
|------|----------|--------------------|
| **形态学运算** | `morphology.cu` | 膨胀/腐蚀邻域遍历，基于 min/max 规约替代加权和 |
| **阈值分割** | `threshold.cu` | 全局阈值、Otsu 自适应阈值、局部均值滤波与数值溢出防护 |
| **高级滤波** | `filters.cu` | 3×3/5×5 高斯滤波、中值滤波（冒泡排序与分支开销思考）、双边滤波 |
| **几何变换** | `geometric.cu` | 旋转/仿射变换矩阵、逆映射坐标计算与边界填充 |
| **色彩空间** | `color_space.cu` | RGB $\leftrightarrow$ 灰度/HSV/YUV 转换、通道交错与对齐处理 |

> 算子实现细节与思考见 [核心算子实现记录](docs/learning-path.md) · [CUDA 技术点速查](docs/cuda-concepts.md)

---

## 快速开始

### 前置要求
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器（GCC 9+ / Clang 10+ / MSVC 2019+）
- NVIDIA GPU（仅运行测试和示例时需要，纯编译不需要）

```bash
git clone https://github.com/build-workbench/cudaimg.git
cd cudaimg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 运行单元测试（无 GPU 环境时 GPU 相关用例会自动 SKIP）
ctest --test-dir build --output-on-failure

# 运行示例程序
./build/bin/example_01_pixel        # Lv1-2 像素运算
./build/bin/example_02_convolution  # Lv3-4 卷积
./build/bin/example_03_histogram    # Lv5 直方图
./build/bin/pipeline_example        # Lv7 多流异步流水线
```

构建选项与测试说明详见 [构建与测试指南](docs/build-and-test.md)。

---

## 代码示例

```cpp
#include "cudaimg/cudaimg.hpp"
using namespace cudaimg;

int main() {
    HostImage host = ImageUtils::createHostImage(1920, 1080, 3);
    // ... 填充 host.data ...

    ImageProcessor proc;                                // 默认同步模式
    CudaImage gpu = proc.loadFromHost(host);            // H2D
    CudaImage blurred = proc.gaussianBlur(gpu, 5, 1.5f);
    CudaImage edges = proc.sobelEdgeDetection(blurred); // 单通道梯度幅值图
    HostImage result = proc.download(edges);            // D2H
}
```

多 Stream 异步流水线（Lv7）：

```cpp
ImageProcessor proc{ImageProcessor::Mode::Async};
CudaImage a = proc.loadFromHost(hostA);      // 在指定 stream 上异步排队
CudaImage b = proc.gaussianBlur(a, 5, 1.0f);
proc.synchronize();                          // 统一等待当前流完成
```

头文件只需引入 `#include "cudaimg/cudaimg.hpp"` 即可使用全部功能。

---

## 项目结构

```
include/cudaimg/
├── cudaimg.hpp              # 统一头文件入口
├── core/                    # Image / DeviceBuffer / ExecutionContext（RAII 封装）
├── operators/               # 各算子头文件声明（Lv1–Lv6 及扩展模块）
├── processing/              # ImageProcessor 门面 + PipelineProcessor（Lv7）
└── io/                      # 图像文件 I/O（基于 stb，可选）
src/                         # GPU kernel 与算子 CPU 参考实现
tests/                       # GoogleTest 单元测试（逐像素对齐验证）
examples/                    # 各阶段调用示例
benchmarks/                  # 简易基准测试工具
docs/                        # 实现笔记、概念速查与踩坑记录
```

---

## 实践笔记与文档

| 文档 | 说明 |
|------|------|
| [核心算子实现记录](docs/learning-path.md) | 逐算子技术细节解析与设计考量 |
| [CUDA 技术点速查](docs/cuda-concepts.md) | 核心概念 $\leftrightarrow$ 源码实现位置映射表 |
| [踩坑与设计反思](docs/pitfalls.md) | 实践过程中遇到的问题、权衡与易错点总结 |
| [构建与测试指南](docs/build-and-test.md) | 编译选项、依赖说明与测试运行指南 |

---

## 后续探索方向 (TODO)

- [ ] 使用 Nsight Compute 与 Nsight Systems 深入分析 kernel 的 occupancy、warp divergence 与内存吞吐
- [ ] 尝试利用 CUDA 纹理内存（Texture Memory）优化插值与边界寻址性能
- [ ] 探索 FP16 / Half 精度计算及 Tensor Core 在部分图像滤波中的可行性
- [ ] 引入 `cudaMallocAsync` 探索 Stream-Ordered 显存分配机制

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
