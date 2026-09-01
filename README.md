# cudaimg — CUDA 图像处理教学代码库

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

> 以图像算子为载体的 CUDA 渐进式教学库。每个 GPU kernel 都配有 CPU 参考实现做逐像素验证 —— 价值不在"能处理图像"，而在**一条可验证、可对照源码自学的 CUDA 主线**。

---

## 定位

| ✅ 是 | ❌ 不是 |
|------|--------|
| 按概念难度递进的 CUDA 学习主线（Lv1 → Lv7） | OpenCV `cv::cuda` 的替代品（后者已工业级优化） |
| 可读性优先、带验证的教学实现 | 追求 occupancy / 带宽极限的高性能库 |
| 源码即教材：概念 → 源码 → 测试 自学 | 手把手视频教程（不提供逐行讲解） |

**适合谁：** 有 C++ 基础（指针/模板/编译链接）、零 CUDA 经验、想系统入门的自学者与工程师。

**学完 Lv1–Lv7 你将能：** 写对 2D grid/block 索引与边界检查 · 用 shared memory + `__syncthreads()` 做 tiling 卷积 · 用 `atomicAdd` 做两级直方图规约 · 实现双线性插值的浮点坐标映射 · 用多 stream + async 组织流水线 · 用 CPU 参考独立验证 kernel 正确性。

---

## 特性

- **渐进式主线** — 7 阶段每阶段只引入一个新概念，从线程模型到多流流水线
- **逐像素可验证** — 每个 GPU kernel 对应 CPU 参考实现，`tests/` 做全量比对而非"能跑就行"
- **可导航** — 三层架构（`core` → `operators` → `processing`）+ 学习路径表，源码本身就是地图
- **可练手扩展** — 5 个自学模块（形态学/阈值/滤波/几何/色彩）复用主线概念，适合独立改造

---

## 学习路径

按顺序阅读，这是本项目的核心主线：

| 阶段 | 核心概念 | 入口源码 |
|------|----------|----------|
| **Lv1** | 2D grid/block、线程索引映射、边界检查 | `pixel_operator.cu` `invertKernelScalar` |
| **Lv2** | `uchar4` 向量化、1D vs 2D grid、dispatch fallback | `pixel_operator.cu` `invertKernelVec4` |
| **Lv3** | shared memory tiling、halo 加载、`__syncthreads()` | `convolution_engine.cu` `convolveKernelShared` |
| **Lv4** | 可分离卷积 O(n²k²)→O(n²k)、两 pass + 中间缓冲 | `convolution_engine.cu` `separableConvolve` |
| **Lv5** | block 级直方图、`atomicAdd` 两级规约 | `histogram_calculator.cu` `histogramKernelShared` |
| **Lv6** | 浮点坐标映射、双线性插值 | `image_resizer.cu` `resizeBilinearKernel` |
| **Lv7** | `cudaStream_t`、async 提交、batch 同步 | `pipeline_processor.cu` + `execution_context.hpp` |

<details>
<summary>自学模块（完成主线后练手，不在核心路径）</summary>

| 模块 | 入口 | 复用概念 | 新挑战 |
|------|------|----------|--------|
| 形态学 | `morphology.cu` | 邻域遍历 | min/max 规约替代加权和 |
| 阈值 | `threshold.cu` | 直方图 | 局部均值、数值溢出防护 |
| 滤波 | `filters.cu` | 邻域遍历 | 排序分支开销、双边权重 |
| 几何 | `geometric.cu` | 坐标逆映射 | 仿射矩阵、输出尺寸计算 |
| 色彩空间 | `color_space.cu` | 逐像素 | 浮点精度、通道拆分 |

</details>

> 详见 [学习路径详解](docs/learning-path.md) · [CUDA 概念速查](docs/cuda-concepts.md)

---

## 快速开始

**前置要求：** CUDA Toolkit 11.0+（含 `nvcc`）· CMake 3.18+ · C++17 编译器 · NVIDIA GPU（仅运行时需要）

```bash
git clone https://github.com/build-workbench/cudaimg.git
cd cudaimg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 测试（无 GPU 时 GPU 用例自动 SKIP，ImageIO 仍会执行）
ctest --test-dir build --output-on-failure

# 示例（按学习级别）
./build/bin/example_01_pixel        # Lv1-2 像素操作
./build/bin/example_02_convolution  # Lv3-4 卷积
./build/bin/example_03_histogram    # Lv5 直方图
./build/bin/pipeline_example        # Lv7 多流流水线
```

更多选项见 [构建与测试](docs/build-and-test.md)。

---

## 用法示例

```cpp
#include "cudaimg/cudaimg.hpp"
using namespace cudaimg;

int main() {
    HostImage host = ImageUtils::createHostImage(1920, 1080, 3);
    // ... 填充 host.data ...

    ImageProcessor proc;                          // 默认同步模式
    CudaImage gpu = proc.loadFromHost(host);      // H2D
    CudaImage blurred = proc.gaussianBlur(gpu, 5, 1.5f);
    CudaImage edges = proc.sobelEdgeDetection(blurred);
    HostImage result = proc.download(edges);      // D2H
}
```

多流流水线（Lv7）：

```cpp
ImageProcessor proc{ImageProcessor::Mode::Async};
CudaImage a = proc.loadFromHost(hostA);  // 同一 stream 上异步排队
CudaImage b = proc.gaussianBlur(a, 5, 1.0f);
proc.synchronize();                      // 统一等待
```

头文件只需 `#include "cudaimg/cudaimg.hpp"`。

---

## 项目结构

```
include/cudaimg/
├── cudaimg.hpp              # 统一入口
├── core/                    # 基础设施：Image / DeviceBuffer / ExecutionContext
├── operators/               # 算子实现（学习重点，Lv1–Lv6）
└── processing/              # 门面层 ImageProcessor + PipelineProcessor（Lv7）
src/                         # 对应 .cu/.cpp 实现
tests/                       # CPU 参考实现逐像素验证
examples/                    # 01_pixel / 02_convolution / 03_histogram / pipeline
benchmarks/                  # 手写计时基准（无外部依赖）
docs/
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [学习路径详解](docs/learning-path.md) | 逐 kernel 概念讲解 + 动手练习 |
| [CUDA 概念速查](docs/cuda-concepts.md) | 概念 → 源码位置映射表 |
| [构建与测试](docs/build-and-test.md) | 构建选项、CI 说明 |
| [坑点记录](docs/pitfalls.md) | 设计权衡与易错点（shared memory 预算、两级规约等） |

---

## 下一步

学完本项目后建议：精读 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 与 Best Practices Guide → 用 Nsight Compute / Systems 分析本项目 kernel 的 occupancy 与带宽 → 进阶 [CUTLASS](https://github.com/NVIDIA/cutlass) / [FlashAttention](https://github.com/Dao-AILab/flash-attention)。

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)
