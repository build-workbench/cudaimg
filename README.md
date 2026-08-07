# cudaimg — CUDA 图像处理入门教程

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

通过图像处理算子学习 CUDA 编程。每个 GPU kernel 都有 CPU 参考实现做正确性验证，按概念难度递进组织。

> **教学项目，已归档。** 代码以可读性优先，不追求极致性能，不替代 OpenCV。

---

## 这是什么

- 一组用 CUDA 实现的图像处理算子（像素操作、卷积、直方图、几何变换、形态学等）
- 每个 GPU kernel 配有 CPU 参考实现，通过逐像素比对验证正确性
- 按 CUDA 概念难度从简单到复杂递进，适合从零开始学 GPU 编程
- 三层架构（基础设施 → 算子 → 门面），代码可导航

## 这不是什么

- **不是 OpenCV 替代品** — `cv::cuda` 已覆盖全部功能且经过工业级优化
- **不是高性能库** — kernel 实现为教学清晰度优先，未做 occupancy 调优
- **不接受功能请求** — 项目已归档，作为学习参考保留

---

## 学习路径

按以下顺序阅读源码，每个阶段引入一个新的 CUDA 概念：

| 阶段 | 源文件 | 学到的 CUDA 概念 |
|------|--------|-----------------|
| **Lv1** | `src/operators/pixel_operator.cu` | 2D grid/block 配置、线程索引映射、边界检查 |
| **Lv2** | `src/operators/pixel_operator.cu`（vec4 路径） | `uchar4` 向量化读写、1D vs 2D grid 选择、dispatch fallback 模式 |
| **Lv3** | `src/operators/convolution_engine.cu` | shared memory tiling、halo 区域加载、`__syncthreads()`、三种边界策略 |
| **Lv4** | `src/operators/convolution_engine.cu`（separable） | 算法优化：O(n²k²) → O(n²k)，两 pass + 中间缓冲 |
| **Lv5** | `src/operators/histogram_calculator.cu` | block 级 shared memory 直方图、`atomicAdd` 规约到全局 |
| **Lv6** | `src/operators/image_resizer.cu` | 浮点坐标映射、双线性插值的 GPU 实现 |
| **Lv7** | `src/processing/pipeline_processor.cu` | 多 stream 创建/销毁、async 提交、batch 同步 |

> 详见 [学习路径详解](docs/learning-path.md)

---

## 快速开始

```bash
git clone https://github.com/AICL-Lab/cudaimg.git
cd cudaimg
cmake -S . -B build
cmake --build build -j$(nproc)

# 运行测试（需要 NVIDIA GPU）
ctest --test-dir build --output-on-failure

# 运行示例（按学习级别）
./build/bin/example_01_pixel
./build/bin/example_02_convolution
./build/bin/example_03_histogram
./build/bin/pipeline_example
```

> 完整构建选项见 [构建与测试](docs/build-and-test.md)

---

## 项目结构

```
include/cudaimg/
├── cudaimg.hpp              # 统一头文件（include 这个就够了）
├── core/
│   ├── image.hpp            # CudaImage / HostImage 数据结构
│   ├── device_buffer.hpp    # RAII 显存管理
│   ├── execution_context.hpp # 执行策略（sync/async/batch）
│   ├── device_kernels.cuh   # 设备端共享工具（clamp、索引等）
│   └── kernel_helpers.hpp   # 主机端 kernel 启动辅助
├── operators/               # CUDA kernel 实现（学习重点）
│   ├── pixel_operator.cu    # Lv1-2：最简单的 kernel + 向量化
│   ├── convolution_engine.cu # Lv3-4：shared memory + 可分离卷积
│   ├── histogram_calculator.cu # Lv5：原子操作 + 规约
│   ├── image_resizer.cu     # Lv6：插值与坐标映射
│   ├── morphology.cu        # 形态学（min/max reduction）
│   ├── threshold.cu         # 阈值处理
│   ├── filters.cu           # 中值/双边/锐化滤波 + 图像算术
│   ├── geometric.cu         # 旋转/翻转/裁剪/仿射
│   └── color_space.cu       # RGB/HSV/YUV 转换
└── processing/
    ├── image_processor.hpp  # 门面层：一行调用一个算子
    └── pipeline_processor.cu # Lv7：多流流水线

tests/                       # 每个算子配 CPU 参考实现做逐像素验证
examples/                    # 按学习级别的渐进示例 + 流水线示例
benchmarks/                  # 手写计时器基准（非 Google Benchmark）
```

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [学习路径详解](docs/learning-path.md) | 逐文件讲解每个 kernel 涉及的 CUDA 概念 |
| [CUDA 概念速查](docs/cuda-concepts.md) | 概念 → 源码位置映射表 |
| [坑点记录](docs/pitfalls.md) | 已知设计权衡和容易踩的坑 |
| [构建与测试](docs/build-and-test.md) | 构建选项、运行测试、CI 说明 |

---

## 下一步学什么

学完本项目后，可以继续深入：

- **GPU 架构** — NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 和 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- **性能分析** — 用 Nsight Compute / Nsight Systems 分析本项目的 kernel，理解 occupancy、memory bandwidth、warp divergence
- **高级 kernel** — [CUTLASS](https://github.com/NVIDIA/cutlass)（矩阵乘法）、[FlashAttention](https://github.com/Dao-AILab/flash-attention)（注意力机制）
- **推理引擎** — vLLM PagedAttention、TensorRT-LLM、量化（INT8/FP8）

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)
