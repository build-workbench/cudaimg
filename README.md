# cudaimg — CUDA 图像处理教学代码库

![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

以图像处理算子为载体、按 CUDA 概念难度递进组织的**教学代码库**。每个 GPU kernel
都配有 CPU 参考实现做逐像素验证——核心价值不是"能处理图像"，而是**一条可验证、
可对照源码自学的 CUDA 学习主线**。

> **定位边界**：这是学 CUDA 的代码库，不是图像处理产品库——代码以可读性优先，
> 不追求极致性能，也不替代 OpenCV。

---

## 适合谁，读完能得到什么

**适合谁**：有 C++ 基础（指针、模板、编译链接都没问题）、想系统学习 CUDA 的
自学者、转行者或工程师。**不需要任何 CUDA 经验**——主线从最简单的 kernel 起步。

**读完核心主线（Lv1–Lv7）后**，你可以：

- 写出正确的 2D grid/block kernel，理解线程索引映射与边界检查
- 用 shared memory + `__syncthreads()` 写 tiling 卷积，清楚 halo 区域的加载方式
- 用 `atomicAdd` 做 block 级直方图规约
- 实现双线性插值等浮点坐标映射的 kernel
- 用多 stream + async 提交组织多步流水线
- 用 CPU 参考实现做逐像素比对，独立验证 GPU kernel 的正确性

## 这是什么

- **渐进式学习主线** — 核心路径（Lv1–Lv7）按 CUDA 概念难度递进，每阶段引入一个新概念，覆盖线程模型到多流流水线
- **可验证** — 每个 GPU kernel 都配有 CPU 参考实现，测试做逐像素比对，不是"能跑就行"
- **可导航** — 三层架构（基础设施 → 算子 → 门面）+ 学习路径表格，源码本身就是阅读地图
- **可扩展自学** — 5 个自学模块（形态学、阈值、滤波、几何、色彩空间）复用主线概念，适合练手改造

## 这不是什么

- **不是 OpenCV 替代品** — `cv::cuda` 已覆盖全部功能且经过工业级优化
- **不是高性能库** — kernel 实现为可读性优先，未做 occupancy 调优
- **不是手把手教程** — 学习路径是源码阅读地图，按"概念 → 源码 → 测试"自学，不提供逐行讲解

---

## 学习路径

按以下顺序阅读源码，每个阶段引入一个新的 CUDA 概念。这是项目的**核心主线**：

| 阶段 | 源文件 | 学到的 CUDA 概念 |
|------|--------|-----------------|
| **Lv1** | `src/operators/pixel_operator.cu` | 2D grid/block 配置、线程索引映射、边界检查 |
| **Lv2** | `src/operators/pixel_operator.cu`（vec4 路径） | `uchar4` 向量化读写、1D vs 2D grid 选择、dispatch fallback 模式 |
| **Lv3** | `src/operators/convolution_engine.cu` | shared memory tiling、halo 区域加载、`__syncthreads()`、三种边界策略 |
| **Lv4** | `src/operators/convolution_engine.cu`（separable） | 算法优化：O(n²k²) → O(n²k)，两 pass + 中间缓冲 |
| **Lv5** | `src/operators/histogram_calculator.cu` | block 级 shared memory 直方图、`atomicAdd` 规约到全局 |
| **Lv6** | `src/operators/image_resizer.cu` | 浮点坐标映射、双线性插值的 GPU 实现 |
| **Lv7** | `src/processing/pipeline_processor.cu` | 多 stream 创建/销毁、async 提交、batch 同步 |

## 自学模块

以下模块不在核心学习路径中，是路径完成后的自学练习材料——每个都复用了
主线的概念，但引入了新的工程问题（分支发散、原子竞争、浮点精度等），
适合作为独立阅读或练手改造的素材：

| 模块 | 源文件 | 复用的概念 | 新的挑战 |
|------|--------|-----------|---------|
| 形态学 | `src/operators/morphology.cu` | Lv1 邻域遍历、结构元素掩码 | min/max 规约替代加权和 |
| 阈值处理 | `src/operators/threshold.cu` | Lv5 直方图（Otsu 复用直方图计算） | 局部窗口均值、数值溢出防护 |
| 中值/双边滤波 | `src/operators/filters.cu` | Lv3 邻域遍历 | 排序找中值的分支开销、双边权重计算 |
| 几何变换 | `src/operators/geometric.cu` | Lv6 坐标逆映射 | 仿射矩阵、输出尺寸计算 |
| 色彩空间 | `src/operators/color_space.cu` | Lv1 逐像素操作 | 浮点精度、通道拆分合并 |

> 详见 [学习路径详解](docs/learning-path.md)

---

## 快速开始

```bash
git clone https://github.com/build-workbench/cudaimg.git
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
