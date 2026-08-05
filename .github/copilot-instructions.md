# GitHub Copilot 指令

Copilot 在 mini-opencv 仓库中工作时使用的项目专属指令。

## 项目概览

**Mini-OpenCV** 是基于 CUDA 的高性能图像处理库，为计算机视觉应用提供 GPU 加速算子。算子以 CUDA 内核实现（shared memory tiling、原子直方图、`uchar4` 向量化）。

- **版本**：3.0.0
- **许可证**：MIT
- **文档**：https://aicl-lab.github.io/mini-opencv/

## 技术栈

| 组件 | 技术 |
|-----------|------------|
| 语言 | C++17、CUDA 17 |
| 构建系统 | CMake 3.18+ |
| CUDA | 11.0+（推荐 12.x） |
| GPU | 计算能力 7.5+（Turing 或更新） |
| 测试 | Google Test v1.14.0 |
| 基准测试 | 自包含手写计时框架（无外部依赖） |
| 图像 I/O | stb（通过 CMake FetchContent，可由 `GPU_IMAGE_ENABLE_IO` 关闭） |
| 文档 | VitePress（中文） |

## 架构

三层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层                                     │
│         ImageProcessor  ·  PipelineProcessor                 │
├─────────────────────────────────────────────────────────────┤
│              算子层 (CUDA Kernels)                           │
│  PixelOperator  │  ConvolutionEngine  │  Geometric          │
│  Morphology     │  ColorSpace         │  Filters            │
│  Threshold      │  HistogramCalculator│  ImageResizer       │
├─────────────────────────────────────────────────────────────┤
│                  基础设施层                                  │
│  DeviceBuffer  ·  GpuImage/HostImage  ·  CudaError          │
│  ImageIO       ·  MemoryManager  ·  ExecutionContext        │
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
mini-opencv/
├── include/gpu_image/          # 公共头文件
│   ├── core/                   # DeviceBuffer、GpuImage、CudaError、memory_manager
│   ├── operators/              # CUDA 算子接口
│   ├── processing/             # ImageProcessor、PipelineProcessor
│   ├── io/                     # ImageIO
│   └── gpu_image_processing.hpp
├── src/                        # 实现（.cpp / .cu）
├── tests/                      # Google Test 测试套件
├── examples/                   # 示例程序
├── benchmarks/                 # 性能基准
├── docs/                       # VitePress 文档（中文）
└── .github/workflows/          # CI/CD
```

## 构建命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

构建选项：`BUILD_TESTS`（ON）、`BUILD_EXAMPLES`（ON）、`BUILD_BENCHMARKS`（OFF）、`GPU_IMAGE_ENABLE_IO`（ON）。

## 测试命令

```bash
# 通过 CTest 运行全套测试
ctest --test-dir build --output-on-failure

# 直接运行一个测试夹具
./build/bin/gpu_image_tests --gtest_filter=FiltersTest.*

# 运行单个测试
./build/bin/gpu_image_tests --gtest_filter=FiltersTest.MedianFilter
```

## 代码风格

- **缩进**：2 空格，80 列上限（`.clang-format` 为权威）
- **命名**：
  - 文件：`snake_case.hpp/.cpp/.cu`
  - 类/结构体/枚举：`PascalCase`
  - 函数/方法：`lowerCamelCase`
  - 私有成员：`snake_case_`（尾随下划线）
- **命名空间**：`gpu_image`
- **头文件保护**：`#pragma once`

## CUDA 内核规则

1. 始终检查边界：`if (x < width && y < height)`
2. 线程块大小：256 线程（2D 用 16×16）
3. 内核启动后调用 `CUDA_CHECK(cudaGetLastError())`
4. 支持异步的算子接受 `cudaStream_t stream = nullptr`

## 测试约定

- 使用 `TEST_F(...)` 夹具；在 `SetUp()` 中检查 `cudaGetDeviceCount()`，无 CUDA 时调用 `GTEST_SKIP()`
- 优先使用确定性输入和明确的期望值
- 无效输入测试使用 `EXPECT_THROW`
- 新测试文件需注册到 `gpu_image_tests` CMake 目标

## API 设计

- 单参数构造函数加 `explicit`
- 不可忽略返回值的访问器加 `[[nodiscard]]`
- 平凡访问器和移动操作加 `noexcept`
- 优先用 `enum class` 而非未作用域枚举
- 只读重型输入用 `const T&`
- 维度/通道用 `int`，字节数用 `size_t`，像素用 `unsigned char`

## 错误处理

- 在每个公共函数顶部校验输入
- 错误的调用者输入用 `std::invalid_argument`
- 运行时失败用 `std::runtime_error`
- 所有 CUDA 运行时调用用 `CUDA_CHECK(...)` 包裹
- 低层算子检查 `cudaGetLastError()`；高层包装在返回前调用 `cudaDeviceSynchronize()`

## 格式化

```bash
# 检查（CI 等价）
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './third_party/*' -print0 | \
  xargs -0 -r clang-format --dry-run --Werror

# 原地修复
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './third_party/*' -print0 | \
  xargs -0 -r clang-format -i
```

## 提交信息

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：`<type>(<scope>): <subject>`。

类型：`feat`、`fix`、`docs`、`refactor`、`perf`、`test`、`chore`、`ci`。
