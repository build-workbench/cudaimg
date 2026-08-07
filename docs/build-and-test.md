# 构建与测试

## 前置要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| CUDA Toolkit | 11.0 | 需要 `nvcc` 编译器 |
| CMake | 3.18 | 支持 `check_language(CUDA)` 和 `FetchContent` |
| C++ 编译器 | C++17 | GCC 9+ / Clang 10+ / MSVC 2019+ |
| NVIDIA GPU | CC 7.5+ (Turing) | 运行时需要，编译不需要 |

验证环境：
```bash
nvcc --version       # CUDA 编译器
cmake --version      # CMake
nvidia-smi           # GPU 驱动和设备
```

## 构建选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_TESTS` | ON | 构建单元测试（需要 GoogleTest，自动 FetchContent） |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_BENCHMARKS` | OFF | 构建基准测试（手写计时器，无外部依赖） |
| `CUDAIMG_ENABLE_IO` | ON | 启用图像文件 I/O（通过 stb，自动 FetchContent） |
| `CUDAIMG_ENABLE_INSTALL` | OFF | 生成 install target 和包配置 |

## 基本构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 运行测试

```bash
ctest --test-dir build --output-on-failure
```

无 GPU 时，GPU 相关测试会标记为 SKIP。ImageIO 测试不需要 GPU，必须通过。

## 运行示例

```bash
./build/bin/basic_example       # 基础算子演示
./build/bin/pipeline_example    # 多步流水线演示
```

## 运行基准测试

```bash
cmake -S . -B build -DBUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
./build/bin/cudaimg_benchmark
```

基准测试使用手写 `std::chrono` 计时器，测量 GPU 绝对延迟。包含 10 次
预热 + 100 次计时取平均。不提供与 OpenCV 的对比数据。

## CI 说明

GitHub Actions CI 在 `nvidia/cuda:12.4.1-devel-ubuntu22.04` 容器中构建。
由于 runner 无 GPU，CI 验证的是：
1. 代码能编译通过（CUDA + C++ + 测试 + 示例）
2. ImageIO 测试通过（CPU 路径）
3. 代码格式符合 clang-format 规范

GPU 测试的正确性需要在本地有 NVIDIA GPU 的机器上运行。
