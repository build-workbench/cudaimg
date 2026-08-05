# 配置说明

Mini-OpenCV 的 CMake 配置选项。

## 构建类型

| 类型 | 描述 |
|------|------|
| `Release` | 优化构建（默认） |
| `Debug` | 调试符号，无优化 |
| `RelWithDebInfo` | 带调试信息的发布版 |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## CMake 选项

### 构建选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `GPU_IMAGE_ENABLE_IO` | ON | 启用图像文件 I/O（通过 stb） |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_TESTS` | ON | 构建 GoogleTest 套件 |
| `BUILD_BENCHMARKS` | OFF | 构建性能基准测试 |

库始终以静态库方式构建——`BUILD_SHARED_LIBS` 无效。

### CUDA 选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `CMAKE_CUDA_ARCHITECTURES` | auto | GPU 架构（如 `"75;80;86;89"`） |

`CMAKE_CUDA_ARCHITECTURES` 是标准 CMake 变量。未设置时，CMake 3.24+ 默认为
`native`（自动检测本机 GPU），否则默认为 `75;80;86;89`。

### 语言标准

| 范围 | 标准 |
|------|------|
| 主机代码（C++） | C++17 |
| 设备代码（CUDA） | C++14（通过 `CMAKE_CUDA_STANDARD`） |

不存在 "CUDA 14" 这个版本——"14" 指的是设备代码所用的 C++ 标准。

## GPU 架构

### 自动检测

默认情况下，CMake 检测已安装的 GPU（CMake 3.24+）：

```bash
cmake -S . -B build  # 自动检测
```

### 手动指定

```bash
# 单一架构
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89

# 多架构
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

### 架构参考

| 架构 | 计算能力 | GPU |
|------|---------|-----|
| Turing | 75 | RTX 20xx, T4 |
| Ampere | 80 | A100 |
| Ampere | 86 | RTX 30xx |
| Ada | 89 | RTX 40xx, L4 |
| Hopper | 90 | H100 |

## 示例配置

### 开发环境

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON
```

### 生产环境

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF
```

### 性能测试

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

## 下一步

- [快速入门](./quickstart) - 基本用法
- [示例代码](../tutorials/examples) - 代码示例
