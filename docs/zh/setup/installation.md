# 安装指南

Mini-OpenCV 完整安装指南。

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CUDA | 11.0 | 12.x |
| CMake | 3.18 | 3.24+ |
| C++ | C++17 | C++17 |
| GPU | CC 7.5+ | RTX 30/40 系列 |

## 安装 CUDA

### Ubuntu

```bash
# 添加 NVIDIA 软件源
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# 安装 CUDA
sudo apt install cuda-toolkit-12-4

# 添加到 PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 验证安装

```bash
nvcc --version
nvidia-smi
```

## 构建 Mini-OpenCV

```bash
# 克隆
git clone https://github.com/AICL-Lab/mini-opencv.git
cd mini-opencv

# 配置选项
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_TESTS=ON

# 构建
cmake --build build -j$(nproc)
```

## CMake 选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `GPU_IMAGE_ENABLE_IO` | ON | 启用图像文件 I/O（通过 stb） |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_TESTS` | ON | 构建测试套件 |
| `BUILD_BENCHMARKS` | OFF | 构建基准测试 |
| `CMAKE_CUDA_ARCHITECTURES` | auto | GPU 架构（标准 CMake 变量） |

库始终以静态库方式构建（`BUILD_SHARED_LIBS` 无效）。
未设置 `CMAKE_CUDA_ARCHITECTURES` 时，CMake 3.24+ 默认为 `native`
（自动检测本机 GPU），否则默认为 `75;80;86;89`。

## 验证构建

```bash
# 运行测试
ctest --test-dir build --output-on-failure

# 运行示例
./build/bin/basic_example
```

## 故障排除

### 找不到 CUDA

```bash
export CUDAToolkit_ROOT=/usr/local/cuda
cmake -S . -B build
```

### GPU 架构错误

```bash
# 显式指定架构
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

## 下一步

- [快速入门](./quickstart) - 基本用法
- [配置说明](./configuration) - 高级选项