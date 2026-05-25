# Mini-OpenCV — GPU Image Processing Library

[![CI](https://github.com/LessUp/mini-opencv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/mini-opencv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/mini-opencv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)
![Version](https://img.shields.io/badge/Version-3.0.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

[English](README.md) | [简体中文](README.zh-CN.md)

A high-performance CUDA-based image processing library providing GPU-accelerated operators for computer vision applications.

> **⚡ Performance**: 30-50x faster than CPU OpenCV for comparable operations
>
> *Tested on RTX 4090 with 4K images vs OpenCV 4.8 CPU implementation. See [benchmarks/](benchmarks/) for details.*

---

## 📚 Quick Links

| Resource | Description |
|----------|-------------|
| [Installation](https://lessup.github.io/mini-opencv/en/setup/installation) | Complete setup guide |
| [Quick Start](https://lessup.github.io/mini-opencv/en/setup/quickstart) | Get started in 5 minutes |
| [Architecture](https://lessup.github.io/mini-opencv/en/architecture/overview) | Three-layer design overview |
| [API Reference](https://lessup.github.io/mini-opencv/en/api/) | Complete API documentation |
| [Examples](https://lessup.github.io/mini-opencv/en/tutorials/examples) | Code examples and tutorials |

**Full Documentation:** https://lessup.github.io/mini-opencv/

---

## ✨ Features

| Category | Operators | Highlights |
|----------|-----------|------------|
| **Pixel Operations** | Invert, grayscale, brightness | Per-pixel parallel |
| **Convolution** | Gaussian blur, Sobel edge detection | Shared memory tiling |
| **Histogram** | Calculation, equalization | Atomic ops + reduction |
| **Geometric** | Rotate, resize, flip, affine | Bilinear interpolation |
| **Morphology** | Erosion, dilation, open/close | Custom structuring elements |
| **Threshold** | Global, adaptive, Otsu | Histogram-driven |
| **Color Space** | RGB/HSV/YUV conversion | Batch conversion |
| **Filters** | Median, bilateral, sharpen | Edge-preserving |
| **Pipeline** | Multi-step async processing | Multi-stream concurrency |

---

## 🏗️ Architecture

> **See the [Architecture Overview](https://lessup.github.io/mini-opencv/en/architecture/overview) for complete design details.**

Three-layer design: **High-level APIs** → **CUDA Kernels** → **Infrastructure**

```
Application Layer → Operator Layer → Infrastructure Layer
```

---

## 🚀 Quick Start

```bash
# Clone and build
git clone https://github.com/LessUp/mini-opencv.git
cd mini-opencv
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Run example
./build/bin/basic_example
```

### Basic Usage

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// Step 1: Create an ImageProcessor
ImageProcessor processor;

// Step 2: Create/Load a host image
HostImage hostImage = ImageUtils::createHostImage(1920, 1080, 3);
// Fill hostImage.data with your pixel data (RGB, 8-bit per channel)

// Step 3: Upload to GPU and process
GpuImage gpu = processor.loadFromHost(hostImage);
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);

// Step 4: Download result back to host
HostImage result = processor.downloadImage(edges);
```

---

## 📋 Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA | 11.0 | 12.x |
| CMake | 3.18 | 3.24+ |
| C++ | C++17 | C++17 |
| GPU | CC 7.5+ (Turing) | RTX 30/40 series |

### Supported Image Formats

| Format | Read | Write |
|--------|------|-------|
| JPEG/JPG | ✓ | ✓ |
| PNG | ✓ | ✓ |
| BMP | ✓ | ✓ |
| TGA | ✓ | ✗ |

**Note:** All formats support 8-bit per channel (Grayscale, RGB, RGBA)

---

## 📖 Documentation

Complete documentation available at [GitHub Pages](https://lessup.github.io/mini-opencv/):

- [Installation Guide](https://lessup.github.io/mini-opencv/en/setup/installation)
- [Quick Start](https://lessup.github.io/mini-opencv/en/setup/quickstart)
- [Architecture Overview](https://lessup.github.io/mini-opencv/en/architecture/overview)
- [Benchmarks](https://lessup.github.io/mini-opencv/en/benchmarks/)
- [API Reference](https://lessup.github.io/mini-opencv/en/api/)
- [FAQ](https://lessup.github.io/mini-opencv/en/tutorials/faq)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## 🛠️ Troubleshooting

**Q: CMake cannot find CUDA / `nvcc` not found**
```bash
# Set CUDA path explicitly
export CUDAToolkit_ROOT=/usr/local/cuda
# Or when configuring:
cmake -S . -B build -DCUDAToolkit_ROOT=/usr/local/cuda
```

**Q: Running examples shows "CUDA is not available"**
- Ensure you have an NVIDIA GPU with Compute Capability 7.5+
- Install CUDA Toolkit 11.0+ from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
- Verify: `nvidia-smi` should show your GPU

**Q: Tests fail or crash during execution**
- Check GPU memory: Large images may require more VRAM
- Try smaller test images or reduce batch size
- Run with `ctest --test-dir build --output-on-failure -V` for verbose output

**Q: How to verify installation is successful?**
```bash
cd build && ctest --output-on-failure
# All tests should pass
```

For more issues, check [GitHub Discussions](https://github.com/LessUp/mini-opencv/discussions).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

**⭐ Star this repo if you find it helpful!**

For support, open an [issue](https://github.com/LessUp/mini-opencv/issues) or start a [discussion](https://github.com/LessUp/mini-opencv/discussions).
