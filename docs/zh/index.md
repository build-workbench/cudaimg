---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">MO</div>
    <div>
      <div class="home-title">Mini-OpenCV</div>
      <div class="home-subtitle">v3.0.0 · CUDA 14 · C++17</div>
    </div>
  </div>
  <div class="home-nav">
    <a href="./setup/quickstart">快速开始</a>
    <a href="./architecture/overview">架构</a>
    <a href="./api/">API</a>
    <a href="https://github.com/LessUp/mini-opencv" target="_blank">GitHub</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    高性能 CUDA 图像处理库，相比 CPU OpenCV 实现 <strong>30-50× 加速</strong>。
    支持 9+ 类算子，采用三层架构设计，提供简洁的 C++17 API。
    完整的 GoogleTest 测试套件和 Google Benchmark 性能基准。
  </div>
  <div class="home-stats">
    <span><strong>30-50×</strong> 加速</span>
    <span><strong>9+</strong> 算子</span>
    <span><strong>MIT</strong> 许可证</span>
  </div>
</div>

## 核心特性

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">⚡ 高性能计算</div>
    <div class="feature-card-desc">
      CUDA 内核优化：共享内存分块、原子操作、warp 级原语，实现最大吞吐量
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Shared Memory</a>
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Atomic Ops</a>
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Warp Primitives</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧠 智能内存管理</div>
    <div class="feature-card-desc">
      零拷贝优化最小化主机-设备传输，内存池复用减少分配开销
    </div>
    <div class="feature-tags">
      <a href="./architecture/memory-model" class="feature-tag">Zero-Copy</a>
      <a href="./architecture/memory-model" class="feature-tag">Memory Pool</a>
      <a href="./architecture/cuda-streams" class="feature-tag">Stream Async</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🏗️ 三层架构</div>
    <div class="feature-card-desc">
      Application → Operator → Infrastructure，清晰的职责分离
    </div>
    <div class="feature-tags">
      <a href="./api/core/image-processor" class="feature-tag">ImageProcessor</a>
      <a href="./api/" class="feature-tag">Operators</a>
      <a href="./api/core/device-buffer" class="feature-tag">DeviceBuffer</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📊 9+ 算子支持</div>
    <div class="feature-card-desc">
      卷积、形态学、几何变换、直方图、阈值、色彩空间、滤波器等
    </div>
    <div class="feature-tags">
      <a href="./api/operators/convolution" class="feature-tag">Convolution</a>
      <a href="./api/operators/morphology" class="feature-tag">Morphology</a>
      <a href="./api/operators/geometric" class="feature-tag">Geometric</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧪 完整测试</div>
    <div class="feature-card-desc">
      GoogleTest 单元测试 + Google Benchmark 性能基准，确保正确性和性能
    </div>
    <div class="feature-tags">
      <a href="./benchmarks/" class="feature-tag">Benchmarks</a>
      <a href="./tutorials/examples" class="feature-tag">Examples</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📖 双语文档</div>
    <div class="feature-card-desc">
      中英文完整的 API 参考、架构指南和教程
    </div>
    <div class="feature-tags">
      <a href="./api/" class="feature-tag">API Docs</a>
      <a href="./tutorials/examples" class="feature-tag">Tutorials</a>
    </div>
  </div>
</div>

## 性能对比

| 操作 | OpenCV CPU | Mini-OpenCV GPU | 加速比 |
|------|:----------:|:---------------:|:------:|
| 高斯模糊 (4K) | 45.2 ms | 1.2 ms | **37.7×** |
| Sobel 边缘检测 (4K) | 38.1 ms | 0.9 ms | **42.3×** |
| 双边滤波 (4K) | 180.5 ms | 4.8 ms | **37.6×** |
| 直方图均衡化 (4K) | 12.3 ms | 0.3 ms | **41.0×** |

*测试环境：RTX 4090 vs Intel i9-13900K，3840×2160 图像*

## 快速开始

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// 创建处理器并加载图像
ImageProcessor processor;
GpuImage gpu = processor.loadFromHost(hostImage);

// 应用操作（全部 GPU 加速）
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);

// 下载结果
HostImage result = processor.downloadImage(edges);
```

## 了解更多

<div class="docs-grid">
  <a href="./whitepaper/overview" class="doc-card">
    <div class="doc-icon">📄</div>
    <h3>技术白皮书</h3>
    <p>项目背景、技术选型、性能优化策略详解。</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./architecture/overview" class="doc-card">
    <div class="doc-icon">🏗️</div>
    <h3>架构设计</h3>
    <p>三层架构设计，CUDA 内核优化详解。</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./api/" class="doc-card">
    <div class="doc-icon">📚</div>
    <h3>API 参考</h3>
    <p>完整的 API 文档和示例代码。</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./benchmarks/" class="doc-card">
    <div class="doc-icon">📊</div>
    <h3>性能基准</h3>
    <p>性能数据和优化技术详解。</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./references/" class="doc-card">
    <div class="doc-icon">📖</div>
    <h3>学术引用</h3>
    <p>相关论文和开源项目。</p>
    <span class="doc-arrow">→</span>
  </a>
</div>
