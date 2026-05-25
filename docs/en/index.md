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
    <a href="./setup/quickstart">Quick Start</a>
    <a href="./architecture/overview">Architecture</a>
    <a href="./api/">API</a>
    <a href="https://github.com/LessUp/mini-opencv" target="_blank">GitHub</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    High-performance CUDA image processing library achieving <strong>30-50× speedup</strong> over CPU OpenCV.
    Supports 9+ operator categories with a three-layer architecture and clean C++17 API.
    Comprehensive GoogleTest suite and Google Benchmark included.
  </div>
  <div class="home-stats">
    <span><strong>30-50×</strong> Speedup</span>
    <span><strong>9+</strong> Operators</span>
    <span><strong>MIT</strong> License</span>
  </div>
</div>

## Core Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">⚡ High Performance</div>
    <div class="feature-card-desc">
      CUDA kernel optimizations: shared memory tiling, atomic operations, warp-level primitives
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Shared Memory</a>
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Atomic Ops</a>
      <a href="./architecture/overview#cuda-optimization" class="feature-tag">Warp Primitives</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧠 Smart Memory</div>
    <div class="feature-card-desc">
      Zero-copy optimization minimizes host-device transfers, memory pool reuse reduces allocation overhead
    </div>
    <div class="feature-tags">
      <a href="./architecture/memory-model" class="feature-tag">Zero-Copy</a>
      <a href="./architecture/memory-model" class="feature-tag">Memory Pool</a>
      <a href="./architecture/cuda-streams" class="feature-tag">Stream Async</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🏗️ Three-Layer Architecture</div>
    <div class="feature-card-desc">
      Application → Operator → Infrastructure, clear separation of concerns
    </div>
    <div class="feature-tags">
      <a href="./api/core/image-processor" class="feature-tag">ImageProcessor</a>
      <a href="./api/" class="feature-tag">Operators</a>
      <a href="./api/core/device-buffer" class="feature-tag">DeviceBuffer</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📊 9+ Operators</div>
    <div class="feature-card-desc">
      Convolution, morphology, geometric transforms, histogram, threshold, color space, filters, etc.
    </div>
    <div class="feature-tags">
      <a href="./api/operators/convolution" class="feature-tag">Convolution</a>
      <a href="./api/operators/morphology" class="feature-tag">Morphology</a>
      <a href="./api/operators/geometric" class="feature-tag">Geometric</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧪 Well Tested</div>
    <div class="feature-card-desc">
      GoogleTest unit tests + Google Benchmark performance baselines ensure correctness and performance
    </div>
    <div class="feature-tags">
      <a href="./benchmarks/" class="feature-tag">Benchmarks</a>
      <a href="./tutorials/examples" class="feature-tag">Examples</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📖 Bilingual Docs</div>
    <div class="feature-card-desc">
      Complete API reference, architecture guides, and tutorials in English and Chinese
    </div>
    <div class="feature-tags">
      <a href="./api/" class="feature-tag">API Docs</a>
      <a href="./tutorials/examples" class="feature-tag">Tutorials</a>
    </div>
  </div>
</div>

## Performance Comparison

| Operation | OpenCV CPU | Mini-OpenCV GPU | Speedup |
|-----------|:----------:|:---------------:|:-------:|
| Gaussian Blur (4K) | 45.2 ms | 1.2 ms | **37.7×** |
| Sobel Edge (4K) | 38.1 ms | 0.9 ms | **42.3×** |
| Bilateral Filter (4K) | 180.5 ms | 4.8 ms | **37.6×** |
| Histogram Equalization (4K) | 12.3 ms | 0.3 ms | **41.0×** |

*Tested on RTX 4090 vs Intel i9-13900K, 3840×2160 images*

## Quick Start

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// Create processor and load image
ImageProcessor processor;
GpuImage gpu = processor.loadFromHost(hostImage);

// Apply operations (all GPU-accelerated)
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);

// Download result
HostImage result = processor.downloadImage(edges);
```

## Learn More

<div class="docs-grid">
  <a href="./whitepaper/overview" class="doc-card">
    <div class="doc-icon">📄</div>
    <h3>Technical Whitepaper</h3>
    <p>Project background, tech stack decisions, and optimization strategies.</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./architecture/overview" class="doc-card">
    <div class="doc-icon">🏗️</div>
    <h3>Architecture</h3>
    <p>Three-layer design with CUDA kernel optimizations.</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./api/" class="doc-card">
    <div class="doc-icon">📚</div>
    <h3>API Reference</h3>
    <p>Complete API documentation with examples.</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./benchmarks/" class="doc-card">
    <div class="doc-icon">📊</div>
    <h3>Benchmarks</h3>
    <p>Performance data and optimization techniques.</p>
    <span class="doc-arrow">→</span>
  </a>

  <a href="./references/" class="doc-card">
    <div class="doc-icon">📖</div>
    <h3>References</h3>
    <p>Academic papers and related projects.</p>
    <span class="doc-arrow">→</span>
  </a>
</div>
