# 学习路径详解

本文按 CUDA 概念难度递进顺序，逐文件讲解每个 kernel 涉及的技术点。
建议按顺序阅读源码，每个阶段先理解概念，再读代码，最后看测试。

---

## Lv1：最简单的 CUDA kernel

**文件**：`src/operators/pixel_operator.cu` — `invertKernelScalar`

这是整个项目中最简单的 kernel，适合作为 CUDA 入门的第一课。

```cpp
__global__ void invertKernelScalar(const unsigned char* input,
                                   unsigned char* output, int width,
                                   int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
      output[idx + c] = 255 - input[idx + c];
    }
  }
}
```

### 学到的概念

1. **grid/block/thread 层次** - `blockIdx`、`threadIdx`、`blockDim` 的关系。每个线程负责一个像素。
2. **2D 索引映射** - `(x, y)` 坐标如何映射到线性内存索引 `(y * width + x) * channels + c`
3. **边界检查** - `if (x < width && y < height)` 防止线程越界访问（grid 维度向上取整会产生多余线程）
4. **kernel 启动** - 主机端如何配置 `dim3 block(16, 16)` 和 `dim3 grid(...)` 并启动 kernel

### 配套测试

`tests/operators/test_pixel_operator.cpp` 中的 CPU 参考实现：

```cpp
// CPU 端：逐像素反色
for (size_t i = 0; i < image.data.size(); ++i) {
  expected.data[i] = 255 - image.data[i];
}
```

GPU 结果与 CPU 参考逐像素比对。这就是验证 CUDA 程序正确性的基本方法。

### 动手练习

1. 修改 `invertKernelScalar`，让它只反转 R 通道（channels == 3 时），G/B 保持不变
2. 把 block 大小从 16×16 改为 32×32，观察是否能正常工作（提示：shared memory 限制）

---

## Lv2：向量化优化与 dispatch

**文件**：`src/operators/pixel_operator.cu` — `invertKernelVec4` + `launchInvert`

同一个反色操作，用 `uchar4` 向量化版本处理 4 个字节/线程：

```cpp
__global__ void invertKernelVec4(const uchar4* input, uchar4* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uchar4 v = input[i];
    output[i] = make_uchar4(255 - v.x, 255 - v.y, 255 - v.z, 255 - v.w);
  }
}
```

### 学到的概念

1. **`uchar4` 向量化** - CUDA 内建向量类型，一条指令读写 4 字节，减少内存事务数量
2. **1D vs 2D grid** - 向量化版本用 1D grid（像素按字节连续排列），标量版本用 2D grid（按 x/y 坐标）
3. **dispatch fallback 模式** - 主机端 `launchInvert` 根据图像尺寸自动选择快路径或 fallback：

```cpp
static inline bool canVectorize(const CudaImage& img) {
  return img.totalBytes() % 4 == 0;  // 总字节数能被 4 整除
}
```

这是真实 CUDA 工程中常见的模式：有快路径时走快路径，不满足条件时退回到通用实现。

### 动手练习

1. 查看 `canVectorize()` 的逻辑，思考：为什么 3 通道 RGB 图像（width=100）可以向量化？
2. 尝试写一个 `toGrayscaleVec4` 版本，用 `uchar4` 同时处理 RGBA 图像的灰度转换

---

## Lv3：Shared Memory 卷积

**文件**：`src/operators/convolution_engine.cu` — `convolveKernelShared`

这是本项目的核心教学 kernel。卷积是 memory-bound 操作，朴素实现每个线程读 k² 个全局内存像素，相邻线程大量重复读取。Shared memory tiling 是标准优化手段。

```cpp
template <int BLOCK_SIZE>
__global__ void convolveKernelShared(...) {
  const int halo = kernelSize / 2;
  const int sharedSize = BLOCK_SIZE + 2 * halo;
  extern __shared__ float sharedMem[];

  // 1. 协作加载 tile + halo 到 shared memory
  for (int dy = ty; dy < sharedSize; dy += BLOCK_SIZE) {
    for (int dx = tx; dx < sharedSize; dx += BLOCK_SIZE) {
      // ... 边界处理（Zero / Mirror / Replicate）...
      sharedMem[dy * sharedSize + dx] = value;
    }
  }
  __syncthreads();  // 等待所有线程加载完成

  // 2. 从 shared memory（~20 cycle）而非 global memory（~200 cycle）读取
  if (x < width && y < height) {
    float sum = 0.0f;
    for (int ky = 0; ky < kernelSize; ++ky) {
      for (int kx = 0; kx < kernelSize; ++kx) {
        sum += sharedMem[...] * kernelData.values[...];
      }
    }
    // ...
  }
  __syncthreads();  // 确保所有线程读完后再进入下一个通道
}
```

### 学到的概念

1. **shared memory** - 每个 block 私有的高速缓存（~20 cycle vs global memory ~200 cycle）
2. **halo 区域** - tile 边缘的线程需要读取相邻 tile 的数据，这些"光环"区域是 tiling 的关键难点
3. **`__syncthreads()`** - block 内线程同步屏障，确保 shared memory 加载完成后再读取
4. **边界策略** - 三种模式的实现差异：Zero（补零）、Mirror（镜像）、Replicate（复制边界）
5. **`extern __shared__`** - 动态 shared memory 分配，大小在 kernel 启动时指定

### 为什么卷积核大小限制 7×7？

`KernelData` 结构体硬编码了 `float values[49]`（7×7=49）。原因：
- shared memory 预算有限（每个 block 通常 48KB）
- 教学简洁性：覆盖 3×3、5×5、7×7 已足够演示概念
- 超过 7×7 时应改用可分离卷积（见 Lv4）

### 动手练习

1. 用 Nsight Compute 对比朴素卷积（`convolveKernelSimple`）和 shared memory 卷积的 L2 cache hit rate
2. 把 `BLOCK_SIZE` 从 16 改为 32，计算 shared memory 用量变化，思考是否会超出限制
3. 实现 `BorderMode::Mirror` 的边界处理逻辑（当前代码已有，尝试理解 mirrorX 的计算）

---

## Lv4：可分离卷积

**文件**：`src/operators/convolution_engine.cu` — `separableConvolve`

如果卷积核可以分解为行核 × 列核（如高斯核），复杂度从 O(n²k²) 降到 O(n²k)：

```
二维卷积：每个像素读 k² 个邻居 -> O(n²k²)
可分离：先水平方向 k 个邻居，再垂直方向 k 个邻居 -> O(n²·2k)
```

实现是两 pass：先用 `separableRowKernel` 做水平卷积到中间缓冲，再用 `separableColKernel` 做垂直卷积到输出。

### 学到的概念

1. **算法层面的优化** - 不是所有优化都在 kernel 内部，算法选择本身就是优化
2. **两 pass 模式** - 中间缓冲区的分配和回收，`ImageUtils::createCudaImage` 分配临时图像
3. **`KernelData` 复用** - 行核和列核共用同一个结构体，只是填入不同的 1D 数据

### 动手练习

1. 用 `separableConvolve` 实现一个 9×9 高斯模糊，对比直接卷积（会超出 7×7 限制）和可分离卷积
2. 思考：为什么可分离卷积需要中间缓冲？能否原地操作？

---

## Lv5：原子操作与直方图

**文件**：`src/operators/histogram_calculator.cu` — `histogramKernelShared`

直方图是原子操作的经典教学案例。每个线程要将像素值对应的 bin 计数 +1，但多个线程可能同时写同一个 bin。

```cpp
__global__ void histogramKernelShared(...) {
  __shared__ int localHist[256];  // block 级局部直方图

  // 1. 初始化局部直方图
  for (int i = tid; i < 256; i += blockSize) {
    localHist[i] = 0;
  }
  __syncthreads();

  // 2. 每个线程对自己的像素做 atomicAdd 到 block 级直方图
  if (x < width && y < height) {
    unsigned char value = ...;  // 计算灰度值
    atomicAdd(&localHist[value], 1);  // block 内原子操作
  }
  __syncthreads();

  // 3. 合并 block 级直方图到全局直方图
  for (int i = tid; i < 256; i += blockSize) {
    if (localHist[i] > 0) {
      atomicAdd(&histogram[i], localHist[i]);  // 跨 block 原子操作
    }
  }
}
```

### 学到的概念

1. **`atomicAdd`** - 保证多个线程对同一地址的加法操作是原子的
2. **两级规约** - 先在 block 内用 shared memory 做局部直方图，再合并到全局。直接对全局内存做 atomicAdd 会产生严重竞争
3. **`__shared__` 数组** - 静态 shared memory 声明（编译时已知大小 256）
4. **直方图均衡化** - `equalizeKernel` 展示了如何用 CDF（累积分布函数）做像素映射

### 为什么用两级规约？

如果 100 万个线程直接对全局 `histogram[256]` 做 `atomicAdd`，竞争会极其严重（256 个 bin 被百万线程争抢）。改为每个 block 先在 shared memory 上做局部直方图（竞争范围小，速度快），最后只有 `gridSize` 个 block 对全局做合并，竞争大幅减少。

### 动手练习

1. 尝试去掉 block 级 shared memory 直方图，直接对全局 `histogram[256]` 做 `atomicAdd`，测量性能下降
2. 思考：如果灰度级不是 256 而是 65536（16-bit 图像），shared memory 还够用吗？

---

## Lv6：双线性插值与坐标映射

**文件**：`src/operators/image_resizer.cu` — `resizeBilinearKernel`

图像缩放的核心是坐标映射 + 插值。

```cpp
// 目标像素 -> 源图像坐标（半像素中心对齐）
float srcX = (x + 0.5f) * srcWidth / dstWidth - 0.5f;
float srcY = (y + 0.5f) * srcHeight / dstHeight - 0.5f;

// 四个相邻像素 + 双线性权重
float v0 = v00 * (1.0f - dx) + v10 * dx;  // 水平插值
float v1 = v01 * (1.0f - dx) + v11 * dx;
return v0 * (1.0f - dy) + v1 * dy;          // 垂直插值
```

### 学到的概念

1. **坐标映射** - 目标图像坐标到源图像坐标的变换，半像素中心对齐避免偏移
2. **`__device__` 函数** - 可被 kernel 调用的设备端函数（`bilinearInterpolate`）
3. **浮点运算在 GPU 上** - `floorf`、`roundf`、`fmaxf`、`fminf` 等 CUDA 数学函数

### 教学说明

本项目未使用 CUDA 纹理内存（`cudaTextureObject_t`），纹理硬件可以自动处理插值和边界。这里手动实现插值是为了让初学者理解原理。纹理内存是可选的下一步优化方向。

### 动手练习

1. 实现最近邻插值版本（`resizeNearestKernel` 已有），对比双线性与最近邻的视觉差异
2. 思考：纹理内存如何自动处理插值和边界？手动实现的优势是什么？

---

## Lv7：多流流水线

**文件**：`src/processing/pipeline_processor.cu` + `include/cudaimg/core/execution_context.hpp`

多流允许 GPU 同时执行多个操作（kernel 计算 + 内存拷贝可以重叠）。

```cpp
PipelineProcessor::PipelineProcessor(int numStreams) {
  streams_.resize(numStreams);
  for (int i = 0; i < numStreams; ++i) {
    cudaStreamCreate(&streams_[i]);  // 创建 CUDA stream
  }
}
```

### 学到的概念

1. **`cudaStream_t`** - 异步执行队列，不同 stream 上的操作可以并发
2. **`cudaStreamCreate` / `cudaStreamDestroy`** - stream 生命周期管理
3. **执行策略** - `ExecutionPolicy` 封装了 Sync / Async / Batch 三种模式：
   - Sync：`cudaDeviceSynchronize()` 阻塞等待
   - Async：调用方管理 stream 生命周期
   - Batch：内部 stream pool，`syncAll()` 统一等待

### 坑点

内存池在多 stream 场景下有数据竞争风险，详见 [坑点记录](pitfalls.md)。
