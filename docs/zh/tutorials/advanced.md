# 高级主题

高级使用模式和优化技术。

## 内存管理

### 缓冲区复用

`ImageProcessor` 门面每次调用都返回新的 `GpuImage`。若要复用输出缓冲区，请下沉到算子层并自行管理输出：

```cpp
// 不好的做法：每次都分配新内存
for (int i = 0; i < 100; i++) {
    GpuImage result = processor.gaussianBlur(gpu, 5, 1.5f);
}

// 好的做法：复用输出缓冲区（算子层 + ExecutionContext）
ExecutionContext ctx(ExecutionPolicy::sync());
GpuImage result = ctx.allocateOutput(gpu);
for (int i = 0; i < 100; i++) {
    ConvolutionEngine::gaussianBlur(gpu, result, 5, 1.5f, ctx.stream());
}
```

`ctx.ensureOutputSize(input, result)` 仅在尺寸变化时才重新分配。

### 内存池

对于高吞吐量应用，启用 `MemoryManager` 中按大小分桶的内存池（池化**默认关闭**）：

```cpp
// 通过门面
ImageProcessor processor;
processor.setMemoryPooling(true);

// 或通过分配器全局开启
ImageAllocator::instance().setPoolingEnabled(true);

// 底层池控制
MemoryManager::instance().setMaxPoolSize(512 * 1024 * 1024);
MemoryStats stats = MemoryManager::instance().getStats();
MemoryManager::instance().clearPool();
```

注意：内存池不跟踪 CUDA 流的完成情况——多线程使用不同流时，回收可能仍被读取的缓冲区前请先同步。

## CUDA 流

### 多流处理

使用 `PipelineProcessor`，它持有一个流池并将图像轮流分配到各条流：

```cpp
PipelineProcessor pipeline(4);  // 4 条流（默认 3 条）

pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});

// 返回前在内部完成同步
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

单个异步操作可使用带 async 策略的 `ExecutionContext`：

```cpp
ExecutionContext ctx(ExecutionPolicy::async());  // 持有内部流
GpuImage out = ctx.allocateOutput(input);
PixelOperator::invert(input, out, ctx.stream());
ctx.synchronize();
```

## 内核优化

### 共享内存分块

卷积和直方图内核使用了共享内存分块。编写自定义内核时可沿用相同模式：

```cpp
__global__ void myKernel(float* output, const float* input, int width) {
    extern __shared__ float sharedMem[];
    
    // 加载到共享内存
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedMem[tid] = input[gid];
    __syncthreads();
    
    // 从共享内存计算
    // ...
}
```

### 实际使用的优化

当前实现依赖以下技术：

- **共享内存分块** —— 卷积（`convolution_engine.cu`）和直方图（`histogram_calculator.cu`）
- **共享内存原子直方图** —— 每个块先写本地直方图，再合并到全局结果
- **`uchar4` 向量化** —— 像素内核每线程处理 4 字节（`pixel_operator.cu`）

纹理内存、warp shuffle/归约、cooperative groups 以及 pinned/zero-copy 主机内存均**未**使用——它们仍是可选的未来优化方向。

## 性能技巧

1. **批量操作**: `PipelineProcessor::processBatch` 跨流处理多张图像
2. **复用缓冲区**: `ensureOutputSize` 避免重复分配
3. **启用池化**: 适用于重复的同尺寸负载
4. **卷积核保持小尺寸**: 仅接受 7×7 以内的奇数尺寸（更大尺寸会抛异常）
5. **先分析**: 使用 Nsight 识别瓶颈

## 调试

### 错误检查

库使用 `CUDA_CHECK` 宏（`gpu_image/core/cuda_error.hpp`）检查每一次 CUDA 调用，失败时抛出 `CudaException`：

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw gpu_image::CudaException(error, __FILE__, __LINE__); \
        } \
    } while (0)
```

按标准异常方式捕获：

```cpp
try {
    GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
} catch (const CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
}
```

### 内存跟踪

```cpp
// 设备级内存
size_t free, total;
cudaMemGetInfo(&free, &total);
printf("GPU 内存: %zu MB 空闲 / %zu MB 总计\n", free/1024/1024, total/1024/1024);

// 库内存池统计（启用池化后有意义）
MemoryStats stats = MemoryManager::instance().getStats();
printf("allocated=%zu pool=%zu peak=%zu\n",
       stats.totalAllocated, stats.poolSize, stats.peakUsage);
```

## 下一步

- [常见问题](./faq) - 常见问题解答
- [架构设计](../architecture/overview) - 系统设计
