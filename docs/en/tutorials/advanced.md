# Advanced Topics

Advanced usage patterns and optimization techniques.

## Memory Management

### Buffer Reuse

The `ImageProcessor` facade returns a new `GpuImage` per call. To reuse an output buffer, drop down to the operator layer and manage the output yourself:

```cpp
// Bad: Allocates new memory each time
for (int i = 0; i < 100; i++) {
    GpuImage result = processor.gaussianBlur(gpu, 5, 1.5f);
}

// Good: Reuse the output buffer (operator layer + ExecutionContext)
ExecutionContext ctx(ExecutionPolicy::sync());
GpuImage result = ctx.allocateOutput(gpu);
for (int i = 0; i < 100; i++) {
    ConvolutionEngine::gaussianBlur(gpu, result, 5, 1.5f, ctx.stream());
}
```

`ctx.ensureOutputSize(input, result)` reallocates only when dimensions change.

### Memory Pool

For high-throughput applications, enable the size-keyed pool in `MemoryManager` (pooling is **disabled by default**):

```cpp
// Via the facade
ImageProcessor processor;
processor.setMemoryPooling(true);

// Or globally via the allocator
ImageAllocator::instance().setPoolingEnabled(true);

// Low-level pool control
MemoryManager::instance().setMaxPoolSize(512 * 1024 * 1024);
MemoryStats stats = MemoryManager::instance().getStats();
MemoryManager::instance().clearPool();
```

Note: the pool does not track CUDA stream completion — with multiple concurrent streams, synchronize before recycling a buffer that may still be read.

## CUDA Streams

### Multi-Stream Processing

Use `PipelineProcessor`, which owns a pool of streams and round-robins images across them:

```cpp
PipelineProcessor pipeline(4);  // 4 streams (default: 3)

pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});

// Synchronizes internally before returning
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

For a single async operation, use an `ExecutionContext` with an async policy:

```cpp
ExecutionContext ctx(ExecutionPolicy::async());  // owns an internal stream
GpuImage out = ctx.allocateOutput(input);
PixelOperator::invert(input, out, ctx.stream());
ctx.synchronize();
```

## Kernel Optimization

### Shared Memory Tiling

The convolution and histogram kernels use shared memory tiling. For custom kernels, follow the same pattern:

```cpp
__global__ void myKernel(float* output, const float* input, int width) {
    extern __shared__ float sharedMem[];
    
    // Load to shared memory
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedMem[tid] = input[gid];
    __syncthreads();
    
    // Compute from shared memory
    // ...
}
```

### Optimizations Actually Used

The current implementation relies on:

- **Shared memory tiling** — convolution (`convolution_engine.cu`) and histogram (`histogram_calculator.cu`)
- **Shared-memory atomic histogram** — per-block local histogram merged into the global result
- **`uchar4` vectorization** — pixel kernels process 4 bytes per thread (`pixel_operator.cu`)

Texture memory, warp shuffles/reductions, cooperative groups, and pinned/zero-copy host memory are **not** used — they remain possible future optimizations.

## Performance Tips

1. **Batch operations**: `PipelineProcessor::processBatch` processes multiple images across streams
2. **Reuse buffers**: `ensureOutputSize` avoids repeated allocations
3. **Enable pooling** for repeated same-size workloads
4. **Keep convolution kernels small**: only odd sizes up to 7×7 are accepted (larger sizes throw)
5. **Profile first**: Use Nsight to identify bottlenecks

## Debugging

### Error Checking

The library checks every CUDA call with the `CUDA_CHECK` macro (`gpu_image/core/cuda_error.hpp`), which throws `CudaException` on failure:

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw gpu_image::CudaException(error, __FILE__, __LINE__); \
        } \
    } while (0)
```

Catch it as a standard exception:

```cpp
try {
    GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
} catch (const CudaException& e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
}
```

### Memory Tracking

```cpp
// Device-wide memory
size_t free, total;
cudaMemGetInfo(&free, &total);
printf("GPU Memory: %zu MB free / %zu MB total\n", free/1024/1024, total/1024/1024);

// Library pool statistics (meaningful when pooling is enabled)
MemoryStats stats = MemoryManager::instance().getStats();
printf("allocated=%zu pool=%zu peak=%zu\n",
       stats.totalAllocated, stats.poolSize, stats.peakUsage);
```

## Next Steps

- [FAQ](./faq) - Common questions
- [Architecture](../architecture/overview) - System design
