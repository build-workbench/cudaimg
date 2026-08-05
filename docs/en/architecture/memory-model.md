# Memory Model

Understanding GPU memory management in Mini-OpenCV.

## Overview

Mini-OpenCV uses a RAII-based memory model. `DeviceBuffer` owns device memory, `GpuImage`/`HostImage` are plain data containers, and `MemoryManager` plus `ImageAllocator` provide optional pooling for output buffers.

## DeviceBuffer

The `DeviceBuffer` class provides RAII-style GPU memory management:

```cpp
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size);  // cudaMalloc
    ~DeviceBuffer();                     // cudaFree (unless moved from)

    void copyFromHost(const void* hostPtr, size_t size);
    void copyToHost(void* hostPtr, size_t size) const;
    void copyFromHostAsync(const void* hostPtr, size_t size, cudaStream_t stream);
    void copyToHostAsync(void* hostPtr, size_t size, cudaStream_t stream) const;

    void* data();
    size_t size() const;
    bool isValid() const;

private:
    void* devicePtr_;
    size_t size_;
};
```

### Key Features

- **Automatic Deallocation**: Memory freed when object goes out of scope
- **Move Semantics**: Efficient transfer of ownership
- **No Copy**: Prevents accidental deep copies

## GpuImage and HostImage

`GpuImage` is a plain struct with public fields — a `DeviceBuffer` member plus image metadata (no getters, no indirection):

```cpp
struct GpuImage {
    DeviceBuffer buffer;
    int width = 0;
    int height = 0;
    int channels = 0;  // 1 grayscale, 3 RGB, 4 RGBA

    size_t pitch() const;       // width * channels
    size_t totalBytes() const;  // width * height * channels
    bool isValid() const;
};
```

`HostImage` stores pixels in an ordinary `std::vector<unsigned char>` (regular pageable host memory — no pinned/zero-copy memory is used):

```cpp
struct HostImage {
    std::vector<unsigned char> data;
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char& at(int x, int y, int c);  // bounds-checked
    size_t totalBytes() const;
    bool isValid() const;
};
```

## Memory Pool (MemoryManager)

`MemoryManager` is a singleton that recycles device allocations, keyed by size (aligned to 256-byte boundaries):

```cpp
// Allocate (may reuse a pooled block), returned by value
DeviceBuffer buf = MemoryManager::instance().allocate(size);

// Return a buffer to the pool
MemoryManager::instance().deallocate(std::move(buf));

// Pool control and statistics
MemoryManager::instance().clearPool();
MemoryManager::instance().setMaxPoolSize(512 * 1024 * 1024);  // default: 512 MB
MemoryStats stats = MemoryManager::instance().getStats();
// stats.totalAllocated / stats.poolSize / stats.peakUsage
```

Notes:

- The pool does **not** track CUDA stream completion. If multiple threads use different streams, synchronize the owning stream before recycling a buffer that may still be read, or stick to a single stream.
- Pooling is gated by `ImageAllocator` and is **disabled by default** (see below), so the pool stays empty unless you opt in.

## ImageAllocator and ExecutionContext

Operators obtain output buffers through `ImageAllocator` (singleton) or, more commonly, through an `ExecutionContext`:

```cpp
ExecutionContext ctx(ExecutionPolicy::sync());

GpuImage out = ctx.allocateOutput(input);       // sized like input
ctx.ensureOutputSize(input, out);               // realloc only on size mismatch
ctx.recycleToPool(std::move(out));              // return buffer for reuse
```

- `ImageAllocator::setPoolingEnabled(bool)` toggles pooling globally; it defaults to **false**, so allocations go straight to `cudaMalloc`/`cudaFree` until you enable it.
- `ImageProcessor::setMemoryPooling(bool)` exposes the same switch on the facade.

## Best Practices

1. **Reuse Outputs**: Keep a `GpuImage` across frames and use `ensureOutputSize` — reallocation happens only when dimensions change
2. **Opt Into Pooling for Repeated Workloads**: Enable pooling for many same-size operations to cut allocation overhead
3. **Batch with Streams**: `PipelineProcessor::processBatch` / `processBatchHost` overlap transfer and compute across streams

## Next Steps

- [CUDA Streams](./cuda-streams) - Async execution
- [Design Decisions](./design-decisions) - Architecture choices
