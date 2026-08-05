# 内存模型

了解 Mini-OpenCV 中的 GPU 内存管理。

## 概述

Mini-OpenCV 使用基于 RAII 的内存模型。`DeviceBuffer` 拥有设备内存，`GpuImage`/`HostImage` 是普通的数据容器，`MemoryManager` 与 `ImageAllocator` 为输出缓冲区提供可选的池化。

## DeviceBuffer

`DeviceBuffer` 类提供 RAII 风格的 GPU 内存管理：

```cpp
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size);  // cudaMalloc
    ~DeviceBuffer();                     // cudaFree（除非已被移动）

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

### 关键特性

- **自动释放**: 对象超出作用域时自动释放内存
- **移动语义**: 高效的所有权转移
- **禁止拷贝**: 防止意外的深拷贝

## GpuImage 与 HostImage

`GpuImage` 是带公开字段的普通 struct——一个 `DeviceBuffer` 成员加图像元数据（无 getter、无间接层）：

```cpp
struct GpuImage {
    DeviceBuffer buffer;
    int width = 0;
    int height = 0;
    int channels = 0;  // 1 灰度, 3 RGB, 4 RGBA

    size_t pitch() const;       // width * channels
    size_t totalBytes() const;  // width * height * channels
    bool isValid() const;
};
```

`HostImage` 将像素存放在普通的 `std::vector<unsigned char>` 中（常规可分页主机内存——未使用 pinned/zero-copy 内存）：

```cpp
struct HostImage {
    std::vector<unsigned char> data;
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char& at(int x, int y, int c);  // 带边界检查
    size_t totalBytes() const;
    bool isValid() const;
};
```

## 内存池 (MemoryManager)

`MemoryManager` 是单例，按大小（对齐到 256 字节边界）回收设备分配：

```cpp
// 分配（可能复用池中的块），按值返回
DeviceBuffer buf = MemoryManager::instance().allocate(size);

// 将缓冲区归还到池中
MemoryManager::instance().deallocate(std::move(buf));

// 池控制与统计
MemoryManager::instance().clearPool();
MemoryManager::instance().setMaxPoolSize(512 * 1024 * 1024);  // 默认 512 MB
MemoryStats stats = MemoryManager::instance().getStats();
// stats.totalAllocated / stats.poolSize / stats.peakUsage
```

注意：

- 内存池**不**跟踪 CUDA 流的完成情况。如果多个线程使用不同的流，在回收可能仍被读取的缓冲区之前，请先同步其所属的流，或者只使用单条流。
- 池化由 `ImageAllocator` 控制开关，且**默认关闭**（见下文），因此除非显式启用，池始终为空。

## ImageAllocator 与 ExecutionContext

算子通过 `ImageAllocator`（单例）获取输出缓冲区，更常见的做法是通过 `ExecutionContext`：

```cpp
ExecutionContext ctx(ExecutionPolicy::sync());

GpuImage out = ctx.allocateOutput(input);       // 与输入同尺寸
ctx.ensureOutputSize(input, out);               // 仅在尺寸不匹配时重新分配
ctx.recycleToPool(std::move(out));              // 归还缓冲区以便复用
```

- `ImageAllocator::setPoolingEnabled(bool)` 全局开关池化；默认值为 **false**，因此在启用之前，分配直接走 `cudaMalloc`/`cudaFree`。
- `ImageProcessor::setMemoryPooling(bool)` 在门面层暴露同一个开关。

## 最佳实践

1. **复用输出**: 跨帧保留 `GpuImage` 并使用 `ensureOutputSize`——仅在尺寸变化时重新分配
2. **重复性负载启用池化**: 对大量同尺寸操作启用池化以降低分配开销
3. **多流批处理**: `PipelineProcessor::processBatch` / `processBatchHost` 跨流重叠传输与计算

## 下一步

- [CUDA 流](./cuda-streams) - 异步执行
- [设计决策](./design-decisions) - 架构选择
