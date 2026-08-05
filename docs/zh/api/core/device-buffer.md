# DeviceBuffer

RAII GPU 内存管理。

## 概述

`DeviceBuffer` 使用 RAII 模式提供自动 GPU 内存分配和释放。

## 构造函数

```cpp
DeviceBuffer() noexcept;             // 空缓冲区（空指针，大小为 0）
explicit DeviceBuffer(size_t size);  // 在 GPU 上分配 `size` 字节
```

`fromRaw` 接管已有分配的所有权（供内存池使用）：

```cpp
static DeviceBuffer fromRaw(void* ptr, size_t size) noexcept;
```

## 析构函数

```cpp
~DeviceBuffer();
```

自动释放 GPU 内存（`cudaFree()`）。

## 数据访问

### data

```cpp
void* data() noexcept;
const void* data() const noexcept;
```

返回 GPU 内存指针。

### dataAs

```cpp
template <typename T> T* dataAs() noexcept;
template <typename T> const T* dataAs() const noexcept;
```

返回带类型的 GPU 内存指针。

### size

```cpp
size_t size() const noexcept;
```

返回分配的字节数。

### isValid

```cpp
bool isValid() const noexcept;
```

缓冲区持有非空指针且 size > 0 时为 true。

## 主机/设备传输

```cpp
void copyFromHost(const void* hostPtr, size_t size);
void copyToHost(void* hostPtr, size_t size) const;
void copyFromDevice(const void* devicePtr, size_t size);  // 设备到设备

// 指定流上的异步版本
void copyFromHostAsync(const void* hostPtr, size_t size, cudaStream_t stream);
void copyToHostAsync(void* hostPtr, size_t size, cudaStream_t stream) const;
```

## 所有权

```cpp
void release() noexcept;                     // 释放内存（cudaFree）
std::pair<void*, size_t> detach() noexcept;  // 放弃所有权但不释放内存
```

## 移动语义

```cpp
DeviceBuffer buf1(1024);
DeviceBuffer buf2 = std::move(buf1);  // buf1 现在为空
```

禁止拷贝以防止意外的深拷贝。

## 示例

```cpp
// 分配 GPU 内存
DeviceBuffer buffer(1920 * 1080 * 3);  // 1080p RGB 图像

// 配合 CUDA 内核使用
myKernel<<<grid, block>>>(buffer.data());

// buffer 超出作用域时自动释放内存
```

## 内部实现

```cpp
class DeviceBuffer {
public:
    DeviceBuffer() noexcept;
    explicit DeviceBuffer(size_t size);
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    void* devicePtr_;
    size_t size_;
};
```

## 另见

- [GpuImage](./gpu-image) - 使用 DeviceBuffer 的图像容器
