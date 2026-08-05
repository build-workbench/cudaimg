#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <utility>

namespace gpu_image {

// 管理单个 GPU 内存缓冲区的生命周期
class DeviceBuffer {
public:
  // 默认构造函数（空缓冲区）
  DeviceBuffer() noexcept : devicePtr_(nullptr), size_(0) {}

  // 构造函数：分配指定大小的 Device 内存
  explicit DeviceBuffer(size_t size);

  // 从已有原始指针接管所有权（用于内存池）
  [[nodiscard]] static DeviceBuffer fromRaw(void* ptr, size_t size) noexcept;

  // 析构函数：自动释放 Device 内存
  ~DeviceBuffer();

  // 禁用拷贝
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // 启用移动语义
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  // 从 Host 复制数据到 Device
  void copyFromHost(const void* hostPtr, size_t size);

  // 从 Device 复制数据到 Host
  void copyToHost(void* hostPtr, size_t size) const;

  // 从另一个 Device 缓冲区复制数据到本缓冲区（Device-to-Device）
  void copyFromDevice(const void* devicePtr, size_t size);

  // 异步版本（使用指定的 stream）
  void copyFromHostAsync(const void* hostPtr, size_t size, cudaStream_t stream);
  void copyToHostAsync(void* hostPtr, size_t size, cudaStream_t stream) const;

  // 获取 Device 指针
  [[nodiscard]] void* data() noexcept { return devicePtr_; }
  [[nodiscard]] const void* data() const noexcept { return devicePtr_; }

  // 模板版本
  template <typename T> [[nodiscard]] T* dataAs() noexcept {
    return static_cast<T*>(devicePtr_);
  }

  template <typename T> [[nodiscard]] const T* dataAs() const noexcept {
    return static_cast<const T*>(devicePtr_);
  }

  // 获取缓冲区大小
  [[nodiscard]] size_t size() const noexcept { return size_; }

  // 检查是否有效
  [[nodiscard]] bool isValid() const noexcept {
    return devicePtr_ != nullptr && size_ > 0;
  }

  // 释放内存（cudaFree）- noexcept for safe destructor use
  void release() noexcept;

  // 放弃所有权，返回原始指针和大小（不释放内存）
  // 调用者负责后续释放
  std::pair<void*, size_t> detach() noexcept;

private:
  void* devicePtr_;
  size_t size_;
};

} // namespace gpu_image
