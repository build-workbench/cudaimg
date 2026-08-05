#include "gpu_image/core/memory_manager.hpp"
#include "gpu_image/core/cuda_error.hpp"
#include <cuda_runtime.h>

namespace gpu_image {

MemoryManager& MemoryManager::instance() {
  static MemoryManager instance;
  return instance;
}

MemoryManager::MemoryManager() = default;

MemoryManager::~MemoryManager() { clearPool(); }

size_t MemoryManager::alignSize(size_t size) {
  const size_t alignment = 256;
  return (size + alignment - 1) & ~(alignment - 1);
}

DeviceBuffer MemoryManager::allocate(size_t size) {
  if (size == 0) {
    return DeviceBuffer();
  }

  size_t alignedSize = alignSize(size);

  if (poolEnabled_) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = memoryPool_.find(alignedSize);
    if (it != memoryPool_.end() && !it->second.empty()) {
      void* ptr = it->second.back();
      it->second.pop_back();
      poolSize_ -= alignedSize;

      // 使用 fromRaw 将池中的原始指针包装为 DeviceBuffer
      return DeviceBuffer::fromRaw(ptr, alignedSize);
    }
  }

  // 从 CUDA 分配新内存（使用对齐后的大小以便回收）
  DeviceBuffer buffer(alignedSize);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    totalAllocated_ += alignedSize; // 累计分配量（单调递增）
  }

  return buffer;
}

void MemoryManager::deallocate(DeviceBuffer&& buffer) {
  if (!buffer.isValid()) {
    return;
  }

  size_t alignedSize = alignSize(buffer.size());

  if (poolEnabled_) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 池未满时回收，否则直接释放
    if (poolSize_ + alignedSize <=
        maxPoolSize_.load(std::memory_order_relaxed)) {
      auto detached = buffer.detach();
      void* ptr = detached.first;
      memoryPool_[alignedSize].push_back(ptr);
      poolSize_ += alignedSize;
      if (poolSize_ > peakUsage_) {
        peakUsage_ = poolSize_; // 记录池占用的历史峰值
      }
      return;
    }
  }

  // 池已满或禁用，buffer 析构时自动 cudaFree
}

void MemoryManager::clearPool() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& [sz, ptrs] : memoryPool_) {
    for (void* ptr : ptrs) {
      cudaError_t err = cudaFree(ptr);
      // 清理时忽略错误但继续执行，避免内存泄漏
      // 错误可能发生在 CUDA 上下文已销毁的情况
      (void)err; // 显式忽略返回值
    }
  }
  memoryPool_.clear();
  poolSize_ = 0;
}

MemoryStats MemoryManager::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return MemoryStats{totalAllocated_, poolSize_, peakUsage_};
}

} // namespace gpu_image
