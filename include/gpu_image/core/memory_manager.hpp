#pragma once

#include "gpu_image/core/device_buffer.hpp"
#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gpu_image {

// 内存使用统计
struct MemoryStats {
  size_t totalAllocated; ///< 累计从 CUDA 分配的字节数（单调递增，不含池复用）
  size_t poolSize;       ///< 当前驻留在内存池中的字节数
  size_t peakUsage;      ///< 内存池占用字节数的历史峰值
};

// 内存管理器单例类
//
// 线程安全性：allocate/deallocate/clearPool/getStats 以及配置项的读写均可
// 安全地并发调用（池映射由互斥锁保护，开关/上限为原子变量）。
//
// ⚠ 并发流限制：池在回收缓冲时不记录/等待所属 CUDA stream 完成。若多个线程
// 各自持有不同 stream 并并发 allocate/deallocate，某条 stream 上仍在读的缓冲
// 可能被另一条 stream 当作输出复用，导致数据竞争。当前的安全用法是：单 stream，
// 或在回收前由调用方自行同步所属 stream。默认配置下池处于关闭状态（见
// ImageAllocator::poolingEnabled_），不受此限制影响。
class MemoryManager {
public:
  static MemoryManager& instance();

  // 禁用拷贝和移动
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  // 分配 Device 内存（可能从池中获取）
  DeviceBuffer allocate(size_t size);

  // 释放内存回池
  void deallocate(DeviceBuffer&& buffer);

  // 清空内存池
  void clearPool();

  // 获取内存使用统计（含义见 MemoryStats 各字段注释）
  MemoryStats getStats() const;

  // 配置（原子，线程安全）
  void setMaxPoolSize(size_t maxSize) {
    maxPoolSize_.store(maxSize, std::memory_order_relaxed);
  }
  void setPoolEnabled(bool enabled) {
    poolEnabled_.store(enabled, std::memory_order_relaxed);
  }

private:
  MemoryManager();
  ~MemoryManager();

  // 将大小对齐到 256 字节边界
  static size_t alignSize(size_t size);

  std::unordered_map<size_t, std::vector<void*>> memoryPool_;
  mutable std::mutex mutex_;

  // 以下三项受 mutex_ 保护
  size_t totalAllocated_ = 0;
  size_t poolSize_ = 0;
  size_t peakUsage_ = 0;

  // 配置项：可被其它线程读写，使用原子变量避免数据竞争
  std::atomic<size_t> maxPoolSize_{512 * 1024 * 1024}; // 512 MB
  std::atomic<bool> poolEnabled_{true};
};

} // namespace gpu_image
