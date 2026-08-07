#pragma once

#include "cudaimg/core/image.hpp"
#include <atomic>
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace cudaimg {

/// ExecutionPolicy: Unified execution strategy abstraction
/// 执行策略：统一的执行策略抽象
///
/// This module hides CUDA stream details from callers, providing a clean
/// interface for synchronous, asynchronous, and batch operations.
///
/// Interface contract:
/// - Sync: Blocks until operation completes
/// - Async: Returns immediately, operation may still be in flight
/// - Batch: Operations queued to internal stream, syncAll() waits for all
class ExecutionPolicy {
public:
  /// Predefined execution strategies
  /// 预定义的执行策略
  enum class Mode {
    Sync,  ///< Synchronous execution (cudaDeviceSynchronize)
    Async, ///< Asynchronous execution (caller manages stream)
    Batch  ///< Batch execution (internal stream pool)
  };

  /// Create a synchronous policy (default)
  /// 创建同步策略（默认）
  [[nodiscard]] static ExecutionPolicy sync() {
    return ExecutionPolicy(Mode::Sync);
  }

  /// Create an asynchronous policy with optional stream
  /// 创建异步策略（可选 stream）
  /// @param stream CUDA stream to use (nullptr creates internal stream)
  [[nodiscard]] static ExecutionPolicy async(cudaStream_t stream = nullptr);

  /// Create a batch policy for pipelined operations
  /// 创建批处理策略用于流水线操作
  [[nodiscard]] static ExecutionPolicy batch();

  // Constructors and destructor
  ExecutionPolicy() : mode_(Mode::Sync), stream_(nullptr), ownsStream_(false) {}
  ~ExecutionPolicy();

  // Move-only (streams cannot be shared)
  ExecutionPolicy(const ExecutionPolicy&) = delete;
  ExecutionPolicy& operator=(const ExecutionPolicy&) = delete;
  ExecutionPolicy(ExecutionPolicy&& other) noexcept;
  ExecutionPolicy& operator=(ExecutionPolicy&& other) noexcept;

  /// Get the execution mode
  [[nodiscard]] Mode mode() const { return mode_; }

  /// Get the CUDA stream (may be null for sync mode)
  [[nodiscard]] cudaStream_t stream() const { return stream_; }

  /// Synchronize: Wait for async/batch operations to complete
  /// 同步：等待异步/批处理操作完成
  void synchronize() const;

  /// Check if this policy owns its stream
  [[nodiscard]] bool ownsStream() const { return ownsStream_; }

private:
  explicit ExecutionPolicy(Mode mode);
  ExecutionPolicy(Mode mode, cudaStream_t stream, bool owns);

  Mode mode_;
  cudaStream_t stream_;
  bool ownsStream_;
};

/// ImageAllocator: Unified image buffer management
/// 图像分配器：统一的图像缓冲区管理
///
/// Provides centralized allocation with optional memory pooling.
/// All operators should use this for output buffer management.
///
/// Interface contract:
/// - Output buffers are automatically sized
/// - Memory pooling is optional (configure per-context)
/// - Thread-safe allocation/deallocation
class ImageAllocator {
public:
  /// Get the global allocator instance
  /// 获取全局分配器实例
  static ImageAllocator& instance();

  /// Allocate a CudaImage with specified dimensions
  /// 分配指定尺寸的 CudaImage
  /// @param usePool If true, use memory pool when available
  [[nodiscard]] CudaImage allocate(int width, int height, int channels,
                                  bool usePool = true);

  /// Ensure output image matches input dimensions
  /// 确保输出图像与输入尺寸匹配
  /// @return true if reallocation occurred
  [[nodiscard]] bool ensureSize(const CudaImage& input, CudaImage& output,
                                bool usePool = true);

  /// Ensure output image has specified dimensions
  /// 确保输出图像具有指定尺寸
  /// @return true if reallocation occurred
  [[nodiscard]] bool ensureSize(CudaImage& output, int width, int height,
                                int channels, bool usePool = true);

  /// Recycle a buffer back to the memory pool for reuse
  /// 将缓冲区回收至内存池以便复用
  ///
  /// @param image The image to recycle (moved-from state after call)
  ///
  /// Note: If pooling is disabled, the buffer is simply destroyed.
  /// This method takes ownership of the image's buffer.
  void recycleToPool(CudaImage&& image);

  /// Enable or disable memory pooling globally
  /// 全局启用或禁用内存池
  /// @note Thread-safe: uses atomic operations
  void setPoolingEnabled(bool enabled) {
    poolingEnabled_.store(enabled, std::memory_order_relaxed);
  }

  /// Check if pooling is enabled
  /// 检查内存池是否启用
  /// @note Thread-safe: uses atomic operations
  [[nodiscard]] bool isPoolingEnabled() const {
    return poolingEnabled_.load(std::memory_order_relaxed);
  }

private:
  ImageAllocator() = default;
  std::atomic<bool> poolingEnabled_{false};
};

/// ExecutionContext: Combined allocation and execution strategy
/// 执行上下文：组合分配和执行策略
///
/// This is the primary interface for operators, providing both
/// memory management and execution policy in one cohesive module.
///
/// Depth: Callers interact with a simple context object instead of
/// managing streams, buffers, and pooling separately.
///
/// Usage:
/// @code
/// ExecutionContext ctx(ExecutionPolicy::async());
/// CudaImage output = ctx.allocateOutput(input);
/// PixelOperator::invert(input, output, ctx.stream());
/// ctx.synchronize();  // Only needed for async
/// @endcode
class ExecutionContext {
public:
  /// Create a context with specified execution policy
  /// 使用指定执行策略创建上下文
  explicit ExecutionContext(ExecutionPolicy policy = ExecutionPolicy::sync());

  /// Create a context with specified mode
  /// 使用指定模式创建上下文
  explicit ExecutionContext(ExecutionPolicy::Mode mode);

  /// Get the execution policy
  [[nodiscard]] const ExecutionPolicy& policy() const { return policy_; }

  /// Get the CUDA stream (convenience)
  [[nodiscard]] cudaStream_t stream() const { return policy_.stream(); }

  /// Allocate output buffer matching input
  /// 分配与输入匹配的输出缓冲区
  [[nodiscard]] CudaImage allocateOutput(const CudaImage& input);

  /// Allocate output buffer with specified dimensions
  /// 分配指定尺寸的输出缓冲区
  [[nodiscard]] CudaImage allocateOutput(int width, int height, int channels);

  /// Ensure output matches input (reallocate if needed)
  /// 确保输出与输入匹配（需要时重新分配）
  [[nodiscard]] bool ensureOutputSize(const CudaImage& input, CudaImage& output);

  /// Ensure output has specified dimensions
  /// 确保输出具有指定尺寸
  [[nodiscard]] bool ensureOutputSize(CudaImage& output, int width, int height,
                                      int channels);

  /// Synchronize (for async/batch modes)
  /// 同步（用于异步/批处理模式）
  void synchronize() const { policy_.synchronize(); }

  /// Recycle a buffer back to the memory pool for reuse
  /// 将缓冲区回收至内存池以便复用
  ///
  /// @param image The image to recycle (moved-from state after call)
  ///
  /// Example:
  /// @code
  /// CudaImage intermediate = ctx.allocateOutput(input);
  /// // ... use intermediate ...
  /// ctx.recycleToPool(std::move(intermediate));  // Return to pool
  /// @endcode
  void recycleToPool(CudaImage&& image);

private:
  ExecutionPolicy policy_;
};

// ===== Convenience Free Functions =====

/// Create an async execution context
/// 创建异步执行上下文
[[nodiscard]] inline ExecutionContext
asyncContext(cudaStream_t stream = nullptr) {
  return ExecutionContext(ExecutionPolicy::async(stream));
}

/// Create a sync execution context
/// 创建同步执行上下文
[[nodiscard]] inline ExecutionContext syncContext() {
  return ExecutionContext(ExecutionPolicy::sync());
}

/// Create a batch execution context
/// 创建批处理执行上下文
[[nodiscard]] inline ExecutionContext batchContext() {
  return ExecutionContext(ExecutionPolicy::batch());
}

} // namespace cudaimg
