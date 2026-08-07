#pragma once

#include "cudaimg/core/execution_context.hpp"
#include "cudaimg/core/image.hpp"
#include "cudaimg/operators/color_space.hpp"
#include "cudaimg/operators/convolution_engine.hpp"
#include "cudaimg/operators/filters.hpp"
#include "cudaimg/operators/geometric.hpp"
#include "cudaimg/operators/histogram_calculator.hpp"
#include "cudaimg/operators/image_resizer.hpp"
#include "cudaimg/operators/morphology.hpp"
#include "cudaimg/operators/pixel_operator.hpp"
#include "cudaimg/operators/threshold.hpp"
#include <array>
#include <functional>

namespace cudaimg {
  /// Lab to RGB

  // ===== Image Arithmetic =====

  /// Add two images
  /// 图像加法
  CudaImage add(const CudaImage& src1, const CudaImage& src2);
  /// Alpha blend
  /// 图像混合
  CudaImage blend(const CudaImage& src1, const CudaImage& src2, float alpha);
  /// Add scalar
  /// 标量加法
  CudaImage addScalar(const CudaImage& input, unsigned char value);

  // ===== Synchronization =====

  /// Synchronize (for async/batch modes)
  /// 同步（用于异步/批处理模式）
  void synchronize();

  /// Check if operations are complete (non-blocking)
  /// 检查操作是否完成（非阻塞）
  bool isComplete() const;

private:
  /// Verify at least one CUDA device exists; throws std::runtime_error
  /// otherwise
  static void ensureCudaAvailable();

  /// Build a context for a mode, verifying CUDA availability first
  static ExecutionContext buildContext(Mode mode);

  /// Build a context from a policy, verifying CUDA availability first
  static ExecutionContext buildContext(ExecutionPolicy policy);

  ExecutionContext context_;

  /// Auto-sync if in sync mode
  void autoSync();

  /// 统一的算子门面调度：分配 output，以当前 stream 调用底层算子，
  /// 按模式自动同步，返回结果。消除每个门面方法的样板代码。
  /// f 签名：void(CudaImage& output, cudaStream_t stream)
  template <typename F> CudaImage invoke(F&& f) {
    CudaImage output;
    std::forward<F>(f)(output, context_.stream());
    autoSync();
    return output;
  }
};

} // namespace cudaimg
