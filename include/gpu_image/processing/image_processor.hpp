#pragma once

#include "gpu_image/core/execution_context.hpp"
#include "gpu_image/core/gpu_image.hpp"
#include "gpu_image/operators/convolution_engine.hpp"
#include "gpu_image/operators/histogram_calculator.hpp"
#include "gpu_image/operators/image_resizer.hpp"
#include "gpu_image/operators/pixel_operator.hpp"
#include <array>

namespace gpu_image {

/// ImageProcessor: High-level image processing facade
/// 图像处理器：高级图像处理门面
///
/// This module provides a unified interface for image processing operations,
/// with integrated memory management and execution policy support.
///
/// Key features:
/// - Automatic memory pooling (configurable, see ImageAllocator)
/// - Unified sync/async/batch execution model (via ExecutionPolicy)
///
/// Depth: Callers interact with simple method calls; all complexity of
/// stream management, buffer allocation, and synchronization is hidden.
class ImageProcessor {
public:
  /// Processing mode: alias of ExecutionPolicy::Mode (single source of truth)
  /// 处理模式：ExecutionPolicy::Mode 的别名（避免重复枚举与手工映射）
  using Mode = ExecutionPolicy::Mode;

  /// Create a processor with default sync mode
  /// 创建默认同步模式的处理器
  ImageProcessor();

  /// Create a processor with specified mode
  /// 创建指定模式的处理器
  explicit ImageProcessor(Mode mode);

  /// Create a processor with custom execution policy
  /// 创建自定义执行策略的处理器
  explicit ImageProcessor(ExecutionPolicy policy);

  // ===== Configuration =====

  /// Enable or disable memory pooling
  /// 启用或禁用内存池
  void setMemoryPooling(bool enabled);

  /// Check if memory pooling is enabled
  bool isMemoryPoolingEnabled() const;

  /// Set execution mode
  /// 设置执行模式
  void setMode(Mode mode);

  /// Get current execution mode
  Mode mode() const { return context_.policy().mode(); }

  /// Get the underlying execution context
  ExecutionContext& context() { return context_; }
  const ExecutionContext& context() const { return context_; }

  // ===== Image Loading/Unloading =====

  /// Load image from raw memory to GPU
  /// 从原始内存加载图像到 GPU
  GpuImage loadFromMemory(const unsigned char* data, int width, int height,
                          int channels);

  /// Load image from HostImage to GPU
  /// 从 HostImage 加载图像到 GPU
  GpuImage loadFromHost(const HostImage& hostImage);

  /// Download GPU image to host
  /// 下载 GPU 图像到主机
  HostImage download(const GpuImage& image);

  /// Download GPU image to pre-allocated buffer
  /// 下载 GPU 图像到预分配缓冲区
  void downloadToBuffer(const GpuImage& image, unsigned char* buffer,
                        size_t bufferSize);

  // ===== Pixel Operations =====

  /// Invert image colors
  /// 反转图像颜色
  GpuImage invert(const GpuImage& input);

  /// Convert to grayscale
  /// 转换为灰度图
  GpuImage toGrayscale(const GpuImage& input);

  /// Adjust brightness
  /// 调整亮度
  GpuImage adjustBrightness(const GpuImage& input, int offset);

  /// In-place invert
  /// 原地反转
  void invertInPlace(GpuImage& image);

  /// In-place brightness adjustment
  /// 原地调整亮度
  void adjustBrightnessInPlace(GpuImage& image, int offset);

  // ===== Convolution Operations =====

  /// Gaussian blur
  /// 高斯模糊
  GpuImage gaussianBlur(const GpuImage& input, int kernelSize = 5,
                        float sigma = 1.0f);

  /// Sobel edge detection
  /// Sobel 边缘检测
  GpuImage sobelEdgeDetection(const GpuImage& input);

  /// General convolution
  /// 通用卷积
  GpuImage convolve(const GpuImage& input, const float* kernel, int kernelSize);

  // ===== Histogram Operations =====

  /// Calculate histogram (grayscale)
  /// 计算直方图（灰度）
  std::array<int, 256> histogram(const GpuImage& input);

  /// Calculate RGB histograms
  /// 计算 RGB 直方图
  std::array<std::array<int, 256>, 3> histogramRGB(const GpuImage& input);

  /// Histogram equalization
  /// 直方图均衡化
  GpuImage histogramEqualize(const GpuImage& input);

  // ===== Geometric Operations =====

  /// Resize to specific dimensions
  /// 调整到指定尺寸
  GpuImage resize(const GpuImage& input, int newWidth, int newHeight);

  /// Resize by scale factors
  /// 按比例因子调整
  GpuImage resizeByScale(const GpuImage& input, float scaleX, float scaleY);

  // ===== Synchronization =====

  /// Synchronize (for async/batch modes)
  /// 同步（用于异步/批处理模式）
  void synchronize();

  /// Check if operations are complete (non-blocking)
  /// 检查操作是否完成（非阻塞）
  bool isComplete() const;

private:
  /// Verify at least one CUDA device exists; throws std::runtime_error otherwise
  static void ensureCudaAvailable();

  /// Build a context for a mode, verifying CUDA availability first
  static ExecutionContext buildContext(Mode mode);

  /// Build a context from a policy, verifying CUDA availability first
  static ExecutionContext buildContext(ExecutionPolicy policy);

  ExecutionContext context_;

  /// Auto-sync if in sync mode
  void autoSync();
};

} // namespace gpu_image
