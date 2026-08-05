#pragma once

#include "gpu_image/core/execution_context.hpp"
#include "gpu_image/core/gpu_image.hpp"
#include "gpu_image/operators/color_space.hpp"
#include "gpu_image/operators/convolution_engine.hpp"
#include "gpu_image/operators/filters.hpp"
#include "gpu_image/operators/geometric.hpp"
#include "gpu_image/operators/histogram_calculator.hpp"
#include "gpu_image/operators/image_resizer.hpp"
#include "gpu_image/operators/morphology.hpp"
#include "gpu_image/operators/pixel_operator.hpp"
#include "gpu_image/operators/threshold.hpp"
#include <array>
#include <functional>

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
  GpuImage convolve(const GpuImage& input, const float* kernel, int kernelSize,
                    BorderMode borderMode = BorderMode::Zero);

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
  GpuImage resize(const GpuImage& input, int newWidth, int newHeight,
                  InterpolationMode mode = InterpolationMode::Bilinear);

  /// Resize by scale factors
  /// 按比例因子调整
  GpuImage resizeByScale(const GpuImage& input, float scaleX, float scaleY,
                         InterpolationMode mode = InterpolationMode::Bilinear);

  // ===== Geometric Transforms (Geometric) =====

  /// Rotate by arbitrary angle (degrees, clockwise)
  /// 任意角度旋转（角度，顺时针）
  GpuImage rotate(const GpuImage& input, float angleDegrees);

  /// Rotate by 90° multiples (more efficient than arbitrary rotate)
  /// 90° 倍数旋转（比任意角度旋转更高效）
  GpuImage rotate90(const GpuImage& input, int times = 1);

  /// Flip image
  /// 图像翻转
  GpuImage flip(const GpuImage& input, FlipDirection direction);

  /// Crop region
  /// 区域裁剪
  GpuImage crop(const GpuImage& input, int x, int y, int width, int height);

  /// Pad image with border
  /// 图像填充
  GpuImage pad(const GpuImage& input, int top, int bottom, int left, int right,
               unsigned char padValue = 0);

  // ===== Morphology =====

  /// Erosion
  /// 腐蚀
  GpuImage erode(const GpuImage& input, int kernelSize = 3,
                 StructuringElement element = StructuringElement::Rectangle);

  /// Dilation
  /// 膨胀
  GpuImage dilate(const GpuImage& input, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle);

  /// Opening (erode then dilate)
  /// 开运算（先腐蚀后膨胀）
  GpuImage
  morphOpen(const GpuImage& input, int kernelSize = 3,
            StructuringElement element = StructuringElement::Rectangle);

  /// Closing (dilate then erode)
  /// 闭运算（先膨胀后腐蚀）
  GpuImage
  morphClose(const GpuImage& input, int kernelSize = 3,
             StructuringElement element = StructuringElement::Rectangle);

  /// Morphological gradient (dilate - erode)
  /// 形态学梯度（膨胀 - 腐蚀）
  GpuImage
  morphGradient(const GpuImage& input, int kernelSize = 3,
                StructuringElement element = StructuringElement::Rectangle);

  // ===== Threshold =====

  /// Global threshold
  /// 全局阈值
  GpuImage threshold(const GpuImage& input, unsigned char thresh,
                     unsigned char maxVal = 255,
                     ThresholdType type = ThresholdType::Binary);

  /// Adaptive threshold
  /// 自适应阈值
  GpuImage adaptiveThreshold(const GpuImage& input, unsigned char maxVal,
                             AdaptiveMethod method, ThresholdType type,
                             int blockSize, int C);

  /// Otsu binarization (auto threshold)
  /// Otsu 自动二值化
  GpuImage otsuBinarize(const GpuImage& input, unsigned char maxVal = 255);

  // ===== Filters =====

  /// Median filter
  /// 中值滤波
  GpuImage medianFilter(const GpuImage& input, int kernelSize = 3);

  /// Bilateral filter (edge-preserving)
  /// 双边滤波（保边去噪）
  GpuImage bilateralFilter(const GpuImage& input, int kernelSize = 5,
                           float sigmaSpace = 10.0f, float sigmaColor = 50.0f);

  /// Box filter (mean blur)
  /// 均值滤波（盒式滤波）
  GpuImage boxFilter(const GpuImage& input, int kernelSize = 3);

  /// Sharpen filter
  /// 锐化滤波
  GpuImage sharpen(const GpuImage& input, float strength = 1.0f);

  /// Laplacian filter (edge enhancement)
  /// 拉普拉斯滤波（边缘增强）
  GpuImage laplacian(const GpuImage& input);

  // ===== Color Space =====

  /// RGB to HSV
  GpuImage rgbToHsv(const GpuImage& input);
  /// HSV to RGB
  GpuImage hsvToRgb(const GpuImage& input);
  /// RGB to YUV
  GpuImage rgbToYuv(const GpuImage& input);
  /// YUV to RGB
  GpuImage yuvToRgb(const GpuImage& input);
  /// RGB to Lab (CIE L*a*b*)
  GpuImage rgbToLab(const GpuImage& input);
  /// Lab to RGB
  GpuImage labToRgb(const GpuImage& input);

  // ===== Image Arithmetic =====

  /// Add two images
  /// 图像加法
  GpuImage add(const GpuImage& src1, const GpuImage& src2);
  /// Subtract two images
  /// 图像减法
  GpuImage subtract(const GpuImage& src1, const GpuImage& src2);
  /// Pixel-wise multiply
  /// 逐像素乘法
  GpuImage multiply(const GpuImage& src1, const GpuImage& src2,
                    float scale = 1.0f);
  /// Alpha blend
  /// 图像混合
  GpuImage blend(const GpuImage& src1, const GpuImage& src2, float alpha);
  /// Weighted sum
  /// 加权和
  GpuImage addWeighted(const GpuImage& src1, float alpha, const GpuImage& src2,
                       float beta, float gamma = 0.0f);
  /// Absolute difference
  /// 绝对差
  GpuImage absDiff(const GpuImage& src1, const GpuImage& src2);
  /// Add scalar
  /// 标量加法
  GpuImage addScalar(const GpuImage& input, unsigned char value);
  /// Multiply by scalar
  /// 标量乘法
  GpuImage multiplyScalar(const GpuImage& input, float scale);

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
  /// f 签名：void(GpuImage& output, cudaStream_t stream)
  template <typename F> GpuImage invoke(F&& f) {
    GpuImage output;
    std::forward<F>(f)(output, context_.stream());
    autoSync();
    return output;
  }
};

} // namespace gpu_image
