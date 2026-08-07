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
  CudaImage loadFromMemory(const unsigned char* data, int width, int height,
                          int channels);

  /// Load image from HostImage to GPU
  /// 从 HostImage 加载图像到 GPU
  CudaImage loadFromHost(const HostImage& hostImage);

  /// Download GPU image to host
  /// 下载 GPU 图像到主机
  HostImage download(const CudaImage& image);

  /// Download GPU image to pre-allocated buffer
  /// 下载 GPU 图像到预分配缓冲区
  void downloadToBuffer(const CudaImage& image, unsigned char* buffer,
                        size_t bufferSize);

  // ===== Pixel Operations =====

  /// Invert image colors
  /// 反转图像颜色
  CudaImage invert(const CudaImage& input);

  /// Convert to grayscale
  /// 转换为灰度图
  CudaImage toGrayscale(const CudaImage& input);

  /// Adjust brightness
  /// 调整亮度
  CudaImage adjustBrightness(const CudaImage& input, int offset);

  /// In-place invert
  /// 原地反转
  void invertInPlace(CudaImage& image);

  /// In-place brightness adjustment
  /// 原地调整亮度
  void adjustBrightnessInPlace(CudaImage& image, int offset);

  // ===== Convolution Operations =====

  /// Gaussian blur
  /// 高斯模糊
  CudaImage gaussianBlur(const CudaImage& input, int kernelSize = 5,
                        float sigma = 1.0f);

  /// Sobel edge detection
  /// Sobel 边缘检测
  CudaImage sobelEdgeDetection(const CudaImage& input);

  /// General convolution
  /// 通用卷积
  CudaImage convolve(const CudaImage& input, const float* kernel, int kernelSize,
                    BorderMode borderMode = BorderMode::Zero);

  // ===== Histogram Operations =====

  /// Calculate histogram (grayscale)
  /// 计算直方图（灰度）
  std::array<int, 256> histogram(const CudaImage& input);

  /// Calculate RGB histograms
  /// 计算 RGB 直方图
  std::array<std::array<int, 256>, 3> histogramRGB(const CudaImage& input);

  /// Histogram equalization
  /// 直方图均衡化
  CudaImage histogramEqualize(const CudaImage& input);

  // ===== Geometric Operations =====

  /// Resize to specific dimensions
  /// 调整到指定尺寸
  CudaImage resize(const CudaImage& input, int newWidth, int newHeight,
                  InterpolationMode mode = InterpolationMode::Bilinear);

  /// Resize by scale factors
  /// 按比例因子调整
  CudaImage resizeByScale(const CudaImage& input, float scaleX, float scaleY,
                         InterpolationMode mode = InterpolationMode::Bilinear);

  // ===== Geometric Transforms (Geometric) =====

  /// Rotate by arbitrary angle (degrees, clockwise)
  /// 任意角度旋转（角度，顺时针）
  CudaImage rotate(const CudaImage& input, float angleDegrees);

  /// Rotate by 90° multiples (more efficient than arbitrary rotate)
  /// 90° 倍数旋转（比任意角度旋转更高效）
  CudaImage rotate90(const CudaImage& input, int times = 1);

  /// Flip image
  /// 图像翻转
  CudaImage flip(const CudaImage& input, FlipDirection direction);

  /// Crop region
  /// 区域裁剪
  CudaImage crop(const CudaImage& input, int x, int y, int width, int height);

  /// Pad image with border
  /// 图像填充
  CudaImage pad(const CudaImage& input, int top, int bottom, int left, int right,
               unsigned char padValue = 0);

  // ===== Morphology =====

  /// Erosion
  /// 腐蚀
  CudaImage erode(const CudaImage& input, int kernelSize = 3,
                 StructuringElement element = StructuringElement::Rectangle);

  /// Dilation
  /// 膨胀
  CudaImage dilate(const CudaImage& input, int kernelSize = 3,
                  StructuringElement element = StructuringElement::Rectangle);

  /// Opening (erode then dilate)
  /// 开运算（先腐蚀后膨胀）
  CudaImage
  morphOpen(const CudaImage& input, int kernelSize = 3,
            StructuringElement element = StructuringElement::Rectangle);

  /// Closing (dilate then erode)
  /// 闭运算（先膨胀后腐蚀）
  CudaImage
  morphClose(const CudaImage& input, int kernelSize = 3,
             StructuringElement element = StructuringElement::Rectangle);

  /// Morphological gradient (dilate - erode)
  /// 形态学梯度（膨胀 - 腐蚀）
  CudaImage
  morphGradient(const CudaImage& input, int kernelSize = 3,
                StructuringElement element = StructuringElement::Rectangle);

  // ===== Threshold =====

  /// Global threshold
  /// 全局阈值
  CudaImage threshold(const CudaImage& input, unsigned char thresh,
                     unsigned char maxVal = 255,
                     ThresholdType type = ThresholdType::Binary);

  /// Adaptive threshold
  /// 自适应阈值
  CudaImage adaptiveThreshold(const CudaImage& input, unsigned char maxVal,
                             AdaptiveMethod method, ThresholdType type,
                             int blockSize, int C);

  /// Otsu binarization (auto threshold)
  /// Otsu 自动二值化
  CudaImage otsuBinarize(const CudaImage& input, unsigned char maxVal = 255);

  // ===== Filters =====

  /// Median filter
  /// 中值滤波
  CudaImage medianFilter(const CudaImage& input, int kernelSize = 3);

  /// Bilateral filter (edge-preserving)
  /// 双边滤波（保边去噪）
  CudaImage bilateralFilter(const CudaImage& input, int kernelSize = 5,
                           float sigmaSpace = 10.0f, float sigmaColor = 50.0f);

  /// Box filter (mean blur)
  /// 均值滤波（盒式滤波）
  CudaImage boxFilter(const CudaImage& input, int kernelSize = 3);

  /// Sharpen filter
  /// 锐化滤波
  CudaImage sharpen(const CudaImage& input, float strength = 1.0f);

  /// Laplacian filter (edge enhancement)
  /// 拉普拉斯滤波（边缘增强）
  CudaImage laplacian(const CudaImage& input);

  // ===== Color Space =====

  /// RGB to HSV
  CudaImage rgbToHsv(const CudaImage& input);
  /// HSV to RGB
  CudaImage hsvToRgb(const CudaImage& input);
  /// RGB to YUV
  CudaImage rgbToYuv(const CudaImage& input);
  /// YUV to RGB
  CudaImage yuvToRgb(const CudaImage& input);
  /// RGB to Lab (CIE L*a*b*)
  CudaImage rgbToLab(const CudaImage& input);
  /// Lab to RGB
  CudaImage labToRgb(const CudaImage& input);

  // ===== Image Arithmetic =====

  /// Add two images
  /// 图像加法
  CudaImage add(const CudaImage& src1, const CudaImage& src2);
  /// Subtract two images
  /// 图像减法
  CudaImage subtract(const CudaImage& src1, const CudaImage& src2);
  /// Pixel-wise multiply
  /// 逐像素乘法
  CudaImage multiply(const CudaImage& src1, const CudaImage& src2,
                    float scale = 1.0f);
  /// Alpha blend
  /// 图像混合
  CudaImage blend(const CudaImage& src1, const CudaImage& src2, float alpha);
  /// Weighted sum
  /// 加权和
  CudaImage addWeighted(const CudaImage& src1, float alpha, const CudaImage& src2,
                       float beta, float gamma = 0.0f);
  /// Absolute difference
  /// 绝对差
  CudaImage absDiff(const CudaImage& src1, const CudaImage& src2);
  /// Add scalar
  /// 标量加法
  CudaImage addScalar(const CudaImage& input, unsigned char value);
  /// Multiply by scalar
  /// 标量乘法
  CudaImage multiplyScalar(const CudaImage& input, float scale);

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
