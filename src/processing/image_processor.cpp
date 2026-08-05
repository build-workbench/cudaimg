#include "gpu_image/processing/image_processor.hpp"
#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/image_utils.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace gpu_image {

// ===== ImageProcessor Implementation =====

namespace {

// 由处理模式构建对应的执行策略
ExecutionPolicy makePolicy(ExecutionPolicy::Mode mode) {
  return mode == ExecutionPolicy::Mode::Sync    ? ExecutionPolicy::sync()
         : mode == ExecutionPolicy::Mode::Async ? ExecutionPolicy::async()
                                                : ExecutionPolicy::batch();
}

} // namespace

void ImageProcessor::ensureCudaAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    throw std::runtime_error("No CUDA devices available");
  }
}

ExecutionContext ImageProcessor::buildContext(Mode mode) {
  ensureCudaAvailable(); // 先校验设备，保证所有模式抛出一致的异常类型
  return ExecutionContext(makePolicy(mode));
}

ExecutionContext ImageProcessor::buildContext(ExecutionPolicy policy) {
  ensureCudaAvailable();
  return ExecutionContext(std::move(policy));
}

ImageProcessor::ImageProcessor() : ImageProcessor(Mode::Sync) {}

ImageProcessor::ImageProcessor(Mode mode) : context_(buildContext(mode)) {}

ImageProcessor::ImageProcessor(ExecutionPolicy policy)
    : context_(buildContext(std::move(policy))) {}

// ===== Configuration =====

void ImageProcessor::setMemoryPooling(bool enabled) {
  ImageAllocator::instance().setPoolingEnabled(enabled);
}

bool ImageProcessor::isMemoryPoolingEnabled() const {
  return ImageAllocator::instance().isPoolingEnabled();
}

void ImageProcessor::setMode(Mode mode) {
  context_ = ExecutionContext(makePolicy(mode));
}

// ===== Image Loading/Unloading =====

GpuImage ImageProcessor::loadFromMemory(const unsigned char* data, int width,
                                        int height, int channels) {
  if (data == nullptr) {
    throw std::invalid_argument("Data pointer is null");
  }
  if (!ImageUtils::validateImageParams(width, height, channels)) {
    throw std::invalid_argument("Invalid image parameters");
  }

  GpuImage image = ImageUtils::createGpuImage(width, height, channels);
  image.buffer.copyFromHost(data, image.totalBytes());

  return image;
}

GpuImage ImageProcessor::loadFromHost(const HostImage& hostImage) {
  return ImageUtils::uploadToGpu(hostImage);
}

HostImage ImageProcessor::download(const GpuImage& image) {
  autoSync();
  return ImageUtils::downloadFromGpu(image);
}

void ImageProcessor::downloadToBuffer(const GpuImage& image,
                                      unsigned char* buffer,
                                      size_t bufferSize) {
  if (buffer == nullptr) {
    throw std::invalid_argument("Buffer is null");
  }
  if (bufferSize < image.totalBytes()) {
    throw std::invalid_argument("Buffer too small");
  }
  autoSync();
  image.buffer.copyToHost(buffer, image.totalBytes());
}

// ===== Pixel Operations =====

GpuImage ImageProcessor::invert(const GpuImage& input) {
  GpuImage output;
  PixelOperator::invert(input, output, context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::toGrayscale(const GpuImage& input) {
  GpuImage output;
  PixelOperator::toGrayscale(input, output, context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::adjustBrightness(const GpuImage& input, int offset) {
  GpuImage output;
  PixelOperator::adjustBrightness(input, output, offset, context_.stream());
  autoSync();
  return output;
}

void ImageProcessor::invertInPlace(GpuImage& image) {
  PixelOperator::invertInPlace(image, context_.stream());
  autoSync();
}

void ImageProcessor::adjustBrightnessInPlace(GpuImage& image, int offset) {
  PixelOperator::adjustBrightnessInPlace(image, offset, context_.stream());
  autoSync();
}

// ===== Convolution Operations =====

GpuImage ImageProcessor::gaussianBlur(const GpuImage& input, int kernelSize,
                                      float sigma) {
  GpuImage output;
  ConvolutionEngine::gaussianBlur(input, output, kernelSize, sigma,
                                  context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::sobelEdgeDetection(const GpuImage& input) {
  GpuImage output;
  ConvolutionEngine::sobelEdgeDetection(input, output, context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::convolve(const GpuImage& input, const float* kernel,
                                  int kernelSize, BorderMode borderMode) {
  GpuImage output;
  ConvolutionEngine::convolve(input, output, kernel, kernelSize, borderMode,
                              context_.stream());
  autoSync();
  return output;
}

// ===== Histogram Operations =====

std::array<int, 256> ImageProcessor::histogram(const GpuImage& input) {
  auto result = HistogramCalculator::calculate(input, context_.stream());
  autoSync();
  return result;
}

std::array<std::array<int, 256>, 3>
ImageProcessor::histogramRGB(const GpuImage& input) {
  auto result = HistogramCalculator::calculateRGB(input, context_.stream());
  autoSync();
  return result;
}

GpuImage ImageProcessor::histogramEqualize(const GpuImage& input) {
  GpuImage output;
  HistogramCalculator::equalize(input, output, context_.stream());
  autoSync();
  return output;
}

// ===== Geometric Operations =====

GpuImage ImageProcessor::resize(const GpuImage& input, int newWidth,
                                int newHeight, InterpolationMode mode) {
  GpuImage output;
  ImageResizer::resize(input, output, newWidth, newHeight, mode,
                       context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::resizeByScale(const GpuImage& input, float scaleX,
                                       float scaleY, InterpolationMode mode) {
  GpuImage output;
  ImageResizer::resizeByScale(input, output, scaleX, scaleY, mode,
                              context_.stream());
  autoSync();
  return output;
}

// ===== Geometric Transforms =====

GpuImage ImageProcessor::rotate(const GpuImage& input, float angleDegrees) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Geometric::rotate(input, o, angleDegrees, s);
  });
}

GpuImage ImageProcessor::rotate90(const GpuImage& input, int times) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Geometric::rotate90(input, o, times, s);
  });
}

GpuImage ImageProcessor::flip(const GpuImage& input, FlipDirection direction) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Geometric::flip(input, o, direction, s);
  });
}

GpuImage ImageProcessor::crop(const GpuImage& input, int x, int y, int width,
                              int height) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Geometric::crop(input, o, x, y, width, height, s);
  });
}

GpuImage ImageProcessor::pad(const GpuImage& input, int top, int bottom,
                             int left, int right, unsigned char padValue) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Geometric::pad(input, o, top, bottom, left, right, padValue, s);
  });
}

// ===== Morphology =====

GpuImage ImageProcessor::erode(const GpuImage& input, int kernelSize,
                               StructuringElement element) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Morphology::erode(input, o, kernelSize, element, s);
  });
}

GpuImage ImageProcessor::dilate(const GpuImage& input, int kernelSize,
                                StructuringElement element) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Morphology::dilate(input, o, kernelSize, element, s);
  });
}

GpuImage ImageProcessor::morphOpen(const GpuImage& input, int kernelSize,
                                   StructuringElement element) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Morphology::open(input, o, kernelSize, element, s);
  });
}

GpuImage ImageProcessor::morphClose(const GpuImage& input, int kernelSize,
                                    StructuringElement element) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Morphology::close(input, o, kernelSize, element, s);
  });
}

GpuImage ImageProcessor::morphGradient(const GpuImage& input, int kernelSize,
                                       StructuringElement element) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Morphology::gradient(input, o, kernelSize, element, s);
  });
}

// ===== Threshold =====

GpuImage ImageProcessor::threshold(const GpuImage& input, unsigned char thresh,
                                   unsigned char maxVal, ThresholdType type) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Threshold::threshold(input, o, thresh, maxVal, type, s);
  });
}

GpuImage ImageProcessor::adaptiveThreshold(const GpuImage& input,
                                           unsigned char maxVal,
                                           AdaptiveMethod method,
                                           ThresholdType type, int blockSize,
                                           int C) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Threshold::adaptiveThreshold(input, o, maxVal, method, type, blockSize, C,
                                 s);
  });
}

GpuImage ImageProcessor::otsuBinarize(const GpuImage& input,
                                      unsigned char maxVal) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Threshold::otsuBinarize(input, o, maxVal, s);
  });
}

// ===== Filters =====

GpuImage ImageProcessor::medianFilter(const GpuImage& input, int kernelSize) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Filters::medianFilter(input, o, kernelSize, s);
  });
}

GpuImage ImageProcessor::bilateralFilter(const GpuImage& input, int kernelSize,
                                         float sigmaSpace, float sigmaColor) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Filters::bilateralFilter(input, o, kernelSize, sigmaSpace, sigmaColor, s);
  });
}

GpuImage ImageProcessor::boxFilter(const GpuImage& input, int kernelSize) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Filters::boxFilter(input, o, kernelSize, s);
  });
}

GpuImage ImageProcessor::sharpen(const GpuImage& input, float strength) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    Filters::sharpen(input, o, strength, s);
  });
}

GpuImage ImageProcessor::laplacian(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { Filters::laplacian(input, o, s); });
}

// ===== Color Space =====

GpuImage ImageProcessor::rgbToHsv(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::rgbToHsv(input, o, s); });
}

GpuImage ImageProcessor::hsvToRgb(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::hsvToRgb(input, o, s); });
}

GpuImage ImageProcessor::rgbToYuv(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::rgbToYuv(input, o, s); });
}

GpuImage ImageProcessor::yuvToRgb(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::yuvToRgb(input, o, s); });
}

GpuImage ImageProcessor::rgbToLab(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::rgbToLab(input, o, s); });
}

GpuImage ImageProcessor::labToRgb(const GpuImage& input) {
  return invoke(
      [&](GpuImage& o, cudaStream_t s) { ColorSpace::labToRgb(input, o, s); });
}

// ===== Image Arithmetic =====

GpuImage ImageProcessor::add(const GpuImage& src1, const GpuImage& src2) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::add(src1, src2, o, s);
  });
}

GpuImage ImageProcessor::subtract(const GpuImage& src1, const GpuImage& src2) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::subtract(src1, src2, o, s);
  });
}

GpuImage ImageProcessor::multiply(const GpuImage& src1, const GpuImage& src2,
                                  float scale) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::multiply(src1, src2, o, scale, s);
  });
}

GpuImage ImageProcessor::blend(const GpuImage& src1, const GpuImage& src2,
                               float alpha) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::blend(src1, src2, o, alpha, s);
  });
}

GpuImage ImageProcessor::addWeighted(const GpuImage& src1, float alpha,
                                     const GpuImage& src2, float beta,
                                     float gamma) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::addWeighted(src1, alpha, src2, beta, o, gamma, s);
  });
}

GpuImage ImageProcessor::absDiff(const GpuImage& src1, const GpuImage& src2) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::absDiff(src1, src2, o, s);
  });
}

GpuImage ImageProcessor::addScalar(const GpuImage& input, unsigned char value) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::addScalar(input, o, value, s);
  });
}

GpuImage ImageProcessor::multiplyScalar(const GpuImage& input, float scale) {
  return invoke([&](GpuImage& o, cudaStream_t s) {
    ImageArithmetic::multiplyScalar(input, o, scale, s);
  });
}

// ===== Synchronization =====

void ImageProcessor::synchronize() { context_.synchronize(); }

bool ImageProcessor::isComplete() const {
  if (context_.policy().mode() == ExecutionPolicy::Mode::Sync) {
    return true;
  }
  cudaError_t err = cudaStreamQuery(context_.stream());
  if (err == cudaSuccess) {
    return true;
  }
  if (err == cudaErrorNotReady) {
    return false; // 操作仍在执行中
  }
  // 真实错误（如无效 stream 句柄）：清除粘性错误状态并抛出，避免被静默吞掉
  cudaGetLastError();
  throw CudaException(err, __FILE__, __LINE__);
}

// ===== Private Helpers =====

void ImageProcessor::autoSync() {
  if (context_.policy().mode() == ExecutionPolicy::Mode::Sync) {
    context_.synchronize();
  }
}

} // namespace gpu_image
