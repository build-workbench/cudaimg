#include "cudaimg/processing/image_processor.hpp"
#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace cudaimg {

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

CudaImage ImageProcessor::loadFromMemory(const unsigned char* data, int width,
                                        int height, int channels) {
  if (data == nullptr) {
    throw std::invalid_argument("Data pointer is null");
  }
  if (!ImageUtils::validateImageParams(width, height, channels)) {
    throw std::invalid_argument("Invalid image parameters");
  }

  CudaImage image = ImageUtils::createCudaImage(width, height, channels);
  image.buffer.copyFromHost(data, image.totalBytes());

  return image;
}

CudaImage ImageProcessor::loadFromHost(const HostImage& hostImage) {
  return ImageUtils::uploadToGpu(hostImage);
}

HostImage ImageProcessor::download(const CudaImage& image) {
  autoSync();
  return ImageUtils::downloadFromGpu(image);
}

void ImageProcessor::downloadToBuffer(const CudaImage& image,
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

CudaImage ImageProcessor::invert(const CudaImage& input) {
  CudaImage output;
  PixelOperator::invert(input, output, context_.stream());
  autoSync();
  return output;
}

CudaImage ImageProcessor::toGrayscale(const CudaImage& input) {
  CudaImage output;
  PixelOperator::toGrayscale(input, output, context_.stream());
  autoSync();
  return output;
}

CudaImage ImageProcessor::adjustBrightness(const CudaImage& input, int offset) {
  CudaImage output;
  PixelOperator::adjustBrightness(input, output, offset, context_.stream());
  autoSync();
  return output;
}

void ImageProcessor::invertInPlace(CudaImage& image) {
  PixelOperator::invertInPlace(image, context_.stream());
  autoSync();
}

void ImageProcessor::adjustBrightnessInPlace(CudaImage& image, int offset) {
  PixelOperator::adjustBrightnessInPlace(image, offset, context_.stream());
  autoSync();
}

// ===== Convolution Operations =====

CudaImage ImageProcessor::gaussianBlur(const CudaImage& input, int kernelSize,
                                      float sigma) {
  CudaImage output;
  ConvolutionEngine::gaussianBlur(input, output, kernelSize, sigma,
                                  context_.stream());
  autoSync();
  return output;
}

CudaImage ImageProcessor::sobelEdgeDetection(const CudaImage& input) {
  CudaImage output;
  ConvolutionEngine::sobelEdgeDetection(input, output, context_.stream());
  autoSync();
  return output;
}

CudaImage ImageProcessor::convolve(const CudaImage& input, const float* kernel,
                                  int kernelSize, BorderMode borderMode) {
  CudaImage output;
  ConvolutionEngine::convolve(input, output, kernel, kernelSize, borderMode,
                              context_.stream());
  autoSync();
  return output;
}

// ===== Histogram Operations =====

std::array<int, 256> ImageProcessor::histogram(const CudaImage& input) {
  auto result = HistogramCalculator::calculate(input, context_.stream());
  autoSync();
  return result;
}

std::array<std::array<int, 256>, 3>
ImageProcessor::histogramRGB(const CudaImage& input) {
  auto result = HistogramCalculator::calculateRGB(input, context_.stream());
  autoSync();
  return result;
}

CudaImage ImageProcessor::histogramEqualize(const CudaImage& input) {
  CudaImage output;
  HistogramCalculator::equalize(input, output, context_.stream());
  autoSync();
  return output;
}

// ===== Geometric Operations =====

CudaImage ImageProcessor::resize(const CudaImage& input, int newWidth,
                                int newHeight, InterpolationMode mode) {
  CudaImage output;
  ImageResizer::resize(input, output, newWidth, newHeight, mode,
                       context_.stream());
  autoSync();
  return output;
}

CudaImage ImageProcessor::resizeByScale(const CudaImage& input, float scaleX,
                                       float scaleY, InterpolationMode mode) {
  CudaImage output;
  ImageResizer::resizeByScale(input, output, scaleX, scaleY, mode,
                              context_.stream());
  autoSync();
  return output;
}

// ===== Geometric Transforms =====

CudaImage ImageProcessor::rotate(const CudaImage& input, float angleDegrees) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Geometric::rotate(input, o, angleDegrees, s);
  });
}

CudaImage ImageProcessor::rotate90(const CudaImage& input, int times) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Geometric::rotate90(input, o, times, s);
  });
}

CudaImage ImageProcessor::flip(const CudaImage& input, FlipDirection direction) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Geometric::flip(input, o, direction, s);
  });
}

CudaImage ImageProcessor::crop(const CudaImage& input, int x, int y, int width,
                              int height) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Geometric::crop(input, o, x, y, width, height, s);
  });
}

CudaImage ImageProcessor::pad(const CudaImage& input, int top, int bottom,
                             int left, int right, unsigned char padValue) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Geometric::pad(input, o, top, bottom, left, right, padValue, s);
  });
}

// ===== Morphology =====

CudaImage ImageProcessor::erode(const CudaImage& input, int kernelSize,
                               StructuringElement element) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Morphology::erode(input, o, kernelSize, element, s);
  });
}

CudaImage ImageProcessor::dilate(const CudaImage& input, int kernelSize,
                                StructuringElement element) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Morphology::dilate(input, o, kernelSize, element, s);
  });
}

CudaImage ImageProcessor::morphOpen(const CudaImage& input, int kernelSize,
                                   StructuringElement element) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Morphology::open(input, o, kernelSize, element, s);
  });
}

CudaImage ImageProcessor::morphClose(const CudaImage& input, int kernelSize,
                                    StructuringElement element) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Morphology::close(input, o, kernelSize, element, s);
  });
}

CudaImage ImageProcessor::morphGradient(const CudaImage& input, int kernelSize,
                                       StructuringElement element) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Morphology::gradient(input, o, kernelSize, element, s);
  });
}

// ===== Threshold =====

CudaImage ImageProcessor::threshold(const CudaImage& input, unsigned char thresh,
                                   unsigned char maxVal, ThresholdType type) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Threshold::threshold(input, o, thresh, maxVal, type, s);
  });
}

CudaImage ImageProcessor::adaptiveThreshold(const CudaImage& input,
                                           unsigned char maxVal,
                                           AdaptiveMethod method,
                                           ThresholdType type, int blockSize,
                                           int C) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Threshold::adaptiveThreshold(input, o, maxVal, method, type, blockSize, C,
                                 s);
  });
}

CudaImage ImageProcessor::otsuBinarize(const CudaImage& input,
                                      unsigned char maxVal) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Threshold::otsuBinarize(input, o, maxVal, s);
  });
}

// ===== Filters =====

CudaImage ImageProcessor::medianFilter(const CudaImage& input, int kernelSize) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Filters::medianFilter(input, o, kernelSize, s);
  });
}

CudaImage ImageProcessor::bilateralFilter(const CudaImage& input, int kernelSize,
                                         float sigmaSpace, float sigmaColor) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Filters::bilateralFilter(input, o, kernelSize, sigmaSpace, sigmaColor, s);
  });
}

CudaImage ImageProcessor::boxFilter(const CudaImage& input, int kernelSize) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Filters::boxFilter(input, o, kernelSize, s);
  });
}

CudaImage ImageProcessor::sharpen(const CudaImage& input, float strength) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    Filters::sharpen(input, o, strength, s);
  });
}

CudaImage ImageProcessor::laplacian(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { Filters::laplacian(input, o, s); });
}

// ===== Color Space =====

CudaImage ImageProcessor::rgbToHsv(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::rgbToHsv(input, o, s); });
}

CudaImage ImageProcessor::hsvToRgb(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::hsvToRgb(input, o, s); });
}

CudaImage ImageProcessor::rgbToYuv(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::rgbToYuv(input, o, s); });
}

CudaImage ImageProcessor::yuvToRgb(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::yuvToRgb(input, o, s); });
}

CudaImage ImageProcessor::rgbToLab(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::rgbToLab(input, o, s); });
}

CudaImage ImageProcessor::labToRgb(const CudaImage& input) {
  return invoke(
      [&](CudaImage& o, cudaStream_t s) { ColorSpace::labToRgb(input, o, s); });
}

// ===== Image Arithmetic =====

CudaImage ImageProcessor::add(const CudaImage& src1, const CudaImage& src2) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::add(src1, src2, o, s);
  });
}

CudaImage ImageProcessor::subtract(const CudaImage& src1, const CudaImage& src2) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::subtract(src1, src2, o, s);
  });
}

CudaImage ImageProcessor::multiply(const CudaImage& src1, const CudaImage& src2,
                                  float scale) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::multiply(src1, src2, o, scale, s);
  });
}

CudaImage ImageProcessor::blend(const CudaImage& src1, const CudaImage& src2,
                               float alpha) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::blend(src1, src2, o, alpha, s);
  });
}

CudaImage ImageProcessor::addWeighted(const CudaImage& src1, float alpha,
                                     const CudaImage& src2, float beta,
                                     float gamma) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::addWeighted(src1, alpha, src2, beta, o, gamma, s);
  });
}

CudaImage ImageProcessor::absDiff(const CudaImage& src1, const CudaImage& src2) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::absDiff(src1, src2, o, s);
  });
}

CudaImage ImageProcessor::addScalar(const CudaImage& input, unsigned char value) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
    ImageArithmetic::addScalar(input, o, value, s);
  });
}

CudaImage ImageProcessor::multiplyScalar(const CudaImage& input, float scale) {
  return invoke([&](CudaImage& o, cudaStream_t s) {
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

} // namespace cudaimg
