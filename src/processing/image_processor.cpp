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
                                       float scaleY,
                                       InterpolationMode mode) {
  GpuImage output;
  ImageResizer::resizeByScale(input, output, scaleX, scaleY, mode,
                              context_.stream());
  autoSync();
  return output;
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
