#include "gpu_image/processing/image_processor.hpp"
#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/image_utils.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace gpu_image {

// ===== ImageProcessor Implementation =====

ImageProcessor::ImageProcessor()
    : mode_(Mode::Sync), context_(ExecutionPolicy::sync()) {
  // Verify CUDA availability
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    throw std::runtime_error("No CUDA devices available");
  }
}

ImageProcessor::ImageProcessor(Mode mode)
    : mode_(mode),
      context_(mode == Mode::Sync ? ExecutionPolicy::sync()
               : mode == Mode::Async ? ExecutionPolicy::async()
                                     : ExecutionPolicy::batch()) {
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    throw std::runtime_error("No CUDA devices available");
  }
}

ImageProcessor::ImageProcessor(ExecutionPolicy policy)
    : mode_(policy.mode() == ExecutionPolicy::Mode::Sync
                ? Mode::Sync
            : policy.mode() == ExecutionPolicy::Mode::Async
                ? Mode::Async
                : Mode::Batch),
      context_(std::move(policy)) {
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    throw std::runtime_error("No CUDA devices available");
  }
}

ImageProcessor::~ImageProcessor() {
  // Recycle temp buffer back to pool if pooling enabled
  if (tempBuffer_.isValid()) {
    context_.recycleToPool(std::move(tempBuffer_));
  }
}

// ===== Configuration =====

void ImageProcessor::setMemoryPooling(bool enabled) {
  ImageAllocator::instance().setPoolingEnabled(enabled);
}

bool ImageProcessor::isMemoryPoolingEnabled() const {
  return ImageAllocator::instance().isPoolingEnabled();
}

void ImageProcessor::setMode(Mode mode) {
  mode_ = mode;
  context_ = ExecutionContext(
      mode == Mode::Sync ? ExecutionPolicy::sync()
      : mode == Mode::Async
          ? ExecutionPolicy::async()
          : ExecutionPolicy::batch());
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
                                  int kernelSize) {
  GpuImage output;
  ConvolutionEngine::convolve(input, output, kernel, kernelSize,
                              BorderMode::Zero, context_.stream());
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
                                int newHeight) {
  GpuImage output;
  ImageResizer::resize(input, output, newWidth, newHeight, context_.stream());
  autoSync();
  return output;
}

GpuImage ImageProcessor::resizeByScale(const GpuImage& input, float scaleX,
                                       float scaleY) {
  GpuImage output;
  ImageResizer::resizeByScale(input, output, scaleX, scaleY,
                              context_.stream());
  autoSync();
  return output;
}

// ===== Synchronization =====

void ImageProcessor::synchronize() { context_.synchronize(); }

bool ImageProcessor::isComplete() const {
  if (mode_ == Mode::Sync) {
    return true;
  }
  cudaError_t err = cudaStreamQuery(context_.stream());
  return err == cudaSuccess;
}

// ===== Private Helpers =====

GpuImage& ImageProcessor::prepareOutput(const GpuImage& input,
                                        GpuImage& output) {
  context_.ensureOutputSize(input, output);
  return output;
}

void ImageProcessor::autoSync() {
  if (mode_ == Mode::Sync) {
    context_.synchronize();
  }
}

} // namespace gpu_image
