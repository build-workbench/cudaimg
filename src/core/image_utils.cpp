#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/execution_context.hpp"
#include "cudaimg/core/image.hpp"
#include <stdexcept>

namespace cudaimg {

// ===== Memory pooling configuration (delegates to ImageAllocator) =====

void ImageUtils::setMemoryPoolingEnabled(bool enabled) {
  ImageAllocator::instance().setPoolingEnabled(enabled);
}

bool ImageUtils::isMemoryPoolingEnabled() {
  return ImageAllocator::instance().isPoolingEnabled();
}

// ===== Image Creation =====

CudaImage ImageUtils::createCudaImage(int width, int height, int channels) {
  return ImageAllocator::instance().allocate(width, height, channels);
}

HostImage ImageUtils::createHostImage(int width, int height, int channels) {
  if (!validateImageParams(width, height, channels)) {
    throw std::invalid_argument("Invalid image parameters");
  }

  HostImage image;
  image.width = width;
  image.height = height;
  image.channels = channels;
  image.data.resize(image.totalBytes());

  return image;
}

// ===== Host-Device Transfer =====

CudaImage ImageUtils::uploadToGpu(const HostImage& hostImage) {
  if (!hostImage.isValid()) {
    throw std::invalid_argument("Invalid host image");
  }

  CudaImage gpuImage =
      createCudaImage(hostImage.width, hostImage.height, hostImage.channels);
  gpuImage.buffer.copyFromHost(hostImage.data.data(), hostImage.totalBytes());

  return gpuImage;
}

HostImage ImageUtils::downloadFromGpu(const CudaImage& gpuImage) {
  if (!gpuImage.isValid()) {
    throw std::invalid_argument("Invalid GPU image");
  }

  HostImage hostImage =
      createHostImage(gpuImage.width, gpuImage.height, gpuImage.channels);
  gpuImage.buffer.copyToHost(hostImage.data.data(), hostImage.totalBytes());

  return hostImage;
}

CudaImage ImageUtils::clone(const CudaImage& gpuImage) {
  if (!gpuImage.isValid()) {
    return CudaImage{};
  }

  CudaImage copy =
      createCudaImage(gpuImage.width, gpuImage.height, gpuImage.channels);
  copy.buffer.copyFromDevice(gpuImage.buffer.data(), gpuImage.totalBytes());

  return copy;
}

void ImageUtils::uploadToGpuAsync(const HostImage& hostImage,
                                  CudaImage& gpuImage, cudaStream_t stream) {
  if (!hostImage.isValid()) {
    throw std::invalid_argument("Invalid host image");
  }

  ensureOutputSize(gpuImage, hostImage.width, hostImage.height,
                   hostImage.channels);

  gpuImage.buffer.copyFromHostAsync(hostImage.data.data(),
                                    hostImage.totalBytes(), stream);
}

void ImageUtils::downloadFromGpuAsync(const CudaImage& gpuImage,
                                      HostImage& hostImage,
                                      cudaStream_t stream) {
  if (!gpuImage.isValid()) {
    throw std::invalid_argument("Invalid GPU image");
  }

  // 复用传入的 HostImage 仅当尺寸字段匹配且底层缓冲确实足够大；
  // 否则重建。缺少容量检查会让手工构造的（尺寸字段匹配但 data 为空/过小的）
  // HostImage 在 copyToHostAsync 处发生越界写（堆破坏）。
  if (hostImage.width != gpuImage.width ||
      hostImage.height != gpuImage.height ||
      hostImage.channels != gpuImage.channels ||
      hostImage.data.size() < gpuImage.totalBytes()) {
    hostImage =
        createHostImage(gpuImage.width, gpuImage.height, gpuImage.channels);
  }

  gpuImage.buffer.copyToHostAsync(hostImage.data.data(), hostImage.totalBytes(),
                                  stream);
}

// ===== Output Buffer Management =====

void ImageUtils::ensureOutputSize(const CudaImage& input, CudaImage& output) {
  ImageAllocator::instance().ensureSize(input, output);
}

void ImageUtils::ensureOutputSize(CudaImage& output, int width, int height,
                                  int channels) {
  ImageAllocator::instance().ensureSize(output, width, height, channels);
}

// ===== Validation =====

bool ImageUtils::validateImageParams(int width, int height, int channels) {
  return width > 0 && height > 0 &&
         (channels == 1 || channels == 3 || channels == 4);
}

} // namespace cudaimg
