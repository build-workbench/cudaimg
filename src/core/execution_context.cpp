#include "cudaimg/core/execution_context.hpp"
#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/memory_manager.hpp"
#include <stdexcept>

namespace cudaimg {

// ===== ExecutionPolicy Implementation =====

ExecutionPolicy::ExecutionPolicy(Mode mode)
    : mode_(mode), stream_(nullptr), ownsStream_(false) {
  if (mode == Mode::Async || mode == Mode::Batch) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    ownsStream_ = true;
  }
}

ExecutionPolicy::ExecutionPolicy(Mode mode, cudaStream_t stream, bool owns)
    : mode_(mode), stream_(stream), ownsStream_(owns) {}

ExecutionPolicy ExecutionPolicy::async(cudaStream_t stream) {
  if (stream != nullptr) {
    return ExecutionPolicy(Mode::Async, stream, false);
  }
  return ExecutionPolicy(Mode::Async);
}

ExecutionPolicy ExecutionPolicy::batch() {
  return ExecutionPolicy(Mode::Batch);
}

ExecutionPolicy::~ExecutionPolicy() {
  if (ownsStream_ && stream_ != nullptr) {
    // Ignore errors in destructor - don't throw
    // Stream may already be invalid if CUDA context was destroyed
    (void)cudaStreamDestroy(stream_);
  }
}

ExecutionPolicy::ExecutionPolicy(ExecutionPolicy&& other) noexcept
    : mode_(other.mode_), stream_(other.stream_),
      ownsStream_(other.ownsStream_) {
  other.mode_ = Mode::Sync;
  other.stream_ = nullptr;
  other.ownsStream_ = false;
}

ExecutionPolicy& ExecutionPolicy::operator=(ExecutionPolicy&& other) noexcept {
  if (this != &other) {
    if (ownsStream_ && stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    mode_ = other.mode_;
    stream_ = other.stream_;
    ownsStream_ = other.ownsStream_;
    other.mode_ = Mode::Sync;
    other.stream_ = nullptr;
    other.ownsStream_ = false;
  }
  return *this;
}

void ExecutionPolicy::synchronize() const {
  if (mode_ == Mode::Sync) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else if (stream_ != nullptr) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
}

// ===== ImageAllocator Implementation =====

ImageAllocator& ImageAllocator::instance() {
  static ImageAllocator instance;
  return instance;
}

CudaImage ImageAllocator::allocate(int width, int height, int channels,
                                  bool usePool) {
  if (width <= 0 || height <= 0 ||
      (channels != 1 && channels != 3 && channels != 4)) {
    throw std::invalid_argument("Invalid image parameters");
  }

  CudaImage image;
  image.width = width;
  image.height = height;
  image.channels = channels;

  size_t size = image.totalBytes();

  if (usePool && poolingEnabled_) {
    image.buffer = MemoryManager::instance().allocate(size);
  } else {
    image.buffer = DeviceBuffer(size);
  }

  return image;
}

bool ImageAllocator::ensureSize(const CudaImage& input, CudaImage& output,
                                bool usePool) {
  if (output.width != input.width || output.height != input.height ||
      output.channels != input.channels || !output.isValid()) {
    output = allocate(input.width, input.height, input.channels, usePool);
    return true;
  }
  return false;
}

bool ImageAllocator::ensureSize(CudaImage& output, int width, int height,
                                int channels, bool usePool) {
  if (output.width != width || output.height != height ||
      output.channels != channels || !output.isValid()) {
    output = allocate(width, height, channels, usePool);
    return true;
  }
  return false;
}

void ImageAllocator::recycleToPool(CudaImage&& image) {
  if (poolingEnabled_.load(std::memory_order_relaxed) && image.isValid()) {
    MemoryManager::instance().deallocate(std::move(image.buffer));
  }
  // If pooling disabled, just let the CudaImage destructor handle it
}

// ===== ExecutionContext Implementation =====

ExecutionContext::ExecutionContext(ExecutionPolicy policy)
    : policy_(std::move(policy)) {}

ExecutionContext::ExecutionContext(ExecutionPolicy::Mode mode)
    : policy_(mode == ExecutionPolicy::Mode::Sync ? ExecutionPolicy::sync()
              : mode == ExecutionPolicy::Mode::Async
                  ? ExecutionPolicy::async()
                  : ExecutionPolicy::batch()) {}

CudaImage ExecutionContext::allocateOutput(const CudaImage& input) {
  return ImageAllocator::instance().allocate(input.width, input.height,
                                             input.channels);
}

CudaImage ExecutionContext::allocateOutput(int width, int height, int channels) {
  return ImageAllocator::instance().allocate(width, height, channels);
}

bool ExecutionContext::ensureOutputSize(const CudaImage& input,
                                        CudaImage& output) {
  return ImageAllocator::instance().ensureSize(input, output);
}

bool ExecutionContext::ensureOutputSize(CudaImage& output, int width, int height,
                                        int channels) {
  return ImageAllocator::instance().ensureSize(output, width, height, channels);
}

void ExecutionContext::recycleToPool(CudaImage&& image) {
  ImageAllocator::instance().recycleToPool(std::move(image));
}

} // namespace cudaimg
