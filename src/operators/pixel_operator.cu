#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/device_kernels.cuh"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/core/kernel_helpers.hpp"
#include "cudaimg/operators/pixel_operator.hpp"
#include <stdexcept>

namespace cudaimg {

// Vectorized kernels: process 4 bytes per thread (uchar4)

__global__ void invertKernelVec4(const uchar4* input, uchar4* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uchar4 v = input[i];
    output[i] = make_uchar4(255 - v.x, 255 - v.y, 255 - v.z, 255 - v.w);
  }
}

__global__ void brightnessKernelVec4(const uchar4* input, uchar4* output, int n,
                                     int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uchar4 v = input[i];
    output[i] =
        make_uchar4(static_cast<unsigned char>(min(max(v.x + offset, 0), 255)),
                    static_cast<unsigned char>(min(max(v.y + offset, 0), 255)),
                    static_cast<unsigned char>(min(max(v.z + offset, 0), 255)),
                    static_cast<unsigned char>(min(max(v.w + offset, 0), 255)));
  }
}

// Scalar fallback kernels: per-pixel, handles any channel count

__global__ void invertKernelScalar(const unsigned char* input,
                                   unsigned char* output, int width, int height,
                                   int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c) {
      output[idx + c] = 255 - input[idx + c];
    }
  }
}

__global__ void brightnessKernelScalar(const unsigned char* input,
                                       unsigned char* output, int width,
                                       int height, int channels, int offset) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = dev::pixelIndex(x, y, width, channels);
    for (int c = 0; c < channels; ++c) {
      output[idx + c] = dev::clampToUchar(input[idx + c] + offset);
    }
  }
}

__global__ void toGrayscaleKernel(const unsigned char* input,
                                  unsigned char* output, int width, int height,
                                  int inputChannels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int inIdx = dev::pixelIndex(x, y, width, inputChannels);
    int outIdx = y * width + x;
    output[outIdx] =
        dev::rgbToGrayUchar(input[inIdx], input[inIdx + 1], input[inIdx + 2]);
  }
}

// ── 教学重点：向量化 dispatch 模式 ─────────────────────────────
//
// 同一个反色操作有两条路径：
//   快路径（invertKernelVec4）：总字节数能被 4 整除时，用 uchar4
//     一次读写 4 字节，1D grid 遍历。内存事务数减少 4 倍。
//   慢路径（invertKernelScalar）：不满足对齐时退回逐字节处理，
//     2D grid 按 (x, y) 坐标遍历。
//
// 这是 CUDA 工程中的常见模式：有快路径时走快路径，不满足条件时
// 退回到通用实现。canVectorize() 是 dispatch 的判据。

static inline bool canVectorize(const CudaImage& img) {
  return img.totalBytes() % 4 == 0;
}

static void launchInvert(const unsigned char* input, unsigned char* output,
                         const CudaImage& img, cudaStream_t stream) {
  if (canVectorize(img)) {
    int n = static_cast<int>(img.totalBytes() / 4);
    int blockSize = kBlockSize1D;
    int gridSize = (n + blockSize - 1) / blockSize;
    invertKernelVec4<<<gridSize, blockSize, 0, stream>>>(
        reinterpret_cast<const uchar4*>(input),
        reinterpret_cast<uchar4*>(output), n);
  } else {
    dim3 grid, block;
    calcGridBlock2D(img.width, img.height, grid, block);
    invertKernelScalar<<<grid, block, 0, stream>>>(input, output, img.width,
                                                   img.height, img.channels);
  }
}

static void launchBrightness(const unsigned char* input, unsigned char* output,
                             const CudaImage& img, int offset,
                             cudaStream_t stream) {
  if (canVectorize(img)) {
    int n = static_cast<int>(img.totalBytes() / 4);
    int blockSize = kBlockSize1D;
    int gridSize = (n + blockSize - 1) / blockSize;
    brightnessKernelVec4<<<gridSize, blockSize, 0, stream>>>(
        reinterpret_cast<const uchar4*>(input),
        reinterpret_cast<uchar4*>(output), n, offset);
  } else {
    dim3 grid, block;
    calcGridBlock2D(img.width, img.height, grid, block);
    brightnessKernelScalar<<<grid, block, 0, stream>>>(
        input, output, img.width, img.height, img.channels, offset);
  }
}

// PixelOperator implementation

void PixelOperator::invert(const CudaImage& input, CudaImage& output,
                           cudaStream_t stream) {
  validateInput(input);
  ImageUtils::ensureOutputSize(input, output);

  launchInvert(input.buffer.dataAs<unsigned char>(),
               output.buffer.dataAs<unsigned char>(), input, stream);

  CUDA_CHECK(cudaGetLastError());
}

void PixelOperator::invertInPlace(CudaImage& image, cudaStream_t stream) {
  validateInput(image);

  auto* ptr = image.buffer.dataAs<unsigned char>();
  launchInvert(ptr, ptr, image, stream);

  CUDA_CHECK(cudaGetLastError());
}

void PixelOperator::toGrayscale(const CudaImage& input, CudaImage& output,
                                cudaStream_t stream) {
  validateInput(input);
  if (input.channels < 3) {
    throw std::invalid_argument(
        "Input must have at least 3 channels for grayscale conversion");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height, 1);

  dim3 grid, block;
  calcGridBlock2D(input.width, input.height, grid, block);

  toGrayscaleKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

void PixelOperator::adjustBrightness(const CudaImage& input, CudaImage& output,
                                     int offset, cudaStream_t stream) {
  validateInput(input);
  ImageUtils::ensureOutputSize(input, output);

  launchBrightness(input.buffer.dataAs<unsigned char>(),
                   output.buffer.dataAs<unsigned char>(), input, offset,
                   stream);

  CUDA_CHECK(cudaGetLastError());
}

void PixelOperator::adjustBrightnessInPlace(CudaImage& image, int offset,
                                            cudaStream_t stream) {
  validateInput(image);

  auto* ptr = image.buffer.dataAs<unsigned char>();
  launchBrightness(ptr, ptr, image, offset, stream);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
