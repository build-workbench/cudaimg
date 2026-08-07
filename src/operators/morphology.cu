#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/morphology.hpp"
#include <algorithm>
#include <stdexcept>

namespace cudaimg {

// 生成结构元素掩码
__device__ bool isInStructuringElement(int dx, int dy, int halfSize,
                                       int elementType) {
  switch (elementType) {
  case 0: // Rectangle
    return true;
  case 1: // Cross
    return dx == 0 || dy == 0;
  case 2: // Ellipse
  {
    float rx = halfSize + 0.5f;
    float ry = halfSize + 0.5f;
    return (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0f;
  }
  default:
    return true;
  }
}

// 腐蚀 Kernel
__global__ void erodeKernel(const unsigned char* input, unsigned char* output,
                            int width, int height, int channels, int kernelSize,
                            int elementType) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    unsigned char minVal = 255;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        if (!isInStructuringElement(kx, ky, half, elementType))
          continue;

        int srcX = x + kx;
        int srcY = y + ky;

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          unsigned char val = input[(srcY * width + srcX) * channels + c];
          minVal = min(minVal, val);
        }
      }
    }

    output[(y * width + x) * channels + c] = minVal;
  }
}

// 膨胀 Kernel
__global__ void dilateKernel(const unsigned char* input, unsigned char* output,
                             int width, int height, int channels,
                             int kernelSize, int elementType) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    unsigned char maxVal = 0;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        if (!isInStructuringElement(kx, ky, half, elementType))
          continue;

        int srcX = x + kx;
        int srcY = y + ky;

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          unsigned char val = input[(srcY * width + srcX) * channels + c];
          maxVal = max(maxVal, val);
        }
      }
    }

    output[(y * width + x) * channels + c] = maxVal;
  }
}

// 图像相减 Kernel
__global__ void subtractKernel(const unsigned char* a, const unsigned char* b,
                               unsigned char* output, int width, int height,
                               int channels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  for (int c = 0; c < channels; ++c) {
    int idx = (y * width + x) * channels + c;
    int diff = static_cast<int>(a[idx]) - static_cast<int>(b[idx]);
    output[idx] = static_cast<unsigned char>(max(0, min(255, diff)));
  }
}

// 辅助函数
static void
launchMorphKernel(void (*kernel)(const unsigned char*, unsigned char*, int, int,
                                 int, int, int),
                  const CudaImage& input, CudaImage& output, int kernelSize,
                  StructuringElement element, cudaStream_t stream) {

  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be positive and odd");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  int elementType = static_cast<int>(element);

  kernel<<<grid, block, 0, stream>>>(input.buffer.dataAs<unsigned char>(),
                                     output.buffer.dataAs<unsigned char>(),
                                     input.width, input.height, input.channels,
                                     kernelSize, elementType);

  CUDA_CHECK(cudaGetLastError());
}

// Morphology 实现
void Morphology::erode(const CudaImage& input, CudaImage& output, int kernelSize,
                       StructuringElement element, cudaStream_t stream) {
  launchMorphKernel(erodeKernel, input, output, kernelSize, element, stream);
}

void Morphology::dilate(const CudaImage& input, CudaImage& output, int kernelSize,
                        StructuringElement element, cudaStream_t stream) {
  launchMorphKernel(dilateKernel, input, output, kernelSize, element, stream);
}

void Morphology::open(const CudaImage& input, CudaImage& output, int kernelSize,
                      StructuringElement element, cudaStream_t stream) {
  CudaImage temp;
  erode(input, temp, kernelSize, element, stream);
  dilate(temp, output, kernelSize, element, stream);
}

void Morphology::close(const CudaImage& input, CudaImage& output, int kernelSize,
                       StructuringElement element, cudaStream_t stream) {
  CudaImage temp;
  dilate(input, temp, kernelSize, element, stream);
  erode(temp, output, kernelSize, element, stream);
}

void Morphology::gradient(const CudaImage& input, CudaImage& output,
                          int kernelSize, StructuringElement element,
                          cudaStream_t stream) {
  CudaImage dilated, eroded;
  dilate(input, dilated, kernelSize, element, stream);
  erode(input, eroded, kernelSize, element, stream);

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  subtractKernel<<<grid, block, 0, stream>>>(
      dilated.buffer.dataAs<unsigned char>(),
      eroded.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

void Morphology::topHat(const CudaImage& input, CudaImage& output, int kernelSize,
                        StructuringElement element, cudaStream_t stream) {
  CudaImage opened;
  open(input, opened, kernelSize, element, stream);

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  subtractKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      opened.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

void Morphology::blackHat(const CudaImage& input, CudaImage& output,
                          int kernelSize, StructuringElement element,
                          cudaStream_t stream) {
  CudaImage closed;
  close(input, closed, kernelSize, element, stream);

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  subtractKernel<<<grid, block, 0, stream>>>(
      closed.buffer.dataAs<unsigned char>(),
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
