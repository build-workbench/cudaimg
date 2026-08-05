#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/operators/filters.hpp"
#include <cmath>
#include <stdexcept>

namespace gpu_image {

// 中值滤波 Kernel（使用排序网络）
__global__ void medianFilterKernel(const unsigned char* input,
                                   unsigned char* output, int width, int height,
                                   int channels, int kernelSize) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int radius = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    // 收集窗口内的值（最大支持 7x7 = 49 个元素）
    unsigned char window[49];
    int count = 0;

    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        int nx = min(max(x + kx, 0), width - 1);
        int ny = min(max(y + ky, 0), height - 1);
        window[count++] = input[(ny * width + nx) * channels + c];
      }
    }

    // 冒泡排序找中值
    for (int i = 0; i <= count / 2; ++i) {
      for (int j = i + 1; j < count; ++j) {
        if (window[j] < window[i]) {
          unsigned char temp = window[i];
          window[i] = window[j];
          window[j] = temp;
        }
      }
    }

    output[(y * width + x) * channels + c] = window[count / 2];
  }
}

// 双边滤波 Kernel
__global__ void bilateralFilterKernel(const unsigned char* input,
                                      unsigned char* output, int width,
                                      int height, int channels, int kernelSize,
                                      float sigmaSpace, float sigmaColor) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int radius = kernelSize / 2;
  float spaceCoeff = -0.5f / (sigmaSpace * sigmaSpace);
  float colorCoeff = -0.5f / (sigmaColor * sigmaColor);

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;
    float weightSum = 0.0f;
    float centerValue = input[(y * width + x) * channels + c];

    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        int nx = min(max(x + kx, 0), width - 1);
        int ny = min(max(y + ky, 0), height - 1);

        float neighborValue = input[(ny * width + nx) * channels + c];

        // 空间权重
        float spatialDist = kx * kx + ky * ky;
        float spatialWeight = expf(spatialDist * spaceCoeff);

        // 颜色权重
        float colorDist =
            (neighborValue - centerValue) * (neighborValue - centerValue);
        float colorWeight = expf(colorDist * colorCoeff);

        float weight = spatialWeight * colorWeight;
        sum += neighborValue * weight;
        weightSum += weight;
      }
    }

    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(fminf(fmaxf(sum / weightSum, 0.0f), 255.0f));
  }
}

// 盒式滤波 Kernel
__global__ void boxFilterKernel(const unsigned char* input,
                                unsigned char* output, int width, int height,
                                int channels, int kernelSize) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int radius = kernelSize / 2;
  float invArea = 1.0f / (kernelSize * kernelSize);

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
      for (int kx = -radius; kx <= radius; ++kx) {
        int nx = min(max(x + kx, 0), width - 1);
        int ny = min(max(y + ky, 0), height - 1);
        sum += input[(ny * width + nx) * channels + c];
      }
    }

    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(sum * invArea + 0.5f);
  }
}

// 锐化 Kernel
__global__ void sharpenKernel(const unsigned char* input, unsigned char* output,
                              int width, int height, int channels,
                              float strength) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // 锐化核: center = 1 + 4*strength, neighbors = -strength
  for (int c = 0; c < channels; ++c) {
    float center = input[(y * width + x) * channels + c];
    float sum = center * (1.0f + 4.0f * strength);

    if (x > 0)
      sum -= strength * input[(y * width + x - 1) * channels + c];
    if (x < width - 1)
      sum -= strength * input[(y * width + x + 1) * channels + c];
    if (y > 0)
      sum -= strength * input[((y - 1) * width + x) * channels + c];
    if (y < height - 1)
      sum -= strength * input[((y + 1) * width + x) * channels + c];

    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
  }
}

// 拉普拉斯 Kernel
__global__ void laplacianKernel(const unsigned char* input,
                                unsigned char* output, int width, int height,
                                int channels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // 拉普拉斯核: [0,-1,0; -1,4,-1; 0,-1,0]
  for (int c = 0; c < channels; ++c) {
    float sum = 4.0f * input[(y * width + x) * channels + c];

    if (x > 0)
      sum -= input[(y * width + x - 1) * channels + c];
    if (x < width - 1)
      sum -= input[(y * width + x + 1) * channels + c];
    if (y > 0)
      sum -= input[((y - 1) * width + x) * channels + c];
    if (y < height - 1)
      sum -= input[((y + 1) * width + x) * channels + c];

    // 取绝对值并缩放
    sum = fabsf(sum);
    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(fminf(sum, 255.0f));
  }
}

// 图像加法 Kernel
__global__ void addKernel(const unsigned char* src1, const unsigned char* src2,
                          unsigned char* output, int size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int sum = src1[idx] + src2[idx];
  output[idx] = static_cast<unsigned char>(min(sum, 255));
}

// 图像减法 Kernel
__global__ void subtractKernel(const unsigned char* src1,
                               const unsigned char* src2, unsigned char* output,
                               int size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int diff = src1[idx] - src2[idx];
  output[idx] = static_cast<unsigned char>(max(diff, 0));
}

// 图像乘法 Kernel
__global__ void multiplyKernel(const unsigned char* src1,
                               const unsigned char* src2, unsigned char* output,
                               int size, float scale) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float result = (src1[idx] * src2[idx] / 255.0f) * scale;
  output[idx] = static_cast<unsigned char>(fminf(fmaxf(result, 0.0f), 255.0f));
}

// 图像混合 Kernel
__global__ void blendKernel(const unsigned char* src1,
                            const unsigned char* src2, unsigned char* output,
                            int size, float alpha) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float result = alpha * src1[idx] + (1.0f - alpha) * src2[idx];
  output[idx] = static_cast<unsigned char>(fminf(fmaxf(result, 0.0f), 255.0f));
}

// 加权和 Kernel
__global__ void addWeightedKernel(const unsigned char* src1,
                                  const unsigned char* src2,
                                  unsigned char* output, int size, float alpha,
                                  float beta, float gamma) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float result = alpha * src1[idx] + beta * src2[idx] + gamma;
  output[idx] = static_cast<unsigned char>(fminf(fmaxf(result, 0.0f), 255.0f));
}

// 绝对差 Kernel
__global__ void absDiffKernel(const unsigned char* src1,
                              const unsigned char* src2, unsigned char* output,
                              int size) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  output[idx] = static_cast<unsigned char>(abs(src1[idx] - src2[idx]));
}

// 标量加法 Kernel
__global__ void addScalarKernel(const unsigned char* input,
                                unsigned char* output, int size, int value) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int result = input[idx] + value;
  output[idx] = static_cast<unsigned char>(min(max(result, 0), 255));
}

// 标量乘法 Kernel
__global__ void multiplyScalarKernel(const unsigned char* input,
                                     unsigned char* output, int size,
                                     float scale) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  float result = input[idx] * scale;
  output[idx] = static_cast<unsigned char>(fminf(fmaxf(result, 0.0f), 255.0f));
}

// Filters 实现
void Filters::medianFilter(const GpuImage& input, GpuImage& output,
                           int kernelSize, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize > 7 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1-7");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  medianFilterKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, kernelSize);

  CUDA_CHECK(cudaGetLastError());
}

void Filters::bilateralFilter(const GpuImage& input, GpuImage& output,
                              int kernelSize, float sigmaSpace,
                              float sigmaColor, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize > 31 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1-31");
  }
  if (sigmaSpace <= 0.0f) {
    throw std::invalid_argument("sigmaSpace must be positive");
  }
  if (sigmaColor <= 0.0f) {
    throw std::invalid_argument("sigmaColor must be positive");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  bilateralFilterKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, kernelSize, sigmaSpace, sigmaColor);

  CUDA_CHECK(cudaGetLastError());
}

void Filters::boxFilter(const GpuImage& input, GpuImage& output, int kernelSize,
                        cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize > 31 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1-31");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  boxFilterKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, kernelSize);

  CUDA_CHECK(cudaGetLastError());
}

void Filters::sharpen(const GpuImage& input, GpuImage& output, float strength,
                      cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (strength <= 0.0f) {
    throw std::invalid_argument("Strength must be positive");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  sharpenKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, strength);

  CUDA_CHECK(cudaGetLastError());
}

void Filters::laplacian(const GpuImage& input, GpuImage& output,
                        cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  laplacianKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

// ImageArithmetic 实现
void ImageArithmetic::add(const GpuImage& src1, const GpuImage& src2,
                          GpuImage& output, cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  addKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::subtract(const GpuImage& src1, const GpuImage& src2,
                               GpuImage& output, cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  subtractKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::multiply(const GpuImage& src1, const GpuImage& src2,
                               GpuImage& output, float scale,
                               cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  multiplyKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size, scale);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::blend(const GpuImage& src1, const GpuImage& src2,
                            GpuImage& output, float alpha,
                            cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }
  if (alpha < 0.0f || alpha > 1.0f) {
    throw std::invalid_argument("Alpha must be in range [0, 1]");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  blendKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size, alpha);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::addWeighted(const GpuImage& src1, float alpha,
                                  const GpuImage& src2, float beta,
                                  GpuImage& output, float gamma,
                                  cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  addWeightedKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size, alpha, beta, gamma);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::absDiff(const GpuImage& src1, const GpuImage& src2,
                              GpuImage& output, cudaStream_t stream) {
  if (!src1.isValid() || !src2.isValid()) {
    throw std::invalid_argument("Invalid input images");
  }
  if (src1.width != src2.width || src1.height != src2.height ||
      src1.channels != src2.channels) {
    throw std::invalid_argument("Image dimensions must match");
  }

  ImageUtils::ensureOutputSize(output, src1.width, src1.height, src1.channels);

  size_t totalSize =
      static_cast<size_t>(src1.width) * src1.height * src1.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  absDiffKernel<<<gridSize, blockSize, 0, stream>>>(
      src1.buffer.dataAs<unsigned char>(), src2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::addScalar(const GpuImage& input, GpuImage& output,
                                unsigned char value, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  size_t totalSize =
      static_cast<size_t>(input.width) * input.height * input.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  addScalarKernel<<<gridSize, blockSize, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size, value);

  CUDA_CHECK(cudaGetLastError());
}

void ImageArithmetic::multiplyScalar(const GpuImage& input, GpuImage& output,
                                     float scale, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  size_t totalSize =
      static_cast<size_t>(input.width) * input.height * input.channels;
  if (totalSize > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Image size too large for processing");
  }
  int size = static_cast<int>(totalSize);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  multiplyScalarKernel<<<gridSize, blockSize, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), size, scale);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu_image
