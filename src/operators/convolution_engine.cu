#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/convolution_engine.hpp"
#include <cmath>
#include <stdexcept>

namespace cudaimg {

namespace {
struct KernelData {
  float values[49] = {0.0f};
};
} // namespace

// ── 教学重点：Shared Memory Tiling ─────────────────────────────
//
// 为什么用 shared memory？
//   朴素卷积：每个线程从 global memory（~200 cycle）读 k² 个像素，
//   相邻线程大量重复读取同一区域（比如 3×3 卷积中，相邻线程有 6/9 的重叠）。
//
//   tiling 后：一个 block 内的线程协作加载 tile + halo 到 shared memory
//   （~20 cycle），之后每个线程从 shared memory 读取，全局内存访问大幅减少。
//
// halo 区域：
//   tile 边缘的卷积窗口需要读取相邻 tile 的数据，这些"光环"区域是
//   tiling 的关键难点。加载时需要处理越界（Zero/Mirror/Replicate）。
//
// 两个 __syncthreads() 的作用：
//   第一个：确保所有线程完成 shared memory 加载后再读取
//   第二个：确保所有线程读完当前通道后，再覆写下一个通道的数据
template <int BLOCK_SIZE>
__global__ void convolveKernelShared(const unsigned char* input,
                                     unsigned char* output, int width,
                                     int height, int channels, int kernelSize,
                                     int borderMode, KernelData kernelData) {

  // Shared memory 大小：block + halo
  const int halo = kernelSize / 2;
  const int sharedSize = BLOCK_SIZE + 2 * halo;

  extern __shared__ float sharedMem[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_SIZE + tx;
  int y = blockIdx.y * BLOCK_SIZE + ty;

  // 对每个通道处理
  for (int c = 0; c < channels; ++c) {
    // 加载数据到 shared memory（包括 halo 区域）
    for (int dy = ty; dy < sharedSize; dy += BLOCK_SIZE) {
      for (int dx = tx; dx < sharedSize; dx += BLOCK_SIZE) {
        int srcX = blockIdx.x * BLOCK_SIZE + dx - halo;
        int srcY = blockIdx.y * BLOCK_SIZE + dy - halo;

        float value = 0.0f;

        // 边界处理
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          value = input[(srcY * width + srcX) * channels + c];
        } else if (borderMode == 1) { // Mirror
          int mirrorX =
              srcX < 0 ? -srcX : (srcX >= width ? 2 * width - srcX - 2 : srcX);
          int mirrorY = srcY < 0
                            ? -srcY
                            : (srcY >= height ? 2 * height - srcY - 2 : srcY);
          mirrorX = max(0, min(mirrorX, width - 1));
          mirrorY = max(0, min(mirrorY, height - 1));
          value = input[(mirrorY * width + mirrorX) * channels + c];
        } else if (borderMode == 2) { // Replicate
          int clampX = max(0, min(srcX, width - 1));
          int clampY = max(0, min(srcY, height - 1));
          value = input[(clampY * width + clampX) * channels + c];
        }
        // borderMode == 0 (Zero) 保持 value = 0

        sharedMem[dy * sharedSize + dx] = value;
      }
    }

    __syncthreads();

    // 执行卷积
    if (x < width && y < height) {
      float sum = 0.0f;

      for (int ky = 0; ky < kernelSize; ++ky) {
        for (int kx = 0; kx < kernelSize; ++kx) {
          int sx = tx + kx;
          int sy = ty + ky;
          sum += sharedMem[sy * sharedSize + sx] *
                 kernelData.values[ky * kernelSize + kx];
        }
      }

      // 裁剪到 [0, 255]
      sum = fminf(fmaxf(sum, 0.0f), 255.0f);
      output[(y * width + x) * channels + c] =
          static_cast<unsigned char>(sum + 0.5f);
    }

    __syncthreads();
  }
}

// 简单卷积 Kernel（不使用 shared memory，用于小图像或调试）
__global__ void convolveKernelSimple(const unsigned char* input,
                                     unsigned char* output, int width,
                                     int height, int channels, int kernelSize,
                                     KernelData kernelData) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int halo = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;

    for (int ky = 0; ky < kernelSize; ++ky) {
      for (int kx = 0; kx < kernelSize; ++kx) {
        int srcX = x + kx - halo;
        int srcY = y + ky - halo;

        float value = 0.0f;
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          value = input[(srcY * width + srcX) * channels + c];
        }

        sum += value * kernelData.values[ky * kernelSize + kx];
      }
    }

    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(sum + 0.5f);
  }
}

// 可分离卷积 - 水平方向 Kernel
__global__ void separableRowKernel(const unsigned char* input,
                                   unsigned char* output, int width, int height,
                                   int channels, int kernelSize,
                                   KernelData kernelData) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;

    for (int k = 0; k < kernelSize; ++k) {
      int srcX = x + k - half;
      float value = 0.0f;
      if (srcX >= 0 && srcX < width) {
        value = input[(y * width + srcX) * channels + c];
      }
      sum += value * kernelData.values[k];
    }

    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(sum + 0.5f);
  }
}

// 可分离卷积 - 垂直方向 Kernel
__global__ void separableColKernel(const unsigned char* input,
                                   unsigned char* output, int width, int height,
                                   int channels, int kernelSize,
                                   KernelData kernelData) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = kernelSize / 2;

  for (int c = 0; c < channels; ++c) {
    float sum = 0.0f;

    for (int k = 0; k < kernelSize; ++k) {
      int srcY = y + k - half;
      float value = 0.0f;
      if (srcY >= 0 && srcY < height) {
        value = input[(srcY * width + x) * channels + c];
      }
      sum += value * kernelData.values[k];
    }

    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    output[(y * width + x) * channels + c] =
        static_cast<unsigned char>(sum + 0.5f);
  }
}

// Sobel 边缘检测 Kernel
__global__ void sobelKernel(const unsigned char* input, unsigned char* output,
                            int width, int height, int channels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Sobel 核
  const float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  // 对于多通道图像，先转换为灰度
  float grayValues[9];

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int srcX = x + kx;
      int srcY = y + ky;

      float gray = 0.0f;
      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        if (channels >= 3) {
          int idx = (srcY * width + srcX) * channels;
          gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                 0.114f * input[idx + 2];
        } else {
          gray = input[(srcY * width + srcX) * channels];
        }
      }
      grayValues[(ky + 1) * 3 + (kx + 1)] = gray;
    }
  }

  // 计算梯度
  float gx = 0.0f, gy = 0.0f;
  for (int i = 0; i < 9; ++i) {
    gx += grayValues[i] * sobelX[i];
    gy += grayValues[i] * sobelY[i];
  }

  // 计算梯度幅值
  float magnitude = sqrtf(gx * gx + gy * gy);
  magnitude = fminf(magnitude, 255.0f);

  // 输出为单通道
  output[y * width + x] = static_cast<unsigned char>(magnitude + 0.5f);
}

// ConvolutionEngine 实现
void ConvolutionEngine::convolve(const CudaImage& input, CudaImage& output,
                                 const float* kernel, int kernelSize,
                                 BorderMode borderMode, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize > 7 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1 and 7");
  }
  if (kernel == nullptr) {
    throw std::invalid_argument("Kernel is null");
  }

  // 确保输出图像大小正确
  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  KernelData kernelData;
  for (int i = 0; i < kernelSize * kernelSize; ++i) {
    kernelData.values[i] = kernel[i];
  }

  constexpr int kBlockSize = 16;
  dim3 block(kBlockSize, kBlockSize);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  int borderModeInt = static_cast<int>(borderMode);
  int halo = kernelSize / 2;
  int sharedSize = kBlockSize + 2 * halo;
  size_t sharedBytes =
      static_cast<size_t>(sharedSize) * sharedSize * sizeof(float);

  // 使用支持边界模式的 shared memory 版本
  convolveKernelShared<kBlockSize><<<grid, block, sharedBytes, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, kernelSize, borderModeInt, kernelData);

  CUDA_CHECK(cudaGetLastError());
}

void ConvolutionEngine::gaussianBlur(const CudaImage& input, CudaImage& output,
                                     int kernelSize, float sigma,
                                     cudaStream_t stream) {
  if (kernelSize < 1 || kernelSize > 7 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1 and 7");
  }
  if (sigma <= 0.0f) {
    throw std::invalid_argument("Sigma must be positive");
  }

  std::vector<float> kernel = generateGaussianKernel(kernelSize, sigma);
  convolve(input, output, kernel.data(), kernelSize, BorderMode::Zero, stream);
}

void ConvolutionEngine::sobelEdgeDetection(const CudaImage& input,
                                           CudaImage& output,
                                           cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  // 输出为单通道
  ImageUtils::ensureOutputSize(output, input.width, input.height, 1);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  sobelKernel<<<grid, block, 0, stream>>>(input.buffer.dataAs<unsigned char>(),
                                          output.buffer.dataAs<unsigned char>(),
                                          input.width, input.height,
                                          input.channels);

  CUDA_CHECK(cudaGetLastError());
}

std::vector<float> ConvolutionEngine::generateGaussianKernel(int size,
                                                             float sigma) {
  if (size < 1 || size % 2 == 0) {
    throw std::invalid_argument("Kernel size must be positive and odd");
  }
  if (sigma <= 0.0f) {
    throw std::invalid_argument("Sigma must be positive");
  }

  std::vector<float> kernel(size * size);
  int half = size / 2;
  float sum = 0.0f;

  for (int y = -half; y <= half; ++y) {
    for (int x = -half; x <= half; ++x) {
      float value = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
      kernel[(y + half) * size + (x + half)] = value;
      sum += value;
    }
  }

  // 归一化
  for (float& v : kernel) {
    v /= sum;
  }

  return kernel;
}

std::vector<float> ConvolutionEngine::generateGaussianKernel1D(int size,
                                                               float sigma) {
  if (size < 1 || size % 2 == 0) {
    throw std::invalid_argument("Kernel size must be positive and odd");
  }
  if (sigma <= 0.0f) {
    throw std::invalid_argument("Sigma must be positive");
  }

  std::vector<float> kernel(size);
  int half = size / 2;
  float sum = 0.0f;

  for (int x = -half; x <= half; ++x) {
    float value = std::exp(-(x * x) / (2.0f * sigma * sigma));
    kernel[x + half] = value;
    sum += value;
  }

  // 归一化
  for (float& v : kernel) {
    v /= sum;
  }

  return kernel;
}

void ConvolutionEngine::separableConvolve(const CudaImage& input,
                                          CudaImage& output,
                                          const float* rowKernel,
                                          const float* colKernel,
                                          int kernelSize, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (kernelSize < 1 || kernelSize > 7 || kernelSize % 2 == 0) {
    throw std::invalid_argument("Kernel size must be odd and between 1 and 7");
  }
  if (rowKernel == nullptr || colKernel == nullptr) {
    throw std::invalid_argument("Kernel is null");
  }

  // 第一步：水平方向卷积（使用 rowKernel）
  CudaImage temp =
      ImageUtils::createCudaImage(input.width, input.height, input.channels);

  KernelData rowKernelData;
  for (int i = 0; i < kernelSize; ++i) {
    rowKernelData.values[i] = rowKernel[i];
  }

  constexpr int kBlockSize = 16;
  dim3 block(kBlockSize, kBlockSize);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  separableRowKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(), temp.buffer.dataAs<unsigned char>(),
      input.width, input.height, input.channels, kernelSize, rowKernelData);
  CUDA_CHECK(cudaGetLastError());

  // 第二步：垂直方向卷积（使用 colKernel）
  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  KernelData colKernelData;
  for (int i = 0; i < kernelSize; ++i) {
    colKernelData.values[i] = colKernel[i];
  }

  separableColKernel<<<grid, block, 0, stream>>>(
      temp.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, kernelSize, colKernelData);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
