#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/operators/histogram_calculator.hpp"
#include "gpu_image/operators/threshold.hpp"
#include <stdexcept>

namespace gpu_image {

// 全局阈值 Kernel
__global__ void thresholdKernel(const unsigned char* input,
                                unsigned char* output, int width, int height,
                                int channels, unsigned char thresh,
                                unsigned char maxVal, int type) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  for (int c = 0; c < channels; ++c) {
    int idx = (y * width + x) * channels + c;
    unsigned char src = input[idx];
    unsigned char dst;

    switch (type) {
    case 0: // Binary
      dst = (src > thresh) ? maxVal : 0;
      break;
    case 1: // BinaryInv
      dst = (src > thresh) ? 0 : maxVal;
      break;
    case 2: // Truncate
      dst = (src > thresh) ? thresh : src;
      break;
    case 3: // ToZero
      dst = (src > thresh) ? src : 0;
      break;
    case 4: // ToZeroInv
      dst = (src > thresh) ? 0 : src;
      break;
    default:
      dst = src;
    }

    output[idx] = dst;
  }
}

// 自适应阈值 Kernel (均值)
__global__ void adaptiveThresholdMeanKernel(const unsigned char* input,
                                            unsigned char* output, int width,
                                            int height, int channels,
                                            unsigned char maxVal, int type,
                                            int blockSize, int C) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = blockSize / 2;

  for (int c = 0; c < channels; ++c) {
    // 计算局部均值
    float sum = 0.0f;
    int count = 0;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        int srcX = x + kx;
        int srcY = y + ky;

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          sum += input[(srcY * width + srcX) * channels + c];
          count++;
        }
      }
    }

    float localThresh = (count > 0) ? (sum / count - C) : 0;

    int idx = (y * width + x) * channels + c;
    unsigned char src = input[idx];
    unsigned char dst;

    if (type == 0) { // Binary
      dst = (src > localThresh) ? maxVal : 0;
    } else { // BinaryInv
      dst = (src > localThresh) ? 0 : maxVal;
    }

    output[idx] = dst;
  }
}

// 自适应阈值 Kernel (高斯)
__global__ void
adaptiveThresholdGaussianKernel(const unsigned char* input,
                                unsigned char* output, int width, int height,
                                int channels, unsigned char maxVal, int type,
                                int blockSize, int C, float sigma) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int half = blockSize / 2;

  for (int c = 0; c < channels; ++c) {
    // 计算高斯加权均值
    float sum = 0.0f;
    float weightSum = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        int srcX = x + kx;
        int srcY = y + ky;

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
          float weight = expf(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
          sum += input[(srcY * width + srcX) * channels + c] * weight;
          weightSum += weight;
        }
      }
    }

    float localThresh = (weightSum > 0) ? (sum / weightSum - C) : 0;

    int idx = (y * width + x) * channels + c;
    unsigned char src = input[idx];
    unsigned char dst;

    if (type == 0) { // Binary
      dst = (src > localThresh) ? maxVal : 0;
    } else { // BinaryInv
      dst = (src > localThresh) ? 0 : maxVal;
    }

    output[idx] = dst;
  }
}

// Threshold 实现
void Threshold::threshold(const GpuImage& input, GpuImage& output,
                          unsigned char thresh, unsigned char maxVal,
                          ThresholdType type, cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  if (output.width != input.width || output.height != input.height ||
      output.channels != input.channels) {
    output =
        ImageUtils::createGpuImage(input.width, input.height, input.channels);
  }

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  thresholdKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels, thresh, maxVal, static_cast<int>(type));

  CUDA_CHECK(cudaGetLastError());
}

void Threshold::adaptiveThreshold(const GpuImage& input, GpuImage& output,
                                  unsigned char maxVal, AdaptiveMethod method,
                                  ThresholdType type, int blockSize, int C,
                                  cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (blockSize < 3 || blockSize % 2 == 0) {
    throw std::invalid_argument("Block size must be odd and >= 3");
  }
  if (type != ThresholdType::Binary && type != ThresholdType::BinaryInv) {
    throw std::invalid_argument(
        "Adaptive threshold only supports Binary and BinaryInv types");
  }

  if (output.width != input.width || output.height != input.height ||
      output.channels != input.channels) {
    output =
        ImageUtils::createGpuImage(input.width, input.height, input.channels);
  }

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  int typeInt = (type == ThresholdType::Binary) ? 0 : 1;

  if (method == AdaptiveMethod::MeanC) {
    adaptiveThresholdMeanKernel<<<grid, block, 0, stream>>>(
        input.buffer.dataAs<unsigned char>(),
        output.buffer.dataAs<unsigned char>(), input.width, input.height,
        input.channels, maxVal, typeInt, blockSize, C);
  } else {
    float sigma = blockSize / 6.0f;
    adaptiveThresholdGaussianKernel<<<grid, block, 0, stream>>>(
        input.buffer.dataAs<unsigned char>(),
        output.buffer.dataAs<unsigned char>(), input.width, input.height,
        input.channels, maxVal, typeInt, blockSize, C, sigma);
  }

  CUDA_CHECK(cudaGetLastError());
}

unsigned char Threshold::otsuThreshold(const GpuImage& input,
                                       cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  // 计算直方图
  auto histogram = HistogramCalculator::calculate(input, stream);

  // 使用 double 累加：int 运算会在大图（约 >8MP、直方图偏斜）下溢出，
  // 导致 Otsu 阈值静默算错（i * histogram[i] 可超过 2^31-1）。
  const double totalPixels =
      static_cast<double>(input.width) * static_cast<double>(input.height);

  // Otsu 算法
  double sum = 0;
  for (int i = 0; i < 256; ++i) {
    sum += static_cast<double>(i) * histogram[i];
  }

  double sumB = 0;
  double wB = 0;
  double wF = 0;

  double maxVariance = 0;
  unsigned char threshold = 0;

  for (int t = 0; t < 256; ++t) {
    wB += histogram[t];
    if (wB == 0)
      continue;

    wF = totalPixels - wB;
    if (wF == 0)
      break;

    sumB += static_cast<double>(t) * histogram[t];

    double mB = sumB / wB;
    double mF = (sum - sumB) / wF;

    double variance = wB * wF * (mB - mF) * (mB - mF);

    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = static_cast<unsigned char>(t);
    }
  }

  return threshold;
}

void Threshold::otsuBinarize(const GpuImage& input, GpuImage& output,
                             unsigned char maxVal, cudaStream_t stream) {
  unsigned char thresh = otsuThreshold(input, stream);
  threshold(input, output, thresh, maxVal, ThresholdType::Binary, stream);
}

} // namespace gpu_image
