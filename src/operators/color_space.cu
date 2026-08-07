#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/color_space.hpp"
#include <cmath>
#include <stdexcept>

namespace cudaimg {

// RGB to HSV Kernel
__global__ void rgbToHsvKernel(const unsigned char* input,
                               unsigned char* output, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = (y * width + x) * 3;

  float r = input[idx] / 255.0f;
  float g = input[idx + 1] / 255.0f;
  float b = input[idx + 2] / 255.0f;

  float maxVal = fmaxf(r, fmaxf(g, b));
  float minVal = fminf(r, fminf(g, b));
  float delta = maxVal - minVal;

  // Value
  float v = maxVal;

  // Saturation
  float s = (maxVal > 0) ? (delta / maxVal) : 0;

  // Hue
  float h = 0;
  if (delta > 0) {
    if (maxVal == r) {
      h = 60.0f * fmodf((g - b) / delta + 6.0f, 6.0f);
    } else if (maxVal == g) {
      h = 60.0f * ((b - r) / delta + 2.0f);
    } else {
      h = 60.0f * ((r - g) / delta + 4.0f);
    }
  }

  // 归一化到 0-255，使用四舍五入减小往返误差
  output[idx] = static_cast<unsigned char>(
      fminf(fmaxf(h * 255.0f / 360.0f + 0.5f, 0.0f), 255.0f));
  output[idx + 1] =
      static_cast<unsigned char>(fminf(fmaxf(s * 255.0f + 0.5f, 0.0f), 255.0f));
  output[idx + 2] =
      static_cast<unsigned char>(fminf(fmaxf(v * 255.0f + 0.5f, 0.0f), 255.0f));
}

// HSV to RGB Kernel
__global__ void hsvToRgbKernel(const unsigned char* input,
                               unsigned char* output, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = (y * width + x) * 3;

  float h = input[idx] * 360.0f / 255.0f;
  float s = input[idx + 1] / 255.0f;
  float v = input[idx + 2] / 255.0f;

  float c = v * s;
  float x_val = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
  float m = v - c;

  float r, g, b;

  if (h < 60) {
    r = c;
    g = x_val;
    b = 0;
  } else if (h < 120) {
    r = x_val;
    g = c;
    b = 0;
  } else if (h < 180) {
    r = 0;
    g = c;
    b = x_val;
  } else if (h < 240) {
    r = 0;
    g = x_val;
    b = c;
  } else if (h < 300) {
    r = x_val;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x_val;
  }

  output[idx] = static_cast<unsigned char>(
      fminf(fmaxf((r + m) * 255.0f + 0.5f, 0.0f), 255.0f));
  output[idx + 1] = static_cast<unsigned char>(
      fminf(fmaxf((g + m) * 255.0f + 0.5f, 0.0f), 255.0f));
  output[idx + 2] = static_cast<unsigned char>(
      fminf(fmaxf((b + m) * 255.0f + 0.5f, 0.0f), 255.0f));
}

// RGB to YUV Kernel
__global__ void rgbToYuvKernel(const unsigned char* input,
                               unsigned char* output, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = (y * width + x) * 3;

  float r = input[idx];
  float g = input[idx + 1];
  float b = input[idx + 2];

  // BT.601 标准
  float Y = 0.299f * r + 0.587f * g + 0.114f * b;
  float U = -0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f;
  float V = 0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f;

  output[idx] =
      static_cast<unsigned char>(fminf(fmaxf(Y + 0.5f, 0.0f), 255.0f));
  output[idx + 1] =
      static_cast<unsigned char>(fminf(fmaxf(U + 0.5f, 0.0f), 255.0f));
  output[idx + 2] =
      static_cast<unsigned char>(fminf(fmaxf(V + 0.5f, 0.0f), 255.0f));
}

// YUV to RGB Kernel
__global__ void yuvToRgbKernel(const unsigned char* input,
                               unsigned char* output, int width, int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = (y * width + x) * 3;

  float Y = input[idx];
  float U = input[idx + 1] - 128.0f;
  float V = input[idx + 2] - 128.0f;

  float r = Y + 1.402f * V;
  float g = Y - 0.344136f * U - 0.714136f * V;
  float b = Y + 1.772f * U;

  output[idx] =
      static_cast<unsigned char>(fminf(fmaxf(r + 0.5f, 0.0f), 255.0f));
  output[idx + 1] =
      static_cast<unsigned char>(fminf(fmaxf(g + 0.5f, 0.0f), 255.0f));
  output[idx + 2] =
      static_cast<unsigned char>(fminf(fmaxf(b + 0.5f, 0.0f), 255.0f));
}


// 通道分离 Kernel
__global__ void splitChannelsKernel(const unsigned char* input,
                                    unsigned char* ch0, unsigned char* ch1,
                                    unsigned char* ch2, int width, int height,
                                    int channels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int srcIdx = (y * width + x) * channels;
  int dstIdx = y * width + x;

  ch0[dstIdx] = input[srcIdx];
  if (channels > 1)
    ch1[dstIdx] = input[srcIdx + 1];
  if (channels > 2)
    ch2[dstIdx] = input[srcIdx + 2];
}

// 通道合并 Kernel
__global__ void mergeChannelsKernel(const unsigned char* ch0,
                                    const unsigned char* ch1,
                                    const unsigned char* ch2,
                                    unsigned char* output, int width,
                                    int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int srcIdx = y * width + x;
  int dstIdx = (y * width + x) * 3;

  output[dstIdx] = ch0[srcIdx];
  output[dstIdx + 1] = ch1[srcIdx];
  output[dstIdx + 2] = ch2[srcIdx];
}

// ColorSpace 实现
void ColorSpace::rgbToHsv(const CudaImage& input, CudaImage& output,
                          cudaStream_t stream) {
  if (!input.isValid() || input.channels != 3) {
    throw std::invalid_argument("Input must be a valid 3-channel image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height, 3);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  rgbToHsvKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height);

  CUDA_CHECK(cudaGetLastError());
}

void ColorSpace::hsvToRgb(const CudaImage& input, CudaImage& output,
                          cudaStream_t stream) {
  if (!input.isValid() || input.channels != 3) {
    throw std::invalid_argument("Input must be a valid 3-channel image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height, 3);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  hsvToRgbKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height);

  CUDA_CHECK(cudaGetLastError());
}

void ColorSpace::rgbToYuv(const CudaImage& input, CudaImage& output,
                          cudaStream_t stream) {
  if (!input.isValid() || input.channels != 3) {
    throw std::invalid_argument("Input must be a valid 3-channel image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height, 3);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  rgbToYuvKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height);

  CUDA_CHECK(cudaGetLastError());
}

void ColorSpace::yuvToRgb(const CudaImage& input, CudaImage& output,
                          cudaStream_t stream) {
  if (!input.isValid() || input.channels != 3) {
    throw std::invalid_argument("Input must be a valid 3-channel image");
  }

  ImageUtils::ensureOutputSize(output, input.width, input.height, 3);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  yuvToRgbKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), input.width, input.height);

  CUDA_CHECK(cudaGetLastError());
}


void ColorSpace::splitChannels(const CudaImage& input, CudaImage& channel0,
                               CudaImage& channel1, CudaImage& channel2,
                               cudaStream_t stream) {
  if (!input.isValid() || input.channels < 3) {
    throw std::invalid_argument(
        "Input must be a valid image with at least 3 channels");
  }

  // 创建单通道输出
  ImageUtils::ensureOutputSize(channel0, input.width, input.height, 1);
  ImageUtils::ensureOutputSize(channel1, input.width, input.height, 1);
  ImageUtils::ensureOutputSize(channel2, input.width, input.height, 1);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  splitChannelsKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      channel0.buffer.dataAs<unsigned char>(),
      channel1.buffer.dataAs<unsigned char>(),
      channel2.buffer.dataAs<unsigned char>(), input.width, input.height,
      input.channels);

  CUDA_CHECK(cudaGetLastError());
}

void ColorSpace::mergeChannels(const CudaImage& channel0,
                               const CudaImage& channel1,
                               const CudaImage& channel2, CudaImage& output,
                               cudaStream_t stream) {
  if (!channel0.isValid() || !channel1.isValid() || !channel2.isValid()) {
    throw std::invalid_argument("All channels must be valid");
  }
  if (channel0.channels != 1 || channel1.channels != 1 ||
      channel2.channels != 1) {
    throw std::invalid_argument("All inputs must be single-channel images");
  }
  if (channel0.width != channel1.width || channel0.width != channel2.width ||
      channel0.height != channel1.height ||
      channel0.height != channel2.height) {
    throw std::invalid_argument("All channels must have the same dimensions");
  }

  ImageUtils::ensureOutputSize(output, channel0.width, channel0.height, 3);

  dim3 block(16, 16);
  dim3 grid((channel0.width + block.x - 1) / block.x,
            (channel0.height + block.y - 1) / block.y);

  mergeChannelsKernel<<<grid, block, 0, stream>>>(
      channel0.buffer.dataAs<unsigned char>(),
      channel1.buffer.dataAs<unsigned char>(),
      channel2.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), channel0.width, channel0.height);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
