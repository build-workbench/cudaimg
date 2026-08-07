#include "cudaimg/core/cuda_error.hpp"
#include "cudaimg/core/image_utils.hpp"
#include "cudaimg/operators/histogram_calculator.hpp"
#include <stdexcept>

namespace cudaimg {

// ── 教学重点：两级原子规约 ─────────────────────────────────────
//
// 为什么不能直接对全局 histogram[256] 做 atomicAdd？
//   100 万个线程争抢 256 个 bin，竞争极其严重，性能灾难。
//
// 两级规约策略：
//   1. 每个 block 在 shared memory 上建局部直方图（localHist[256]）
//      block 内线程数远少于全局，竞争范围小
//   2. 每个线程对自己的像素做 atomicAdd 到 localHist（block 级原子）
//   3. block 级直方图完成后，合并到全局直方图（仅 gridSize 个 block
//      做合并，竞争大幅减少）
//
// 关键同步点：
//   __syncthreads() #1：确保 localHist 初始化完成
//   __syncthreads() #2：确保所有线程完成 atomicAdd 后再合并
__global__ void histogramKernelShared(const unsigned char* input,
                                      int* histogram, int width, int height,
                                      int channels, int targetChannel) {

  // 每个 block 的局部直方图
  __shared__ int localHist[256];

  // 初始化局部直方图
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;

  for (int i = tid; i < 256; i += blockSize) {
    localHist[i] = 0;
  }
  __syncthreads();

  // 计算像素位置
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    unsigned char value;

    if (targetChannel < 0) {
      // 灰度转换
      if (channels >= 3) {
        int idx = (y * width + x) * channels;
        float gray = 0.299f * input[idx] + 0.587f * input[idx + 1] +
                     0.114f * input[idx + 2];
        value = static_cast<unsigned char>(min(max(gray + 0.5f, 0.0f), 255.0f));
      } else {
        value = input[(y * width + x) * channels];
      }
    } else {
      // 指定通道
      value = input[(y * width + x) * channels + targetChannel];
    }

    atomicAdd(&localHist[value], 1);
  }

  __syncthreads();

  // 合并到全局直方图
  for (int i = tid; i < 256; i += blockSize) {
    if (localHist[i] > 0) {
      atomicAdd(&histogram[i], localHist[i]);
    }
  }
}

// 直方图均衡化 Kernel
__global__ void equalizeKernel(const unsigned char* input,
                               unsigned char* output, const int* cdf, int width,
                               int height, int channels, int cdfMin,
                               int totalPixels) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    for (int c = 0; c < channels; ++c) {
      int idx = (y * width + x) * channels + c;
      unsigned char value = input[idx];

      // 直方图均衡化公式
      float newValue =
          (float)(cdf[value] - cdfMin) / (totalPixels - cdfMin) * 255.0f;
      output[idx] =
          static_cast<unsigned char>(min(max(newValue + 0.5f, 0.0f), 255.0f));
    }
  }
}

// HistogramCalculator 实现
std::array<int, HistogramCalculator::NUM_BINS>
HistogramCalculator::calculate(const CudaImage& input, cudaStream_t stream) {

  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  // 分配 GPU 直方图内存（在用户 stream 上清零，避免默认流串行化/PTDS 竞态）
  DeviceBuffer histBuffer(NUM_BINS * sizeof(int));
  CUDA_CHECK(
      cudaMemsetAsync(histBuffer.data(), 0, NUM_BINS * sizeof(int), stream));

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  histogramKernelShared<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(), histBuffer.dataAs<int>(),
      input.width, input.height, input.channels, -1 // -1 表示灰度
  );

  CUDA_CHECK(cudaGetLastError());

  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // 复制结果到 Host
  std::array<int, NUM_BINS> histogram;
  histBuffer.copyToHost(histogram.data(), NUM_BINS * sizeof(int));

  return histogram;
}

std::array<std::array<int, HistogramCalculator::NUM_BINS>, 3>
HistogramCalculator::calculateRGB(const CudaImage& input, cudaStream_t stream) {

  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (input.channels < 3) {
    throw std::invalid_argument("Input must have at least 3 channels");
  }

  std::array<std::array<int, NUM_BINS>, 3> histograms;

  for (int c = 0; c < 3; ++c) {
    histograms[c] = calculateChannel(input, c, stream);
  }

  return histograms;
}

std::array<int, HistogramCalculator::NUM_BINS>
HistogramCalculator::calculateChannel(const CudaImage& input, int channel,
                                      cudaStream_t stream) {

  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (channel < 0 || channel >= input.channels) {
    throw std::invalid_argument("Invalid channel index");
  }

  DeviceBuffer histBuffer(NUM_BINS * sizeof(int));
  CUDA_CHECK(
      cudaMemsetAsync(histBuffer.data(), 0, NUM_BINS * sizeof(int), stream));

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  histogramKernelShared<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(), histBuffer.dataAs<int>(),
      input.width, input.height, input.channels, channel);

  CUDA_CHECK(cudaGetLastError());

  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::array<int, NUM_BINS> histogram;
  histBuffer.copyToHost(histogram.data(), NUM_BINS * sizeof(int));

  return histogram;
}

void HistogramCalculator::equalize(const CudaImage& input, CudaImage& output,
                                   cudaStream_t stream) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  // 确保输出图像大小正确
  ImageUtils::ensureOutputSize(output, input.width, input.height,
                               input.channels);

  // 计算直方图
  auto histogram = calculate(input, stream);

  // 计算 CDF
  std::array<int, NUM_BINS> cdf;
  cdf[0] = histogram[0];
  for (int i = 1; i < NUM_BINS; ++i) {
    cdf[i] = cdf[i - 1] + histogram[i];
  }

  // 找到 CDF 最小非零值
  int cdfMin = 0;
  for (int i = 0; i < NUM_BINS; ++i) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }

  int totalPixels = input.width * input.height;
  if (totalPixels == cdfMin) {
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(output.buffer.data(), input.buffer.data(),
                            input.totalBytes(), cudaMemcpyDeviceToDevice));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(output.buffer.data(), input.buffer.data(),
                                 input.totalBytes(), cudaMemcpyDeviceToDevice,
                                 stream));
    }
    return;
  }

  // 上传 CDF 到 GPU（走用户 stream，保证与后续 kernel
  // 的顺序且不被默认流串行化）
  DeviceBuffer cdfBuffer(NUM_BINS * sizeof(int));
  cdfBuffer.copyFromHostAsync(cdf.data(), NUM_BINS * sizeof(int), stream);

  dim3 block(16, 16);
  dim3 grid((input.width + block.x - 1) / block.x,
            (input.height + block.y - 1) / block.y);

  equalizeKernel<<<grid, block, 0, stream>>>(
      input.buffer.dataAs<unsigned char>(),
      output.buffer.dataAs<unsigned char>(), cdfBuffer.dataAs<int>(),
      input.width, input.height, input.channels, cdfMin, totalPixels);

  CUDA_CHECK(cudaGetLastError());
}

} // namespace cudaimg
