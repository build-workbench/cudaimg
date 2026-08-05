#include "gpu_image/core/cuda_error.hpp"
#include "gpu_image/core/image_utils.hpp"
#include "gpu_image/processing/pipeline_processor.hpp"
#include <stdexcept>

namespace gpu_image {

PipelineProcessor::PipelineProcessor(int numStreams) : numStreams_(numStreams) {
  if (numStreams < 1) {
    throw std::invalid_argument("Number of streams must be at least 1");
  }
  createStreams();
}

PipelineProcessor::~PipelineProcessor() { destroyStreams(); }

void PipelineProcessor::createStreams() {
  streams_.resize(numStreams_);
  for (int i = 0; i < numStreams_; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams_[i]));
  }
}

void PipelineProcessor::destroyStreams() noexcept {
  for (cudaStream_t stream : streams_) {
    cudaError_t err = cudaStreamDestroy(stream);
    if (err != cudaSuccess) {
      // Don't throw in destructor - silently handle
    }
  }
  streams_.clear();
}

void PipelineProcessor::addStep(ProcessingStep step) {
  if (!step) {
    throw std::invalid_argument("Processing step is null");
  }
  steps_.push_back(std::move(step));
}

void PipelineProcessor::clearSteps() { steps_.clear(); }

GpuImage PipelineProcessor::process(const GpuImage& input) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }
  if (steps_.empty()) {
    // 没有处理步骤，返回输入的副本
    GpuImage output =
        ImageUtils::createGpuImage(input.width, input.height, input.channels);
    CUDA_CHECK(cudaMemcpy(output.buffer.data(), input.buffer.data(),
                          input.totalBytes(), cudaMemcpyDeviceToDevice));
    return output;
  }

  cudaStream_t stream = streams_[0];

  // 创建工作图像
  GpuImage current =
      ImageUtils::createGpuImage(input.width, input.height, input.channels);
  CUDA_CHECK(cudaMemcpyAsync(current.buffer.data(), input.buffer.data(),
                             input.totalBytes(), cudaMemcpyDeviceToDevice,
                             stream));

  // 执行所有处理步骤
  for (const auto& step : steps_) {
    step(current, stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return current;
}

HostImage PipelineProcessor::processHost(const HostImage& input) {
  if (!input.isValid()) {
    throw std::invalid_argument("Invalid input image");
  }

  cudaStream_t stream = streams_[0];

  // 上传到 GPU
  GpuImage gpuImage =
      ImageUtils::createGpuImage(input.width, input.height, input.channels);
  gpuImage.buffer.copyFromHostAsync(input.data.data(), input.totalBytes(),
                                    stream);

  // 执行处理步骤
  for (const auto& step : steps_) {
    step(gpuImage, stream);
  }

  // 下载结果
  HostImage output = ImageUtils::createHostImage(
      gpuImage.width, gpuImage.height, gpuImage.channels);
  gpuImage.buffer.copyToHostAsync(output.data.data(), output.totalBytes(),
                                  stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return output;
}

std::vector<GpuImage>
PipelineProcessor::processBatch(const std::vector<GpuImage>& inputs) {
  if (inputs.empty()) {
    return {};
  }

  // 先校验全部输入：一旦入队开始，中途抛异常会让仍在飞的异步工作
  // 与已分配缓冲的析构产生竞态
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isValid()) {
      throw std::invalid_argument("Invalid input image at index " +
                                  std::to_string(i));
    }
  }

  std::vector<GpuImage> outputs;
  outputs.reserve(inputs.size());

  // 使用多个 stream 并行处理
  for (size_t i = 0; i < inputs.size(); ++i) {
    cudaStream_t stream = streams_[i % numStreams_];

    // 创建输出图像
    GpuImage output = ImageUtils::createGpuImage(
        inputs[i].width, inputs[i].height, inputs[i].channels);

    // 复制输入
    CUDA_CHECK(cudaMemcpyAsync(output.buffer.data(), inputs[i].buffer.data(),
                               inputs[i].totalBytes(), cudaMemcpyDeviceToDevice,
                               stream));

    // 执行处理步骤
    for (const auto& step : steps_) {
      step(output, stream);
    }

    outputs.push_back(std::move(output));
  }

  // 同步所有 streams
  synchronize();

  return outputs;
}

std::vector<HostImage>
PipelineProcessor::processBatchHost(const std::vector<HostImage>& inputs) {
  if (inputs.empty()) {
    return {};
  }

  std::vector<HostImage> outputs;
  outputs.reserve(inputs.size());

  // 先校验全部输入（理由同 processBatch）
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isValid()) {
      throw std::invalid_argument("Invalid input image at index " +
                                  std::to_string(i));
    }
  }

  // 为每个输入创建 GPU 图像
  std::vector<GpuImage> gpuImages;
  gpuImages.reserve(inputs.size());

  // 流水线处理：上传 -> 处理 -> 下载
  for (size_t i = 0; i < inputs.size(); ++i) {
    cudaStream_t stream = streams_[i % numStreams_];

    // 创建 GPU 图像并上传
    GpuImage gpuImage = ImageUtils::createGpuImage(
        inputs[i].width, inputs[i].height, inputs[i].channels);
    gpuImage.buffer.copyFromHostAsync(inputs[i].data.data(),
                                      inputs[i].totalBytes(), stream);

    // 执行处理步骤
    for (const auto& step : steps_) {
      step(gpuImage, stream);
    }

    gpuImages.push_back(std::move(gpuImage));

    // 创建输出并下载
    HostImage output = ImageUtils::createHostImage(
        gpuImages[i].width, gpuImages[i].height, gpuImages[i].channels);
    gpuImages[i].buffer.copyToHostAsync(output.data.data(), output.totalBytes(),
                                        stream);

    outputs.push_back(std::move(output));
  }

  // 同步所有 streams
  synchronize();

  return outputs;
}

void PipelineProcessor::synchronize() {
  for (cudaStream_t stream : streams_) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

} // namespace gpu_image
