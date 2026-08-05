#pragma once

#include "gpu_image/core/gpu_image.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace gpu_image {

// 流水线处理器类
class PipelineProcessor {
public:
  // 处理步骤的函数类型
  using ProcessingStep = std::function<void(GpuImage&, cudaStream_t)>;

  explicit PipelineProcessor(int numStreams = 3);
  ~PipelineProcessor();

  // 禁用拷贝
  PipelineProcessor(const PipelineProcessor&) = delete;
  PipelineProcessor& operator=(const PipelineProcessor&) = delete;

  // 添加处理步骤
  void addStep(ProcessingStep step);

  // 清空处理步骤
  void clearSteps();

  // 获取步骤数量
  size_t stepCount() const { return steps_.size(); }

  // 处理单张图像（使用流水线）
  GpuImage process(const GpuImage& input);

  // 处理 HostImage（包含上传和下载）
  HostImage processHost(const HostImage& input);

  // 批量处理多张图像（流水线并行）
  std::vector<GpuImage> processBatch(const std::vector<GpuImage>& inputs);

  // 批量处理 HostImage
  std::vector<HostImage> processBatchHost(const std::vector<HostImage>& inputs);

  // 同步等待所有操作完成
  void synchronize();

private:
  int numStreams_;
  std::vector<cudaStream_t> streams_;
  std::vector<ProcessingStep> steps_;

  void createStreams();
  void destroyStreams() noexcept;
};

} // namespace gpu_image
