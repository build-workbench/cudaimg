#pragma once

#include "cudaimg/core/image.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace cudaimg {

// 流水线处理器类
class PipelineProcessor {
public:
  // 处理步骤的函数类型
  using ProcessingStep = std::function<void(CudaImage&, cudaStream_t)>;

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
  CudaImage process(const CudaImage& input);

  // 处理 HostImage（包含上传和下载）
  HostImage processHost(const HostImage& input);

  // 批量处理多张图像（流水线并行）
  std::vector<CudaImage> processBatch(const std::vector<CudaImage>& inputs);

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

} // namespace cudaimg
