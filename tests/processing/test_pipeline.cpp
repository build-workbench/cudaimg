/**
 * PipelineProcessor 单元测试
 *
 * Feature: gpu-image-processing
 * Property 9: 流水线处理结果一致性
 */

#include "cudaimg/cudaimg.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace cudaimg;

class PipelineTest : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  HostImage createTestImage(int width, int height, int channels) {
    HostImage image = ImageUtils::createHostImage(width, height, channels);
    unsigned int seed = 42;
    for (size_t i = 0; i < image.data.size(); ++i) {
      seed = seed * 1103515245 + 12345;
      image.data[i] = static_cast<unsigned char>((seed >> 16) & 0xFF);
    }
    return image;
  }
};

// 测试基本流水线
TEST_F(PipelineTest, BasicPipeline) {
  PipelineProcessor pipeline(2);

  // 添加反色步骤
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  EXPECT_EQ(pipeline.stepCount(), 1u);

  HostImage input = createTestImage(32, 32, 3);
  HostImage output = pipeline.processHost(input);

  // 验证反色
  for (size_t i = 0; i < input.data.size(); ++i) {
    EXPECT_EQ(output.data[i], 255 - input.data[i]);
  }
}

// Property 9: 流水线处理结果一致性
// 流水线处理和顺序处理应该产生相同结果
TEST_F(PipelineTest, PipelineConsistency) {
  const int width = 64;
  const int height = 64;
  const int channels = 3;

  HostImage input = createTestImage(width, height, channels);

  // 顺序处理
  ImageProcessor processor;
  CudaImage gpuInput = processor.loadFromHost(input);
  CudaImage step1 = processor.adjustBrightness(gpuInput, 30);
  CudaImage step2 = processor.invert(step1);
  HostImage sequentialResult = processor.download(step2);

  // 流水线处理
  PipelineProcessor pipeline(2);

  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    CudaImage temp;
    PixelOperator::adjustBrightness(img, temp, 30, stream);
    img = std::move(temp);
  });

  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  HostImage pipelineResult = pipeline.processHost(input);

  // 比较结果
  ASSERT_EQ(sequentialResult.data.size(), pipelineResult.data.size());

  int maxDiff = 0;
  for (size_t i = 0; i < sequentialResult.data.size(); ++i) {
    int diff = std::abs(static_cast<int>(sequentialResult.data[i]) -
                        static_cast<int>(pipelineResult.data[i]));
    maxDiff = std::max(maxDiff, diff);
  }

  // 允许小的数值差异
  EXPECT_LE(maxDiff, 1) << "Max difference: " << maxDiff;
}

// 测试批量处理
TEST_F(PipelineTest, BatchProcessing) {
  const int numImages = 5;
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  std::vector<HostImage> inputs;
  for (int i = 0; i < numImages; ++i) {
    HostImage img = ImageUtils::createHostImage(width, height, channels);
    for (size_t j = 0; j < img.data.size(); ++j) {
      img.data[j] = static_cast<unsigned char>((i * 50 + j) % 256);
    }
    inputs.push_back(std::move(img));
  }

  PipelineProcessor pipeline(3);
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  std::vector<HostImage> outputs = pipeline.processBatchHost(inputs);

  ASSERT_EQ(outputs.size(), static_cast<size_t>(numImages));

  // 验证每个输出
  for (int i = 0; i < numImages; ++i) {
    EXPECT_EQ(outputs[i].width, width);
    EXPECT_EQ(outputs[i].height, height);
    EXPECT_EQ(outputs[i].channels, channels);

    // 验证反色
    for (size_t j = 0; j < inputs[i].data.size(); ++j) {
      EXPECT_EQ(outputs[i].data[j], 255 - inputs[i].data[j])
          << "Image " << i << ", index " << j;
    }
  }
}

// 测试空流水线
TEST_F(PipelineTest, EmptyPipeline) {
  PipelineProcessor pipeline(2);

  EXPECT_EQ(pipeline.stepCount(), 0u);

  HostImage input = createTestImage(32, 32, 3);
  HostImage output = pipeline.processHost(input);

  // 空流水线应该返回输入的副本
  EXPECT_EQ(input.data, output.data);
}

// 测试清空步骤
TEST_F(PipelineTest, ClearSteps) {
  PipelineProcessor pipeline(2);

  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  EXPECT_EQ(pipeline.stepCount(), 1u);

  pipeline.clearSteps();

  EXPECT_EQ(pipeline.stepCount(), 0u);
}

// 测试多步骤流水线
TEST_F(PipelineTest, MultiStepPipeline) {
  PipelineProcessor pipeline(2);

  // 步骤 1: 亮度 +50
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    CudaImage temp;
    PixelOperator::adjustBrightness(img, temp, 50, stream);
    img = std::move(temp);
  });

  // 步骤 2: 反色
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  // 步骤 3: 亮度 -30
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    CudaImage temp;
    PixelOperator::adjustBrightness(img, temp, -30, stream);
    img = std::move(temp);
  });

  EXPECT_EQ(pipeline.stepCount(), 3u);

  HostImage input = createTestImage(32, 32, 1);
  HostImage output = pipeline.processHost(input);

  // 手动计算预期结果
  for (size_t i = 0; i < input.data.size(); ++i) {
    int v = input.data[i];
    v = std::min(255, v + 50); // 亮度 +50
    v = 255 - v;               // 反色
    v = std::max(0, v - 30);   // 亮度 -30

    EXPECT_NEAR(output.data[i], v, 1) << "Index " << i;
  }
}

// 测试无效参数
TEST_F(PipelineTest, InvalidParameters) {
  // 无效的 stream 数量
  EXPECT_THROW(PipelineProcessor(0), std::invalid_argument);

  // 空步骤
  PipelineProcessor pipeline(2);
  EXPECT_THROW(pipeline.addStep(nullptr), std::invalid_argument);

  // 无效图像
  HostImage invalid;
  EXPECT_THROW(pipeline.processHost(invalid), std::invalid_argument);
}

// 测试同步
TEST_F(PipelineTest, Synchronization) {
  PipelineProcessor pipeline(4);

  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  std::vector<HostImage> inputs;
  for (int i = 0; i < 10; ++i) {
    inputs.push_back(createTestImage(64, 64, 3));
  }

  auto outputs = pipeline.processBatchHost(inputs);

  // 同步后所有结果应该可用
  pipeline.synchronize();

  EXPECT_EQ(outputs.size(), inputs.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_TRUE(outputs[i].isValid());
  }
}

// 测试 GPU 图像批处理
TEST_F(PipelineTest, CudaImageBatchProcessing) {
  const int numImages = 3;
  const int width = 32;
  const int height = 32;
  const int channels = 3;

  std::vector<CudaImage> gpuInputs;
  std::vector<HostImage> hostInputs;

  for (int i = 0; i < numImages; ++i) {
    HostImage host = createTestImage(width, height, channels);
    hostInputs.push_back(host);
    gpuInputs.push_back(ImageUtils::uploadToGpu(host));
  }

  PipelineProcessor pipeline(2);
  pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
  });

  auto gpuOutputs = pipeline.processBatch(gpuInputs);

  ASSERT_EQ(gpuOutputs.size(), static_cast<size_t>(numImages));

  for (int i = 0; i < numImages; ++i) {
    HostImage output = ImageUtils::downloadFromGpu(gpuOutputs[i]);

    for (size_t j = 0; j < hostInputs[i].data.size(); ++j) {
      EXPECT_EQ(output.data[j], 255 - hostInputs[i].data[j]);
    }
  }
}
