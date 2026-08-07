/**
 * ExecutionContext and OperatorPipeline Tests
 * 执行上下文和操作符管道测试
 *
 * Feature: gpu-image-processing
 * Property: Architectural depth verification
 */

#include "cudaimg/cudaimg.hpp"
#include <gtest/gtest.h>

using namespace cudaimg;

class ExecutionContextTest : public ::testing::Test {
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
    for (size_t i = 0; i < image.data.size(); ++i) {
      image.data[i] = static_cast<unsigned char>(i % 256);
    }
    return image;
  }
};

// Test ExecutionPolicy modes
TEST_F(ExecutionContextTest, ExecutionPolicyModes) {
  // Sync policy
  auto syncPolicy = ExecutionPolicy::sync();
  EXPECT_EQ(syncPolicy.mode(), ExecutionPolicy::Mode::Sync);
  EXPECT_EQ(syncPolicy.stream(), nullptr);

  // Async policy
  auto asyncPolicy = ExecutionPolicy::async();
  EXPECT_EQ(asyncPolicy.mode(), ExecutionPolicy::Mode::Async);
  EXPECT_NE(asyncPolicy.stream(), nullptr);
  EXPECT_TRUE(asyncPolicy.ownsStream());

  // Batch policy
  auto batchPolicy = ExecutionPolicy::batch();
  EXPECT_EQ(batchPolicy.mode(), ExecutionPolicy::Mode::Batch);
  EXPECT_NE(batchPolicy.stream(), nullptr);
}

// Test ExecutionPolicy move semantics
TEST_F(ExecutionContextTest, ExecutionPolicyMove) {
  auto policy1 = ExecutionPolicy::async();
  cudaStream_t stream = policy1.stream();

  auto policy2 = std::move(policy1);

  EXPECT_EQ(policy2.stream(), stream);
  EXPECT_TRUE(policy2.ownsStream());
  EXPECT_EQ(policy1.stream(), nullptr);
  EXPECT_FALSE(policy1.ownsStream());
}

// Test ExecutionContext creation
TEST_F(ExecutionContextTest, ContextCreation) {
  ExecutionContext syncCtx(ExecutionPolicy::sync());
  EXPECT_EQ(syncCtx.stream(), nullptr);

  ExecutionContext asyncCtx(ExecutionPolicy::async());
  EXPECT_NE(asyncCtx.stream(), nullptr);
}

// Test ExecutionContext allocation
TEST_F(ExecutionContextTest, ContextAllocation) {
  ExecutionContext ctx(ExecutionPolicy::sync());

  CudaImage output = ctx.allocateOutput(64, 64, 3);

  EXPECT_EQ(output.width, 64);
  EXPECT_EQ(output.height, 64);
  EXPECT_EQ(output.channels, 3);
  EXPECT_TRUE(output.isValid());
}

// Test ExecutionContext ensureOutputSize
TEST_F(ExecutionContextTest, EnsureOutputSize) {
  ExecutionContext ctx(ExecutionPolicy::sync());

  CudaImage input = ctx.allocateOutput(64, 64, 3);
  CudaImage output;

  bool reallocated = ctx.ensureOutputSize(input, output);

  EXPECT_TRUE(reallocated);
  EXPECT_EQ(output.width, input.width);
  EXPECT_EQ(output.height, input.height);
  EXPECT_EQ(output.channels, input.channels);

  // Second call should not reallocate
  reallocated = ctx.ensureOutputSize(input, output);
  EXPECT_FALSE(reallocated);
}

// Test ImageAllocator singleton
TEST_F(ExecutionContextTest, AllocatorSingleton) {
  auto& alloc1 = ImageAllocator::instance();
  auto& alloc2 = ImageAllocator::instance();

  EXPECT_EQ(&alloc1, &alloc2);
}

// Test memory pooling toggle
TEST_F(ExecutionContextTest, MemoryPoolingToggle) {
  auto& allocator = ImageAllocator::instance();

  bool originalState = allocator.isPoolingEnabled();

  allocator.setPoolingEnabled(true);
  EXPECT_TRUE(allocator.isPoolingEnabled());

  allocator.setPoolingEnabled(false);
  EXPECT_FALSE(allocator.isPoolingEnabled());

  // Restore original state
  allocator.setPoolingEnabled(originalState);
}

// Test ImageProcessor modes
TEST_F(ExecutionContextTest, ImageProcessorModes) {
  ImageProcessor syncProcessor(ImageProcessor::Mode::Sync);
  EXPECT_EQ(syncProcessor.mode(), ImageProcessor::Mode::Sync);

  ImageProcessor asyncProcessor(ImageProcessor::Mode::Async);
  EXPECT_EQ(asyncProcessor.mode(), ImageProcessor::Mode::Async);

  ImageProcessor batchProcessor(ImageProcessor::Mode::Batch);
  EXPECT_EQ(batchProcessor.mode(), ImageProcessor::Mode::Batch);
}

// Test ImageProcessor with custom policy
TEST_F(ExecutionContextTest, ImageProcessorWithPolicy) {
  auto policy = ExecutionPolicy::async();
  cudaStream_t stream = policy.stream();

  ImageProcessor processor(std::move(policy));

  EXPECT_EQ(processor.mode(), ImageProcessor::Mode::Async);
  EXPECT_EQ(processor.context().stream(), stream);
}

// Test ImageProcessor isComplete
TEST_F(ExecutionContextTest, ImageProcessorIsComplete) {
  ImageProcessor syncProcessor(ImageProcessor::Mode::Sync);
  EXPECT_TRUE(syncProcessor.isComplete()); // Sync always complete

  ImageProcessor asyncProcessor(ImageProcessor::Mode::Async);

  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = asyncProcessor.loadFromHost(input);
  CudaImage output = asyncProcessor.invert(gpuInput);

  // After async operation, may or may not be complete
  // Just verify the method doesn't throw
  asyncProcessor.isComplete();

  // After synchronize, should be complete
  asyncProcessor.synchronize();
  EXPECT_TRUE(asyncProcessor.isComplete());
}

class OperatorPipelineTest : public ::testing::Test {
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
    for (size_t i = 0; i < image.data.size(); ++i) {
      image.data[i] = static_cast<unsigned char>(i % 256);
    }
    return image;
  }
};

// Test OperatorPipeline basic construction
TEST_F(OperatorPipelineTest, PipelineConstruction) {
  OperatorPipeline pipeline;
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(pipeline.size(), 0u);
}

// Test OperatorPipeline with operators
TEST_F(OperatorPipelineTest, PipelineWithOperators) {
  OperatorPipeline pipeline;

  pipeline.then<InvertOperator>().then<GrayscaleOperator>();

  EXPECT_FALSE(pipeline.empty());
  EXPECT_EQ(pipeline.size(), 2u);
}

// Test OperatorPipeline execution
TEST_F(OperatorPipelineTest, PipelineExecution) {
  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  OperatorPipeline pipeline;
  pipeline.then<InvertOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());
  CudaImage output = pipeline.apply(gpuInput, ctx);

  EXPECT_TRUE(output.isValid());
  EXPECT_EQ(output.width, input.width);
  EXPECT_EQ(output.height, input.height);
  EXPECT_EQ(output.channels, input.channels);

  // Verify invert was applied
  HostImage result = ImageUtils::downloadFromGpu(output);
  for (size_t i = 0; i < input.data.size(); ++i) {
    EXPECT_EQ(result.data[i], static_cast<unsigned char>(255 - input.data[i]));
  }
}

// Test OperatorPipeline with grayscale (changes channels)
TEST_F(OperatorPipelineTest, PipelineGrayscale) {
  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  OperatorPipeline pipeline;
  pipeline.then<GrayscaleOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());
  CudaImage output = pipeline.apply(gpuInput, ctx);

  EXPECT_EQ(output.channels, 1); // Grayscale output
}

// Test OperatorPipeline chaining
TEST_F(OperatorPipelineTest, PipelineChaining) {
  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  // Grayscale -> Blur -> Sobel (edge detection pipeline)
  OperatorPipeline pipeline;
  pipeline.then<GrayscaleOperator>()
      .then<GaussianBlurOperator>(5, 1.0f)
      .then<SobelOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());
  CudaImage output = pipeline.apply(gpuInput, ctx);

  EXPECT_TRUE(output.isValid());
  EXPECT_EQ(output.channels, 1); // Sobel outputs single channel

  // Verify pipeline traits
  auto traits = pipeline.traits();
  EXPECT_TRUE(traits.name.find("grayscale") != std::string::npos);
  EXPECT_TRUE(traits.changesChannels);
}

// Test OperatorPipeline clone
TEST_F(OperatorPipelineTest, PipelineClone) {
  OperatorPipeline original;
  original.then<InvertOperator>().then<GrayscaleOperator>();

  auto cloned = original.clone();

  EXPECT_EQ(cloned->traits().name, original.traits().name);
}

// Test OperatorPipeline as edge detection chain (replaces PipelineBuilder test)
TEST_F(OperatorPipelineTest, PipelineEdgeDetection) {
  HostImage input = createTestImage(64, 64, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  OperatorPipeline pipeline;
  pipeline.then<GrayscaleOperator>()
      .then<GaussianBlurOperator>(5, 1.0f)
      .then<SobelOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());
  CudaImage result = pipeline.apply(gpuInput, ctx);

  EXPECT_TRUE(result.isValid());
  EXPECT_EQ(result.channels, 1);
}

// Test OperatorPipeline invert with download (replaces
// PipelineBuilderWithDownload)
TEST_F(OperatorPipelineTest, PipelineInvertWithDownload) {
  HostImage input = createTestImage(32, 32, 3);
  CudaImage gpuInput = ImageUtils::uploadToGpu(input);

  OperatorPipeline pipeline;
  pipeline.then<InvertOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());
  CudaImage result = pipeline.apply(gpuInput, ctx);
  HostImage downloaded = ImageUtils::downloadFromGpu(result);

  EXPECT_TRUE(downloaded.isValid());
  for (size_t i = 0; i < input.data.size(); ++i) {
    EXPECT_EQ(downloaded.data[i],
              static_cast<unsigned char>(255 - input.data[i]));
  }
}

// Test Operator traits via constructors
TEST_F(OperatorPipelineTest, OperatorTraits) {
  InvertOperator invert;
  auto invertTraits = invert.traits();
  EXPECT_TRUE(invertTraits.inPlaceCapable);
  EXPECT_FALSE(invertTraits.changesDimensions);
  EXPECT_FALSE(invertTraits.changesChannels);

  GrayscaleOperator grayscale;
  auto grayscaleTraits = grayscale.traits();
  EXPECT_FALSE(grayscaleTraits.inPlaceCapable);
  EXPECT_TRUE(grayscaleTraits.changesChannels);

  ResizeOperator resize(100, 100);
  auto resizeTraits = resize.traits();
  EXPECT_TRUE(resizeTraits.changesDimensions);
}

// Test BrightnessOperator with parameters
TEST_F(OperatorPipelineTest, BrightnessOperatorParams) {
  BrightnessOperator brightness(50);
  EXPECT_EQ(brightness.offset(), 50);

  brightness.setOffset(-30);
  EXPECT_EQ(brightness.offset(), -30);
}

// Test GaussianBlurOperator parameters
TEST_F(OperatorPipelineTest, GaussianBlurOperatorParams) {
  GaussianBlurOperator blur(7, 2.5f);
  EXPECT_EQ(blur.kernelSize(), 7);
  EXPECT_FLOAT_EQ(blur.sigma(), 2.5f);

  blur.setKernelSize(9);
  blur.setSigma(3.0f);
  EXPECT_EQ(blur.kernelSize(), 9);
  EXPECT_FLOAT_EQ(blur.sigma(), 3.0f);
}

// Test ResizeOperator named constructors
TEST_F(OperatorPipelineTest, ResizeOperatorNamedConstructors) {
  auto resizeByDim = ResizeOperator::byDimensions(100, 200);
  EXPECT_FALSE(resizeByDim.isScaleMode());
  EXPECT_EQ(resizeByDim.width(), 100);
  EXPECT_EQ(resizeByDim.height(), 200);

  auto resizeByScale = ResizeOperator::byScale(0.5f, 0.5f);
  EXPECT_TRUE(resizeByScale.isScaleMode());
  EXPECT_FLOAT_EQ(resizeByScale.scaleX(), 0.5f);
  EXPECT_FLOAT_EQ(resizeByScale.scaleY(), 0.5f);

  resizeByDim.setScale(2.0f, 2.0f);
  EXPECT_TRUE(resizeByDim.isScaleMode());
  EXPECT_FLOAT_EQ(resizeByDim.scaleX(), 2.0f);

  resizeByScale.setDimensions(300, 400);
  EXPECT_FALSE(resizeByScale.isScaleMode());
  EXPECT_EQ(resizeByScale.width(), 300);
  EXPECT_EQ(resizeByScale.height(), 400);
}

// Test OperatorPipeline with invalid input
TEST_F(OperatorPipelineTest, PipelineWithInvalidInput) {
  OperatorPipeline pipeline;
  pipeline.then<InvertOperator>();

  ExecutionContext ctx(ExecutionPolicy::sync());

  CudaImage invalidInput;
  CudaImage output = pipeline.apply(invalidInput, ctx);

  EXPECT_FALSE(output.isValid());
}

// Test recycleToPool method
TEST_F(ExecutionContextTest, RecycleToPoolMethod) {
  ExecutionContext ctx(ExecutionPolicy::sync());

  CudaImage image = ctx.allocateOutput(64, 64, 3);
  EXPECT_TRUE(image.isValid());

  ctx.recycleToPool(std::move(image));
  EXPECT_FALSE(image.isValid());
}
