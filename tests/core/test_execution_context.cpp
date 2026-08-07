/**
 * ExecutionContext Tests
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
