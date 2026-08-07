/**
 * cudaimg - 流水线处理示例
 *
 * 本示例演示了流水线处理功能：
 * 1. 创建处理流水线
 * 2. 添加多个处理步骤
 * 3. 批量处理多张图像
 * 4. 利用 CUDA Streams 实现并行处理
 */

#include "cudaimg/cudaimg.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cudaimg;

// 创建随机测试图像
HostImage createRandomImage(int width, int height, int channels,
                            unsigned int seed) {
  HostImage image = ImageUtils::createHostImage(width, height, channels);

  // 简单的伪随机数生成
  unsigned int state = seed;
  auto nextRandom = [&state]() {
    state = state * 1103515245 + 12345;
    return static_cast<unsigned char>((state >> 16) & 0xFF);
  };

  for (size_t i = 0; i < image.data.size(); ++i) {
    image.data[i] = nextRandom();
  }

  return image;
}

// 计时辅助类
class Timer {
public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  double elapsedMs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
};

int main() {
  std::cout << "=== GPU Image Processing - Pipeline Demo ===" << std::endl;
  std::cout << "Version: " << getVersionString() << std::endl;

  if (!isCudaAvailable()) {
    std::cerr << "Error: CUDA is not available!" << std::endl;
    return 1;
  }

  std::cout << getDeviceInfo() << std::endl;
  std::cout << std::endl;

  try {
    // 测试参数
    const int imageWidth = 512;
    const int imageHeight = 512;
    const int channels = 3;
    const int numImages = 10;

    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Image size: " << imageWidth << "x" << imageHeight << "x"
              << channels << std::endl;
    std::cout << "  Number of images: " << numImages << std::endl;
    std::cout << std::endl;

    // 创建测试图像
    std::cout << "Creating test images..." << std::endl;
    std::vector<HostImage> testImages;
    for (int i = 0; i < numImages; ++i) {
      testImages.push_back(
          createRandomImage(imageWidth, imageHeight, channels, i * 12345));
    }

    // ========== 方法 1: 顺序处理 ==========
    std::cout << "\n--- Sequential Processing ---" << std::endl;

    ImageProcessor processor;
    Timer timer;

    timer.start();
    std::vector<HostImage> sequentialResults;

    for (const auto& hostImage : testImages) {
      // 上传
      CudaImage gpuImage = processor.loadFromHost(hostImage);

      // 处理步骤
      CudaImage result = processor.adjustBrightness(gpuImage, 20);
      CudaImage blurred = processor.gaussianBlur(result, 3, 1.0f);
      CudaImage final_result = processor.invert(blurred);

      // 下载
      sequentialResults.push_back(processor.download(final_result));
    }

    double sequentialTime = timer.elapsedMs();
    std::cout << "Sequential time: " << std::fixed << std::setprecision(2)
              << sequentialTime << " ms" << std::endl;
    std::cout << "Average per image: " << sequentialTime / numImages << " ms"
              << std::endl;

    // ========== 方法 2: 流水线处理 ==========
    std::cout << "\n--- Pipeline Processing ---" << std::endl;

    // 创建流水线处理器（使用 4 个 streams）
    PipelineProcessor pipeline(4);

    // 添加处理步骤
    pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
      CudaImage temp;
      PixelOperator::adjustBrightness(img, temp, 20, stream);
      img = std::move(temp);
    });

    pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
      CudaImage temp;
      ConvolutionEngine::gaussianBlur(img, temp, 3, 1.0f, stream);
      img = std::move(temp);
    });

    pipeline.addStep([](CudaImage& img, cudaStream_t stream) {
      PixelOperator::invertInPlace(img, stream);
    });

    std::cout << "Pipeline steps: " << pipeline.stepCount() << std::endl;

    timer.start();
    std::vector<HostImage> pipelineResults =
        pipeline.processBatchHost(testImages);
    double pipelineTime = timer.elapsedMs();

    std::cout << "Pipeline time: " << std::fixed << std::setprecision(2)
              << pipelineTime << " ms" << std::endl;
    std::cout << "Average per image: " << pipelineTime / numImages << " ms"
              << std::endl;

    // 计算加速比
    double speedup = sequentialTime / pipelineTime;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup
              << "x" << std::endl;

    // ========== 验证结果一致性 ==========
    std::cout << "\n--- Verifying Results ---" << std::endl;

    bool resultsMatch = true;
    for (int i = 0; i < numImages; ++i) {
      if (sequentialResults[i].data.size() != pipelineResults[i].data.size()) {
        resultsMatch = false;
        std::cout << "Image " << i << ": Size mismatch!" << std::endl;
        continue;
      }

      int diffCount = 0;
      int maxDiff = 0;
      for (size_t j = 0; j < sequentialResults[i].data.size(); ++j) {
        int diff = std::abs(static_cast<int>(sequentialResults[i].data[j]) -
                            static_cast<int>(pipelineResults[i].data[j]));
        if (diff > 0) {
          diffCount++;
          maxDiff = std::max(maxDiff, diff);
        }
      }

      if (diffCount > 0) {
        std::cout << "Image " << i << ": " << diffCount
                  << " pixels differ, max diff = " << maxDiff << std::endl;
        // 允许小的数值差异（由于浮点运算顺序）
        if (maxDiff > 2) {
          resultsMatch = false;
        }
      }
    }

    if (resultsMatch) {
      std::cout << "All results match (within tolerance)!" << std::endl;
    } else {
      std::cout << "Warning: Some results differ significantly!" << std::endl;
    }

    // ========== 不同 Stream 数量的性能对比 ==========
    std::cout << "\n--- Stream Count Comparison ---" << std::endl;

    for (int numStreams : {1, 2, 4, 8}) {
      PipelineProcessor testPipeline(numStreams);

      testPipeline.addStep([](CudaImage& img, cudaStream_t stream) {
        CudaImage temp;
        PixelOperator::adjustBrightness(img, temp, 20, stream);
        img = std::move(temp);
      });

      testPipeline.addStep([](CudaImage& img, cudaStream_t stream) {
        CudaImage temp;
        ConvolutionEngine::gaussianBlur(img, temp, 3, 1.0f, stream);
        img = std::move(temp);
      });

      testPipeline.addStep([](CudaImage& img, cudaStream_t stream) {
        PixelOperator::invertInPlace(img, stream);
      });

      timer.start();
      auto results = testPipeline.processBatchHost(testImages);
      double time = timer.elapsedMs();

      std::cout << "  " << numStreams << " stream(s): " << std::fixed
                << std::setprecision(2) << time << " ms" << std::endl;
    }

    std::cout << "\n=== Pipeline Demo Complete ===" << std::endl;

  } catch (const CudaException& e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
