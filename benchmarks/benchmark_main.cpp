/**
 * GPU Image Processing Library - 性能基准测试
 *
 * 测试各种图像处理操作的性能
 */

#include "gpu_image/gpu_image_processing.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace gpu_image;

// 计时器类
class Benchmark {
public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  double elapsedMs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
};

// 创建测试图像
HostImage createTestImage(int width, int height, int channels) {
  HostImage image = ImageUtils::createHostImage(width, height, channels);
  unsigned int seed = 42;
  for (size_t i = 0; i < image.data.size(); ++i) {
    seed = seed * 1103515245 + 12345;
    image.data[i] = static_cast<unsigned char>((seed >> 16) & 0xFF);
  }
  return image;
}

// 运行基准测试
template <typename Func>
double runBenchmark(const std::string& name, Func func, int iterations = 100) {
  // 预热
  for (int i = 0; i < 10; ++i) {
    func();
  }
  cudaDeviceSynchronize();

  // 计时
  Benchmark timer;
  timer.start();

  for (int i = 0; i < iterations; ++i) {
    func();
  }
  cudaDeviceSynchronize();

  double totalMs = timer.elapsedMs();
  double avgMs = totalMs / iterations;

  std::cout << std::setw(30) << std::left << name << std::setw(12) << std::right
            << std::fixed << std::setprecision(3) << avgMs << " ms"
            << std::endl;

  return avgMs;
}

int main() {
  std::cout << "=== GPU Image Processing Benchmark ===" << std::endl;
  std::cout << getDeviceInfo() << std::endl;
  std::cout << std::endl;

  if (!isCudaAvailable()) {
    std::cerr << "CUDA not available!" << std::endl;
    return 1;
  }

  // 测试不同尺寸
  std::vector<std::pair<int, int>> sizes = {
      {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096}};

  for (const auto& size : sizes) {
    int width = size.first;
    int height = size.second;

    std::cout << "\n--- Image Size: " << width << "x" << height << " ---"
              << std::endl;
    std::cout << std::setw(30) << std::left << "Operation" << std::setw(12)
              << std::right << "Time" << std::endl;
    std::cout << std::string(42, '-') << std::endl;

    HostImage hostImage = createTestImage(width, height, 3);
    GpuImage gpuImage = ImageUtils::uploadToGpu(hostImage);
    GpuImage output;

    // 像素操作
    runBenchmark("Invert", [&]() { PixelOperator::invert(gpuImage, output); });

    runBenchmark("Grayscale",
                 [&]() { PixelOperator::toGrayscale(gpuImage, output); });

    runBenchmark("Brightness (+50)", [&]() {
      PixelOperator::adjustBrightness(gpuImage, output, 50);
    });

    // 卷积操作
    runBenchmark("Gaussian Blur 3x3", [&]() {
      ConvolutionEngine::gaussianBlur(gpuImage, output, 3, 1.0f);
    });

    runBenchmark("Gaussian Blur 5x5", [&]() {
      ConvolutionEngine::gaussianBlur(gpuImage, output, 5, 1.5f);
    });

    runBenchmark("Sobel Edge Detection", [&]() {
      ConvolutionEngine::sobelEdgeDetection(gpuImage, output);
    });

    // 直方图
    runBenchmark("Histogram",
                 [&]() { HistogramCalculator::calculate(gpuImage); });

    runBenchmark("Histogram Equalize",
                 [&]() { HistogramCalculator::equalize(gpuImage, output); });

    // 缩放
    runBenchmark("Resize 2x (Bilinear)", [&]() {
      ImageResizer::resize(gpuImage, output, width * 2, height * 2);
    });

    runBenchmark("Resize 0.5x (Bilinear)", [&]() {
      ImageResizer::resize(gpuImage, output, width / 2, height / 2);
    });

    // 形态学操作
    runBenchmark("Erode 3x3",
                 [&]() { Morphology::erode(gpuImage, output, 3); });

    runBenchmark("Dilate 3x3",
                 [&]() { Morphology::dilate(gpuImage, output, 3); });

    // 阈值
    runBenchmark("Threshold",
                 [&]() { Threshold::threshold(gpuImage, output, 128); });

    runBenchmark("Otsu Binarize",
                 [&]() { Threshold::otsuBinarize(gpuImage, output); });

    // 颜色空间
    runBenchmark("RGB to HSV",
                 [&]() { ColorSpace::rgbToHsv(gpuImage, output); });

    runBenchmark("RGB to YUV",
                 [&]() { ColorSpace::rgbToYuv(gpuImage, output); });

    // 数据传输
    runBenchmark("Upload (H2D)",
                 [&]() { GpuImage temp = ImageUtils::uploadToGpu(hostImage); });

    runBenchmark("Download (D2H)", [&]() {
      HostImage temp = ImageUtils::downloadFromGpu(gpuImage);
    });
  }

  // 流水线性能测试
  std::cout << "\n--- Pipeline Performance ---" << std::endl;

  const int batchSize = 10;
  const int pipelineWidth = 512;
  const int pipelineHeight = 512;

  std::vector<HostImage> batchImages;
  for (int i = 0; i < batchSize; ++i) {
    batchImages.push_back(createTestImage(pipelineWidth, pipelineHeight, 3));
  }

  // 顺序处理
  ImageProcessor processor;
  Benchmark timer;

  timer.start();
  for (const auto& img : batchImages) {
    GpuImage gpu = processor.loadFromHost(img);
    GpuImage step1 = processor.adjustBrightness(gpu, 20);
    GpuImage step2 = processor.gaussianBlur(step1, 3, 1.0f);
    GpuImage step3 = processor.invert(step2);
    HostImage result = processor.download(step3);
    (void)result;
  }
  double sequentialTime = timer.elapsedMs();

  std::cout << "Sequential (" << batchSize << " images): " << std::fixed
            << std::setprecision(2) << sequentialTime << " ms" << std::endl;

  // 流水线处理
  for (int numStreams : {1, 2, 4, 8}) {
    PipelineProcessor pipeline(numStreams);

    pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
      GpuImage temp;
      PixelOperator::adjustBrightness(img, temp, 20, stream);
      img = std::move(temp);
    });

    pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
      GpuImage temp;
      ConvolutionEngine::gaussianBlur(img, temp, 3, 1.0f, stream);
      img = std::move(temp);
    });

    pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
      PixelOperator::invertInPlace(img, stream);
    });

    timer.start();
    auto results = pipeline.processBatchHost(batchImages);
    double pipelineTime = timer.elapsedMs();

    double speedup = sequentialTime / pipelineTime;

    std::cout << "Pipeline " << numStreams << " streams: " << std::fixed
              << std::setprecision(2) << pipelineTime << " ms"
              << " (speedup: " << speedup << "x)" << std::endl;
  }

  std::cout << "\n=== Benchmark Complete ===" << std::endl;

  return 0;
}
