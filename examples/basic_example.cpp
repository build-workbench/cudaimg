/**
 * GPU Image Processing Library - 基础使用示例
 *
 * 本示例演示了库的基本功能：
 * 1. 创建和加载图像
 * 2. 像素级操作（反色、灰度化、亮度调整）
 * 3. 卷积操作（高斯模糊、边缘检测）
 * 4. 直方图计算
 * 5. 图像缩放
 */

#include "gpu_image/gpu_image_processing.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace gpu_image;

// 创建测试图像（渐变图案）
HostImage createTestImage(int width, int height, int channels) {
  HostImage image = ImageUtils::createHostImage(width, height, channels);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (channels >= 3) {
        // RGB 渐变
        image.at(x, y, 0) = static_cast<unsigned char>(x * 255 / width);  // R
        image.at(x, y, 1) = static_cast<unsigned char>(y * 255 / height); // G
        image.at(x, y, 2) =
            static_cast<unsigned char>((x + y) * 128 / (width + height)); // B
        if (channels == 4) {
          image.at(x, y, 3) = 255; // A
        }
      } else {
        // 灰度渐变
        image.at(x, y, 0) =
            static_cast<unsigned char>((x + y) * 255 / (width + height));
      }
    }
  }

  return image;
}

// 打印图像统计信息
void printImageStats(const HostImage& image, const std::string& name) {
  if (!image.isValid()) {
    std::cout << name << ": Invalid image" << std::endl;
    return;
  }

  int minVal = 255, maxVal = 0;
  long long sum = 0;

  for (size_t i = 0; i < image.data.size(); ++i) {
    int val = image.data[i];
    minVal = std::min(minVal, val);
    maxVal = std::max(maxVal, val);
    sum += val;
  }

  double avg = static_cast<double>(sum) / image.data.size();

  std::cout << name << ": " << image.width << "x" << image.height << "x"
            << image.channels << ", min=" << minVal << ", max=" << maxVal
            << ", avg=" << avg << std::endl;
}

int main() {
  std::cout << "=== GPU Image Processing Library Demo ===" << std::endl;
  std::cout << "Version: " << getVersionString() << std::endl;

  // 检查 CUDA 可用性
  if (!isCudaAvailable()) {
    std::cerr << "Error: CUDA is not available!" << std::endl;
    return 1;
  }

  std::cout << getDeviceInfo() << std::endl;
  std::cout << std::endl;

  try {
    ImageProcessor processor;

    // 创建测试图像
    const int width = 256;
    const int height = 256;
    const int channels = 3;

    std::cout << "Creating test image..." << std::endl;
    HostImage testImage = createTestImage(width, height, channels);
    printImageStats(testImage, "Original");

    // 上传到 GPU
    std::cout << "\nUploading to GPU..." << std::endl;
    GpuImage gpuImage = processor.loadFromHost(testImage);

    // 1. 反色操作
    std::cout << "\n--- Invert Operation ---" << std::endl;
    GpuImage inverted = processor.invert(gpuImage);
    HostImage invertedHost = processor.download(inverted);
    printImageStats(invertedHost, "Inverted");

    // 验证反色：再次反色应该恢复原图
    GpuImage doubleInverted = processor.invert(inverted);
    HostImage doubleInvertedHost = processor.download(doubleInverted);
    printImageStats(doubleInvertedHost, "Double Inverted");

    // 2. 灰度化
    std::cout << "\n--- Grayscale Conversion ---" << std::endl;
    GpuImage grayscale = processor.toGrayscale(gpuImage);
    HostImage grayscaleHost = processor.download(grayscale);
    printImageStats(grayscaleHost, "Grayscale");

    // 3. 亮度调整
    std::cout << "\n--- Brightness Adjustment ---" << std::endl;
    GpuImage brighter = processor.adjustBrightness(gpuImage, 50);
    HostImage brighterHost = processor.download(brighter);
    printImageStats(brighterHost, "Brighter (+50)");

    GpuImage darker = processor.adjustBrightness(gpuImage, -50);
    HostImage darkerHost = processor.download(darker);
    printImageStats(darkerHost, "Darker (-50)");

    // 4. 高斯模糊
    std::cout << "\n--- Gaussian Blur ---" << std::endl;
    GpuImage blurred = processor.gaussianBlur(gpuImage, 5, 1.5f);
    HostImage blurredHost = processor.download(blurred);
    printImageStats(blurredHost, "Blurred (5x5, sigma=1.5)");

    // 5. Sobel 边缘检测
    std::cout << "\n--- Sobel Edge Detection ---" << std::endl;
    GpuImage edges = processor.sobelEdgeDetection(gpuImage);
    HostImage edgesHost = processor.download(edges);
    printImageStats(edgesHost, "Edges");

    // 6. 直方图计算
    std::cout << "\n--- Histogram Calculation ---" << std::endl;
    auto histogram = processor.histogram(grayscale);

    // 打印直方图摘要
    int totalPixels = 0;
    int nonZeroBins = 0;
    for (int i = 0; i < 256; ++i) {
      totalPixels += histogram[i];
      if (histogram[i] > 0)
        nonZeroBins++;
    }
    std::cout << "Histogram: " << nonZeroBins << " non-zero bins, "
              << totalPixels << " total pixels" << std::endl;

    // 7. 直方图均衡化
    std::cout << "\n--- Histogram Equalization ---" << std::endl;
    GpuImage equalized = processor.histogramEqualize(grayscale);
    HostImage equalizedHost = processor.download(equalized);
    printImageStats(equalizedHost, "Equalized");

    // 8. 图像缩放
    std::cout << "\n--- Image Resize ---" << std::endl;
    GpuImage resizedUp = processor.resize(gpuImage, 512, 512);
    HostImage resizedUpHost = processor.download(resizedUp);
    printImageStats(resizedUpHost, "Resized (512x512)");

    GpuImage resizedDown = processor.resize(gpuImage, 128, 128);
    HostImage resizedDownHost = processor.download(resizedDown);
    printImageStats(resizedDownHost, "Resized (128x128)");

    std::cout << "\n=== Demo Complete ===" << std::endl;

  } catch (const CudaException& e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
