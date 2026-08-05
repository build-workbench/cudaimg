# Examples

Code examples for common use cases.

## Basic Operations

### Gaussian Blur

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

int main() {
    ImageProcessor processor;
    
    HostImage input = ImageIO::loadFromFile("input.jpg");
    GpuImage gpu = processor.loadFromHost(input);
    
    // Apply Gaussian blur with 5x5 kernel, sigma=1.5
    GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
    
    HostImage output = processor.download(blurred);
    ImageIO::saveToFile(output, "blurred.jpg");
}
```

### Edge Detection

```cpp
// Sobel edge detection
GpuImage edges = processor.sobelEdgeDetection(gpu);

// Convert to grayscale first for better results
GpuImage gray = processor.toGrayscale(gpu);
GpuImage grayEdges = processor.sobelEdgeDetection(gray);
```

### Resize

```cpp
// Resize to specific dimensions
GpuImage resized = processor.resize(gpu, 1920, 1080);

// Scale by factor
GpuImage doubled = processor.resizeByScale(gpu, 2.0f, 2.0f);
```

## Image Processing Pipeline

```cpp
// Multi-step processing
GpuImage gray = processor.toGrayscale(gpu);
GpuImage blurred = processor.gaussianBlur(gray, 5, 1.0f);
GpuImage edges = processor.sobelEdgeDetection(blurred);
```

## Batch Processing

```cpp
// Process multiple images
std::vector<std::string> files = {"img1.jpg", "img2.jpg", "img3.jpg"};

for (const auto& file : files) {
    HostImage input = ImageIO::loadFromFile(file);
    GpuImage gpu = processor.loadFromHost(input);
    GpuImage processed = processor.gaussianBlur(gpu, 5, 1.5f);
    HostImage output = processor.download(processed);
    
    ImageIO::saveToFile(output, "processed_" + file);
}
```

## Async / Multi-Stream Processing

```cpp
// PipelineProcessor owns a pool of CUDA streams
PipelineProcessor pipeline(4);  // 4 streams (default: 3)

// Steps receive the image and the stream assigned to it
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    GpuImage temp;
    ConvolutionEngine::gaussianBlur(img, temp, 5, 1.5f, stream);
    img = std::move(temp);
});
pipeline.addStep([](GpuImage& img, cudaStream_t stream) {
    PixelOperator::invertInPlace(img, stream);
});

// Images are round-robined across streams; synchronizes before returning
std::vector<HostImage> results = pipeline.processBatchHost(images);
```

## Morphological Operations

Morphology is available at the operator layer (not on the `ImageProcessor` facade):

```cpp
GpuImage eroded, dilated, opened, closed;
Morphology::erode(gpu, eroded, 5);    // 5x5 rectangle element
Morphology::dilate(gpu, dilated, 5);

// Opening (erosion followed by dilation)
Morphology::open(gpu, opened, 5);

// Closing (dilation followed by erosion)
Morphology::close(gpu, closed, 5);
```

## Color Space Conversion

Color conversion is also an operator-layer API:

```cpp
GpuImage hsv, yuv;
ColorSpace::rgbToHsv(gpu, hsv);
ColorSpace::rgbToYuv(gpu, yuv);
```

## Histogram Operations

```cpp
// Calculate histogram (256 bins, fixed-size array)
std::array<int, 256> hist = processor.histogram(gpu);

// Histogram equalization
GpuImage equalized = processor.histogramEqualize(gpu);
```

## More Examples

See the `examples/` directory in the repository for complete programs:

- `basic_example.cpp` - Pixel ops, convolution, histogram, and resize via `ImageProcessor`
- `pipeline_example.cpp` - Multi-stream batch processing with `PipelineProcessor`
