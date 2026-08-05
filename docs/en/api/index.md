# API Reference

Complete API documentation for Mini-OpenCV.

## Core Classes

| Class | Description |
|-------|-------------|
| [ImageProcessor](./core/image-processor) | Main entry point for image operations |
| [GpuImage](./core/gpu-image) | GPU memory image container |
| [DeviceBuffer](./core/device-buffer) | RAII GPU memory management |

## Operators

| Category | Operations |
|----------|------------|
| [Convolution](./operators/convolution) | Gaussian blur, Sobel, custom kernels |
| [Filters](./operators/filters) | Median, bilateral, box, sharpen |
| [Geometric](./operators/geometric) | Resize, rotate, flip, affine |
| [Morphology](./operators/morphology) | Erosion, dilation, open/close |

## Image I/O

All `ImageIO` functions are static.

| Function | Description |
|----------|-------------|
| `ImageIO::loadFromFile(filepath)` | Load image from file, returns `HostImage` |
| `ImageIO::saveToFile(image, filepath)` | Save image to file (image first, path second), returns `bool` |
| `ImageIO::loadFromMemory(data, size)` | Decode image from memory buffer |
| `ImageIO::encodeToMemory(image, format)` | Encode image to memory buffer |
| `ImageIO::getSupportedFormats()` | List readable formats |
| `ImageIO::getWritableFormats()` | List writable formats |
| `ImageIO::isFormatSupported(filepath)` | Check if a file format can be read |

Supported formats:

- Read: JPEG, PNG, BMP, TGA, PSD, GIF, HDR, PNM
- Write: JPEG, PNG, BMP, TGA

## Quick Reference

```cpp
#include "gpu_image/gpu_image_processing.hpp"
using namespace gpu_image;

// Create processor
ImageProcessor processor;

// Load and upload
HostImage host = ImageIO::loadFromFile("image.jpg");
GpuImage gpu = processor.loadFromHost(host);

// Operations
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(gpu);
GpuImage resized = processor.resize(gpu, 1920, 1080);

// Download and save
HostImage result = processor.download(gpu);
ImageIO::saveToFile(result, "output.jpg");
```
