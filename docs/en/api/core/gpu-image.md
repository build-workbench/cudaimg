# GpuImage

GPU memory image container.

## Overview

`GpuImage` is a lightweight struct pairing a GPU memory buffer with image
metadata. Memory lifetime is managed by the embedded `DeviceBuffer` (RAII).

## Definition

```cpp
struct GpuImage {
    DeviceBuffer buffer;   // GPU memory
    int width = 0;         // Image width in pixels
    int height = 0;        // Image height in pixels
    int channels = 0;      // Number of color channels (1, 3, or 4)

    size_t pitch() const;       // Bytes per row (width * channels)
    size_t totalBytes() const;  // Total bytes (width * height * channels)
    bool isValid() const;       // True if buffer is valid and dimensions are set
    size_t pixelCount() const;  // Total pixels (width * height)
};
```

All fields are public — read `gpu.width`, `gpu.height`, `gpu.channels`
directly (they are fields, not accessor methods).

## Creation

GpuImage objects are typically created by `ImageProcessor`:

```cpp
ImageProcessor processor;
GpuImage gpu = processor.loadFromHost(hostImage);
```

## Memory Management

### RAII Pattern

Memory is automatically freed when `GpuImage` goes out of scope
(its `DeviceBuffer` member is released):

```cpp
{
    GpuImage gpu = processor.loadFromHost(host);
    // Use gpu...
}  // Memory freed here
```

### Move Semantics

`GpuImage` is movable; moving transfers buffer ownership:

```cpp
GpuImage gpu1 = processor.loadFromHost(host);
GpuImage gpu2 = std::move(gpu1);  // gpu1 no longer owns the buffer
```

## Usage with Operations

```cpp
GpuImage gpu = processor.loadFromHost(host);

// Operations return new GpuImage
GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);

// Download to host
HostImage result = processor.download(blurred);
```

## HostImage

The host-side counterpart `HostImage` stores pixel data in a `std::vector`:

```cpp
struct HostImage {
    std::vector<unsigned char> data;
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char& at(int x, int y, int c);  // Bounds-checked pixel access
    size_t totalBytes() const;
    bool isValid() const;
};
```

## See Also

- [DeviceBuffer](./device-buffer) - GPU memory management
- [ImageProcessor](./image-processor) - Main API
