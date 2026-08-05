# ImageProcessor

Main entry point for GPU image operations.

## Constructors

```cpp
ImageProcessor();
explicit ImageProcessor(Mode mode);
explicit ImageProcessor(ExecutionPolicy policy);
```

- `ImageProcessor()` creates a processor in default `Sync` mode.
- `Mode` is an alias of `ExecutionPolicy::Mode`: `Sync`, `Async`, `Batch`.
- Pass an `ExecutionPolicy` for full control (mode + stream).

## Configuration

```cpp
void setMemoryPooling(bool enabled);
bool isMemoryPoolingEnabled() const;
void setMode(Mode mode);
Mode mode() const;
```

## Image Transfer

### loadFromMemory

```cpp
GpuImage loadFromMemory(const unsigned char* data, int width, int height,
                        int channels);
```

Uploads raw pixel data to GPU memory.

### loadFromHost

```cpp
GpuImage loadFromHost(const HostImage& hostImage);
```

Uploads image data from host to GPU memory.

**Parameters:**
- `hostImage`: Host image data

**Returns:** `GpuImage` with data on GPU

### download

```cpp
HostImage download(const GpuImage& image);
```

Downloads image data from GPU to host memory.

**Parameters:**
- `image`: GPU image

**Returns:** `HostImage` with data on CPU

### downloadToBuffer

```cpp
void downloadToBuffer(const GpuImage& image, unsigned char* buffer,
                      size_t bufferSize);
```

Downloads image data into a pre-allocated host buffer.

## Pixel Operations

### invert

```cpp
GpuImage invert(const GpuImage& input);
```

Inverts image colors.

### toGrayscale

```cpp
GpuImage toGrayscale(const GpuImage& input);
```

Converts RGB image to grayscale.

### adjustBrightness

```cpp
GpuImage adjustBrightness(const GpuImage& input, int offset);
```

Adjusts image brightness by adding `offset` to each channel value
(clamped to 0–255). The parameter is an integer offset, not a factor.

### In-place variants

```cpp
void invertInPlace(GpuImage& image);
void adjustBrightnessInPlace(GpuImage& image, int offset);
```

Modify the image in place instead of returning a new one.

## Convolution

### gaussianBlur

```cpp
GpuImage gaussianBlur(const GpuImage& input, int kernelSize = 5,
                      float sigma = 1.0f);
```

Applies Gaussian blur.

**Parameters:**
- `kernelSize`: Must be odd, ≤ 7 (throws `std::invalid_argument` otherwise)
- `sigma`: Standard deviation, >0

### sobelEdgeDetection

```cpp
GpuImage sobelEdgeDetection(const GpuImage& input);
```

Detects edges using Sobel operator.

### convolve

```cpp
GpuImage convolve(const GpuImage& input, const float* kernel, int kernelSize);
```

Applies a custom convolution kernel (`kernelSize × kernelSize` weights,
odd, ≤ 7).

## Histogram

### histogram

```cpp
std::array<int, 256> histogram(const GpuImage& input);
```

Computes a grayscale histogram (256 bins).

### histogramRGB

```cpp
std::array<std::array<int, 256>, 3> histogramRGB(const GpuImage& input);
```

Computes per-channel RGB histograms.

### histogramEqualize

```cpp
GpuImage histogramEqualize(const GpuImage& input);
```

Applies histogram equalization.

## Geometric

### resize

```cpp
GpuImage resize(const GpuImage& input, int newWidth, int newHeight);
```

Resizes image using bilinear interpolation.

### resizeByScale

```cpp
GpuImage resizeByScale(const GpuImage& input, float scaleX, float scaleY);
```

Resizes image by scale factors.

## Synchronization

```cpp
void synchronize();        // Block until async/batch operations complete
bool isComplete() const;   // Non-blocking completion check
```

## Example

```cpp
ImageProcessor processor;

HostImage input = ImageIO::loadFromFile("input.jpg");
GpuImage gpu = processor.loadFromHost(input);

GpuImage blurred = processor.gaussianBlur(gpu, 5, 1.5f);
GpuImage edges = processor.sobelEdgeDetection(blurred);
GpuImage gray = processor.toGrayscale(gpu);

HostImage output = processor.download(edges);
ImageIO::saveToFile(output, "output.jpg");
```
