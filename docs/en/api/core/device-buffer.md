# DeviceBuffer

RAII GPU memory management.

## Overview

`DeviceBuffer` provides automatic GPU memory allocation and deallocation using RAII pattern.

## Constructors

```cpp
DeviceBuffer() noexcept;             // Empty buffer (null pointer, size 0)
explicit DeviceBuffer(size_t size);  // Allocates `size` bytes on GPU
```

`fromRaw` takes ownership of an existing allocation (used by the memory pool):

```cpp
static DeviceBuffer fromRaw(void* ptr, size_t size) noexcept;
```

## Destructor

```cpp
~DeviceBuffer();
```

Automatically frees GPU memory (`cudaFree()`).

## Data Access

### data

```cpp
void* data() noexcept;
const void* data() const noexcept;
```

Returns pointer to GPU memory.

### dataAs

```cpp
template <typename T> T* dataAs() noexcept;
template <typename T> const T* dataAs() const noexcept;
```

Returns a typed pointer to GPU memory.

### size

```cpp
size_t size() const noexcept;
```

Returns allocated size in bytes.

### isValid

```cpp
bool isValid() const noexcept;
```

True if the buffer holds a non-null pointer with size > 0.

## Host/Device Transfers

```cpp
void copyFromHost(const void* hostPtr, size_t size);
void copyToHost(void* hostPtr, size_t size) const;
void copyFromDevice(const void* devicePtr, size_t size);  // Device-to-device

// Async variants on a given stream
void copyFromHostAsync(const void* hostPtr, size_t size, cudaStream_t stream);
void copyToHostAsync(void* hostPtr, size_t size, cudaStream_t stream) const;
```

## Ownership

```cpp
void release() noexcept;                     // Free memory (cudaFree)
std::pair<void*, size_t> detach() noexcept;  // Release ownership without freeing
```

## Move Semantics

```cpp
DeviceBuffer buf1(1024);
DeviceBuffer buf2 = std::move(buf1);  // buf1 is now empty
```

Copy is disabled to prevent accidental deep copies.

## Example

```cpp
// Allocate GPU memory
DeviceBuffer buffer(1920 * 1080 * 3);  // 1080p RGB image

// Use with CUDA kernel
myKernel<<<grid, block>>>(buffer.data());

// Memory automatically freed when buffer goes out of scope
```

## Internal Implementation

```cpp
class DeviceBuffer {
public:
    DeviceBuffer() noexcept;
    explicit DeviceBuffer(size_t size);
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    void* devicePtr_;
    size_t size_;
};
```

## See Also

- [GpuImage](./gpu-image) - Image container using DeviceBuffer
