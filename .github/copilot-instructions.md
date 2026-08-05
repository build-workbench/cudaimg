# GitHub Copilot Instructions

Project-specific instructions for GitHub Copilot when working in the mini-opencv repository.

## Project Overview

**Mini-OpenCV** is a CUDA-based high-performance image processing library providing GPU-accelerated operators for computer vision applications. It delivers 30-50x faster performance than CPU OpenCV.

- **Version:** 3.0.0
- **License:** MIT
- **Docs:** https://aicl-lab.github.io/mini-opencv/

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C++17, CUDA 14 |
| Build System | CMake 3.18+ |
| CUDA | 11.0+ (Recommended: 12.x) |
| GPU | Compute Capability 7.5+ (Turing or newer) |
| Testing | Google Test v1.14.0 |
| Benchmarking | Google Benchmark v1.8.3 |
| Image I/O | stb (via CMake FetchContent) |
| Documentation | VitePress (bilingual EN/ZH) |

## Architecture

Three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│         ImageProcessor  ·  PipelineProcessor                 │
├─────────────────────────────────────────────────────────────┤
│              Operator Layer (CUDA Kernels)                   │
│  PixelOperator  │  ConvolutionEngine  │  Geometric          │
│  Morphology     │  ColorSpace         │  Filters            │
│  Threshold      │  HistogramCalculator│  ImageResizer       │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  DeviceBuffer  ·  GpuImage/HostImage  ·  CudaError          │
│  ImageIO       ·  StreamManager                              │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mini-opencv/
├── include/gpu_image/          # Public headers
│   ├── core/                   # DeviceBuffer, GpuImage, CudaError, memory_manager
│   ├── operators/              # CUDA operator interfaces
│   ├── processing/             # ImageProcessor, PipelineProcessor
│   ├── io/                     # ImageIO
│   └── gpu_image_processing.hpp
├── src/                        # Implementations (.cpp / .cu)
├── tests/                      # Google Test suite
├── examples/                   # Example programs
├── benchmarks/                 # Performance benchmarks
├── docs/                       # VitePress documentation (EN + ZH)
└── .github/workflows/          # CI/CD
```

## Build Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Build options: `BUILD_TESTS` (ON), `BUILD_EXAMPLES` (ON), `BUILD_BENCHMARKS` (OFF), `GPU_IMAGE_ENABLE_IO` (ON).

## Test Commands

```bash
# Full suite via CTest
ctest --test-dir build --output-on-failure

# Run one fixture directly
./build/bin/gpu_image_tests --gtest_filter=FiltersTest.*

# Run one test
./build/bin/gpu_image_tests --gtest_filter=FiltersTest.MedianFilter
```

## Code Style

- **Indentation**: 2 spaces, 80-column limit (`.clang-format` is authoritative)
- **Naming**:
  - Files: `snake_case.hpp/.cpp/.cu`
  - Classes/Structs/Enums: `PascalCase`
  - Functions/Methods: `lowerCamelCase`
  - Private members: `snake_case_` (trailing underscore)
- **Namespace**: `gpu_image`
- **Header guard**: `#pragma once`

## CUDA Kernel Rules

1. Always check boundaries: `if (x < width && y < height)`
2. Thread block size: 256 threads (16×16 for 2D)
3. Call `CUDA_CHECK(cudaGetLastError())` after kernel launch
4. Async-capable operators accept `cudaStream_t stream = nullptr`

## Testing Conventions

- Use `TEST_F(...)` fixtures; check `cudaGetDeviceCount()` in `SetUp()` and call `GTEST_SKIP()` when CUDA is unavailable
- Prefer deterministic inputs and explicit expected values
- Use `EXPECT_THROW` for invalid-input tests
- Register new test files in the `gpu_image_tests` CMake target

## API Design

- `explicit` on single-argument constructors
- `[[nodiscard]]` on accessors whose result must not be ignored
- `noexcept` on trivial accessors and move operations
- `enum class` over unscoped enums
- `const T&` for read-only heavy inputs
- `int` for dimensions/channels, `size_t` for byte counts, `unsigned char` for pixels

## Error Handling

- Validate inputs at the top of every public function
- `std::invalid_argument` for bad caller input
- `std::runtime_error` for runtime failures
- Wrap all CUDA runtime calls with `CUDA_CHECK(...)`
- Low-level operators check `cudaGetLastError()`; high-level wrappers call `cudaDeviceSynchronize()` before returning

## Formatting

```bash
# Check (CI-equivalent)
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './third_party/*' -print0 | \
  xargs -0 -r clang-format-14 --dry-run --Werror

# Fix in-place
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -not -path './third_party/*' -print0 | \
  xargs -0 -r clang-format-14 -i
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <subject>`.

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`.
