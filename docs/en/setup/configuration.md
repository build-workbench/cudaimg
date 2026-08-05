# Configuration

CMake configuration options for Mini-OpenCV.

## Build Types

| Type | Description |
|------|-------------|
| `Release` | Optimized build (default) |
| `Debug` | Debug symbols, no optimization |
| `RelWithDebInfo` | Release with debug info |

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## CMake Options

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `GPU_IMAGE_ENABLE_IO` | ON | Enable image file I/O (via stb) |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_TESTS` | ON | Build GoogleTest suite |
| `BUILD_BENCHMARKS` | OFF | Build performance benchmarks |

The library is always built as a static library — `BUILD_SHARED_LIBS` has no effect.

### CUDA Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_CUDA_ARCHITECTURES` | auto | GPU architectures (e.g., `"75;80;86;89"`) |

`CMAKE_CUDA_ARCHITECTURES` is a standard CMake variable. When it is not set,
the project defaults to `native` on CMake 3.24+ (auto-detects the local GPU),
otherwise `75;80;86;89`.

### Language Standards

| Scope | Standard |
|-------|----------|
| Host code (C++) | C++17 |
| Device code (CUDA) | C++14 (via `CMAKE_CUDA_STANDARD`) |

There is no "CUDA 14" — the "14" refers to the C++ standard used for device code.

## GPU Architecture

### Auto Detection

By default, CMake detects the installed GPU (CMake 3.24+):

```bash
cmake -S . -B build  # Auto-detect
```

### Manual Specification

```bash
# Single architecture
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89

# Multiple architectures
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

### Architecture Reference

| Architecture | Compute | GPUs |
|--------------|---------|------|
| Turing | 75 | RTX 20xx, T4 |
| Ampere | 80 | A100 |
| Ampere | 86 | RTX 30xx |
| Ada | 89 | RTX 40xx, L4 |
| Hopper | 90 | H100 |

## Example Configurations

### Development

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON
```

### Production

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF
```

### Benchmarking

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARKS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

## Next Steps

- [Quick Start](./quickstart) - Basic usage
- [Examples](../tutorials/examples) - Code examples
