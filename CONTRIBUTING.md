# Contributing to Mini-OpenCV

Thank you for your interest in contributing to Mini-OpenCV! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style and Guidelines](#code-style-and-guidelines)
- [Commit Message Format](#commit-message-format)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- CUDA Toolkit 11.0+ installed
- CMake 3.18+
- C++17 compatible compiler (GCC, Clang, or MSVC)
- NVIDIA GPU (Compute Capability 7.5+)
- Git

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/mini-opencv.git
cd mini-opencv

# Add upstream remote
git remote add upstream https://github.com/AICL-Lab/mini-opencv.git

# Build the project
mkdir build && cd build
cmake -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

---

## Development Workflow

### 1. Create a Branch

```bash
git fetch upstream
git checkout master
git merge upstream/master

# Create a new feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-commented code
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Build
cmake --build build -j$(nproc)

# Run tests
cd build
ctest --output-on-failure

# Run specific test
./bin/gpu_image_tests --gtest_filter=YourTestName

# Format check
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -print0 | xargs -0 -r clang-format-14 --dry-run --Werror

# Format code (if needed)
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -print0 | xargs -0 -r clang-format-14 -i
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new convolution operator"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub using the [PR template](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

---

## Code Style and Guidelines

### C++/CUDA Code Style

- Use `.editorconfig` and `.clang-format` for consistent formatting
- Use `snake_case` for filenames
- Use `PascalCase` for classes, structs, enums
- Use `lowerCamelCase` for functions and methods
- Use trailing underscores for private members (`devicePtr_`)
- Use `#pragma once` in headers
- Wrap code in `namespace gpu_image`

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | `snake_case` | `image_processor.cpp` |
| Classes/Structs | `PascalCase` | `DeviceBuffer`, `GpuImage` |
| Functions | `lowerCamelCase` | `copyFromHost`, `downloadImage` |
| Enum Types | `PascalCase` | `ThresholdType::Binary` |
| Enum Values | `PascalCase` | `ThresholdType::Otsu` |
| Private Members | `snake_case_` | `devicePtr_`, `size_` |

### API Design Guidelines

- Mark single-argument constructors `explicit`
- Use `[[nodiscard]]` for accessors whose result should not be ignored
- Use `noexcept` for trivial operations when appropriate
- Prefer `enum class` over unscoped enums
- Use `const T&` for read-only heavy inputs
- Use non-const reference output parameters for operator-style APIs

### Error Handling

- Validate inputs at the top of public functions
- Use `std::invalid_argument` for bad caller input
- Use `std::runtime_error` for runtime failures
- Wrap CUDA calls with `CUDA_CHECK(...)`
- Keep error messages short and specific

### CUDA Specific Guidelines

- Check `cudaGetLastError()` after kernel launches
- Async-capable operators accept `cudaStream_t stream = nullptr`
- Use shared memory optimization for compute-intensive kernels
- Consider coalesced memory access patterns
- Test on multiple GPU architectures when possible

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Changes that do not affect the code meaning (formatting, etc.) |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `perf` | A code change that improves performance |
| `test` | Adding or updating tests |
| `chore` | Changes to the build process or auxiliary tools |
| `ci` | Changes to CI configuration files |

### Examples

```bash
feat(convolution): add bilateral filter operator

Implements bilateral filter for edge-preserving smoothing with
configurable spatial and range parameters.

Closes #123
```

```bash
fix(geometric): correct bilinear interpolation boundary handling

Fixes issue where boundary pixels were incorrectly zeroed out
when using bilinear interpolation for scaling operations.
```

---

## Testing

### Writing Tests

Tests are located in the `tests/` directory and use Google Test framework.

```cpp
#include <gtest/gtest.h>
#include "gpu_image/gpu_image_processing.hpp"

TEST(PixelOperatorTest, Invert) {
    // Arrange
    ImageProcessor processor;
    HostImage input = /* create test image */;

    // Act
    GpuImage gpuInput = processor.loadFromHost(input);
    GpuImage result = processor.invert(gpuInput);
    HostImage output = processor.downloadImage(result);

    // Assert
    EXPECT_EQ(output.width, input.width);
    EXPECT_EQ(output.height, input.height);
    // Verify pixel values
}

TEST(PixelOperatorTest, InvalidInput) {
    ImageProcessor processor;
    HostImage empty = {0, 0, 0, {}};

    EXPECT_THROW(processor.loadFromHost(empty), std::invalid_argument);
}
```

### Test Coverage

- Write unit tests for all new public APIs
- Test edge cases (empty input, minimum/maximum values)
- Test error conditions (invalid parameters, out of memory)
- Ensure existing tests still pass
- Aim for >80% code coverage for new features

---

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass (`ctest --output-on-failure`)
- [ ] Code is formatted with `clang-format`
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages follow the convention
- [ ] PR description clearly explains the changes
- [ ] PR links to any related issues

### Review Process

1. Automated checks (CI, format) must pass
2. Maintainers review the code and provide feedback
3. Address any review comments
4. Once approved, the PR will be merged

---

## Reporting Issues

### Bug Reports

Use the [bug report template](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, CUDA version, GPU, etc.)
- Minimal reproducible example

### Feature Requests

Use the [feature request template](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Description of the feature
- Use case and why it's needed
- Proposed solution
- Alternatives considered
- Any API suggestions

### Documentation Issues

Use the [documentation template](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/documentation.md) and include:

- Location of the issue
- Current content (if applicable)
- Suggested improvement

---

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

### Getting Help

- Check the [documentation](https://aicl-lab.github.io/mini-opencv/)
- Search [existing issues](https://github.com/AICL-Lab/mini-opencv/issues)
- Join discussions in [GitHub Discussions](https://github.com/AICL-Lab/mini-opencv/discussions)
- Ask a new question or issue

---

## Additional Resources

- [Project README](README.md)
- [Documentation](https://aicl-lab.github.io/mini-opencv/)
- [ChangeLog](CHANGELOG.md)

---

Thank you for contributing to Mini-OpenCV! 🎉
