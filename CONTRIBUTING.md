# 贡献指南

感谢你对 Mini-OpenCV 的兴趣！本文档说明如何参与贡献。

---

## 准备环境

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器（GCC、Clang 或 MSVC）
- NVIDIA GPU（计算能力 7.5+）
- Git

```bash
# Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/mini-opencv.git
cd mini-opencv

# 添加上游远程
git remote add upstream https://github.com/AICL-Lab/mini-opencv.git

# 构建
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)

# 运行测试
ctest --test-dir build --output-on-failure
```

---

## 开发流程

1. 基于最新 master 创建分支：`git checkout -b feature/your-feature-name`
2. 编写代码，为新功能添加测试，按需更新文档
3. 本地验证：构建、测试、格式化
4. 提交（遵循下面的提交信息规范）
5. 推送并创建 PR

---

## 代码风格

- 使用 `.editorconfig` 和 `.clang-format` 保持格式一致
- 文件名 `snake_case`，类/结构体/枚举 `PascalCase`，函数/方法 `lowerCamelCase`
- 私有成员尾随下划线（`devicePtr_`）
- 头文件用 `#pragma once`
- 代码置于 `namespace gpu_image`

### API 设计

- 单参数构造函数加 `explicit`
- 不可忽略返回值的访问器加 `[[nodiscard]]`
- 平凡访问器和移动操作加 `noexcept`
- 优先用 `enum class`
- 只读重型输入用 `const T&`

### 错误处理

- 在公共函数顶部校验输入
- 错误的调用者输入用 `std::invalid_argument`
- 运行时失败用 `std::runtime_error`
- CUDA 调用用 `CUDA_CHECK(...)` 包裹

### CUDA 规则

- 内核启动后检查 `cudaGetLastError()`
- 支持异步的算子接受 `cudaStream_t stream = nullptr`
- 计算密集型内核使用 shared memory 优化
- 注意合并内存访问模式

---

## 格式化

```bash
# 检查
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -print0 | xargs -0 -r clang-format --dry-run --Werror

# 原地修复
find . -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) \
  -not -path './build/*' -print0 | xargs -0 -r clang-format -i
```

---

## 提交信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <subject>

<body>

<footer>
```

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 仅文档变更 |
| `style` | 不影响代码语义的格式变更 |
| `refactor` | 既非修 Bug 也非加功能的重构 |
| `perf` | 提升性能的变更 |
| `test` | 新增或更新测试 |
| `chore` | 构建过程或辅助工具变更 |
| `ci` | CI 配置变更 |

示例：

```
feat(convolution): add bilateral filter operator

Implements bilateral filter for edge-preserving smoothing with
configurable spatial and range parameters.

Closes #123
```

---

## 测试

测试位于 `tests/` 目录，使用 Google Test 框架。

```cpp
#include <gtest/gtest.h>
#include "gpu_image/gpu_image_processing.hpp"

TEST(PixelOperatorTest, Invert) {
    ImageProcessor processor;
    HostImage input = /* 创建测试图像 */;
    GpuImage gpuInput = processor.loadFromHost(input);
    GpuImage result = processor.invert(gpuInput);
    HostImage output = processor.download(result);
    EXPECT_EQ(output.width, input.width);
}

TEST(PixelOperatorTest, InvalidInput) {
    ImageProcessor processor;
    HostImage empty = {0, 0, 0, {}};
    EXPECT_THROW(processor.loadFromHost(empty), std::invalid_argument);
}
```

要求：
- 为所有新公共 API 编写单元测试
- 覆盖边界情况（空输入、最小/最大值）和错误条件
- 确保现有测试仍通过

---

## PR 自检清单

提交 PR 前请确认：

- [ ] 代码遵循项目风格
- [ ] 全部测试通过（`ctest --output-on-failure`）
- [ ] 已用 `clang-format` 格式化
- [ ] 文档已同步更新
- [ ] 提交信息符合规范
- [ ] PR 描述清晰说明改动
- [ ] 关联了相关 Issue

---

## 报告问题

- Bug 报告：使用 [Bug 报告模板](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/bug_report.md)
- 功能建议：使用[功能建议模板](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/feature_request.md)
- 文档问题：使用[文档问题模板](https://github.com/AICL-Lab/mini-opencv/blob/main/.github/ISSUE_TEMPLATE/documentation.md)

---

## 获取帮助

- 查看[文档](https://aicl-lab.github.io/mini-opencv/)
- 搜索[现有 Issue](https://github.com/AICL-Lab/mini-opencv/issues)
- 在 [GitHub Discussions](https://github.com/AICL-Lab/mini-opencv/discussions) 提问

---

感谢你为 Mini-OpenCV 贡献力量！
