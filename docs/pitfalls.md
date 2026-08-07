# 坑点记录

本项目在开发过程中遇到的设计权衡和容易踩的坑。这些记录的教学价值
往往高于完美代码本身--初学者最容易在这些地方犯错。

---

## 1. 内存池多 stream 并发数据竞争

**位置**：`include/cudaimg/core/memory_manager.hpp`

内存池回收缓冲区时不检查该缓冲区所属 stream 是否已完成。如果多个线程
各自持有不同 stream 并并发 allocate/deallocate，某条 stream 上仍在读
的缓冲可能被另一条 stream 当作输出复用，导致数据竞争。

**当前处理**：内存池默认关闭（`poolEnabled_` 默认 `false`）。安全用法
是单 stream，或在回收前由调用方自行同步所属 stream。

**教学价值**：这是 CUDA 内存池设计的经典难题。生产级实现（如 CUDA
Stream-Ordered Memory Allocator `cudaMallocAsync`）通过 stream 亲和性
解决此问题。

---

## 2. 卷积核大小硬编码 7×7 上限

**位置**：`src/operators/convolution_engine.cu` - `KernelData` 结构体

```cpp
struct KernelData {
  float values[49] = {0.0f};  // 7×7 = 49
};
```

`KernelData` 通过值传递给 kernel，大小在编译时固定。超过 7×7 会抛异常。

**原因**：
- shared memory 预算有限
- 教学简洁性：3×3、5×5、7×7 已覆盖常见用例
- 超过 7×7 应改用可分离卷积（O(n²k²) -> O(n²k)）

**教学价值**：CUDA kernel 不能直接接受动态大小的数组参数。传递可变
大小数据到 kernel 需要 `cudaMemcpy` 到 device 内存或使用动态 shared
memory。这里用固定大小结构体是最简单的教学方案。

---

## 3. 中值滤波使用冒泡排序

**位置**：`src/operators/filters.cu` - `medianFilterKernel`

```cpp
// 冒泡排序找中值
for (int i = 0; i <= count / 2; ++i) {
  for (int j = i + 1; j < count; ++j) {
    if (window[j] < window[i]) { /* swap */ }
  }
}
```

冒泡排序时间复杂度 O(k²)，对于 7×7 窗口（49 个元素）需要 ~1225 次比较。

**生产级替代方案**：
- 排序网络（sorting network）：固定比较模式，可完全并行化
- 直方图法：对 256 级灰度做局部直方图，找中值只需累积计数
- 共享内存 + 并行归约

**教学价值**：这是 "正确但不高效" 的典型例子。初学者应理解为什么
冒泡排序在 GPU 上表现差（分支发散、串行依赖），以及替代方案如何
利用 GPU 的并行性。

---

## 4. HostImage::at() 的负索引 UB 修复

**位置**：`include/cudaimg/core/image.hpp` - `checkBounds()`

早期版本 `checkBounds` 未检查负索引。由于索引计算使用 `size_t`（无符号），
负值会转换为巨大的正数，导致越界读写 UB（未定义行为）。

```cpp
// 修复前：负索引经 size_t 转换变成巨值 -> UB
// 修复后：
void checkBounds(int x, int y, int c) const {
  if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c >= channels) {
    throw std::out_of_range("CudaImage::at: index out of range");
  }
}
```

**教学价值**：C++ 中 `int` 到 `size_t` 的隐式转换是经典的坑。CUDA
编程中 kernel 内部的坐标计算如果不做边界检查，同样会导致越界访问。

---

## 5. Sobel 输出为互相关而非卷积

**位置**：`include/cudaimg/operators/convolution_engine.hpp` - `convolve()` 注释

`convolve()` 的实现是互相关（不翻转卷积核），不是严格的数学卷积。
对称核（如高斯）结果一致；方向性核（如 Sobel）梯度符号相反，但幅值不受影响。

```cpp
// 注意：实现为互相关（不翻转卷积核）。对称核结果与卷积一致；
// 方向性核梯度符号相反，但幅值不受影响。
```

**教学价值**：很多图像处理库的 "卷积" 实际都是互相关。理解两者的
区别有助于在梯度方向敏感的场景（如光流、HOG 特征）中避免错误。

---

## 6. 自适应阈值未使用 integral image

**位置**：`src/operators/threshold.cu` - `adaptiveThresholdMeanKernel`

当前实现为每个像素遍历 blockSize×blockSize 窗口计算局部均值，
复杂度 O(n²k²)。生产级实现应使用积分图（integral image），
将局部均值降到 O(1)。

**教学价值**：这是算法选择影响性能的另一个例子。积分图是图像处理中
的经典数据结构，适合作为进阶练习实现。

---

## 7. CI 中 GPU 测试会跳过

**位置**：`.github/workflows/ci.yml`

GitHub Actions 的 `ubuntu-latest` runner 没有 GPU。GPU 测试在运行时
通过 `cudaGetDeviceCount` 检测，无设备时 `GTEST_SKIP()` 跳过。

CI 分两步：
1. CPU-only 测试（ImageIO）必须通过 -- 提供真实数值信号
2. 完整测试套件 -- GPU 测试跳过，仅作编译门禁

**教学价值**：CI 环境通常没有 GPU。将 CPU 可验证的部分单独分离出来
强制运行，是保证 CI 有意义的好实践。避免 "全部跳过 = 永远绿" 的假象。
