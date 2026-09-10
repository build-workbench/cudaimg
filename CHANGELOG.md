# 更新日志

cudaimg 是基于现代 C++17 与 CUDA 的轻量级 GPU 图像处理练手项目，以常见图像算子为载体实践
GPU 核心编程，并为每个 kernel 提供逐像素对比的纯 CPU 参考实现。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- 从零实现完整 GPU 图像处理库：`DeviceBuffer` / `ExecutionContext` 以 RAII 管理显存与
  CUDA Stream，`ImageProcessor` 门面统一调度各算子。
- 点运算：标量 2D 线程映射与 `uchar4` 向量化反色，示范对齐向量化读写与降级处理。
- 卷积：Shared Memory Tiling（含 Halo 协作加载）卷积与可分离卷积，把复杂度从 O(K²) 降到
  O(2K)。
- 直方图：Block 局部直方图与 `atomicAdd` 两级规约合并，并实现直方图均衡化。
- 几何与采样：旋转 / 翻转 / 仿射变换 / 裁剪 / 填充，以及双线性插值缩放。
- 扩展算子：形态学（膨胀 / 腐蚀）、阈值（全局 / Otsu / 局部均值）、高斯 / 中值 / 双边等
  高级滤波、RGB ↔ 灰度 / HSV / YUV / Lab 色彩空间转换。
- 异步流水线：基于 `cudaStream_t` 的多流并发与批次同步提交；`ImageIO` 基于可选
  header-only `stb_image`。
- 为每个 GPU kernel 手写纯 CPU 黄金参考实现，并用 GoogleTest 做逐像素 / 逐通道断言
  （几何变换测试已扩展到全像素校验）。
- 拆分 `examples` 为 `01_pixel` / `02_convolution` / `03_histogram` 三个由浅入深的可运行
  示例，并保留性能基准 `benchmarks/benchmark_main.cpp`。
- 中文教学文档：`docs/learning-path.md`、`docs/cuda-concepts.md`、`docs/pitfalls.md` 等，
  记录技术演进与实际踩坑。

### 变更

- 项目定位由产品级图像库转型为个人 CUDA 练手 / 教学实验场，命名从 `gpu_image` 迁移为
  `cudaimg`。
- 抽取设备端共享工具 `device_kernels.cuh` 与 host 侧 `kernel_helpers`，示范消除 kernel
  样板代码。
- `ImageProcessor` 门面补全全部算子类别并统一返回 `GpuImage`；`convolve` / `resize` 开放
  插值与边界模式选择。
- 各算子采用统一的 `ensureOutputSize` 尺寸校验，减少重复代码并修正个别算子正确性。
- 构建减负：CMake 升级到 C++17，install 目标可选化，FetchContent 改为浅克隆，移除外部
  benchmark 依赖。
- 统一 clang-format 风格并精简中文化工程 / 协作文档，README 多次重写以对齐项目新定位。

### 修复

- 修复 `CudaException` 未定义行为、`MemoryManager` 显存池生命周期与 `DeviceBuffer::fromRaw` /
  `detach` 的语义问题。
- 加固核心错误处理、参数校验与整数溢出防护，修正 pipeline 批次校验与 resize 调用。

### 移除

- 删除 `MemoryManager` 显存池与 `concrete_operators` / `operator_pipeline` / `image_operator`
  等冗余抽象层。
- 删除旧 `gpu_image` 目录、产品级 artifacts（文档站与治理文件）、英文 README 与 `docs/en`。
- 移除对 GoogleTest 主函数与外部 benchmark 的依赖，以及自动依赖更新（dependabot）配置。
