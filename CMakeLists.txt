cmake_minimum_required(VERSION 3.18)
project(cudaimg
    VERSION 1.0.0
    DESCRIPTION "CUDA 图像处理入门教学项目"
    LANGUAGES CXX
)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA toolkit with nvcc is required to build cudaimg. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH.")
endif()
enable_language(CUDA)

# ── C++ / CUDA 标准 ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── CUDA 配置 ────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# 自动检测 GPU 架构（CMake 3.24+），否则使用常见架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
endif()

# ── 输出目录 ─────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ── 主库 ─────────────────────────────────────────────────────────
add_library(cudaimg STATIC
    # Core infrastructure
    src/core/cuda_error.cpp
    src/core/device_buffer.cu
    src/core/image_utils.cpp
    src/core/execution_context.cpp
    # Operators
    src/operators/pixel_operator.cu
    src/operators/convolution_engine.cu
    src/operators/histogram_calculator.cu
    src/operators/image_resizer.cu
    # Processing
    src/processing/image_processor.cpp
    src/processing/pipeline_processor.cu
    # More Operators
    src/operators/morphology.cu
    src/operators/threshold.cu
    src/operators/color_space.cu
    src/operators/geometric.cu
    src/operators/filters.cu
    # I/O
    src/io/image_io.cpp
)

add_library(cudaimg::cudaimg ALIAS cudaimg)

target_include_directories(cudaimg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaimg PUBLIC CUDA::cudart)

target_compile_options(cudaimg
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# ── 可选：stb 图像 I/O ──────────────────────────────────────────
option(CUDAIMG_ENABLE_IO "Enable image file I/O via stb" ON)
if(CUDAIMG_ENABLE_IO)
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        # stb master HEAD（2026-07 核实存在；旧 tag ae721c54 在上游不存在，导致配置失败）
        GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
        GIT_SHALLOW ON
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
    )
    FetchContent_MakeAvailable(stb)
    target_include_directories(cudaimg PRIVATE ${stb_SOURCE_DIR})
    target_compile_definitions(cudaimg PRIVATE CUDAIMG_HAS_STB)
endif()

# 版本宏注入
target_compile_definitions(cudaimg PUBLIC
    CUDAIMG_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CUDAIMG_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CUDAIMG_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# ── 测试配置 ─────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW ON
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    add_executable(cudaimg_tests
        tests/core/test_main.cpp
        tests/core/test_device_buffer.cpp
        tests/core/test_execution_context.cpp
        tests/operators/test_pixel_operator.cpp
        tests/operators/test_convolution.cpp
        tests/operators/test_histogram.cpp
        tests/operators/test_resizer.cpp
        tests/processing/test_pipeline.cpp
        tests/operators/test_morphology.cpp
        tests/operators/test_threshold.cpp
        tests/operators/test_color_space.cpp
        tests/operators/test_geometric.cpp
        tests/operators/test_filters.cpp
        tests/io/test_io.cpp
    )
    
    # 注意：不要链接 GTest::gtest_main —— tests/core/test_main.cpp 已提供 main，
    # 同时链接会导致 main 重复定义、链接失败。
    target_link_libraries(cudaimg_tests
        PRIVATE
            cudaimg
            GTest::gtest
    )
    
    include(GoogleTest)
    gtest_discover_tests(cudaimg_tests)
endif()

# ── 示例程序 ─────────────────────────────────────────────────────
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_executable(example_01_pixel examples/01_pixel.cu)
    target_link_libraries(example_01_pixel PRIVATE cudaimg)

    add_executable(example_02_convolution examples/02_convolution.cu)
    target_link_libraries(example_02_convolution PRIVATE cudaimg)

    add_executable(example_03_histogram examples/03_histogram.cu)
    target_link_libraries(example_03_histogram PRIVATE cudaimg)

    add_executable(pipeline_example examples/pipeline_example.cpp)
    target_link_libraries(pipeline_example PRIVATE cudaimg)
endif()

# ── 基准测试 ─────────────────────────────────────────────────────
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    # 基准测试为自包含的手写计时框架，无需外部 benchmark 依赖
    add_executable(cudaimg_benchmark benchmarks/benchmark_main.cpp)
    target_link_libraries(cudaimg_benchmark PRIVATE cudaimg)
endif()
