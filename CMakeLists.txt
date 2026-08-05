cmake_minimum_required(VERSION 3.18)
project(gpu_image_processing
    VERSION 3.0.0
    DESCRIPTION "CUDA-based high-performance image processing library"
    LANGUAGES CXX
)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA toolkit with nvcc is required to build mini-opencv. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH.")
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
add_library(gpu_image_processing STATIC
    # Core infrastructure
    src/core/cuda_error.cpp
    src/core/device_buffer.cu
    src/core/memory_manager.cpp
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

add_library(gpu_image::gpu_image_processing ALIAS gpu_image_processing)

target_include_directories(gpu_image_processing
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(gpu_image_processing PUBLIC CUDA::cudart)

target_compile_options(gpu_image_processing
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# ── 可选：stb 图像 I/O ──────────────────────────────────────────
option(GPU_IMAGE_ENABLE_IO "Enable image file I/O via stb" ON)
if(GPU_IMAGE_ENABLE_IO)
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
    target_include_directories(gpu_image_processing PRIVATE ${stb_SOURCE_DIR})
    target_compile_definitions(gpu_image_processing PRIVATE GPU_IMAGE_HAS_STB)
endif()

# 版本宏注入
target_compile_definitions(gpu_image_processing PUBLIC
    GPU_IMAGE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GPU_IMAGE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GPU_IMAGE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
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
    
    add_executable(gpu_image_tests
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
    target_link_libraries(gpu_image_tests
        PRIVATE
            gpu_image_processing
            GTest::gtest
    )
    
    include(GoogleTest)
    gtest_discover_tests(gpu_image_tests)
endif()

# ── 示例程序 ─────────────────────────────────────────────────────
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_executable(basic_example examples/basic_example.cpp)
    target_link_libraries(basic_example PRIVATE gpu_image_processing)
    
    add_executable(pipeline_example examples/pipeline_example.cpp)
    target_link_libraries(pipeline_example PRIVATE gpu_image_processing)
endif()

# ── 基准测试 ─────────────────────────────────────────────────────
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    # 基准测试为自包含的手写计时框架，无需外部 benchmark 依赖
    add_executable(gpu_image_benchmark benchmarks/benchmark_main.cpp)
    target_link_libraries(gpu_image_benchmark PRIVATE gpu_image_processing)
endif()

# ── Install ──────────────────────────────────────────────────────
# 默认关闭：业余项目多数不需要 `cmake --install` 生成下游可 find_package 的配置。
# 需要 `find_package(gpu_image_processing)` 的下游，请以
# `-DGPU_IMAGE_ENABLE_INSTALL=ON` 配置，再 `cmake --install build`。
option(GPU_IMAGE_ENABLE_INSTALL "Generate install targets and package config" OFF)
if(GPU_IMAGE_ENABLE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS gpu_image_processing
        EXPORT gpu_image_processing-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(DIRECTORY include/gpu_image
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # 包配置文件：声明对 CUDAToolkit 的依赖（库 PUBLIC 链接 CUDA::cudart），
    # 否则下游 find_package(gpu_image_processing) 会因找不到 CUDA::cudart 失败。
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gpu_image_processing-config.cmake
        "include(CMakeFindDependencyMacro)\n"
        "find_dependency(CUDAToolkit)\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/gpu_image_processing-targets.cmake\")\n"
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/gpu_image_processing-config-version.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/gpu_image_processing-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/gpu_image_processing-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpu_image_processing
    )

    install(EXPORT gpu_image_processing-targets
        FILE gpu_image_processing-targets.cmake
        NAMESPACE gpu_image::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpu_image_processing
    )
endif()
