cmake_minimum_required(VERSION 3.10)
project(RayTracerHW2)

# Build option: Defaults to CPU (OFF)
option(ENABLE_GPU "Enable CUDA implementation" OFF)

# Define source files
set(SOURCES
    src/main.cu
    include/bvh.cu
    include/query.cu
)

# Include directories
# Matches: -I ../third_party/glm/ 
include_directories(${CMAKE_SOURCE_DIR}/third_party/glm)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

if(ENABLE_GPU)
    message(STATUS "Building for GPU (CUDA Enabled)")
    
    enable_language(CUDA)
    
    add_executable(bvh_viz ${SOURCES})
    
    # Add CUDA specific flags (extended-lambda, relaxed-constexpr, fast-math)
    target_compile_options(bvh_viz PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;--use_fast_math>)
    
    # Ensure C++11 or higher (CUDA usually supports C++14/17 in recent versions)
    set_target_properties(bvh_viz PROPERTIES CUDA_STANDARD 14)

else()
    message(STATUS "Building for CPU (Default)")
    
    enable_language(CXX)
    
    # Tell CMake to treat .cu files as C++ sources
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    
    add_executable(bvh_viz ${SOURCES})
    
    # Force GCC/Clang to treat .cu files as C++
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bvh_viz PRIVATE -x c++)
    endif()
    
    set_target_properties(bvh_viz PROPERTIES CXX_STANDARD 14)
endif()
