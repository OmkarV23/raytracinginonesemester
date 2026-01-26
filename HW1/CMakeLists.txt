cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Place executables in the root of the project (HW1) so output.png appears there
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Option to enable GPU build (Default is OFF -> CPU only)
option(BUILD_GPU "Build with CUDA support" OFF)

if(BUILD_GPU)
    enable_language(CUDA OPTIONAL)
    
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "Building for GPU (CUDA detected)")
        add_executable(raytracer src/main.cu)
        
        set_target_properties(raytracer PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 11
            CUDA_STANDARD_REQUIRED ON
        )
    else()
        message(FATAL_ERROR "BUILD_GPU is ON but no CUDA compiler was found.")
    endif()

else()
    message(STATUS "Building for CPU")
    
    # CPU build: Use a symlink to trick CMake into treating main.cu as C++
    set(CPU_SRC_LINK ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)
    file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu ${CPU_SRC_LINK} SYMBOLIC)
    
    add_executable(raytracer ${CPU_SRC_LINK})
endif()

target_include_directories(raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
