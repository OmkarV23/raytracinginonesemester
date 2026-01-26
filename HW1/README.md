# CUDA Ray Tracer

A simple ray tracer implementation that runs on both CPU (C++) and GPU (CUDA).

## Project Structure

```
HW1/
├── CMakeLists.txt       # Build configuration
├── src/
│   ├── main.cu          # Main entry point (CPU & GPU logic)
│   ├── raytracer.h      # Core ray tracing headers
│   └── stb_image_write.h # Image writing library
└── build/               # Build directory (generated)
```

## Prerequisites

*   **CMake** (3.18 or higher)
*   **C++ Compiler** (GCC/Clang/MSVC) with C++11 support
*   **NVIDIA CUDA Toolkit** (Optional, for GPU support)

## Building

1.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```

2.  Configure the project:
    *   **CPU Build (Default)**:
        ```bash
        cmake ..
        ```
    *   **GPU Build**:
        ```bash
        cmake .. -DBUILD_GPU=ON
        ```

3.  Build the target:
    ```bash
    cmake --build .
    ```

This will generate a single executable `raytracer` in the project root (`HW1/`).

## Usage

Run the executable from the project root. By default, it outputs `output.png`.

```bash
./raytracer
```

### Options

*   **-o <filename>**: Specify the output image filename.

```bash
./raytracer -o my_render.png
```

## Implementation Details

*   **Hybrid Code**: The `src/main.cu` file contains both CPU and GPU implementations.
    *   The CPU version uses standard C++ loops.
    *   The GPU version launches a CUDA kernel.
*   **Vector Library**: A unified vector header (`raytracer.h`) handles `float3` for CUDA and a custom struct for CPU.
*   **CMake Trick**: The CPU build uses a symbolic link (`main.cpp` -> `main.cu`) to trick CMake into compiling the CUDA source file with the standard C++ compiler, enabling a single-source hybrid codebase.
