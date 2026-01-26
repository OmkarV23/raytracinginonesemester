#include "raytracer.h"
#include <vector>
#include <iostream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#ifdef __CUDACC__

__global__ void render_kernel(Vec3* image_buffer, int width, int height, 
                              Vec3 origin, Vec3 lower_left, 
                              Vec3 horizontal, Vec3 vertical, Light light) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    // Normalized coordinates
    float u = float(i) / float(width);
    float v = float(j) / float(height);

    Ray r;
    r.origin = origin;
    r.dir = lower_left + horizontal * u + vertical * v - origin;

    HitRecord rec = world_intersect(r);
    Vec3 col = shade(r, rec, light);

    // Flatten index
    int pixel_index = (height - 1 - j) * width + i;
    image_buffer[pixel_index] = col;
}

void launch_renderer(std::vector<Vec3>& h_image, int width, int height,
                     Vec3 o, Vec3 ll, Vec3 h, Vec3 v, Light l) 
{
    // std::cout << "[GPU MODE] Allocating memory..." << std::endl;
    
    Vec3* d_image;
    size_t size = width * height * sizeof(Vec3);
    cudaMalloc((void**)&d_image, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // std::cout << "[GPU MODE] Launching Kernel..." << std::endl;
    render_kernel<<<numBlocks, threadsPerBlock>>>(d_image, width, height, o, ll, h, v, l);
    cudaDeviceSynchronize();
    
    // std::cout << "[GPU MODE] Copying back to CPU..." << std::endl;
    cudaMemcpy(h_image.data(), d_image, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
}

#else

void launch_renderer(std::vector<Vec3>& image, int width, int height,
                     Vec3 origin, Vec3 lower_left, Vec3 horizontal, Vec3 vertical, Light light) 
{
    // std::cout << "[CPU MODE] Starting sequential render..." << std::endl;
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            float u = float(i) / float(width);
            float v = float(j) / float(height);
            
            Ray r;
            r.origin = origin;
            r.dir = lower_left + horizontal * u + vertical * v - origin;

            HitRecord rec = world_intersect(r);
            Vec3 col = shade(r, rec, light);

            image[(height - 1 - j) * width + i] = col;
        }
    }
}

#endif


int main(int argc, char** argv) {
    std::string output_filename = "output.png";

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-o" && i + 1 < argc) {
            output_filename = argv[++i];
        }
    }

    int width = 2000;
    int height = 1000;
    std::vector<Vec3> image(width * height);

    Vec3 origin = make_vec3(0, 0, 0);
    Vec3 lower_left = make_vec3(-2.0f, -1.0f, -1.0f);
    Vec3 horizontal = make_vec3(4.0f, 0.0f, 0.0f);
    Vec3 vertical   = make_vec3(0.0f, 2.0f, 0.0f);

    Light light;
    light.position = make_vec3(2.0f, 2.0f, 1.0f);
    light.color = make_vec3(1.0f, 1.0f, 1.0f);

    launch_renderer(image, width, height, origin, lower_left, horizontal, vertical, light);

    // std::cout << "Saving output.png..." << std::endl;
    std::vector<unsigned char> png_data(width * height * 3);
    for (int k = 0; k < width * height; ++k) {
        png_data[k*3 + 0] = (unsigned char)(255.99f * image[k].x);
        png_data[k*3 + 1] = (unsigned char)(255.99f * image[k].y);
        png_data[k*3 + 2] = (unsigned char)(255.99f * image[k].z);
    }
    stbi_write_png(output_filename.c_str(), width, height, 3, png_data.data(), width * 3);
    
    // std::cout << "Done!" << std::endl;
    return 0;
}