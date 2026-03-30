#ifndef TEXTURE_H
#define TEXTURE_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdio>

#include "vec3.h"
#include "imports.h"

// ============================================================
// TextureData — lightweight, GPU-copyable view (no ownership)
// This is what gets uploaded to device memory and used in kernels.
// ============================================================
struct TextureData {
    int width    = 0;
    int height   = 0;
    int channels = 0;
    const unsigned char* data = nullptr;  // pointer to pixel bytes (RGB or RGBA)

    // Sample the texture at (u, v) in [0,1]^2.  Returns linear-space RGB.
    // Uses bilinear filtering with wrap addressing.
    HYBRID_FUNC inline Vec3 sample(float u, float v) const {
        if (data == nullptr || width <= 0 || height <= 0) {
            return make_vec3(1.0f, 0.0f, 1.0f);  // magenta = missing texture
        }

        // Wrap UVs to [0, 1)
        u = u - floorf(u);
        v = v - floorf(v);

        // Flip V: OBJ convention has (0,0) at bottom-left, image has it at top-left
        v = 1.0f - v;

        // Pixel coordinates (continuous)
        float fx = u * (float)width  - 0.5f;
        float fy = v * (float)height - 0.5f;

        int x0 = (int)floorf(fx);
        int y0 = (int)floorf(fy);
        float dx = fx - (float)x0;
        float dy = fy - (float)y0;

        // Wrap pixel coords
        int x1 = (x0 + 1) % width;
        int y1 = (y0 + 1) % height;
        x0 = ((x0 % width) + width) % width;
        y0 = ((y0 % height) + height) % height;

        // Fetch four texels
        auto fetch = [&](int px, int py) -> Vec3 {
            int idx = (py * width + px) * channels;
            float r = (float)data[idx + 0] / 255.0f;
            float g = (float)data[idx + 1] / 255.0f;
            float b = (float)data[idx + 2] / 255.0f;
            return make_vec3(r, g, b);
        };

        Vec3 c00 = fetch(x0, y0);
        Vec3 c10 = fetch(x1, y0);
        Vec3 c01 = fetch(x0, y1);
        Vec3 c11 = fetch(x1, y1);

        // Bilinear interpolation
        Vec3 top    = c00 * (1.0f - dx) + c10 * dx;
        Vec3 bottom = c01 * (1.0f - dx) + c11 * dx;
        return top * (1.0f - dy) + bottom * dy;
    }

    // Sample a tangent-space normal map.  Returns a Vec3 in [-1, 1]^3.
    HYBRID_FUNC inline Vec3 sampleNormal(float u, float v) const {
        Vec3 rgb = sample(u, v);
        // Convert from [0,1] to [-1,1]
        return make_vec3(rgb.x * 2.0f - 1.0f,
                         rgb.y * 2.0f - 1.0f,
                         rgb.z * 2.0f - 1.0f);
    }
};


// ============================================================
// Texture — host-side owner of pixel data
// ============================================================
struct Texture {
    int width    = 0;
    int height   = 0;
    int channels = 0;                       // 3 or 4
    std::vector<unsigned char> data;         // interleaved RGB/RGBA, row-major

    TextureData sampled;                     // lightweight view for GPU

    void refreshSampledView() {
        sampled.width    = width;
        sampled.height   = height;
        sampled.channels = channels;
        sampled.data     = data.empty() ? nullptr : data.data();
    }
};

// Global texture cache (host side) — keyed by file path.
extern std::unordered_map<std::string, std::unique_ptr<Texture>> g_textureCache;

// ============================================================
// Texture loading helper (uses stb_image, call from host only)
// Returns a pointer to the cached Texture, or nullptr on failure.
// ============================================================
#ifndef STB_IMAGE_IMPLEMENTATION
// We only declare the function; stb_image.h must be included with
// STB_IMAGE_IMPLEMENTATION defined in exactly one .cpp / .cu file.
extern "C" unsigned char* stbi_load(const char*, int*, int*, int*, int);
extern "C" void stbi_image_free(void*);
#endif

inline Texture* LoadTexture(const std::string& path) {
    // Check cache first
    auto it = g_textureCache.find(path);
    if (it != g_textureCache.end()) {
        return it->second.get();
    }

    int w, h, c;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!pixels) {
        std::fprintf(stderr, "Warning: failed to load texture '%s'\n", path.c_str());
        return nullptr;
    }

    // Force to 3 or 4 channels (keep alpha if present)
    auto tex = std::make_unique<Texture>();
    tex->width    = w;
    tex->height   = h;
    tex->channels = c;  // 3 or 4 typically
    tex->data.assign(pixels, pixels + w * h * c);
    stbi_image_free(pixels);

    tex->refreshSampledView();

    Texture* ptr = tex.get();
    g_textureCache[path] = std::move(tex);
    return ptr;
}

#endif // TEXTURE_H