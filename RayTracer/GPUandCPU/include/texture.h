#ifndef TEXTURE_H
#define TEXTURE_H

#include <memory>
#include "vec3.h"   // assumes vec3.h defines Vec3

struct TextureData {
    int width = 0;
    int height = 0;
    int channels = 0;
    const unsigned char* data = nullptr;
};

struct Texture {
    int width = 0;
    int height = 0;
    int channels = 0;              // 3 or 4
    std::vector<unsigned char> data; // interleaved RGB/RGBA, row-major
    TextureData sampled;

    void refreshSampledView() {
        sampled.width = width;
        sampled.height = height;
        sampled.channels = channels;
        sampled.data = data.empty() ? nullptr : data.data();
    }
};

extern std::unordered_map<std::string, std::unique_ptr<Texture>> g_textureCache;

#endif // TEXTURE_H