#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <string>
#include <vector>
#include "vec3.h"
#include "texture.h"   // NEW;

struct Material {
    // Diffuse (Lambert)
    Vec3  albedo = make_vec3(0.8f, 0.8f, 0.8f);   // diffuse reflectance (rho)
    float kd     = 1.0f;                          // diffuse weight

    // Specular lobe (BRDF)
    Vec3  specularColor = make_vec3(0.04f, 0.04f, 0.04f);   // specular tint
    float ks            = 0.0f;                             // specular weight
    float shininess     = 32.0f;                            // Blinn-Phong exponent

    // Reflectance
    float kr            = 0.0f;

    // Emission (we will need to update this later, basic placeholder for now)
    Vec3  emission      = make_vec3(0.0f, 0.0f, 0.0f);


    std::string albedoMapPath;
    Texture* albedo_map = nullptr;

    std::string bumpMapPath;
    Texture* bump_map = nullptr;

    std::string normalMapPath;
    Texture* normal_map = nullptr;
};

// POD-like material payload used by CPU/GPU ray traversal and shading code.
// Keep this free of std::string so device code can safely copy/use it.
struct MaterialData {
    Vec3  albedo = make_vec3(0.8f, 0.8f, 0.8f);
    float kd     = 1.0f;

    Vec3  specularColor = make_vec3(0.04f, 0.04f, 0.04f);
    float ks            = 0.0f;
    float shininess     = 32.0f;

    float kr            = 0.0f;
    Vec3  emission      = make_vec3(0.0f, 0.0f, 0.0f);

    TextureData* albedo_map = nullptr;
    TextureData* bump_map   = nullptr;
    TextureData* normal_map = nullptr;
};

inline MaterialData ToMaterialData(const Material& m) {
    MaterialData out;
    out.albedo = m.albedo;
    out.kd = m.kd;
    out.specularColor = m.specularColor;
    out.ks = m.ks;
    out.shininess = m.shininess;
    out.kr = m.kr;
    out.emission = m.emission;
    out.albedo_map = m.albedo_map ? &m.albedo_map->sampled : nullptr;
    out.bump_map = m.bump_map ? &m.bump_map->sampled : nullptr;
    out.normal_map = m.normal_map ? &m.normal_map->sampled : nullptr;
    return out;
}

#endif