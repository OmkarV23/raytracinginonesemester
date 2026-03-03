// brdf.h
#ifndef BRDF_H
#define BRDF_H

#include <cmath>
#include "vec3.h"
#include "material.h"

HYBRID_FUNC inline float saturate(float x) { return (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x); }

HYBRID_FUNC inline Vec3 sampleTexture(const TextureData* tex, Vec2 uv)
{
    if (!tex || tex->width == 0 || tex->data == nullptr) return {1.0f, 1.0f, 1.0f};

    float u = uv.x - floorf(uv.x);  // Wrap
    float v = uv.y - floorf(uv.y);
    
    int x = static_cast<int>(u * (tex->width - 1));
    int y = static_cast<int>((1.0f - v) * (tex->height - 1));  // Flip Y
    
    x = x < 0 ? 0 : (x > tex->width - 1 ? tex->width - 1 : x);
    y = y < 0 ? 0 : (y > tex->height - 1 ? tex->height - 1 : y);
    
    int idx = (y * tex->width + x) * tex->channels;
    const unsigned char* p = tex->data + idx;
    
    if (tex->channels >= 3) {
        return {
            p[0] / 255.0f,
            p[1] / 255.0f, 
            p[2] / 255.0f
        };
    }
    return {1.0f, 1.0f, 1.0f};
}






// Returns f(wo, wi) (does NOT include N·L)
HYBRID_FUNC inline Vec3 EvaluateBRDF(const HitRecord& rec,
                                    const Vec3& V,  // to viewer
                                    const Vec3& L,
                                const Vec3& Nshading)  // to light
{

    const MaterialData& m = rec.mat;
    Vec3 N = Nshading;
    Vec3 albedo = m.albedo;


    if (m.albedo_map) {
        albedo = albedo * sampleTexture(m.albedo_map, {rec.u, rec.v});
    }


    

    const float NdotL = fmaxf(dot(N, L), 0.0f);
    const float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.f || NdotV <= 0.f) return make_vec3(0,0,0);

    // const float checker_scale = 16.0f;
    // // Use uv only if reasonable -- default (0,0) will produce consistent color.
    // float uu = rec.u - floorf(rec.u);
    // float vv = rec.v - floorf(rec.v);
    // int iu = static_cast<int>(floorf(uu * checker_scale));
    // int iv = static_cast<int>(floorf(vv * checker_scale));
    // if (((iu + iv) & 1) == 1) {
    //     // darken alternate squares
    //     albedo = albedo * 0.6f;
    // }

    // Lambertian diffuse: rho/pi
    const float invPi = 0.31830988618f;
    Vec3 fd = albedo * (m.kd * invPi);

    // Blinn-Phong specular lobe (simple for now, will update)
    Vec3 H = unit_vector(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);

    // Normalized Blinn-Phong: (n+2)/(2π) * (N·H)^n
    const float inv2Pi = 0.15915494309f;
    float specNorm = (m.shininess + 2.0f) * inv2Pi;
    float specLobe = specNorm * powf(NdotH, m.shininess);

    Vec3 fs = m.specularColor * (m.ks * specLobe);

    return fd + fs;
}

#endif