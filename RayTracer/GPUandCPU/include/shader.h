#ifndef SHADER_H
#define SHADER_H

#include <cmath>
#include <limits>
#include "ray.h"
#include "brdf.h"
#include "MeshOBJ.h"
#include "scene.h"
#include "bvh.h"
#include "query.h"


HYBRID_FUNC inline void SearchBVH(
    const int numTriangles,
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    HitRecord& hitRecord);

static constexpr float RT_EPS = 1e-3f;

HYBRID_FUNC inline Vec3 clamp(Vec3 color) {
    if (color.x > 1.0f) color.x = 1.0f;
    if (color.y > 1.0f) color.y = 1.0f;
    if (color.z > 1.0f) color.z = 1.0f;
    if (color.x < 0.0f) color.x = 0.0f;
    if (color.y < 0.0f) color.y = 0.0f;
    if (color.z < 0.0f) color.z = 0.0f;
    return color;
}

HYBRID_FUNC inline float length3(const Vec3& v) {
    return sqrtf(dot(v, v));
}

HYBRID_FUNC inline Vec3 reflect_dir(const Vec3& I, const Vec3& N) {
    // I points *in the ray direction* (from origin toward scene)
    // Reflection: R = I - 2*(I·N)*N
    return I - (2.0f * dot(I, N)) * N;
}

HYBRID_FUNC inline bool IsInShadow(const Vec3& P,
                       const Vec3& N,
                       const Light& light,
                       const Triangle* tris,
                       int triCount,
                       const BVHNode* nodes,
                       const AABB* aabbs)
{
    Vec3 toL = light.position - P;
    float distToL = length3(toL);
    if (distToL <= 0.0f) return false;

    Vec3 Ldir = toL / distToL;
    Ray shadowRay(P + N * RT_EPS, Ldir);

    HitRecord shadowHit{};
    SearchBVH(triCount, shadowRay, nodes, aabbs, tris, shadowHit);
    return shadowHit.hit && shadowHit.t < distToL;
}

HYBRID_FUNC inline float sampleBumpHeight(const TextureData* tex, float u, float v)
{
    if (!tex || tex->width == 0 || tex->data == nullptr) return 0.5f;  // Mid-gray = flat

    u = u - floorf(u);  // Wrap
    v = v - floorf(v);
    
    int x = static_cast<int>(u * (tex->width - 1));
    int y = static_cast<int>((1.0f - v) * (tex->height - 1));
    
    x = x < 0 ? 0 : (x > tex->width - 1 ? tex->width - 1 : x);
    y = y < 0 ? 0 : (y > tex->height - 1 ? tex->height - 1 : y);

    // printf("Bump map:, size=%dx%d, channels=%d\n", tex->width, tex->height, tex->channels);
    
    int idx = (y * tex->width + x) * tex->channels;
    const unsigned char* p = tex->data + idx;
    
    // Grayscale: use luminance (works for RGB/RGBA)
    float gray = 0.299f * (p[0]/255.0f) + 
                 0.587f * (p[1]/255.0f) + 
                 0.114f * (p[2]/255.0f);
    
    return gray;  // 0=black(dent), 1=white(bump)
}

HYBRID_FUNC inline Vec3 computeBumpNormal(const Vec3& N, const Vec3& T, const Vec3& B, 
                              float u, float v, const TextureData* bumpMap, float strength = 100.0f)
{
    float H = sampleBumpHeight(bumpMap, u, v);
    float H_dx = sampleBumpHeight(bumpMap, u + 0.01f, v);
    float H_dy = sampleBumpHeight(bumpMap, u, v + 0.01f);

    float dHdx = H_dx - H;
    float dHdy = H_dy - H;
    
    Vec3 bump = strength * normalize(make_vec3(-dHdx, dHdy, 1.0f));

    return normalize(T * bump.x + B * bump.y + N * bump.z);
}


HYBRID_FUNC inline Vec3 ShadeDirect(const Ray& r,
                        const HitRecord& rec,
                        const Light* lights,
                        const int numLights,
                        const int numTriangles,
                        const BVHNode* nodes,
                        const AABB* aabbs,
                        const Triangle* triangles)
{
    // Assuming rec.hit == true already
    Vec3 N = unit_vector(rec.normal);
    Vec3 V = unit_vector(r.origin() - rec.p);

    Vec3 Lo = make_vec3(0,0,0);

    // small ambient (looks nicer)
    Vec3 ambient = rec.mat.albedo * 0.05f;
    Lo = Lo + ambient;

    // add emission (placeholder math for now)
    Lo = Lo + rec.mat.emission;


    Vec3 T = unit_vector(rec.tangent);
    Vec3 B = unit_vector(rec.bitangent);

    Vec3 Ns = N;
    if (rec.mat.bump_map) {
        Ns = computeBumpNormal(unit_vector(rec.normal), T, B, rec.u, rec.v, rec.mat.bump_map, 100.0f);
    }

    for (int i = 0; i < numLights; ++i) {
        const Light& light = lights[i];
        Vec3 L = unit_vector(light.position - rec.p);
        float NdotL = fmaxf(dot(Ns, L), 0.0f);
        if (NdotL <= 0.0f) continue;

        // Hard shadows
        if (IsInShadow(rec.p, Ns, light, triangles, 
                        numTriangles, nodes, aabbs)) {
            continue;
        }

        // BRDF value
        Vec3 f = EvaluateBRDF(rec, V, L, Ns);

        // Light contribution
        Vec3 radiance = light.color * light.intensity;
        Vec3 direct = (radiance * f) * NdotL;

        Lo = Lo + direct;
    }

    return Lo;
}

#endif
