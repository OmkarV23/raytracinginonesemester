// brdf.h
#ifndef BRDF_H
#define BRDF_H

#include <cmath>
#include "vec3.h"
#include "material.h"

HYBRID_FUNC inline float saturate_f(float x) { return (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x); }

HYBRID_FUNC inline float BlinnPhongSpecularPDF(const Vec3& N,
                                               const Vec3& wo,
                                               const Vec3& wi,
                                               float shininess)
{
    const Vec3 Hraw = wo + wi;
    const float hlen2 = dot(Hraw, Hraw);
    if (hlen2 <= 1e-12f) return 0.0f;

    const Vec3 H = Hraw * (1.0f / sqrtf(hlen2));
    const float NdotH = fmaxf(dot(normalize(N), H), 0.0f);
    const float VdotH = fmaxf(dot(wo, H), 0.0f);
    const float inv2Pi = 0.15915494309f;
    return (VdotH > 1e-6f)
         ? (shininess + 2.0f) * inv2Pi * powf(NdotH, shininess) * NdotH / (4.0f * VdotH)
         : 0.0f;
}

// Returns f(wo, wi) (does NOT include N·L)
HYBRID_FUNC inline Vec3 EvaluateBRDF(const HitRecord& rec,
                                    const Vec3& V,  // to viewer
                                    const Vec3& L)  // to light
{

    const Material& m = rec.mat;
    const Vec3 N = rec.normal;

    const float NdotL = fmaxf(dot(N, L), 0.0f);
    const float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.f || NdotV <= 0.f) return make_vec3(0,0,0);

    // Lambertian diffuse: rho/pi
    const float invPi = 0.31830988618f;
    Vec3 fd = m.albedo * (m.kd * invPi);

    // Blinn-Phong specular lobe (simple for now, will update)
    Vec3 H = unit_vector(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);

    // Normalized Blinn-Phong: (n+2)/(2π) * (N·H)^n
    const float inv2Pi = 0.15915494309f;
    float specNorm = (m.shininess + 2.0f) * inv2Pi;
    float specLobe = specNorm * powf(NdotH, m.shininess);

    Vec3 fs = m.specularColor * (m.ks) * specLobe;

    return fd + fs;
}

HYBRID_FUNC inline float BRDFpdf(const HitRecord& rec,
                                  const Vec3& wo,   // toward viewer
                                  const Vec3& wi)   // sampled direction
{
    const Material& m = rec.mat;
    const Vec3  N     = normalize(rec.normal);
    const float kd    = m.kd;
    const float ks    = m.ks;
    const float total = kd + ks;
    if (total <= 0.0f) return 0.0f;

    const float NdotWi  = fmaxf(dot(N, wi),  0.0f);
    const float invPi   = 0.31830988618f;
    const float pdf_diff = NdotWi * invPi;   // cosine-weighted PDF

    const float pdf_spec = BlinnPhongSpecularPDF(N, wo, wi, m.shininess);

    return (kd / total) * pdf_diff + (ks / total) * pdf_spec;
}

#endif
