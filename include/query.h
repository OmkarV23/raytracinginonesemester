// include/query.h
#pragma once

#include "camera.h"
#include "ray.h"
#include "MeshOBJ.h"
#include "brdf.h"
#include "shader.h"
#include "bvh.h"
#include "antialias.h"
#include "medium.h"
#include "texture.h"

struct Light;

// GPU-friendly volume region struct (mirrors VolumeRegion from scene.h)
struct VolumeRegionGPU {
    Vec3 min_bounds;
    Vec3 max_bounds;
    HomogeneousMedium medium;

    HYBRID_FUNC inline bool contains(const Vec3& p) const {
        return p.x >= min_bounds.x && p.x <= max_bounds.x &&
               p.y >= min_bounds.y && p.y <= max_bounds.y &&
               p.z >= min_bounds.z && p.z <= max_bounds.z;
    }
};

void render(
    const size_t numTriangles,
    int W, int H,
    const Camera cam,
    const Vec3 missColor,
    const int max_depth,
    const int spp,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const bool diffuse_bounce,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    Vec3* __restrict__ output,
    Vec3* __restrict__ albedo_aov = nullptr,
    Vec3* __restrict__ normal_aov = nullptr,
    int nee_mode = 2,
    const HomogeneousMedium* __restrict__ objectMedia = nullptr,
    int numObjectMedia = 0,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0,
    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
    int numVolumeRegions = 0);


HYBRID_FUNC inline float rng_next(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    unsigned int h = state;
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h ^= h >> 4u;
    h *= 0x27d4eb2du;
    h ^= h >> 15u;
    return float(h) / float(0xFFFFFFFFu);
}

HYBRID_FUNC inline unsigned int make_rng_seed(int x, int y, int sample) {
    return (unsigned int)x * 73856093u
         ^ (unsigned int)y * 19349663u
         ^ (unsigned int)sample * 83492791u;
}

HYBRID_FUNC inline Vec3 random_unit_vector(unsigned int& state) {
    for (;;) {
        float x = 2.0f * rng_next(state) - 1.0f;
        float y = 2.0f * rng_next(state) - 1.0f;
        float z = 2.0f * rng_next(state) - 1.0f;
        float lensq = x*x + y*y + z*z;
        if (lensq > 1e-10f && lensq <= 1.0f) {
            float inv = 1.0f / sqrtf(lensq);
            return make_vec3(x * inv, y * inv, z * inv);
        }
    }
}

HYBRID_FUNC inline Vec3 random_on_hemisphere(const Vec3& normal, unsigned int& state) {
    Vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0f) return on_unit_sphere;
    return make_vec3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
}

HYBRID_FUNC inline void build_onb(const Vec3& N, Vec3& tangent, Vec3& bitangent) {
    Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0.0f, 0.0f, 1.0f)
                                   : make_vec3(1.0f, 0.0f, 0.0f);
    Vec3 cr  = cross(up, N);
    float len = sqrtf(dot(cr, cr));
    tangent   = (len > 1e-10f) ? cr * (1.0f / len) : make_vec3(1.0f, 0.0f, 0.0f);
    bitangent = cross(N, tangent);
}

HYBRID_FUNC inline Vec3 cosine_hemisphere_sample(const Vec3& N, unsigned int& rng_state) {
    const float u1  = rng_next(rng_state);
    const float u2  = rng_next(rng_state);
    const float r   = sqrtf(u1);
    const float phi = 6.28318530717958647692f * u2;
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));
    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);
    Vec3 wi = tangent * x + bitangent * y + N * z;
    float wilen = sqrtf(dot(wi, wi));
    return (wilen > 1e-10f) ? wi * (1.0f / wilen) : N;
}

HYBRID_FUNC inline float BRDFSamplingPdf(const HitRecord& rec,
                                          const Vec3& wo,
                                          const Vec3& wi,
                                          bool allow_diffuse = true)
{
    if (!allow_diffuse && rec.mat.ks <= 0.0f) return 0.0f;
    return allow_diffuse
         ? BRDFpdf(rec, wo, wi)
         : BlinnPhongSpecularPDF(normalize(rec.normal), wo, wi, rec.mat.shininess);
}

HYBRID_FUNC inline Vec3 sample_blinn_phong(const Vec3& N, const Vec3& V,
                                             float shininess,
                                             unsigned int& rng_state, float& pdf)
{
    const float u1    = rng_next(rng_state);
    const float u2    = rng_next(rng_state);
    const float twoPi = 6.28318530717958647692f;

    const float cos_theta_H = powf(fmaxf(u1, 1e-10f), 1.0f / (shininess + 2.0f));
    const float sin_theta_H = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_H * cos_theta_H));
    const float phi         = twoPi * u2;

    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);

    Vec3 H_raw = tangent   * (sin_theta_H * cosf(phi))
               + bitangent * (sin_theta_H * sinf(phi))
               + N         * cos_theta_H;
    float hlen = sqrtf(dot(H_raw, H_raw));
    Vec3  H    = (hlen > 1e-10f) ? H_raw * (1.0f / hlen) : N;

    const float VdotH = fmaxf(dot(V, H), 0.0f);
    Vec3 wi_raw = H * (2.0f * VdotH) - V;
    float wilen = sqrtf(dot(wi_raw, wi_raw));
    Vec3  wi    = (wilen > 1e-10f) ? wi_raw * (1.0f / wilen) : N;

    pdf = BlinnPhongSpecularPDF(N, V, wi, shininess);
    return wi;
}

HYBRID_FUNC inline Vec3 SampleBRDF(const HitRecord& rec, const Vec3& Vo,
                                     unsigned int& rng_state, float& out_pdf,
                                     bool allow_diffuse = true)
{
    const Vec3  N      = normalize(rec.normal);
    const float kd     = allow_diffuse ? rec.mat.kd : 0.0f;
    const float ks     = rec.mat.ks;
    const float total  = kd + ks;
    if (total <= 0.0f) { out_pdf = 0.0f; return make_vec3(0,0,0); }

    Vec3 wi;
    if (kd > 0.0f && rng_next(rng_state) < kd / total) {
        wi = cosine_hemisphere_sample(N, rng_state);
    } else {
        float dummy;
        wi = sample_blinn_phong(N, Vo, rec.mat.shininess, rng_state, dummy);
    }
    out_pdf = BRDFSamplingPdf(rec, Vo, wi, allow_diffuse);
    return wi;
}

HYBRID_FUNC inline HitRecord intersectTriangle(const Ray& r,
                                                const Triangle& tri,
                                                float tmin, float tmax)
{
    HitRecord rec{};
    rec.triangleIdx = -1;

    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;
    const Vec3 pvec = cross(r.direction(), e2);
    const float det = dot(e1, pvec);
    if (fabsf(det) < 1e-8f) { rec.hit = false; return rec; }
    const float invDet = 1.0f / det;

    const Vec3 tvec = r.origin() - tri.v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) { rec.hit = false; return rec; }

    const Vec3 qvec = cross(tvec, e1);
    const float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) { rec.hit = false; return rec; }

    const float t = dot(e2, qvec) * invDet;
    if (t < tmin || t > tmax) { rec.hit = false; return rec; }

    rec.hit  = true;
    rec.t    = t;
    rec.p    = r.origin() + r.direction() * t;

    Vec3 geomN = normalize(cross(e1, e2));
    rec.front_face = dot(r.direction(), geomN) < 0.0f;
    if (!rec.front_face) geomN = -geomN;

    Vec3 shadingN = (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    if (length_squared(shadingN) < 1e-12f) {
        shadingN = geomN;
    } else {
        shadingN = normalize(shadingN);
        if (dot(shadingN, geomN) < 0.0f) shadingN = -shadingN;
    }
    rec.normal = shadingN;
    rec.mat    = Material();

    float w0 = 1.0f - u - v;
    rec.uv.x = w0 * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x;
    rec.uv.y = w0 * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y;

    return rec;
}

HYBRID_FUNC inline void assignMaterialToHit(
    HitRecord& hitRecord,
    const int numTriangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0)
{
    if (!hitRecord.hit || triObjectIds == nullptr || objectMaterials == nullptr ||
        hitRecord.triangleIdx < 0 || hitRecord.triangleIdx >= numTriangles)
        return;

    const int objId = triObjectIds[hitRecord.triangleIdx];
    if (objId >= 0 && objId < numObjectMaterials)
        hitRecord.mat = objectMaterials[objId];

    ApplyTextures(hitRecord, textures, numTextures);
}

HYBRID_FUNC inline int binary_search_cdf(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// -----------------------------------------------------------------------
// findVolumeAtPoint
// Inline helper: returns the first VolumeRegionGPU that contains p, or a
// disabled medium if no region matches.  Written as a plain inline
// function rather than a lambda so it compiles cleanly under CUDA.
// -----------------------------------------------------------------------
HYBRID_FUNC inline HomogeneousMedium findVolumeAtPoint(
    const Vec3& p,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions)
{
    if (volumeRegions != nullptr) {
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (volumeRegions[i].contains(p))
                return volumeRegions[i].medium;
        }
    }
    HomogeneousMedium none;
    none.enabled = false;
    return none;
}

// -----------------------------------------------------------------------
// TraceRayIterative
//
// Volume path tracing from Lecture 14 (EEE 598):
//
//   Per bounce:
//     1. Check if current ray origin is inside a volume region.
//     2. If inside:
//          sample free path t = -ln(1-xi)/sigma_t          [slide 55]
//          if t < t_surface: VOLUME SCATTER
//            throughput *= sigma_s/sigma_t  (albedo)        [slide 59]
//            sample new dir from HG phase function          [slide 53]
//            continue to next depth
//          else: SURFACE HIT
//            apply transmittance Tr(t_surf)/P(z) = 1        [slide 59]
//            proceed with normal surface shading
//     3. If not inside a volume: normal surface shading only.
// -----------------------------------------------------------------------
HYBRID_FUNC inline Vec3 TraceRayIterative(
    const Ray& primaryRay,
    const int maxDepth,
    const Vec3 missColor,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    unsigned int rng_state = 42u,
    bool diffuse_bounce = true,
    int nee_mode = 2,
    const HomogeneousMedium* __restrict__ objectMedia = nullptr,
    int numObjectMedia = 0,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0,
    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
    int numVolumeRegions = 0)
{
    if (maxDepth <= 0) return make_vec3(0.0f, 0.0f, 0.0f);

    Ray  ray        = primaryRay;
    Vec3 radiance   = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
    float prev_pdf   = 0.0f;
    bool  prev_delta = false;
    Vec3  prev_N     = make_vec3(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < maxDepth; ++depth) {

        // ----------------------------------------------------------------
        // 1. Determine if the current ray origin is inside a volume region.
        //    We re-check every bounce because the ray may have scattered out.
        // ----------------------------------------------------------------
        HomogeneousMedium activeMedium =
            findVolumeAtPoint(ray.origin(), volumeRegions, numVolumeRegions);

        // ----------------------------------------------------------------
        // 2. BVH traversal to find the nearest surface hit
        // ----------------------------------------------------------------
        HitRecord hitRecord;
        SearchBVH(numTriangles, ray, nodes, aabbs, triangles, hitRecord);

        const float t_surf = hitRecord.hit ? hitRecord.t : 1e30f;

        // ----------------------------------------------------------------
        // 3. Volume scatter decision  [Lecture 14, slides 54-59]
        // ----------------------------------------------------------------
        if (activeMedium.enabled && activeMedium.sigma_t > 0.0f) {

            // Sample free-path distance:  t = -ln(1-xi) / sigma_t  [slide 55]
            const float xi_t  = rng_next(rng_state);
            const float t_vol = activeMedium.sampleFreePath(xi_t);

            if (t_vol < t_surf) {
                // ---- VOLUME SCATTER EVENT [slide 59, if t < tmax] ----
                //
                // The full weight before simplification is:
                //   Tr(t) / p(t)  *  sigma_s  *  f_p(omega, omega_i) / p(omega_i)
                //
                // With HG sampled proportional to f_p:   f_p / p(omega_i) = 1
                // And Tr(t)/p(t) = exp(-sigma_t*t) / (sigma_t*exp(-sigma_t*t)) = 1/sigma_t
                // So the full factor reduces to:  sigma_s / sigma_t  (the albedo)
                throughput = throughput * activeMedium.albedo();

                // Move ray origin to the scatter point
                const Vec3 scatter_pos = ray.origin() + ray.direction() * t_vol;

                // ---- VOLUME NEE: direct illumination at scatter point ----
                //
                // At scatter point x, the in-scattered direct contribution is:
                //   L_direct = throughput * p_HG(wo, wi_l) * Tr(x->light) * Le / pdf_area
                //
                // throughput already carries sigma_s/sigma_t from the albedo above, so
                // we only need the phase function value and transmittance explicitly.
                if (numEmissiveTris > 0) {
                    const float u_sel = rng_next(rng_state);
                    const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                    const EmissiveTriInfo& emi = emissiveTris[eidx];
                    const Triangle& eTri = triangles[emi.triangleIdx];

                    float u1 = rng_next(rng_state), u2 = rng_next(rng_state);
                    if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                    const Vec3 lightPoint = eTri.v0*(1.0f-u1-u2) + eTri.v1*u1 + eTri.v2*u2;

                    Vec3 toLight = lightPoint - scatter_pos;
                    const float r2 = dot(toLight, toLight);
                    if (r2 > 1e-8f) {
                        const float r_dist  = sqrtf(r2);
                        const Vec3 wi_light = toLight * (1.0f / r_dist);
                        const float cosLight = fabsf(dot(emi.normal, -wi_light));

                        if (cosLight > 1e-6f) {
                            Ray shadowRay(scatter_pos + wi_light * RT_EPS, wi_light);
                            HitRecord shadowHit{}; shadowHit.hit = false;
                            SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);

                            if (!shadowHit.hit || shadowHit.t >= r_dist - RT_EPS) {
                                // Phase function for wo -> wi_light direction
                                const Vec3 wo_vol = make_vec3(-ray.direction().x,
                                                               -ray.direction().y,
                                                               -ray.direction().z);
                                const float cos_theta_l = dot(normalize(wo_vol), wi_light);
                                const float phase_l = activeMedium.phaseHG(cos_theta_l);

                                // Area PDF converted to solid angle
                                const float G = cosLight / r2;
                                const float pdf_area_sa = (G > 1e-10f)
                                    ? (1.0f / totalEmissiveArea) / G : 0.0f;

                                // Transmittance along shadow ray through the medium
                                const float Tr_light = activeMedium.transmittance(r_dist);

                                if (pdf_area_sa > 1e-10f) {
                                    radiance = radiance + throughput *
                                               (emi.emission * phase_l * Tr_light / pdf_area_sa);
                                }
                            }
                        }
                    }
                }
                // ---- END VOLUME NEE ----

                // Sample scattered direction from HG phase function [slide 53]
                // Convention: pass the incoming direction (pointing back toward camera)
                const float xi1 = rng_next(rng_state);
                const float xi2 = rng_next(rng_state);
                const Vec3 wi_in = make_vec3(-ray.direction().x,
                                             -ray.direction().y,
                                             -ray.direction().z);
                const Vec3 new_dir = activeMedium.samplePhaseHG(wi_in, xi1, xi2);

                ray = Ray(scatter_pos, new_dir);
                prev_pdf   = activeMedium.phaseHG(dot(normalize(wi_in), normalize(new_dir)));
                prev_delta = false;

                // Russian roulette
                const float p_survive = fminf(luminance(throughput), 0.95f);
                if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
                throughput = throughput * (1.0f / p_survive);

                continue;  // next depth iteration — no surface shading
            }

            // ---- SURFACE HIT THROUGH MEDIUM [slide 59, else branch] ----
            // Tr(t_surf) / P(z) cancels to 1 for a homogeneous medium
            // (both equal exp(-sigma_t * t_surf)), so throughput is unchanged.
            // We fall through to the normal surface shading code below.
        }

        // ----------------------------------------------------------------
        // 4. No hit → sky / miss color
        // ----------------------------------------------------------------
        if (!hitRecord.hit) {
            radiance = radiance + throughput * missColor;
            break;
        }

        assignMaterialToHit(hitRecord, numTriangles, triObjectIds,
                            objectMaterials, numObjectMaterials,
                            textures, numTextures);

        // ----------------------------------------------------------------
        // 5. Emissive surface
        // ----------------------------------------------------------------
        const Vec3 Le = hitRecord.mat.emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f) {
            if (depth == 0 || prev_delta) {
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 1) {
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 2 && numEmissiveTris > 0 && prev_pdf > 1e-6f) {
                Vec3 hitGeomN = normalize(cross(
                    triangles[hitRecord.triangleIdx].v1 - triangles[hitRecord.triangleIdx].v0,
                    triangles[hitRecord.triangleIdx].v2 - triangles[hitRecord.triangleIdx].v0));
                float cosLight = fabsf(dot(hitGeomN, -unit_vector(ray.direction())));
                float dist2    = hitRecord.t * hitRecord.t;
                float G        = (dist2 > 1e-10f) ? cosLight / dist2 : 0.0f;
                float pdf_area_sa = (G > 1e-10f) ? (1.0f / totalEmissiveArea) / G : 0.0f;
                float NdotWi_prev = fmaxf(dot(prev_N, unit_vector(ray.direction())), 0.0f);
                float pdf_cos     = NdotWi_prev * 0.31830988618f;
                float pdf_nee     = (pdf_area_sa + pdf_cos + prev_pdf) / 3.0f;
                float p2_brdf     = prev_pdf * prev_pdf;
                float p2_nee      = pdf_nee  * pdf_nee;
                float w_brdf = (p2_brdf + p2_nee > 0.0f) ? p2_brdf / (p2_brdf + p2_nee) : 0.0f;
                radiance = radiance + throughput * Le * w_brdf;
            }
        }

        // ----------------------------------------------------------------
        // 6. Direct lighting (point lights)
        // ----------------------------------------------------------------
        Vec3 direct = ShadeDirect(ray, hitRecord, lights, numLights,
                                  numTriangles, nodes, aabbs, triangles,
                                  objectMedia, numObjectMedia, triObjectIds);
        radiance = radiance + throughput * direct;

        const Vec3  N  = normalize(hitRecord.normal);
        const Vec3  Vo = unit_vector(-ray.direction());
        prev_N = N;

        // ----------------------------------------------------------------
        // 7. NEE (next event estimation for emissive triangles)
        // ----------------------------------------------------------------
        if (numEmissiveTris > 0 && nee_mode != 1) {
            int strategy;
            if (nee_mode == 0) {
                strategy = 0;
            } else {
                const float strategy_u = rng_next(rng_state);
                strategy = (strategy_u < 0.333333f) ? 0
                         : (strategy_u < 0.666667f) ? 1 : 2;
            }

            Vec3  wi_nee       = make_vec3(0,0,0);
            Vec3  Le_nee       = make_vec3(0,0,0);
            float dist_nee     = 0.0f;
            float cosLight_nee = 0.0f;
            float NdotL_nee    = 0.0f;
            bool  nee_valid    = false;

            if (strategy == 0) {
                const float u_sel = rng_next(rng_state);
                const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                const EmissiveTriInfo& emi = emissiveTris[eidx];
                const Triangle& eTri = triangles[emi.triangleIdx];
                float u1 = rng_next(rng_state), u2 = rng_next(rng_state);
                if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                const Vec3 lightPoint = eTri.v0 * (1.0f - u1 - u2) + eTri.v1 * u1 + eTri.v2 * u2;
                Vec3 toLight = lightPoint - hitRecord.p;
                const float r2 = dot(toLight, toLight);
                if (r2 > 1e-10f) {
                    const float r_dist = sqrtf(r2);
                    wi_nee       = toLight * (1.0f / r_dist);
                    NdotL_nee    = fmaxf(dot(N, wi_nee), 0.0f);
                    cosLight_nee = fabsf(dot(emi.normal, -wi_nee));
                    if (NdotL_nee > 0.0f && cosLight_nee > 0.0f) {
                        Ray shadowRay(hitRecord.p + N * RT_EPS, wi_nee);
                        HitRecord shadowHit{}; shadowHit.hit = false;
                        SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);
                        if (!shadowHit.hit || shadowHit.t >= r_dist - RT_EPS) {
                            Le_nee    = emi.emission;
                            dist_nee  = r_dist;
                            nee_valid = true;
                        }
                    }
                }
            } else {
                if (strategy == 1)
                    wi_nee = cosine_hemisphere_sample(N, rng_state);
                else {
                    float dummy_pdf;
                    wi_nee = SampleBRDF(hitRecord, Vo, rng_state, dummy_pdf, diffuse_bounce);
                }
                NdotL_nee = fmaxf(dot(N, wi_nee), 0.0f);
                if (NdotL_nee > 0.0f) {
                    Ray neeRay(hitRecord.p + N * RT_EPS, wi_nee);
                    HitRecord neeHit; neeHit.hit = false;
                    SearchBVH(numTriangles, neeRay, nodes, aabbs, triangles, neeHit);
                    if (neeHit.hit) {
                        assignMaterialToHit(neeHit, numTriangles, triObjectIds,
                                            objectMaterials, numObjectMaterials,
                                            textures, numTextures);
                        const Vec3 Le_hit = neeHit.mat.emission;
                        if (Le_hit.x > 0.0f || Le_hit.y > 0.0f || Le_hit.z > 0.0f) {
                            Vec3 geomN = normalize(cross(
                                triangles[neeHit.triangleIdx].v1 - triangles[neeHit.triangleIdx].v0,
                                triangles[neeHit.triangleIdx].v2 - triangles[neeHit.triangleIdx].v0));
                            cosLight_nee = fabsf(dot(geomN, -wi_nee));
                            dist_nee     = neeHit.t;
                            Le_nee       = Le_hit;
                            nee_valid    = true;
                        }
                    }
                }
            }

            if (nee_valid && dist_nee > 1e-6f && cosLight_nee > 1e-6f) {
                const float r2          = dist_nee * dist_nee;
                const float G           = cosLight_nee / r2;
                const float pdf_area_sa = (1.0f / totalEmissiveArea) / G;
                const Vec3  f_nee       = EvaluateBRDF(hitRecord, Vo, wi_nee);

                // Apply medium transmittance along NEE shadow ray if in volume
                float Tr_nee = 1.0f;
                if (activeMedium.enabled)
                    Tr_nee = activeMedium.transmittance(dist_nee);

                if (nee_mode == 0) {
                    if (pdf_area_sa > 1e-10f) {
                        radiance = radiance + throughput *
                                   (Le_nee * f_nee * (NdotL_nee * Tr_nee / pdf_area_sa));
                    }
                } else {
                    const float pdf_cos      = NdotL_nee * 0.31830988618f;
                    const float pdf_brdf     = BRDFSamplingPdf(hitRecord, Vo, wi_nee, diffuse_bounce);
                    const float pdf_combined = (pdf_area_sa + pdf_cos + pdf_brdf) / 3.0f;
                    if (pdf_combined > 1e-10f) {
                        const float p2_nee  = pdf_combined * pdf_combined;
                        const float p2_brdf = pdf_brdf     * pdf_brdf;
                        const float w_nee   = (p2_nee + p2_brdf > 0.0f)
                                            ? p2_nee / (p2_nee + p2_brdf) : 0.0f;
                        radiance = radiance + throughput *
                                   (Le_nee * f_nee * (NdotL_nee * Tr_nee / pdf_combined) * w_nee);
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // 8. Sample next bounce direction (BRDF, mirror, or glass)
        // ----------------------------------------------------------------
        const float kd  = hitRecord.mat.kd;
        const float ks  = hitRecord.mat.ks;
        const float kr  = hitRecord.mat.kr;
        const float ior = hitRecord.mat.ior;

        if (kd > 1e-6f || ks > 1e-6f) {
            // ---- Diffuse / glossy BRDF ----
            float pdf;
            Vec3 wi = SampleBRDF(hitRecord, Vo, rng_state, pdf, diffuse_bounce);
            if (pdf < 1e-6f || dot(wi, N) < 0.0f) break;
            ray = Ray(hitRecord.p + N * RT_EPS, wi);
            const float NdotWi = fmaxf(dot(N, wi), 0.0f);
            const Vec3  f      = EvaluateBRDF(hitRecord, Vo, wi);
            throughput = throughput * (f * (NdotWi / pdf));
            prev_pdf   = pdf;
            prev_delta = false;
        } else if (ior > 1.0f) {
            // ---- Dielectric glass (Snell's law + Schlick Fresnel) ----
            //
            // hitRecord.normal is always oriented toward the incident side (i.e. toward
            // the ray origin) for both entry and exit — this matches the convention that
            // refract_dir and the Schlick formula expect for N.
            const bool  entering  = hitRecord.front_face;
            const float eta       = entering ? (1.0f / ior) : ior;  // n_incident / n_transmitted
            const Vec3  rayDir    = unit_vector(ray.direction());

            // cos(θᵢ): always positive since N points toward incident side
            const float cos_i     = fminf(-dot(rayDir, N), 1.0f);

            // Schlick Fresnel (r0 is symmetric in n1<->n2, so ior suffices)
            const float F         = schlick(cos_i, ior);

            const Vec3  refractDir = refract_dir(rayDir, N, eta);
            const bool  tir        = (refractDir.x == 0.0f && refractDir.y == 0.0f
                                      && refractDir.z == 0.0f);

            if (!tir && rng_next(rng_state) > F) {
                // Transmit: offset past the surface into the transmitted side (opposite N)
                ray = Ray(hitRecord.p - N * RT_EPS, refractDir);
            } else {
                // Fresnel reflection or total internal reflection: stay on incident side
                const Vec3 reflDir = reflect_dir(rayDir, N);
                ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            }
            throughput = throughput * hitRecord.mat.specularColor;
            prev_pdf   = 0.0f;
            prev_delta = true;
        } else {
            // ---- Perfect mirror ----
            const Vec3 reflDir = reflect_dir(unit_vector(ray.direction()), N);
            ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            throughput = throughput * (hitRecord.mat.specularColor * kr);
            prev_pdf   = 0.0f;
            prev_delta = true;
        }

        // Russian roulette
        const float p_survive = fminf(luminance(throughput), 0.95f);
        if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
        throughput = throughput * (1.0f / p_survive);
    }

    return radiance;
}


HYBRID_FUNC inline void SearchBVH(
    const int numTriangles,
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    HitRecord& hitRecord)
{
    constexpr float kRayTMin = 1e-4f;
    const float tmin = kRayTMin;
    float bestT = FLT_MAX;
    HitRecord bestHit;
    bestHit.triangleIdx = -1;
    bestHit.hit        = false;
    bestHit.t          = -1.0;
    bestHit.p          = make_vec3(0,0,0);
    bestHit.normal     = make_vec3(0,0,0);
    bestHit.front_face = false;
    bestHit.mat        = Material();
    bestHit.uv         = make_vec2(0.0f, 0.0f);

    constexpr int STACK_CAPACITY = 512;
    std::uint32_t stack[STACK_CAPACITY];
    std::uint32_t* stack_ptr = stack;
    bool stackOverflow = false;
    *stack_ptr++ = 0;

    while (stack_ptr > stack) {
        const std::uint32_t nodeIdx = *--stack_ptr;
        if (!intersectAABB(ray, aabbs[nodeIdx], tmin, bestT)) continue;

        const BVHNode       node    = nodes[nodeIdx];
        const std::uint32_t obj_idx = node.object_idx;

        if (obj_idx != 0xFFFFFFFF) {
            if (obj_idx < static_cast<std::uint32_t>(numTriangles)) {
                HitRecord rec = intersectTriangle(ray, triangles[obj_idx], tmin, bestT);
                if (rec.hit) {
                    rec.triangleIdx = static_cast<int>(obj_idx);
                    bestT   = rec.t;
                    bestHit = rec;
                }
            }
            continue;
        }

        const std::uint32_t left_idx  = node.left_idx;
        const std::uint32_t right_idx = node.right_idx;

        if (left_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[left_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) *stack_ptr++ = left_idx;
                else stackOverflow = true;
            }
        }
        if (right_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[right_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) *stack_ptr++ = right_idx;
                else stackOverflow = true;
            }
        }
    }

    if (stackOverflow) {
        for (int i = 0; i < numTriangles; ++i) {
            HitRecord rec = intersectTriangle(ray, triangles[i], tmin, bestT);
            if (rec.hit) {
                rec.triangleIdx = i;
                bestT   = rec.t;
                bestHit = rec;
            }
        }
    }

    hitRecord = bestHit;
}