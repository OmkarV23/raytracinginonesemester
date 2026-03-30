#pragma once

#include "camera.h"
#include "ray.h"
#include "MeshOBJ.h"
#include "brdf.h"
#include "shader.h"
#include "bvh.h"
#include "antialias.h"

struct Light;

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
    int nee_mode = 2);


HYBRID_FUNC inline float rng_next(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    // xorshift-style mixing for better distribution
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
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    return make_vec3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
}

// Build an orthonormal basis (tangent, bitangent) aligned with surface normal N.
HYBRID_FUNC inline void build_onb(const Vec3& N, Vec3& tangent, Vec3& bitangent) {
    Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0.0f, 0.0f, 1.0f)
                                   : make_vec3(1.0f, 0.0f, 0.0f);
    Vec3 cr  = cross(up, N);
    float len = sqrtf(dot(cr, cr));
    tangent   = (len > 1e-10f) ? cr * (1.0f / len) : make_vec3(1.0f, 0.0f, 0.0f);
    bitangent = cross(N, tangent);
}

// Cosine-weighted hemisphere sampling (Malley's method).
// PDF = cos(theta)/pi
HYBRID_FUNC inline Vec3 cosine_hemisphere_sample(const Vec3& N, unsigned int& rng_state) {
    const float u1  = rng_next(rng_state);
    const float u2  = rng_next(rng_state);
    const float r   = sqrtf(u1);
    const float phi = 6.28318530717958647692f * u2;   // 2*pi
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1)); // cos(theta)
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

// Sample the Blinn-Phong specular lobe by importance-sampling D(H) * (N·H).
// Returns the outgoing direction wi and writes the solid-angle PDF.
HYBRID_FUNC inline Vec3 sample_blinn_phong(const Vec3& N, const Vec3& V,
                                            float shininess,
                                            unsigned int& rng_state, float& pdf)
{
    const float u1     = rng_next(rng_state);
    const float u2     = rng_next(rng_state);
    const float twoPi  = 6.28318530717958647692f;

    // Match the PDF used throughout the integrator:
    // pdf_H(H) = D(H) * (N·H) = (n + 2) / (2pi) * (N·H)^(n + 1).
    const float cos_theta_H = powf(fmaxf(u1, 1e-10f), 1.0f / (shininess + 2.0f));
    const float sin_theta_H = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_H * cos_theta_H));
    const float phi         = twoPi * u2;

    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);

    // H in world space
    Vec3 H_raw = tangent   * (sin_theta_H * cosf(phi))
               + bitangent * (sin_theta_H * sinf(phi))
               + N         * cos_theta_H;
    float hlen = sqrtf(dot(H_raw, H_raw));
    Vec3  H    = (hlen > 1e-10f) ? H_raw * (1.0f / hlen) : N;

    // Reflect -V around H to get outgoing direction wi
    const float VdotH = fmaxf(dot(V, H), 0.0f);
    Vec3 wi_raw = H * (2.0f * VdotH) - V;
    float wilen = sqrtf(dot(wi_raw, wi_raw));
    Vec3  wi    = (wilen > 1e-10f) ? wi_raw * (1.0f / wilen) : N;

    pdf = BlinnPhongSpecularPDF(N, V, wi, shininess);
    return wi;
}

// Unified BRDF importance sampling: picks diffuse or specular lobe proportional
// to kd/ks weights, then computes the mixed PDF.
HYBRID_FUNC inline Vec3 SampleBRDF(const HitRecord& rec, const Vec3& Vo,
                                    unsigned int& rng_state, float& out_pdf,
                                    bool allow_diffuse = true)
{
    const Vec3  N      = normalize(rec.normal);
    const float kd     = allow_diffuse ? rec.mat.kd : 0.0f;
    const float ks     = rec.mat.ks;
    const float total  = kd + ks;
    if (total <= 0.0f) { out_pdf = 0.0f; return make_vec3(0.0f, 0.0f, 0.0f); }

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
                                          float tmin,
                                          float tmax)
{
    HitRecord rec{};
    rec.triangleIdx = -1;

    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;
    const Vec3 pvec = cross(r.direction(), e2);
    const float det = dot(e1, pvec);

    if (fabsf(det) < 1e-8f) {
        rec.hit = false;
        return rec;
    }
    const float invDet = 1.0f / det;

    const Vec3 tvec = r.origin() - tri.v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        rec.hit = false;
        return rec;
    }

    const Vec3 qvec = cross(tvec, e1);
    const float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        rec.hit = false;
        return rec;
    }
    const float t = dot(e2, qvec) * invDet;
    if (t < tmin || t > tmax) {
        rec.hit = false;
        return rec;
    }

    rec.hit = true;
    rec.t = t;
    rec.p = r.origin() + r.direction() * t;

    // Use geometric normal for sidedness (robust), shading normal for BRDF.
    Vec3 geomN = normalize(cross(e1, e2));
    rec.front_face = dot(r.direction(), geomN) < 0.0f;
    if (!rec.front_face) geomN = -geomN;

    Vec3 shadingN = (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    if (length_squared(shadingN) < 1e-12f) {
        shadingN = geomN;
    } else {
        shadingN = normalize(shadingN);
        // Keep shading normal in same hemisphere as geometric normal.
        if (dot(shadingN, geomN) < 0.0f) shadingN = -shadingN;
    }

    rec.normal = shadingN;
    rec.mat = Material();

    return rec;
}

HYBRID_FUNC inline void assignMaterialToHit(
    HitRecord& hitRecord,
    const int numTriangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials)
{
    if (!hitRecord.hit ||
        triObjectIds == nullptr ||
        objectMaterials == nullptr ||
        hitRecord.triangleIdx < 0 ||
        hitRecord.triangleIdx >= numTriangles) {
        return;
    }

    const int objId = triObjectIds[hitRecord.triangleIdx];
    if (objId >= 0 && objId < numObjectMaterials) {
        hitRecord.mat = objectMaterials[objId];
    }
}


// Binary search on a CDF array — returns index of the selected entry.
HYBRID_FUNC inline int binary_search_cdf(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

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
    int nee_mode = 2)           // 0 = area only, 1 = brdf only, 2 = 3-strategy MIS
{
    if (maxDepth <= 0) return make_vec3(0.0f, 0.0f, 0.0f);

    Ray ray = primaryRay;
    Vec3 radiance = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
    float prev_pdf   = 0.0f;   // BRDF PDF of the bounce that generated current ray
    bool  prev_delta = false;   // true if previous bounce was a delta (mirror) reflection
    Vec3  prev_N     = make_vec3(0.0f, 0.0f, 0.0f); // normal at previous hit (for cosine PDF in MIS)

    for (int depth = 0; depth < maxDepth; ++depth) {
        HitRecord hitRecord;
        SearchBVH(numTriangles, ray, nodes, aabbs, triangles, hitRecord);
        if (!hitRecord.hit) {
            radiance = radiance + throughput * missColor;
            break;
        }

        assignMaterialToHit(hitRecord, numTriangles, triObjectIds, objectMaterials, numObjectMaterials);

        // ------------------------------------------------------------------
        // Emissive surface hit: add emission (mode-dependent)
        //   mode 0 (area only):  skip — all emission comes from NEE
        //   mode 1 (BRDF only):  full weight — this IS the only path
        //   mode 2 (MIS):        MIS weight between BRDF bounce and NEE
        // ------------------------------------------------------------------
        const Vec3 Le = hitRecord.mat.emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f) {
            if (depth == 0 || prev_delta) {
                // Primary ray or delta bounce — always add, all modes
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 0) {
                // Area-only: BRDF bounce should NOT add emission (NEE handles it)
                // Skip entirely to avoid double-counting
            } else if (nee_mode == 1) {
                // BRDF-only: no NEE exists, so BRDF bounce gets full weight
                radiance = radiance + throughput * Le;
            } else if (numEmissiveTris > 0 && prev_pdf > 1e-6f) {
                // MIS: weight between BRDF bounce PDF and combined NEE PDF
                Vec3 hitGeomN = normalize(cross(
                    triangles[hitRecord.triangleIdx].v1 - triangles[hitRecord.triangleIdx].v0,
                    triangles[hitRecord.triangleIdx].v2 - triangles[hitRecord.triangleIdx].v0));
                float cosLight = fabsf(dot(hitGeomN, -unit_vector(ray.direction())));
                float dist2 = (float)(hitRecord.t * hitRecord.t);
                float G = (dist2 > 1e-10f) ? cosLight / dist2 : 0.0f;

                float pdf_area_sa = (G > 1e-10f) ? (1.0f / totalEmissiveArea) / G : 0.0f;
                float NdotWi_prev = fmaxf(dot(prev_N, unit_vector(ray.direction())), 0.0f);
                float pdf_cos = NdotWi_prev * 0.31830988618f;

                float pdf_nee = (pdf_area_sa + pdf_cos + prev_pdf) / 3.0f;

                float p2_brdf = prev_pdf * prev_pdf;
                float p2_nee  = pdf_nee  * pdf_nee;
                float w_brdf  = (p2_brdf + p2_nee > 0.0f)
                              ? p2_brdf / (p2_brdf + p2_nee) : 0.0f;

                radiance = radiance + throughput * Le * w_brdf;
            }
        }

        // Direct lighting: point lights via ShadeDirect
        Vec3 direct = ShadeDirect(ray, hitRecord, lights, numLights,
                                  numTriangles, nodes, aabbs, triangles);
        radiance = radiance + throughput * direct;

        const Vec3  N  = normalize(hitRecord.normal);
        const Vec3  Vo = unit_vector(-ray.direction());  // outgoing (toward camera)

        // ------------------------------------------------------------------
        // Next-Event Estimation (mode-dependent):
        //   nee_mode 0: Area sampling only — full weight, no MIS
        //   nee_mode 1: BRDF only — skip NEE entirely (emission via bounce)
        //   nee_mode 2: 3-strategy one-sample MIS (area + cosine + BRDF)
        // ------------------------------------------------------------------
        if (numEmissiveTris > 0 && nee_mode != 1) {
            // Pick strategy
            int strategy;
            if (nee_mode == 0) {
                strategy = 0;  // always area
            } else {
                // MIS: randomly pick one of 3 strategies
                const float strategy_u = rng_next(rng_state);
                strategy = (strategy_u < 0.333333f) ? 0
                         : (strategy_u < 0.666667f) ? 1 : 2;
            }

            Vec3  wi_nee       = make_vec3(0.0f, 0.0f, 0.0f);
            Vec3  Le_nee       = make_vec3(0.0f, 0.0f, 0.0f);
            float dist_nee     = 0.0f;
            float cosLight_nee = 0.0f;
            float NdotL_nee    = 0.0f;
            bool  nee_valid    = false;

            if (strategy == 0) {
                // --- Area sampling: pick emissive triangle via CDF, sample point ---
                const float u_sel = rng_next(rng_state);
                const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                const EmissiveTriInfo& emi = emissiveTris[eidx];
                const Triangle& eTri = triangles[emi.triangleIdx];

                float u1 = rng_next(rng_state);
                float u2 = rng_next(rng_state);
                if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                const Vec3 lightPoint = eTri.v0 * (1.0f - u1 - u2)
                                      + eTri.v1 * u1
                                      + eTri.v2 * u2;

                Vec3 toLight = lightPoint - hitRecord.p;
                const float r2 = dot(toLight, toLight);
                if (r2 > 1e-10f) {
                    const float r = sqrtf(r2);
                    wi_nee     = toLight * (1.0f / r);
                    NdotL_nee  = fmaxf(dot(N, wi_nee), 0.0f);
                    cosLight_nee = fabsf(dot(emi.normal, -wi_nee));

                    if (NdotL_nee > 0.0f && cosLight_nee > 0.0f) {
                        Ray shadowRay(hitRecord.p + N * RT_EPS, wi_nee);
                        HitRecord shadowHit{};
                        shadowHit.hit = false;
                        SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);

                        if (!shadowHit.hit || shadowHit.t >= r - RT_EPS) {
                            Le_nee   = emi.emission;
                            dist_nee = r;
                            nee_valid = true;
                        }
                    }
                }
            } else {
                // --- Cosine or BRDF direction sampling: trace to find emitter ---
                if (strategy == 1) {
                    wi_nee = cosine_hemisphere_sample(N, rng_state);
                } else {
                    float dummy_pdf;
                    wi_nee = SampleBRDF(hitRecord, Vo, rng_state, dummy_pdf, diffuse_bounce);
                }

                NdotL_nee = fmaxf(dot(N, wi_nee), 0.0f);
                if (NdotL_nee > 0.0f) {
                    Ray neeRay(hitRecord.p + N * RT_EPS, wi_nee);
                    HitRecord neeHit;
                    neeHit.hit = false;
                    SearchBVH(numTriangles, neeRay, nodes, aabbs, triangles, neeHit);

                    if (neeHit.hit) {
                        assignMaterialToHit(neeHit, numTriangles, triObjectIds,
                                            objectMaterials, numObjectMaterials);
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

            // Evaluate contribution
            if (nee_valid && dist_nee > 1e-6f && cosLight_nee > 1e-6f) {
                const float r2 = dist_nee * dist_nee;
                const float G  = cosLight_nee / r2;

                const float pdf_area_sa = (1.0f / totalEmissiveArea) / G;
                const Vec3  f_nee       = EvaluateBRDF(hitRecord, Vo, wi_nee);

                if (nee_mode == 0) {
                    // Area only: full weight, no MIS — just Le * f * NdotL / pdf
                    if (pdf_area_sa > 1e-10f) {
                        const Vec3 nee_contrib = Le_nee * f_nee
                                               * (NdotL_nee / pdf_area_sa);
                        radiance = radiance + throughput * nee_contrib;
                    }
                } else {
                    // MIS mode: 3-strategy combined PDF with power heuristic
                    const float pdf_cos  = NdotL_nee * 0.31830988618f;
                    const float pdf_brdf = BRDFSamplingPdf(hitRecord, Vo, wi_nee, diffuse_bounce);
                    const float pdf_combined = (pdf_area_sa + pdf_cos + pdf_brdf) / 3.0f;

                    if (pdf_combined > 1e-10f) {
                        const float p2_nee  = pdf_combined * pdf_combined;
                        const float p2_brdf = pdf_brdf     * pdf_brdf;
                        const float w_nee   = (p2_nee + p2_brdf > 0.0f)
                                            ? p2_nee / (p2_nee + p2_brdf) : 0.0f;

                        const Vec3 nee_contrib = Le_nee * f_nee
                                               * (NdotL_nee / pdf_combined) * w_nee;
                        radiance = radiance + throughput * nee_contrib;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Choose next bounce direction via unified BRDF importance sampling
        // ------------------------------------------------------------------
        prev_N = N;  // save for cosine PDF in emission-on-hit MIS next bounce

        const float kd = hitRecord.mat.kd;
        const float ks = hitRecord.mat.ks;
        const float kr = hitRecord.mat.kr;

        if (kd > 1e-6f || ks > 1e-6f) {
            float pdf;
            Vec3 wi = SampleBRDF(hitRecord, Vo, rng_state, pdf, diffuse_bounce);
            if (pdf < 1e-6f || dot(wi, N) < 0.0f) break;
            ray = Ray(hitRecord.p + N * RT_EPS, wi);
            const float NdotWi = fmaxf(dot(N, wi), 0.0f);
            const Vec3  f      = EvaluateBRDF(hitRecord, Vo, wi);
            throughput = throughput * (f * (NdotWi / pdf));
            prev_pdf   = pdf;
            prev_delta = false;
        } else {
            // Perfect mirror fallback for kr-only materials (delta BRDF, no PDF)
            const Vec3 reflDir = reflect_dir(unit_vector(ray.direction()), N);
            ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            throughput = throughput * (hitRecord.mat.specularColor * kr);
            prev_pdf   = 0.0f;
            prev_delta = true;
        }

        // ------------------------------------------------------------------
        // Russian Roulette (unbiased termination)
        // ------------------------------------------------------------------
        const float p_survive = fminf(luminance(throughput), 0.95f);
        if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
        throughput = throughput * (1.0f / p_survive);
    }

    // Return raw HDR radiance — tone mapping applied in main.cu at output time.
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
    bestHit.hit = false;
    bestHit.t = -1.0;
    bestHit.p = make_vec3(0.0f, 0.0f, 0.0f);
    bestHit.normal = make_vec3(0.0f, 0.0f, 0.0f);
    bestHit.front_face = false;
    bestHit.mat = Material();

    constexpr int STACK_CAPACITY = 512;
    std::uint32_t stack[STACK_CAPACITY];
    std::uint32_t* stack_ptr = stack;
    bool stackOverflow = false;
    *stack_ptr++ = 0; // root node is always 0


    while (stack_ptr > stack) {
        const std::uint32_t nodeIdx = *--stack_ptr;

        if (!intersectAABB(ray, aabbs[nodeIdx], tmin, bestT)) {
            continue;
        }

        const BVHNode node = nodes[nodeIdx];
        const std::uint32_t obj_idx = node.object_idx;

        if (obj_idx != 0xFFFFFFFF) {
            if (obj_idx < static_cast<std::uint32_t>(numTriangles)) {
                HitRecord rec = intersectTriangle(ray, triangles[obj_idx], tmin, bestT);
                if (rec.hit) {
                    rec.triangleIdx = static_cast<int>(obj_idx);
                    bestT = rec.t;
                    bestHit = rec;
                }
            }
            continue;
        }

        const std::uint32_t left_idx = node.left_idx;
        const std::uint32_t right_idx = node.right_idx;

        if (left_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[left_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) {
                    *stack_ptr++ = left_idx;
                } else {
                    stackOverflow = true;
                }
            }
        }

        if (right_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[right_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) {
                    *stack_ptr++ = right_idx;
                } else {
                    stackOverflow = true;
                }
            }
        }
    }

    // Safety fallback: if traversal overflowed, complete with brute-force test to avoid artifacts.
    if (stackOverflow) {
        for (int i = 0; i < numTriangles; ++i) {
            HitRecord rec = intersectTriangle(ray, triangles[i], tmin, bestT);
            if (rec.hit) {
                rec.triangleIdx = i;
                bestT = rec.t;
                bestHit = rec;
            }
        }
    }

    hitRecord = bestHit;
}
