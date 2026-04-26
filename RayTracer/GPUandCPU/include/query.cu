// include/query.cu
#include "buffers.h"
#include "query.h"
#include "bdpt.h"
#include "scene.h"
#include "shader.h"

#include <cfloat>
#include <cmath>


#ifdef __CUDACC__

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBatchCUDA(const int numTriangles,
       int W, int H,
       int max_depth,
       int sample_begin,
       int sample_end,
       const Camera cam,
       const Vec3 missColor,
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
       Vec3* __restrict__ albedo_aov,
       Vec3* __restrict__ normal_aov,
       int nee_mode,
       const HomogeneousMedium* __restrict__ objectMedia,
       int numObjectMedia,
       const TextureData* __restrict__ textures,
       int numTextures,
       // NEW: volume regions
       const VolumeRegionGPU* __restrict__ volumeRegions,
       int numVolumeRegions,
       Vec3* __restrict__ splat_buffer)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int pix_id = y * W + x;
    const unsigned int pixel_seed = (unsigned int)x * 73856093u ^ (unsigned int)y * 19349663u;

    Vec3 batch_accum = make_vec3(0.0f, 0.0f, 0.0f);

    for (int s = sample_begin; s < sample_end; ++s) {
        unsigned int h = pixel_seed ^ (unsigned int)(s * 83492791u);
        float jx = wang_hash_float(h) - 0.5f;
        h = h * 1664525u + 1013904223u;
        float jy = wang_hash_float(h) - 0.5f;

        Ray ray = cam.get_ray((float)x + jx, (float)y + jy);

        unsigned int rng = make_rng_seed(x, y, s);
        // BDPT integrator. Unused: lights[], diffuse_bounce, nee_mode, objectMedia.
        (void)lights; (void)numLights; (void)diffuse_bounce; (void)nee_mode;
        (void)objectMedia; (void)numObjectMedia;
        Vec3 color = bdpt_li(
            ray,
            max_depth,
            missColor,
            numTriangles,
            nodes, aabbs, triangles,
            triObjectIds, objectMaterials, numObjectMaterials,
            emissiveTris, emissiveCDF, numEmissiveTris, totalEmissiveArea,
            rng,
            textures, numTextures,
            volumeRegions, numVolumeRegions,
            splat_buffer, &cam, W, H
        );
        batch_accum = batch_accum + color;

        // Write AOV buffers on the very first sample
        if (s == 0 && albedo_aov != nullptr && normal_aov != nullptr) {
            HitRecord aovHit;
            SearchBVH(numTriangles, ray, nodes, aabbs, triangles, aovHit);
            if (aovHit.hit) {
                assignMaterialToHit(aovHit, numTriangles, triObjectIds,
                                    objectMaterials, numObjectMaterials,
                                    textures, numTextures);
                albedo_aov[pix_id] = aovHit.mat.albedo;
                normal_aov[pix_id] = normalize(aovHit.normal);
            } else {
                albedo_aov[pix_id] = missColor;
                normal_aov[pix_id] = make_vec3(0.0f, 0.0f, 0.0f);
            }
        }
    }

    output[pix_id] = output[pix_id] + batch_accum;
}

__global__ void normalizeCUDA(int W, int H, int spp, Vec3* __restrict__ output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix_id = y * W + x;
    output[pix_id] = output[pix_id] / float(spp);
}

// Merge t=1 light-tracing splat buffer into the output: output += splat / spp.
// Splat is accumulated unnormalized during rendering (one entry per (s, t=1)
// strategy hit per camera sample); dividing by spp gives the average per
// camera sample, which is the right per-pixel BDPT estimator since every
// pixel runs spp independent light paths whose t=1 splats land somewhere.
__global__ void mergeSplatCUDA(int W, int H, int spp,
                               Vec3* __restrict__ output,
                               const Vec3* __restrict__ splat) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix_id = y * W + x;
    Vec3 s = splat[pix_id] * (1.0f / float(spp));
    output[pix_id] = output[pix_id] + s;
}

#endif

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
    Vec3* __restrict__ albedo_aov,
    Vec3* __restrict__ normal_aov,
    int nee_mode,
    const HomogeneousMedium* __restrict__ objectMedia,
    int numObjectMedia,
    const TextureData* __restrict__ textures,
    int numTextures,
    // NEW: volume regions
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions)
{
#ifdef __CUDACC__
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Allocate + zero the t=1 splat buffer. One entry per pixel, atomic-added
    // by light_tracing_splat() inside bdpt_li.
    Vec3* d_splat = nullptr;
    const size_t splat_bytes = sizeof(Vec3) * size_t(W) * size_t(H);
    if (cudaMalloc(reinterpret_cast<void**>(&d_splat), splat_bytes) != cudaSuccess) {
        d_splat = nullptr;
    } else {
        cudaMemset(d_splat, 0, splat_bytes);
    }

    for (int s = 0; s < spp; s += SAMPLES_PER_BATCH) {
        int batch_end = s + SAMPLES_PER_BATCH;
        if (batch_end > spp) batch_end = spp;

        renderBatchCUDA<<<tile_grid, block>>>(
            numTriangles,
            W, H,
            max_depth,
            s,
            batch_end,
            cam,
            missColor,
            nodes,
            aabbs,
            triangles,
            triObjectIds,
            objectMaterials,
            numObjectMaterials,
            lights,
            numLights,
            diffuse_bounce,
            emissiveTris,
            emissiveCDF,
            numEmissiveTris,
            totalEmissiveArea,
            output,
            albedo_aov,
            normal_aov,
            nee_mode,
            objectMedia, numObjectMedia,
            textures, numTextures,
            volumeRegions, numVolumeRegions,
            d_splat
        );
    }

    normalizeCUDA<<<tile_grid, block>>>(W, H, spp, output);
    if (d_splat != nullptr) {
        mergeSplatCUDA<<<tile_grid, block>>>(W, H, spp, output, d_splat);
        cudaFree(d_splat);
    }
    CHECK_CUDA((cudaDeviceSynchronize()), true);

#else
    if (nodes == nullptr || aabbs == nullptr || triangles == nullptr || output == nullptr)
        return;

    const int triCount = static_cast<int>(numTriangles);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix_id = W * y + x;
            Vec3 col{0,0,0};

            auto offsets = jittered_samples(spp, 42u);

            for (int si = 0; si < (int)offsets.size(); ++si) {
                float px = float(x) + offsets[si].first;
                float py = float(y) + offsets[si].second;

                const Ray ray = cam.get_ray(px, py);

                unsigned int rng = make_rng_seed(x, y, si);
                (void)lights; (void)numLights; (void)diffuse_bounce; (void)nee_mode;
                (void)objectMedia; (void)numObjectMedia;
                col = col + bdpt_li(
                    ray,
                    max_depth,
                    missColor,
                    triCount,
                    nodes, aabbs, triangles,
                    triObjectIds, objectMaterials, numObjectMaterials,
                    emissiveTris, emissiveCDF, numEmissiveTris, totalEmissiveArea,
                    rng,
                    textures, numTextures,
                    volumeRegions, numVolumeRegions
                );

                // Write AOVs on first sample
                if (si == 0 && albedo_aov != nullptr && normal_aov != nullptr) {
                    HitRecord aovHit;
                    SearchBVH(triCount, ray, nodes, aabbs, triangles, aovHit);
                    if (aovHit.hit) {
                        assignMaterialToHit(aovHit, triCount, triObjectIds,
                                            objectMaterials, numObjectMaterials,
                                            textures, numTextures);
                        albedo_aov[pix_id] = aovHit.mat.albedo;
                        normal_aov[pix_id] = normalize(aovHit.normal);
                    } else {
                        albedo_aov[pix_id] = missColor;
                        normal_aov[pix_id] = make_vec3(0.0f, 0.0f, 0.0f);
                    }
                }
            }
            output[pix_id] = col / float(spp);
        }
    }
#endif
}
