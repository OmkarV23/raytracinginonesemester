#include "MeshOBJ.h"
#include "buffers.h"
#include "bvh.h"
#include "visualizer.h"
#include "warmup.h"
#include "scene.h"
#include "camera.h"
#include "query.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>

#ifdef __CUDACC__
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK(call)                                                         \
    do {                                                                          \
        OptixResult res = call;                                                   \
        if (res != OPTIX_SUCCESS) {                                               \
            std::fprintf(stderr, "OptiX error: %s at %s:%d\n",                   \
                         optixGetErrorString(res), __FILE__, __LINE__);           \
        }                                                                         \
    } while (0)
#endif

#ifdef __CUDACC__
__global__ void buildTrianglesKernel(const MeshView mesh, Triangle* out, int numTriangles) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    const uint32_t i0 = mesh.indices[idx * 3 + 0];
    const uint32_t i1 = mesh.indices[idx * 3 + 1];
    const uint32_t i2 = mesh.indices[idx * 3 + 2];

    const Vec3 v0 = mesh.positions[i0];
    const Vec3 v1 = mesh.positions[i1];
    const Vec3 v2 = mesh.positions[i2];

    Vec3 n0 = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 n1 = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 n2 = make_vec3(0.0f, 0.0f, 0.0f);
    if (mesh.normals != nullptr) {
        n0 = mesh.normals[i0];
        n1 = mesh.normals[i1];
        n2 = mesh.normals[i2];
    }

    out[idx] = Triangle(v0, v1, v2, n0, n1, n2);
}

#endif

RayTracer::BVHState RayTracer::BVHState::fromChunk(char*& chunk, size_t P)
{
    BVHState state;
    obtain(chunk, state.Nodes, 2 * P - 1, 128);
    obtain(chunk, state.AABBs, 2 * P - 1, 128);
    return state;
}

static inline float deg2rad(const float d) {
    return d * 0.01745329251994329577f;
}

static inline Vec3 rotateXYZ(Vec3 v, const Vec3& rotationDeg) {
    const float rx = deg2rad(rotationDeg.x);
    const float ry = deg2rad(rotationDeg.y);
    const float rz = deg2rad(rotationDeg.z);

    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);

    // Rotate around X
    v = make_vec3(v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z);
    // Rotate around Y
    v = make_vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    // Rotate around Z
    v = make_vec3(cz * v.x - sz * v.y, sz * v.x + cz * v.y, v.z);
    return v;
}

static inline void applyObjectTransform(Mesh& mesh, const SceneObject& obj) {
    for (auto& p : mesh.positions) {
        Vec3 scaled = make_vec3(p.x * obj.scale.x, p.y * obj.scale.y, p.z * obj.scale.z);
        Vec3 rotated = rotateXYZ(scaled, obj.rotation);
        p = rotated + obj.position;
    }

    for (auto& n : mesh.normals) {
        Vec3 nScaled = n;
        if (fabsf(obj.scale.x) > 1e-8f) nScaled.x /= obj.scale.x;
        if (fabsf(obj.scale.y) > 1e-8f) nScaled.y /= obj.scale.y;
        if (fabsf(obj.scale.z) > 1e-8f) nScaled.z /= obj.scale.z;
        Vec3 nRot = rotateXYZ(nScaled, obj.rotation);
        const float len2 = dot(nRot, nRot);
        if (len2 > 1e-12f) {
            const float invLen = 1.0f / sqrtf(len2);
            n = nRot * invLen;
        } else {
            n = make_vec3(0.0f, 0.0f, 1.0f);
        }
    }
}

int main(int argc, char** argv)
{

    using vec3 = Vec3;
    using point3 = Vec3;

    // Parse optional flags
    bool use_denoiser = false;
    std::string output_filename = "render.png";
    int nee_mode = 2;  // 0 = area only, 1 = brdf only, 2 = 3-strategy MIS (default)
    std::vector<char*> positional_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--denoise" || arg == "-d") {
            use_denoiser = true;
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_filename = argv[++i];
        } else if (arg == "--nee-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "area")      nee_mode = 0;
            else if (mode == "brdf") nee_mode = 1;
            else if (mode == "mis")  nee_mode = 2;
            else { printf("Unknown --nee-mode '%s'. Use: area, brdf, mis\n", mode.c_str()); return 1; }
        } else {
            positional_args.push_back(argv[i]);
        }
    }
    const char* nee_names[] = {"area", "brdf", "mis"};
    printf("NEE mode: %s\n", nee_names[nee_mode]);
    // Rebuild argc/argv as positional-only for scene loading below
    int pos_argc = static_cast<int>(positional_args.size()) + 1;

    std::vector<SceneObject> load_objects;
    Scene scene;
    bool has_scene = false;
    if (pos_argc >= 2) {
        std::string first = positional_args[0];
        const bool is_scene =
            (first.size() >= 5 && first.substr(first.size() - 5) == ".json") ||
            (first.size() >= 6 && first.substr(first.size() - 6) == ".scene");
        if (is_scene) {
            std::string err;
            if (!SceneIO::LoadSceneFromFile(first, scene, &err)) {
                std::cerr << "Failed to load scene: " << err << "\n";
                return 1;
            }
            has_scene = true;
            const std::string base_dir = SceneIO::dirname(first);
            const std::string project_dir = SceneIO::dirname(SceneIO::dirname(base_dir));
            auto file_exists = [](const std::string& p) {
                std::ifstream f(p);
                return static_cast<bool>(f);
            };
            for (const auto& obj : scene.objects) {
                if (!obj.type.empty() && obj.type != "mesh") continue;
                SceneObject resolved = obj;
                std::string path = resolved.path;
                if (!SceneIO::is_abs_path(path)) {
                    const std::string scene_relative = SceneIO::join_path(base_dir, path);
                    std::string project_relative = path;
                    if (project_relative.rfind("./", 0) == 0) {
                        project_relative = project_relative.substr(2);
                    }
                    project_relative = SceneIO::join_path(project_dir, project_relative);

                    if (file_exists(scene_relative)) {
                        path = scene_relative;
                    } else if (file_exists(path)) {
                        // Keep cwd-relative path as-is.
                    } else if (file_exists(project_relative)) {
                        path = project_relative;
                    } else {
                        // Fall back to scene-relative for clearer diagnostics.
                        path = scene_relative;
                    }
                }
                resolved.path = path;
                load_objects.push_back(resolved);
            }
        } else {
            for (const auto& parg : positional_args) {
                SceneObject obj;
                obj.path = parg;
                load_objects.push_back(obj);
            }
        }
    } else {
        SceneObject obj;
        obj.path = "../assets/meshes/frog.obj";
        load_objects.push_back(obj);
    }

    Mesh globalMesh;
    std::vector<Material> objectMaterials;
    int nextObjectId = 0;

    for (const auto& obj : load_objects)
    {
        const std::string& path = obj.path;
        std::printf("Loading OBJ: %s\n", path.c_str());
        Mesh tempMesh;
        const int objIdBegin = nextObjectId;
        if (!LoadOBJ_ToMesh(path, tempMesh, nextObjectId))
        {
            std::cerr << "Failed to load OBJ: " << path << "\n";
            continue;
        }
        applyObjectTransform(tempMesh, obj);

        if (objectMaterials.size() < static_cast<size_t>(nextObjectId)) {
            objectMaterials.resize(nextObjectId, Material());
        }
        for (int oid = objIdBegin; oid < nextObjectId; ++oid) {
            objectMaterials[oid] = obj.material;
        }

        std::printf("  -> Loaded %zu triangles.\n", tempMesh.indices.size() / 3);
        AppendMesh(globalMesh, tempMesh);
    }

    if (globalMesh.positions.empty())
    {
        std::cerr << "No valid geometry loaded.\n";
        return 1;
    }
    AABB sceneAABB;

    size_t P = globalMesh.indices.size() / 3;
    size_t bvh_chunk_size = required<RayTracer::BVHState>(P);
    char* bvh_chunk = nullptr;
#ifdef __CUDACC__
    cudaError_t alloc_err = cudaMalloc(&bvh_chunk, bvh_chunk_size);
    if (alloc_err != cudaSuccess) {
        std::fprintf(stderr, "Failed to allocate device memory for BVH: %s\n", cudaGetErrorString(alloc_err));
        return 1;
    }
#else
    bvh_chunk = new char[bvh_chunk_size];
#endif
    RayTracer::BVHState bvhState = RayTracer::BVHState::fromChunk(bvh_chunk, P);

    AccStruct::BVH bvh;
    
#ifdef __CUDACC__
    Vec3* d_positions = nullptr;
    Vec3* d_normals = nullptr;
    uint32_t* d_indices = nullptr;
    int32_t* d_triangle_obj_ids = nullptr;
    Material* d_object_materials = nullptr;

    const size_t bytesPos = globalMesh.positions.size() * sizeof(Vec3);
    const size_t bytesIdx = globalMesh.indices.size() * sizeof(uint32_t);
    const size_t bytesNrm = globalMesh.normals.size() * sizeof(Vec3);
    const size_t bytesTriObj = globalMesh.triangleObjIds.size() * sizeof(int32_t);
    const size_t bytesObjMat = objectMaterials.size() * sizeof(Material);

    CHECK_CUDA((cudaMalloc(&d_positions, bytesPos)), true);
    CHECK_CUDA((cudaMalloc(&d_indices, bytesIdx)), true);
    CHECK_CUDA((cudaMalloc(&d_triangle_obj_ids, bytesTriObj)), true);
    CHECK_CUDA((cudaMalloc(&d_object_materials, bytesObjMat)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMalloc(&d_normals, bytesNrm)), true);
    }

    CHECK_CUDA((cudaMemcpy(d_positions, globalMesh.positions.data(), bytesPos, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_indices, globalMesh.indices.data(), bytesIdx, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_triangle_obj_ids, globalMesh.triangleObjIds.data(), bytesTriObj, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_object_materials, objectMaterials.data(), bytesObjMat, cudaMemcpyHostToDevice)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMemcpy(d_normals, globalMesh.normals.data(), bytesNrm, cudaMemcpyHostToDevice)), true);
    }

    MeshView d_mesh{};
    d_mesh.positions = d_positions;
    d_mesh.normals = d_normals;
    d_mesh.uvs = nullptr;
    d_mesh.indices = d_indices;
    d_mesh.triangleObjIds = d_triangle_obj_ids;
    d_mesh.numVertices = globalMesh.positions.size();
    d_mesh.numIndices = globalMesh.indices.size();
    d_mesh.numTriangles = P;

    CHECK_CUDA(bvh.calculateAABBs(d_mesh, bvhState.AABBs), true);
#else
    MeshView h_mesh = globalMesh.getView();
    bvh.calculateAABBs(h_mesh, bvhState.AABBs);
#endif

    AABB SceneBoundingBox;
    #ifdef __CUDACC__
        AABB default_aabb;
        // Reduce directly on device memory
        SceneBoundingBox = thrust::reduce(
            thrust::device_pointer_cast(bvhState.AABBs + (P - 1)),
            thrust::device_pointer_cast(bvhState.AABBs + (2*P - 1)),
            default_aabb,
            [] __device__ __host__ (const AABB& lhs, const AABB& rhs) {
                return AABB::merge(lhs, rhs);
            });

        thrust::device_vector<unsigned int> TriangleIndices(P);
        thrust::copy(thrust::make_counting_iterator<std::uint32_t>(0),
            thrust::make_counting_iterator<std::uint32_t>(P),
            TriangleIndices.begin());

        // --- GPU Warmup (Lightweight) ---
        // This initializes Thrust's internal allocators and kernels without touching real data.
        warmupGPU();

        auto start_gpu = std::chrono::high_resolution_clock::now();

        bvh.buildBVH(
            bvhState.Nodes,
            bvhState.AABBs,
            SceneBoundingBox,
            &TriangleIndices,
            static_cast<int>(P)
        );
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_gpu = end_gpu - start_gpu;
        printf("GPU LBVH Build Time: %.3f ms\n", ms_gpu.count());

    #else
        SceneBoundingBox = std::accumulate(
            bvhState.AABBs + (P - 1),
            bvhState.AABBs + (2 * P - 1),
            AABB(),
            [](const AABB& lhs, const AABB& rhs) {
                return AABB::merge(lhs, rhs);
            });

        std::vector<unsigned int> TriangleIndices(P);
        std::iota(TriangleIndices.begin(), TriangleIndices.end(), 0);
        auto start_cpu =  std::chrono::high_resolution_clock::now();
        bvh.buildBVH(
            bvhState.Nodes,
            bvhState.AABBs,
            SceneBoundingBox,
            TriangleIndices,
            static_cast<int>(P)
        );

        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;
        printf("CPU LBVH Build Time: %.3f ms\n", ms_cpu.count());

    #endif

    // --- Camera and Ray Generation ---
    int max_depth = has_scene ? scene.settings.max_depth : 1;
    int spp = has_scene ? scene.settings.spp : 1;
    bool diffuse_bounce = has_scene ? scene.settings.diffuse_bounce : true;

    Vec3 miss_color = has_scene ? scene.miss_color : make_vec3(0.0f, 0.0f, 0.0f);
    Camera cam = has_scene ? scene.camera : Camera();
    std::vector<Light> render_lights = scene.lights;
    // Filter out area lights (type==1) — they are now handled as emissive meshes.
    // Keep only point lights (type==0).
    {
        std::vector<Light> point_only;
        for (const auto& l : render_lights) {
            if (l.type == 0) point_only.push_back(l);
        }
        render_lights = std::move(point_only);
    }
    const int num_lights = static_cast<int>(render_lights.size());
    const int num_object_materials = static_cast<int>(objectMaterials.size());

    // ---- Build emissive triangle list (Mitsuba-style area lights) ----
    std::vector<EmissiveTriInfo> h_emissiveTris;
    std::vector<float> h_emissiveCDF;
    float totalEmissiveArea = 0.0f;
    {
        // We need triangle geometry on the host to compute areas/normals.
        // Build a temporary host triangle list (reused later for CPU path).
        std::vector<Triangle> tmpTris(P);
        for (size_t i = 0; i < P; ++i) {
            const uint32_t i0 = globalMesh.indices[i * 3 + 0];
            const uint32_t i1 = globalMesh.indices[i * 3 + 1];
            const uint32_t i2 = globalMesh.indices[i * 3 + 2];
            Vec3 n0 = make_vec3(0,0,0), n1 = make_vec3(0,0,0), n2 = make_vec3(0,0,0);
            if (!globalMesh.normals.empty()) {
                n0 = globalMesh.normals[i0];
                n1 = globalMesh.normals[i1];
                n2 = globalMesh.normals[i2];
            }
            tmpTris[i] = Triangle(globalMesh.positions[i0], globalMesh.positions[i1],
                                  globalMesh.positions[i2], n0, n1, n2);
        }

        for (size_t i = 0; i < P; ++i) {
            const int objId = globalMesh.triangleObjIds[i];
            if (objId < 0 || objId >= num_object_materials) continue;
            const Vec3& em = objectMaterials[objId].emission;
            if (em.x <= 0.0f && em.y <= 0.0f && em.z <= 0.0f) continue;

            const Triangle& tri = tmpTris[i];
            Vec3 e1 = tri.v1 - tri.v0;
            Vec3 e2 = tri.v2 - tri.v0;
            Vec3 cr = cross(e1, e2);
            float triArea = 0.5f * sqrtf(dot(cr, cr));
            if (triArea < 1e-12f) continue;

            float crLen = sqrtf(dot(cr, cr));
            Vec3 faceN = cr * (1.0f / crLen);

            EmissiveTriInfo info;
            info.triangleIdx = static_cast<int>(i);
            info.emission = em;
            info.area = triArea;
            info.normal = faceN;
            h_emissiveTris.push_back(info);
            totalEmissiveArea += triArea;
        }

        // Build area-weighted CDF for sampling
        h_emissiveCDF.resize(h_emissiveTris.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < h_emissiveTris.size(); ++i) {
            cumulative += h_emissiveTris[i].area;
            h_emissiveCDF[i] = cumulative / totalEmissiveArea;
        }
        if (!h_emissiveCDF.empty()) h_emissiveCDF.back() = 1.0f; // ensure exact 1.0

        printf("Emissive triangles: %zu  (total area: %.4f)\n",
               h_emissiveTris.size(), totalEmissiveArea);
    }
    const int numEmissiveTris = static_cast<int>(h_emissiveTris.size());

    const int img_w = cam.pixel_width;
    const int img_h = cam.pixel_height;
    const int num_pixels = img_w * img_h;
    // const int num_rays = num_pixels * spp;
    std::vector<Vec3> image(num_pixels, make_vec3(0.0f, 0.0f, 0.0f));

#ifdef __CUDACC__
    Triangle* d_tris = nullptr;
    Vec3* d_image = nullptr;
    Light* d_lights = nullptr;

    CHECK_CUDA((cudaMalloc(&d_tris, sizeof(Triangle) * P)), true);
    CHECK_CUDA((cudaMalloc(&d_image, sizeof(Vec3) * img_w * img_h)), true);
    CHECK_CUDA((cudaMalloc(&d_lights, sizeof(Light) * num_lights)), true);
    CHECK_CUDA((cudaMemcpy(d_lights, render_lights.data(), sizeof(Light) * num_lights, cudaMemcpyHostToDevice)), true);

    // Upload emissive triangle data
    EmissiveTriInfo* d_emissiveTris = nullptr;
    float* d_emissiveCDF = nullptr;
    if (numEmissiveTris > 0) {
        CHECK_CUDA((cudaMalloc(&d_emissiveTris, sizeof(EmissiveTriInfo) * numEmissiveTris)), true);
        CHECK_CUDA((cudaMemcpy(d_emissiveTris, h_emissiveTris.data(), sizeof(EmissiveTriInfo) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
        CHECK_CUDA((cudaMalloc(&d_emissiveCDF, sizeof(float) * numEmissiveTris)), true);
        CHECK_CUDA((cudaMemcpy(d_emissiveCDF, h_emissiveCDF.data(), sizeof(float) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
    }

    // AOV buffers for OptiX denoiser (only allocated when --denoise is used)
    Vec3* d_albedo_aov = nullptr;
    Vec3* d_normal_aov = nullptr;
    if (use_denoiser) {
        CHECK_CUDA((cudaMalloc(&d_albedo_aov, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMalloc(&d_normal_aov, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMemset(d_albedo_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMemset(d_normal_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
    }

    const int threads = 256;
    const int tri_blocks = (static_cast<int>(P) + threads - 1) / threads;
    buildTrianglesKernel<<<tri_blocks, threads>>>(d_mesh, d_tris, static_cast<int>(P));
    CHECK_CUDA((cudaDeviceSynchronize()), true);

    // Warm up with a tiny launch to pay first-launch/JIT cost without full-frame work.
    render(P, 1, 1, cam, miss_color, max_depth, 1, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce,
           d_emissiveTris, d_emissiveCDF, numEmissiveTris, totalEmissiveArea,
           d_image, nullptr, nullptr, nee_mode);

    // Zero buffers before the real render (warmup wrote into them)
    CHECK_CUDA((cudaMemset(d_image, 0, sizeof(Vec3) * img_w * img_h)), true);
    if (use_denoiser) {
        CHECK_CUDA((cudaMemset(d_albedo_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMemset(d_normal_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
    }

    // clock results for render
    auto start_render = std::chrono::high_resolution_clock::now();
    render(P, img_w, img_h, cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce,
           d_emissiveTris, d_emissiveCDF, numEmissiveTris, totalEmissiveArea,
           d_image, d_albedo_aov, d_normal_aov, nee_mode);

    auto end_render = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
    printf("GPU Render Time: %.3f ms\n", ms_render.count());

    // ---- OptiX AI Denoiser (optional: --denoise / -d) ----
    if (use_denoiser) {
        auto start_denoise = std::chrono::high_resolution_clock::now();

        // Initialize OptiX
        OPTIX_CHECK(optixInit());

        CUcontext cuCtx = nullptr;  // use current CUDA context
        OptixDeviceContext optixCtx = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, nullptr, &optixCtx));

        // Create denoiser — HDR model with albedo + normal guides
        OptixDenoiserOptions denoiserOptions = {};
        denoiserOptions.guideAlbedo = 1;
        denoiserOptions.guideNormal = 1;
        denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

        OptixDenoiser denoiser = nullptr;
        OPTIX_CHECK(optixDenoiserCreate(optixCtx, OPTIX_DENOISER_MODEL_KIND_HDR,
                                        &denoiserOptions, &denoiser));

        // Compute memory requirements
        OptixDenoiserSizes denoiserSizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser,
                        static_cast<unsigned int>(img_w),
                        static_cast<unsigned int>(img_h),
                        &denoiserSizes));

        CUdeviceptr d_denoiserState = 0;
        CUdeviceptr d_denoiserScratch = 0;
        CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_denoiserState),
                               denoiserSizes.stateSizeInBytes)), true);
        CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_denoiserScratch),
                               denoiserSizes.withoutOverlapScratchSizeInBytes)), true);

        OPTIX_CHECK(optixDenoiserSetup(denoiser, nullptr,
                        static_cast<unsigned int>(img_w),
                        static_cast<unsigned int>(img_h),
                        d_denoiserState,   denoiserSizes.stateSizeInBytes,
                        d_denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));

        // Compute HDR intensity for stable denoising
        CUdeviceptr d_hdrIntensity = 0;
        CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_hdrIntensity), sizeof(float))), true);

        // Scratch buffer for intensity computation
        CUdeviceptr d_intensityScratch = 0;
        CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_intensityScratch),
                               denoiserSizes.computeIntensitySizeInBytes)), true);

        // Helper lambda to set up an OptixImage2D from a device Vec3* buffer
        auto makeImage2D = [&](CUdeviceptr ptr) -> OptixImage2D {
            OptixImage2D img = {};
            img.data               = ptr;
            img.width              = static_cast<unsigned int>(img_w);
            img.height             = static_cast<unsigned int>(img_h);
            img.rowStrideInBytes   = static_cast<unsigned int>(img_w * sizeof(Vec3));
            img.pixelStrideInBytes = static_cast<unsigned int>(sizeof(Vec3));
            img.format             = OPTIX_PIXEL_FORMAT_FLOAT3;
            return img;
        };

        OptixImage2D colorImg  = makeImage2D(reinterpret_cast<CUdeviceptr>(d_image));
        OptixImage2D albedoImg = makeImage2D(reinterpret_cast<CUdeviceptr>(d_albedo_aov));
        OptixImage2D normalImg = makeImage2D(reinterpret_cast<CUdeviceptr>(d_normal_aov));

        OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser, nullptr,
                        &colorImg, d_hdrIntensity,
                        d_intensityScratch, denoiserSizes.computeIntensitySizeInBytes));

        // Denoised output buffer (in-place: overwrite d_image)
        OptixImage2D outputImg = colorImg;

        OptixDenoiserGuideLayer guideLayer = {};
        guideLayer.albedo = albedoImg;
        guideLayer.normal = normalImg;

        OptixDenoiserLayer layer = {};
        layer.input  = colorImg;
        layer.output = outputImg;

        OptixDenoiserParams params = {};
        params.hdrIntensity  = d_hdrIntensity;
        params.blendFactor   = 0.0f;  // 0 = fully denoised
        params.hdrAverageColor = 0;
        params.temporalModeUsePreviousLayers = 0;

        OPTIX_CHECK(optixDenoiserInvoke(denoiser, nullptr, &params,
                        d_denoiserState, denoiserSizes.stateSizeInBytes,
                        &guideLayer, &layer, 1,
                        0, 0,  // input offset x, y
                        d_denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));

        CHECK_CUDA((cudaDeviceSynchronize()), true);

        auto end_denoise = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_denoise = end_denoise - start_denoise;
        printf("OptiX Denoise Time: %.3f ms\n", ms_denoise.count());

        // Cleanup denoiser resources
        cudaFree(reinterpret_cast<void*>(d_denoiserState));
        cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
        cudaFree(reinterpret_cast<void*>(d_hdrIntensity));
        cudaFree(reinterpret_cast<void*>(d_intensityScratch));
        optixDenoiserDestroy(denoiser);
        optixDeviceContextDestroy(optixCtx);
    }

    CHECK_CUDA((cudaMemcpy(image.data(), d_image, sizeof(Vec3) * img_w * img_h, cudaMemcpyDeviceToHost)), true);

    cudaFree(d_tris);
    cudaFree(d_image);
    if (d_albedo_aov) cudaFree(d_albedo_aov);
    if (d_normal_aov) cudaFree(d_normal_aov);
    cudaFree(d_lights);
    if (d_emissiveTris) cudaFree(d_emissiveTris);
    if (d_emissiveCDF) cudaFree(d_emissiveCDF);
    cudaFree(d_positions);
    cudaFree(d_normals);
    cudaFree(d_indices);
    cudaFree(d_triangle_obj_ids);
    cudaFree(d_object_materials);
#else
    std::vector<Triangle> h_tris(P);
    for (size_t i = 0; i < P; ++i) {
        const uint32_t i0 = globalMesh.indices[i * 3 + 0];
        const uint32_t i1 = globalMesh.indices[i * 3 + 1];
        const uint32_t i2 = globalMesh.indices[i * 3 + 2];

        Vec3 n0 = make_vec3(0.0f, 0.0f, 0.0f);
        Vec3 n1 = make_vec3(0.0f, 0.0f, 0.0f);
        Vec3 n2 = make_vec3(0.0f, 0.0f, 0.0f);
        if (!globalMesh.normals.empty()) {
            n0 = globalMesh.normals[i0];
            n1 = globalMesh.normals[i1];
            n2 = globalMesh.normals[i2];
        }

        h_tris[i] = Triangle(globalMesh.positions[i0], globalMesh.positions[i1], globalMesh.positions[i2], n0, n1, n2);
    }
    auto start_render = std::chrono::high_resolution_clock::now();
    render(P, img_w, img_h, cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, h_tris.data(),
           globalMesh.triangleObjIds.data(), objectMaterials.data(), num_object_materials,
           render_lights.data(), num_lights, diffuse_bounce,
           h_emissiveTris.data(), h_emissiveCDF.data(), numEmissiveTris, totalEmissiveArea,
           image.data(), nullptr, nullptr, nee_mode);
    auto end_render = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
    printf("CPU Render Time: %.3f ms\n", ms_render.count());
#endif


//     std::cout << "Exporting all AABBs to file...\n";
//     size_t totalNodes = 2 * P - 1;
//     std::vector<AABB> allAABBs(totalNodes);
// #ifdef __CUDACC__
//     cudaMemcpy(allAABBs.data(), bvhState.AABBs, sizeof(AABB) * totalNodes, cudaMemcpyDeviceToHost);
// #else
//     for(size_t i=0; i<totalNodes; ++i) allAABBs[i] = bvhState.AABBs[i];
// #endif
//     ExportAABBsToOBJ("bvh_.obj", allAABBs.data(), totalNodes);

    // Write image to disk with Reinhard tone mapping.
    auto reinhard = [](float c) -> unsigned char {
        float mapped = c / (1.0f + c);                   // compress [0,∞) → [0,1)
        mapped = powf(fmaxf(mapped, 0.0f), 1.0f / 2.2f); // gamma correction
        return static_cast<unsigned char>(255.0f * fminf(mapped, 1.0f));
    };
    std::vector<unsigned char> img_data(num_pixels * 3);
    for (size_t i = 0; i < num_pixels; ++i) {
        img_data[i * 3 + 0] = reinhard(image[i].x);
        img_data[i * 3 + 1] = reinhard(image[i].y);
        img_data[i * 3 + 2] = reinhard(image[i].z);
    }
    stbi_write_png(output_filename.c_str(), img_w, img_h, 3, img_data.data(), img_w * 3);
    printf("Image saved to %s\n", output_filename.c_str());

    return 0;
}
