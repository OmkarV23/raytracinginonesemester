#include "MeshOBJ.h"
#include "buffers.h"
#include "bvh.h"
#include "visualizer.h"
#include "warmup.h"
#include "scene.h"
#include "camera.h"
#include "query.h"
#include "material.h"
#include "texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>
#include <filesystem>


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

    
    Vec2 uv0 = make_vec2(0.0f, 0.0f);
    Vec2 uv1 = make_vec2(0.0f, 0.0f);
    Vec2 uv2 = make_vec2(0.0f, 0.0f);
    if (mesh.uvs != nullptr) {
        uv0 = mesh.uvs[i0];
        uv1 = mesh.uvs[i1];
        uv2 = mesh.uvs[i2];
    }

    Triangle tri = Triangle(v0, v1, v2, n0, n1, n2);
    tri.uv0 = uv0;
    tri.uv1 = uv1;
    tri.uv2 = uv2;
    out[idx] = tri;
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


Texture* loadTexture(const std::string& path)
{

    static std::unordered_map<std::string, std::unique_ptr<Texture>> g_textureCache;
    // 1. Check if texture is already in the cache.
    auto it = g_textureCache.find(path);
    if (it != g_textureCache.end()) {
        return it->second.get();  // return existing Texture*
    }

    // 2. Not in cache: load from disk.
    int w, h, n;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!pixels) {
        std::fprintf(stderr, "Failed to load texture: %s\n", path.c_str());
        return nullptr;
    }

    // 3. Create a new Texture object and fill it.
    auto tex = std::make_unique<Texture>();
    tex->width = w;
    tex->height = h;
    tex->channels = n;
    tex->data.assign(pixels, pixels + (w * h * n));
    tex->refreshSampledView();

    stbi_image_free(pixels);

    Texture* ptr = tex.get();
    g_textureCache[path] = std::move(tex);
    return ptr;
}

int main(int argc, char** argv)
{

    using vec3 = Vec3;
    using point3 = Vec3;

    std::vector<SceneObject> load_objects;
    Scene scene;
    bool has_scene = false;
    if (argc >= 2) {
        std::string first = argv[1];
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
            auto resolve_scene_relative_path = [&](const std::string& in_path) -> std::string {
                if (in_path.empty() || SceneIO::is_abs_path(in_path)) {
                    return in_path;
                }
                const std::string scene_relative = SceneIO::join_path(base_dir, in_path);
                std::string project_relative = in_path;
                if (project_relative.rfind("./", 0) == 0) {
                    project_relative = project_relative.substr(2);
                }
                project_relative = SceneIO::join_path(project_dir, project_relative);

                if (file_exists(scene_relative)) return scene_relative;
                if (file_exists(in_path)) return in_path; // cwd-relative
                if (file_exists(project_relative)) return project_relative;
                return scene_relative; // keep best-effort path for diagnostics
            };
            for (const auto& obj : scene.objects) {
                if (!obj.type.empty() && obj.type != "mesh") continue;
                SceneObject resolved = obj;
                resolved.path = resolve_scene_relative_path(resolved.path);
                resolved.material.albedoMapPath = resolve_scene_relative_path(resolved.material.albedoMapPath);
                resolved.material.bumpMapPath = resolve_scene_relative_path(resolved.material.bumpMapPath);
                resolved.material.normalMapPath = resolve_scene_relative_path(resolved.material.normalMapPath);
                load_objects.push_back(resolved);
            }
        } else {
            for (int i = 1; i < argc; ++i) {
                SceneObject obj;
                obj.path = argv[i];
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

    for (auto& obj : load_objects)
    {
                if (!obj.material.albedoMapPath.empty()) {
            obj.material.albedo_map = loadTexture(obj.material.albedoMapPath);
        } else {
            obj.material.albedo_map = nullptr;
        }

        if (!obj.material.bumpMapPath.empty()) {
            obj.material.bump_map = loadTexture(obj.material.bumpMapPath);
        } else {
            obj.material.bump_map = nullptr;
        }

        if (!obj.material.normalMapPath.empty()) {
            obj.material.normal_map = loadTexture(obj.material.normalMapPath);
        } else {
            obj.material.normal_map = nullptr;
        }
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

    std::vector<MaterialData> renderMaterials;
    renderMaterials.reserve(objectMaterials.size());
    for (const auto& m : objectMaterials) {
        renderMaterials.push_back(ToMaterialData(m));
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
    Vec2* d_uvs = nullptr;
    uint32_t* d_indices = nullptr;
    int32_t* d_triangle_obj_ids = nullptr;
    MaterialData* d_object_materials = nullptr;
    std::vector<MaterialData> renderMaterialsGPU = renderMaterials;
    std::vector<unsigned char*> d_texture_pixels;
    std::vector<TextureData*> d_texture_views;
    std::unordered_map<const TextureData*, TextureData*> texture_view_map;

    auto uploadTextureView = [&](TextureData* hostView) -> TextureData* {
        if (hostView == nullptr || hostView->data == nullptr ||
            hostView->width <= 0 || hostView->height <= 0 || hostView->channels <= 0) {
            return nullptr;
        }
        auto it = texture_view_map.find(hostView);
        if (it != texture_view_map.end()) {
            return it->second;
        }

        const size_t numBytes = static_cast<size_t>(hostView->width) *
                                static_cast<size_t>(hostView->height) *
                                static_cast<size_t>(hostView->channels);
        unsigned char* d_pixels = nullptr;
        TextureData* d_view = nullptr;
        CHECK_CUDA((cudaMalloc(&d_pixels, numBytes)), true);
        CHECK_CUDA((cudaMemcpy(d_pixels, hostView->data, numBytes, cudaMemcpyHostToDevice)), true);

        TextureData devView = *hostView;
        devView.data = d_pixels;
        CHECK_CUDA((cudaMalloc(&d_view, sizeof(TextureData))), true);
        CHECK_CUDA((cudaMemcpy(d_view, &devView, sizeof(TextureData), cudaMemcpyHostToDevice)), true);

        d_texture_pixels.push_back(d_pixels);
        d_texture_views.push_back(d_view);
        texture_view_map[hostView] = d_view;
        return d_view;
    };

    for (auto& m : renderMaterialsGPU) {
        m.albedo_map = uploadTextureView(m.albedo_map);
        m.bump_map = uploadTextureView(m.bump_map);
        m.normal_map = uploadTextureView(m.normal_map);
    }

    const size_t bytesPos = globalMesh.positions.size() * sizeof(Vec3);
    const size_t bytesIdx = globalMesh.indices.size() * sizeof(uint32_t);
    const size_t bytesNrm = globalMesh.normals.size() * sizeof(Vec3);
    const size_t bytesUV = globalMesh.uvs.size() * sizeof(Vec2);
    const size_t bytesTriObj = globalMesh.triangleObjIds.size() * sizeof(int32_t);
    const size_t bytesObjMat = renderMaterialsGPU.size() * sizeof(MaterialData);

    CHECK_CUDA((cudaMalloc(&d_positions, bytesPos)), true);
    CHECK_CUDA((cudaMalloc(&d_indices, bytesIdx)), true);
    CHECK_CUDA((cudaMalloc(&d_triangle_obj_ids, bytesTriObj)), true);
    CHECK_CUDA((cudaMalloc(&d_object_materials, bytesObjMat)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMalloc(&d_normals, bytesNrm)), true);
    }
    if (!globalMesh.uvs.empty()) {
        CHECK_CUDA((cudaMalloc(&d_uvs, bytesUV)), true);
    }

    CHECK_CUDA((cudaMemcpy(d_positions, globalMesh.positions.data(), bytesPos, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_indices, globalMesh.indices.data(), bytesIdx, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_triangle_obj_ids, globalMesh.triangleObjIds.data(), bytesTriObj, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_object_materials, renderMaterialsGPU.data(), bytesObjMat, cudaMemcpyHostToDevice)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMemcpy(d_normals, globalMesh.normals.data(), bytesNrm, cudaMemcpyHostToDevice)), true);
    }
    if (!globalMesh.uvs.empty()) {
        CHECK_CUDA((cudaMemcpy(d_uvs, globalMesh.uvs.data(), bytesUV, cudaMemcpyHostToDevice)), true);
    }

    MeshView d_mesh{};
    d_mesh.positions = d_positions;
    d_mesh.normals = d_normals;
    d_mesh.uvs = d_uvs;
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
        auto render_start_cpu =  std::chrono::high_resolution_clock::now();
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
    if (render_lights.empty()) {
        Light fallback;
        fallback.position = make_vec3(-3.0f, 0.0f, 1.0f);
        fallback.color = make_vec3(1.0f, 1.0f, 1.0f);
        fallback.intensity = 1;
        render_lights.push_back(fallback);
        printf("No lights in scene, using fallback light.\n");
    }
    const int num_lights = static_cast<int>(render_lights.size());
    const int num_object_materials = static_cast<int>(renderMaterials.size());

    // Scene statistics (shared by CPU & GPU paths)
    printf("Scene stats: %zu triangles, %d light(s)\n", P, num_lights);

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

    const int threads = 256;
    const int tri_blocks = (static_cast<int>(P) + threads - 1) / threads;
    buildTrianglesKernel<<<tri_blocks, threads>>>(d_mesh, d_tris, static_cast<int>(P));
    CHECK_CUDA((cudaDeviceSynchronize()), true);

    // Warm up with a tiny launch to pay first-launch/JIT cost without full-frame work.
    render(P, 1, 1, cam, miss_color, max_depth, 1, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce, d_image);

    // Zero image buffer before the real render (warmup wrote into it)
    CHECK_CUDA((cudaMemset(d_image, 0, sizeof(Vec3) * img_w * img_h)), true);

    // clock results for render
    auto start_render = std::chrono::high_resolution_clock::now();
    render(P, img_w, img_h, cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce, d_image);
    CHECK_CUDA((cudaMemcpy(image.data(), d_image, sizeof(Vec3) * img_w * img_h, cudaMemcpyDeviceToHost)), true);

    auto end_render = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
    printf("GPU Render Time: %.3f ms\n", ms_render.count());
    cudaFree(d_tris);
    cudaFree(d_image);
    cudaFree(d_lights);
    cudaFree(d_positions);
    cudaFree(d_normals);
    cudaFree(d_indices);
    cudaFree(d_uvs);
    cudaFree(d_triangle_obj_ids);
    cudaFree(d_object_materials);
    for (TextureData* d_view : d_texture_views) {
        cudaFree(d_view);
    }
    for (unsigned char* d_pixels : d_texture_pixels) {
        cudaFree(d_pixels);
    }
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

        Vec2 uv0 = make_vec2(0.0f, 0.0f);
        Vec2 uv1 = make_vec2(0.0f, 0.0f);
        Vec2 uv2 = make_vec2(0.0f, 0.0f);
        if (!globalMesh.uvs.empty()) {
            uv0 = globalMesh.uvs[i0];
            uv1 = globalMesh.uvs[i1];
            uv2 = globalMesh.uvs[i2];
        }

        Triangle tri = Triangle(globalMesh.positions[i0], globalMesh.positions[i1], globalMesh.positions[i2], n0, n1, n2);
        tri.uv0 = uv0;
        tri.uv1 = uv1;
        tri.uv2 = uv2;
        h_tris[i] = tri;
    }
    auto start_render = std::chrono::high_resolution_clock::now();
    render(P, img_w, img_h, cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, h_tris.data(),
           globalMesh.triangleObjIds.data(), renderMaterials.data(), num_object_materials,
           render_lights.data(), num_lights, diffuse_bounce, image.data());
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

    // write image to disk
    std::vector<unsigned char> img_data(num_pixels * 3);
    for (size_t i = 0; i < num_pixels; ++i)    {
        img_data[i * 3 + 0] = static_cast<unsigned char>(255.0f * std::min(image[i].x, 1.0f));
        img_data[i * 3 + 1] = static_cast<unsigned char>(255.0f * std::min(image[i].y, 1.0f));
        img_data[i * 3 + 2] = static_cast<unsigned char>(255.0f * std::min(image[i].z, 1.0f));
    }
    stbi_write_png("render.png", img_w, img_h, 3, img_data.data(), img_w * 3);
    printf("Image saved to render.png\n");


    return 0;
}
