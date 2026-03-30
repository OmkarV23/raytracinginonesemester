#pragma once

#include "warmup.h"
#include "bvh.h"
#include "MeshOBJ.h"
#include "buffers.h"
#include <iostream>
#include <vector>

inline void warmupGPU() {
#ifdef __CUDACC__
    
    // 1. Create dummy mesh (single triangle)
    Mesh dummyMesh;
    dummyMesh.positions.push_back(make_vec3(0,0,0));
    dummyMesh.positions.push_back(make_vec3(1,0,0));
    dummyMesh.positions.push_back(make_vec3(0,1,0));
    dummyMesh.indices.push_back(0);
    dummyMesh.indices.push_back(1);
    dummyMesh.indices.push_back(2);
    
    size_t P = 1;

    // 2. Allocate Device Memory for BVH
    // calling required<BVHState> works because template is in header
    size_t size = required<RayTracer::BVHState>(P);
    char* d_buffer = nullptr;
    if (cudaMalloc(&d_buffer, size) != cudaSuccess) {
        return;
    }
    
    // We replicate the obtain logic to be self-contained and avoid linking issues
    // with BVHState::fromChunk if it's defined in another translation unit.
    RayTracer::BVHState state;
    char* walker = d_buffer;
    RayTracer::obtain(walker, state.Nodes, 2 * P - 1, 128);
    RayTracer::obtain(walker, state.AABBs, 2 * P - 1, 128);
    
    // 3. Run Pipeline
    AccStruct::BVH bvh;
    // calculateAABBs expects device pointers in MeshView, so create a device copy.
    Vec3* d_positions = nullptr;
    uint32_t* d_indices = nullptr;
    cudaMalloc(&d_positions, dummyMesh.positions.size() * sizeof(Vec3));
    cudaMalloc(&d_indices, dummyMesh.indices.size() * sizeof(uint32_t));
    cudaMemcpy(d_positions, dummyMesh.positions.data(),
               dummyMesh.positions.size() * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, dummyMesh.indices.data(),
               dummyMesh.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    MeshView d_mesh{};
    d_mesh.positions = d_positions;
    d_mesh.normals = nullptr;
    d_mesh.uvs = nullptr;
    d_mesh.indices = d_indices;
    d_mesh.triangleObjIds = nullptr;
    d_mesh.numVertices = dummyMesh.positions.size();
    d_mesh.numIndices = dummyMesh.indices.size();
    d_mesh.numTriangles = P;

    CHECK_CUDA(bvh.calculateAABBs(d_mesh, state.AABBs), true);

    // Thrust Reduce (instantiates thrust::reduce for AABB)
    AABB default_aabb;
    AABB sceneBox = thrust::reduce(
        thrust::device_pointer_cast(state.AABBs + (P - 1)),
        thrust::device_pointer_cast(state.AABBs + (2*P - 1)),
        default_aabb,
        [] __device__ __host__ (const AABB& lhs, const AABB& rhs) {
            return AABB::merge(lhs, rhs);
        });

    thrust::device_vector<unsigned int> indices(P);
    thrust::sequence(indices.begin(), indices.end());

    // Build (exercises sort, gather, etc.)
    bvh.buildBVH(
        state.Nodes,
        state.AABBs,
        sceneBox,
        &indices,
        (int)P
    );
    
    cudaDeviceSynchronize();
    cudaFree(d_positions);
    cudaFree(d_indices);
    cudaFree(d_buffer);
#endif
}
