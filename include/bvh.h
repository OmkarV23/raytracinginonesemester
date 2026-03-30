#pragma once

#include "imports.h"
#include "ray.h"
#include "MeshOBJ.h"

struct BVHNode 
{
    std::uint32_t parent_idx; // parent node
    std::uint32_t left_idx;   // index of left  child node
    std::uint32_t right_idx;  // index of right child node
    std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
};

#ifndef __CUDACC__
struct uint2 {
    unsigned int x;
    unsigned int y;
};

inline uint2 make_uint2(unsigned int x, unsigned int y) {
    uint2 t; 
    t.x = x; t.y = y; 
    return t;
}
#endif

struct AABB 
{
    Vec3 minCorner;
    Vec3 maxCorner;

    HYBRID_FUNC AABB() : minCorner(make_vec3(INFINITY, INFINITY, INFINITY)), maxCorner(make_vec3(-INFINITY, -INFINITY, -INFINITY)) {}
    HYBRID_FUNC AABB(Vec3 minCorner, Vec3 maxCorner) : minCorner(minCorner), maxCorner(maxCorner) {}

    HYBRID_FUNC static inline AABB merge(const AABB& a, const AABB& b) {
        AABB result;
        result.minCorner = make_vec3(
            fminf(a.minCorner.x, b.minCorner.x),
            fminf(a.minCorner.y, b.minCorner.y),
            fminf(a.minCorner.z, b.minCorner.z)
        );
        result.maxCorner = make_vec3(
            fmaxf(a.maxCorner.x, b.maxCorner.x),
            fmaxf(a.maxCorner.y, b.maxCorner.y),
            fmaxf(a.maxCorner.z, b.maxCorner.z)
        );

        return result;
    }
    
};

HYBRID_FUNC inline void aabb_of_triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                             AABB& outBox, float eps = 0.0f) {
    AABB box;
    box.minCorner.x = fminf(a.x, fminf(b.x, c.x));
    box.minCorner.y = fminf(a.y, fminf(b.y, c.y));
    box.minCorner.z = fminf(a.z, fminf(b.z, c.z));

    box.maxCorner.x = fmaxf(a.x, fmaxf(b.x, c.x));
    box.maxCorner.y = fmaxf(a.y, fmaxf(b.y, c.y));
    box.maxCorner.z = fmaxf(a.z, fmaxf(b.z, c.z));

    // Optional: pad to avoid zero-thickness boxes causing numeric issues
    box.minCorner.x -= eps; box.minCorner.y -= eps; box.minCorner.z -= eps;
    box.maxCorner.x += eps; box.maxCorner.y += eps; box.maxCorner.z += eps;

    outBox = box;
    
}


HYBRID_FUNC inline bool aabbContains(const AABB& box, const Vec3& p) {
    const float epsilon = 0.0f;
    return (p.x >= box.minCorner.x - epsilon && p.x <= box.maxCorner.x + epsilon &&
            p.y >= box.minCorner.y - epsilon && p.y <= box.maxCorner.y + epsilon &&
            p.z >= box.minCorner.z - epsilon && p.z <= box.maxCorner.z + epsilon);
}

HYBRID_FUNC inline bool intersectAABB(const Ray& r, const AABB& aabb, double tmin, double tmax) {
    const Vec3 orig = r.origin();
    const Vec3 dir  = r.direction();
    const float eps = 1e-8f;

    double t0 = tmin;
    double t1 = tmax;

    // X slab
    if (fabsf(dir.x) < eps) {
        if (orig.x < aabb.minCorner.x || orig.x > aabb.maxCorner.x) return false;
    } else {
        const double inv = 1.0 / static_cast<double>(dir.x);
        double tNear = (static_cast<double>(aabb.minCorner.x) - static_cast<double>(orig.x)) * inv;
        double tFar  = (static_cast<double>(aabb.maxCorner.x) - static_cast<double>(orig.x)) * inv;
        if (tNear > tFar) { const double tmp = tNear; tNear = tFar; tFar = tmp; }
        if (tNear > t0) t0 = tNear;
        if (tFar  < t1) t1 = tFar;
        if (t0 > t1) return false;
    }

    // Y slab
    if (fabsf(dir.y) < eps) {
        if (orig.y < aabb.minCorner.y || orig.y > aabb.maxCorner.y) return false;
    } else {
        const double inv = 1.0 / static_cast<double>(dir.y);
        double tNear = (static_cast<double>(aabb.minCorner.y) - static_cast<double>(orig.y)) * inv;
        double tFar  = (static_cast<double>(aabb.maxCorner.y) - static_cast<double>(orig.y)) * inv;
        if (tNear > tFar) { const double tmp = tNear; tNear = tFar; tFar = tmp; }
        if (tNear > t0) t0 = tNear;
        if (tFar  < t1) t1 = tFar;
        if (t0 > t1) return false;
    }

    // Z slab
    if (fabsf(dir.z) < eps) {
        if (orig.z < aabb.minCorner.z || orig.z > aabb.maxCorner.z) return false;
    } else {
        const double inv = 1.0 / static_cast<double>(dir.z);
        double tNear = (static_cast<double>(aabb.minCorner.z) - static_cast<double>(orig.z)) * inv;
        double tFar  = (static_cast<double>(aabb.maxCorner.z) - static_cast<double>(orig.z)) * inv;
        if (tNear > tFar) { const double tmp = tNear; tNear = tFar; tFar = tmp; }
        if (tNear > t0) t0 = tNear;
        if (tFar  < t1) t1 = tFar;
        if (t0 > t1) return false;
    }

    return true;
}

HYBRID_FUNC inline std::uint32_t bitExpansion(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
HYBRID_FUNC inline std::uint32_t ComputeMortonCode(Vec3 xyz, float resolution = 1024.0f) noexcept
{
    xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const std::uint32_t xx = bitExpansion(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = bitExpansion(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = bitExpansion(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

#ifdef __CUDACC__
__device__ inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept
{
    return ::__clz(lhs ^ rhs);
}
__device__ inline int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept
{
    return ::__clzll(lhs ^ rhs);
}

template<typename UInt>
__device__
inline uint2 determine_range(UInt const* node_code,
        const unsigned int num_leaves, unsigned int idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const UInt self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = thrust::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        thrust::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

template<typename UInt>
__device__
inline unsigned int find_split(UInt const* node_code, const unsigned int num_leaves,
    const unsigned int first, const unsigned int last) noexcept
{
    const UInt first_code = node_code[first];
    const UInt last_code  = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}


template<typename UInt>
inline void construct_internal_nodes(BVHNode* BVHNodes,
        UInt const* node_code, const unsigned int num_objects)
{
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(num_objects - 1),
        [BVHNodes, node_code, num_objects] __device__ (const unsigned int idx)
        {
            BVHNodes[idx].object_idx = 0xFFFFFFFF; //  internal nodes

            const uint2 ij  = determine_range(node_code, num_objects, idx);
            const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

            BVHNodes[idx].left_idx  = gamma;
            BVHNodes[idx].right_idx = gamma + 1;
            if(thrust::min(ij.x, ij.y) == gamma)
            {
                BVHNodes[idx].left_idx += num_objects - 1;
            }
            if(thrust::max(ij.x, ij.y) == gamma + 1)
            {
                BVHNodes[idx].right_idx += num_objects - 1;
            }
            BVHNodes[BVHNodes[idx].left_idx].parent_idx  = idx;
            BVHNodes[BVHNodes[idx].right_idx].parent_idx = idx;
            return;
        });
    return;
}

#else
inline int common_upper_bits_cpu(const unsigned int lhs, const unsigned int rhs) noexcept
{
    unsigned int diff = lhs ^ rhs;
    return (diff == 0) ? 32 : __builtin_clz(diff);
}
inline int common_upper_bits_cpu(const unsigned long long int lhs, const unsigned long long int rhs) noexcept
{
    unsigned long long int diff = lhs ^ rhs;
    return (diff == 0) ? 64 : __builtin_clzll(diff);
}

template<typename UInt>
inline uint2 determine_range_cpu(UInt const* node_code,
        const unsigned int num_leaves, unsigned int idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const UInt self_code = node_code[idx];
    const int L_delta = common_upper_bits_cpu(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits_cpu(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range
    const int delta_min = std::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits_cpu(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits_cpu(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits_cpu(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        std::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

template<typename UInt>
inline unsigned int find_split_cpu(UInt const* node_code, const unsigned int num_leaves,
    const unsigned int first, const unsigned int last)
{
    const UInt first_code = node_code[first];
    const UInt last_code  = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits_cpu(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits_cpu(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}

inline void refit_cpu(int node_idx, BVHNode* nodes, AABB* aabbs, int num_leaves) {
    if (node_idx >= num_leaves - 1) { // Leaf node (indices start at numTriangles - 1)
        return;
    }
    
    BVHNode& node = nodes[node_idx];
    refit_cpu(node.left_idx, nodes, aabbs, num_leaves);
    refit_cpu(node.right_idx, nodes, aabbs, num_leaves);
    
    aabbs[node_idx] = AABB::merge(aabbs[node.left_idx], aabbs[node.right_idx]);
}

#endif

namespace AccStruct {

class BVH
{
public:
    void calculateAABBs(const MeshView& mesh, AABB* aabbs);

#ifdef __CUDACC__
    void buildBVH(
        BVHNode* BVHNodes,
        AABB* aabbs,
        AABB SceneBoundingBox,
        thrust::device_vector<unsigned int>* triangleIndices,
        int numTriangles
    );
#else
    void buildBVH(
        BVHNode* BVHNodes,
        AABB* aabbs,
        AABB SceneBoundingBox,
        std::vector<unsigned int>& triangleIndices,
        int numTriangles
    );
#endif

private:
#ifdef __CUDACC__
    // Reusable GPU workspace to avoid per-call allocations
    thrust::device_vector<unsigned long long int> mortonCodes64;
    thrust::device_vector<AABB> sorted_leaves;
    thrust::device_vector<int> flag_container;
#endif
};

} // namespace AccStruct
