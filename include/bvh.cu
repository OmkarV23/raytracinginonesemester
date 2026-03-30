#include "vec3.h"
#include "bvh.h"
#include "MeshOBJ.h"


#ifdef __CUDACC__
__global__ void computeAABB(
    const MeshView mesh,
    const size_t triCount,
    AABB* aabbs)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= triCount) return;
    if (mesh.positions == nullptr || mesh.indices == nullptr) return;

    uint32_t idx0 = mesh.indices[idx * 3 + 0];
    uint32_t idx1 = mesh.indices[idx * 3 + 1];
    uint32_t idx2 = mesh.indices[idx * 3 + 2];

    if (idx0 >= mesh.numVertices || idx1 >= mesh.numVertices || idx2 >= mesh.numVertices) return;

    Vec3 v0 = mesh.positions[idx0];
    Vec3 v1 = mesh.positions[idx1];
    Vec3 v2 = mesh.positions[idx2];
    
    AABB triAABB;
    aabb_of_triangle(v0, v1, v2, triAABB, 0.0f);
    
    // Leaves start at index (triCount - 1) in the 2*P-1 array
    aabbs[(triCount - 1) + idx] = triAABB;
}

__global__ void ComputeMortonCodes(const AABB* aabbs, 
                                   int numTriangles,
                                   AABB SceneBoundingBox, 
                                   unsigned long long int* mortonCodes)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numTriangles) return;

    int triangle_idx = numTriangles - 1 + idx;

    float3 centroid = (aabbs[triangle_idx].minCorner + aabbs[triangle_idx].maxCorner) / 2.0f;

    // Normalize centroid
    float3 normalizedCentroid = (centroid - SceneBoundingBox.minCorner) / (SceneBoundingBox.maxCorner - SceneBoundingBox.minCorner);

    // Convert normalized centroid to morton code
    unsigned int code = ComputeMortonCode(normalizedCentroid);
    unsigned long long int key = code;
    key <<= 32;
    key |= idx;
    mortonCodes[idx] = key;
}

#endif


void AccStruct::BVH::calculateAABBs(
    const MeshView& mesh,
    AABB* aabbs)
{
    const size_t triCount = mesh.numTriangles;

#ifdef __CUDACC__
    const size_t threadsPerBlock = 256;
    const size_t blocks = (triCount + threadsPerBlock - 1) / threadsPerBlock;

    computeAABB<<<blocks, threadsPerBlock>>>(
        mesh, triCount, aabbs);
    
#else
    for (size_t i = 0; i < triCount; ++i) {
        uint32_t idx0 = mesh.indices[i * 3 + 0];
        uint32_t idx1 = mesh.indices[i * 3 + 1];
        uint32_t idx2 = mesh.indices[i * 3 + 2];

        Vec3 v0 = mesh.positions[idx0];
        Vec3 v1 = mesh.positions[idx1];
        Vec3 v2 = mesh.positions[idx2];

        AABB triAABB;
        aabb_of_triangle(v0, v1, v2, triAABB, 0.0f);

        // Leaves start at index (triCount - 1)
        aabbs[(triCount - 1) + i] = triAABB;
    }
#endif
}

#ifdef __CUDACC__
void AccStruct::BVH::buildBVH(
    BVHNode* BVHNodes,
    AABB* aabbs,
    AABB SceneBoundingBox,
    thrust::device_vector<unsigned int>* triangleIndices,
    int numTriangles
)
{
    // Use 64-bit keys directly: (MortonCode << 32) | TriangleIndex
    // thrust::device_vector<unsigned long long int> mortonCodes64(numTriangles);
    mortonCodes64.resize(numTriangles);

    dim3 blockSize(256);
    dim3 gridSize((numTriangles + blockSize.x - 1) / blockSize.x);

    CHECK_CUDA((ComputeMortonCodes<<<gridSize, blockSize>>>(
        aabbs,
        numTriangles,
        SceneBoundingBox,
        mortonCodes64.data().get())), true)

    thrust::sort_by_key(
        mortonCodes64.begin(), 
        mortonCodes64.end(),
        triangleIndices->begin()
    );

    sorted_leaves.resize(numTriangles);
    
    thrust::gather(
        triangleIndices->begin(),
        triangleIndices->end(),
        thrust::device_pointer_cast(aabbs) + numTriangles - 1,
        sorted_leaves.begin()
    );

    thrust::copy(
        sorted_leaves.begin(),
        sorted_leaves.end(),
        thrust::device_pointer_cast(aabbs) + numTriangles - 1
    );
    
    BVHNode default_node;
    default_node.object_idx = 0xFFFFFFFF;
    default_node.left_idx = 0xFFFFFFFF;
    default_node.right_idx = 0xFFFFFFFF;
    default_node.parent_idx = 0xFFFFFFFF;

    thrust::fill(thrust::device, BVHNodes, BVHNodes + (2 * numTriangles - 1), default_node);

    thrust::transform(
        triangleIndices->begin(),
        triangleIndices->end(),
        thrust::device_pointer_cast(BVHNodes) + numTriangles - 1,
        [] __device__ (const std::uint32_t idx) {
            BVHNode node;
            // node.object_idx = (*triangleIndices)[idx];
            node.object_idx = idx;
            node.left_idx = 0xFFFFFFFF;
            node.right_idx = 0xFFFFFFFF;
            node.parent_idx = 0xFFFFFFFF;
            return node;
        }
    );

    // Initialize internal node AABBs to "invalid" state
    AABB invalid_aabb;
    thrust::fill(thrust::device, 
        thrust::device_pointer_cast(aabbs), 
        thrust::device_pointer_cast(aabbs) + numTriangles - 1, 
        invalid_aabb);

    const unsigned long long int* node_code = mortonCodes64.data().get();
    construct_internal_nodes(BVHNodes, node_code, numTriangles);

    // thrust::device_vector<int> flag_container(numTriangles, 0);
    flag_container.assign(numTriangles, 0);
    const auto flags = flag_container.data().get();

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<std::uint32_t>(numTriangles-1),
        thrust::make_counting_iterator<std::uint32_t>(2*numTriangles-1),
        [BVHNodes, aabbs, flags] __device__ (const std::uint32_t idx) {
            
            unsigned int parent = BVHNodes[idx].parent_idx;
            while(parent != 0xFFFFFFFF)
            {
                // Ensure our AABB write (from a previous iteration) is
                // visible to all threads before we advertise it via the flag.
                __threadfence();

                const int old = atomicCAS(flags + parent, 0, 1);
                if(old==0)
                {
                    // First child to arrive — the sibling isn't done yet.
                    return;
                }
                assert(old==1);

                // Second child to arrive — the sibling's AABB write is
                // guaranteed visible because its __threadfence() completed
                // before its atomicCAS, and we observed that CAS result.
                const auto left_idx = BVHNodes[parent].left_idx;
                const auto right_idx = BVHNodes[parent].right_idx;
                const AABB left_aabb = aabbs[left_idx];
                const AABB right_aabb = aabbs[right_idx];
                aabbs[parent] = AABB::merge(left_aabb, right_aabb);
                parent = BVHNodes[parent].parent_idx;
            }
            return;
        });


}
#else

void AccStruct::BVH::buildBVH(
    BVHNode* BVHNodes,
    AABB* aabbs,
    AABB SceneBoundingBox,
    std::vector<unsigned int>& triangleIndices,
    int numTriangles
)
{
    if (numTriangles <= 0) {
        return;
    }

    // Initialize all nodes to a known state (matches GPU path behavior).
    const int totalNodes = 2 * numTriangles - 1;
    for (int i = 0; i < totalNodes; ++i) {
        BVHNodes[i].object_idx = 0xFFFFFFFF;
        BVHNodes[i].left_idx   = 0xFFFFFFFF;
        BVHNodes[i].right_idx  = 0xFFFFFFFF;
        BVHNodes[i].parent_idx = 0xFFFFFFFF;
    }

    // 1. Compute Morton Codes
    std::vector<unsigned long long int> mortonCodes(numTriangles);
    
    // We need pairs of (code, original_index) to sort
    // Actually triangleIndices already contains 0..N-1, we can just sort that array 
    // using a custom comparator that looks up the code.
    // BUT we need strictly sorted codes for the construction step.
    
    for(int i = 0; i < numTriangles; ++i) {
        // Look up AABB for this specific triangle.
        // NOTE: calculateAABBs fills aabbs[ (N-1) + i ] for i in 0..N-1
        // We assume 'aabbs' is already populated with leaves at the end.
        
        int leaf_idx = (numTriangles - 1) + i;
        AABB box = aabbs[leaf_idx];
        Vec3 centroid = (box.minCorner + box.maxCorner) * 0.5f;
        
        Vec3 normalized = (centroid - SceneBoundingBox.minCorner) / (SceneBoundingBox.maxCorner - SceneBoundingBox.minCorner);
        
        unsigned int code = ComputeMortonCode(normalized);
        unsigned long long int key = code;
        key <<= 32;
        key |= i;
        mortonCodes[i] = key;
    }
    
    // 2. Sort Indices
    // We actually want to sort the "triangleIndices" array based on the keys in 'mortonCodes'.
    // Currently triangleIndices should be 0, 1, 2... if called from main.
    // We'll create a pair vector to make it easier.
    std::vector<std::pair<unsigned long long int, unsigned int>> sortPairs(numTriangles);
    for(int i=0; i<numTriangles; ++i) {
        sortPairs[i] = {mortonCodes[i], triangleIndices[i]};
    }
    
    std::sort(sortPairs.begin(), sortPairs.end());
    
    // Write back sorted indices and keeping sorted codes
    std::vector<unsigned long long int> sortedCodes(numTriangles);
    for(int i=0; i<numTriangles; ++i) {
        sortedCodes[i] = sortPairs[i].first;
        triangleIndices[i] = sortPairs[i].second;
    }
    
    // 3. Permute Leaf AABBs
    // We need to shuffle the leaf AABBs in memory to match the Morton order
    std::vector<AABB> leafCopy(numTriangles);
    for(int i=0; i<numTriangles; ++i) {
        leafCopy[i] = aabbs[(numTriangles - 1) + triangleIndices[i]]; // Grab AABB from original index
    }
    // Write back
    for(int i=0; i<numTriangles; ++i) {
        aabbs[(numTriangles - 1) + i] = leafCopy[i];
    }

    // 3b. Initialize leaves with the triangle index they represent.
    // Leaves are stored in Morton order, but object_idx must reference original triangle IDs.
    for (int i = 0; i < numTriangles; ++i) {
        const int leafNodeIdx = (numTriangles - 1) + i;
        BVHNodes[leafNodeIdx].object_idx = triangleIndices[i];
    }
    
    // 4. Construct Internal Nodes
    // Loop over internal nodes 0 to N-2
    for(int idx = 0; idx < numTriangles - 1; ++idx) {
        BVHNodes[idx].object_idx = 0xFFFFFFFF; // internal
        
        const uint2 ij = determine_range_cpu(sortedCodes.data(), numTriangles, idx);
        const int gamma = find_split_cpu(sortedCodes.data(), numTriangles, ij.x, ij.y);
        
        BVHNodes[idx].left_idx = gamma;
        BVHNodes[idx].right_idx = gamma + 1;
        
        if(std::min(ij.x, ij.y) == gamma) {
            BVHNodes[idx].left_idx += numTriangles - 1; // It's a leaf
        }
        if(std::max(ij.x, ij.y) == gamma + 1) {
            BVHNodes[idx].right_idx += numTriangles - 1; // It's a leaf
        }
        
        BVHNodes[BVHNodes[idx].left_idx].parent_idx = idx;
        BVHNodes[BVHNodes[idx].right_idx].parent_idx = idx;
    }
    
    // 5. Refit AABBs
    // Recurse from root. For numTriangles==1, root is leaf index 0 and refit is a no-op.
    refit_cpu(0, BVHNodes, aabbs, numTriangles);
}
#endif
