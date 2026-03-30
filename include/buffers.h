#pragma once

#include "imports.h"
#include "bvh.h"

#define BLOCK_X 16
#define BLOCK_Y 16

namespace RayTracer
{

    template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    struct BVHState 
    {
        BVHNode* Nodes; // (Total nodes + internal + leaf)
        AABB* AABBs;  // Pointer to the BVH stored in the buffer

        static BVHState fromChunk(char*& chunk, size_t P);
    };

}


template<typename T> 
size_t required(size_t P)
{
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}