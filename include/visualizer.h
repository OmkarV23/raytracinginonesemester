#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "bvh.h"

// Helper to write AABBs to an OBJ file as wireframe boxes
inline void ExportAABBsToOBJ(const std::string& filename, const AABB* aabbs, size_t count, size_t offset = 0)
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "[Visualizer] Failed to open " << filename << "\n";
        return;
    }

    std::cout << "[Visualizer] Exporting " << count << " AABBs to " << filename << "...\n";

    size_t validCount = 0;
    // OBJ uses 1-based indexing
    size_t vIdx = 1; 

    for (size_t i = 0; i < count; ++i)
    {
        const AABB& box = aabbs[offset + i];

        // Skip uninitialized or invalid boxes
        if (box.minCorner.x > box.maxCorner.x) continue;

        validCount++;

        // 8 corners
        // 0: min.x, min.y, min.z
        // 1: max.x, min.y, min.z
        // 2: max.x, max.y, min.z
        // 3: min.x, max.y, min.z
        // 4: min.x, min.y, max.z
        // 5: max.x, min.y, max.z
        // 6: max.x, max.y, max.z
        // 7: min.x, max.y, max.z

        float x0 = box.minCorner.x, y0 = box.minCorner.y, z0 = box.minCorner.z;
        float x1 = box.maxCorner.x, y1 = box.maxCorner.y, z1 = box.maxCorner.z;

        out << "v " << x0 << " " << y0 << " " << z0 << "\n";
        out << "v " << x1 << " " << y0 << " " << z0 << "\n";
        out << "v " << x1 << " " << y1 << " " << z0 << "\n";
        out << "v " << x0 << " " << y1 << " " << z0 << "\n";
        out << "v " << x0 << " " << y0 << " " << z1 << "\n";
        out << "v " << x1 << " " << y0 << " " << z1 << "\n";
        out << "v " << x1 << " " << y1 << " " << z1 << "\n";
        out << "v " << x0 << " " << y1 << " " << z1 << "\n";

        // 12 lines
        // Bottom face
        out << "l " << vIdx+0 << " " << vIdx+1 << "\n";
        out << "l " << vIdx+1 << " " << vIdx+2 << "\n";
        out << "l " << vIdx+2 << " " << vIdx+3 << "\n";
        out << "l " << vIdx+3 << " " << vIdx+0 << "\n";

        // Top face
        out << "l " << vIdx+4 << " " << vIdx+5 << "\n";
        out << "l " << vIdx+5 << " " << vIdx+6 << "\n";
        out << "l " << vIdx+6 << " " << vIdx+7 << "\n";
        out << "l " << vIdx+7 << " " << vIdx+4 << "\n";

        // Vertical pillars
        out << "l " << vIdx+0 << " " << vIdx+4 << "\n";
        out << "l " << vIdx+1 << " " << vIdx+5 << "\n";
        out << "l " << vIdx+2 << " " << vIdx+6 << "\n";
        out << "l " << vIdx+3 << " " << vIdx+7 << "\n";

        vIdx += 8;
    }

    out.close();
    std::cout << "[Visualizer] Done. Exported " << validCount << " boxes.\n";
}
