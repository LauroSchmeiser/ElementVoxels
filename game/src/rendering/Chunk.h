// In Chunk.h
#pragma once

#include "VoxelStructures.h"
#include <vector>
#include <glad/glad.h>

namespace gl3 {
    struct Chunk {
        // Core voxel data (CHUNK_SIZE + 2)^3 for padding
        Voxel voxels[CHUNK_SIZE + 2][CHUNK_SIZE + 2][CHUNK_SIZE + 2];

        // Additional data
        std::vector<VoxelLight> emissiveLights;
        bool meshDirty = true;
        bool lightingDirty = true;
        ChunkCoord coord;
        uint32_t vertexCount = 0;

        // Helper methods
        void clear() {
            // Initialize all voxels to air
            for (int x = 0; x < CHUNK_SIZE + 2; ++x) {
                for (int y = 0; y < CHUNK_SIZE + 2; ++y) {
                    for (int z = 0; z < CHUNK_SIZE + 2; ++z) {
                        voxels[x][y][z].type = 0; // Air
                        voxels[x][y][z].density = -1000.0f; // Far outside
                        voxels[x][y][z].color = glm::vec3(0.0f);
                    }
                }
            }
        }
    };
}