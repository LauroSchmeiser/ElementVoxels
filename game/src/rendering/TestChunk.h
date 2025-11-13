// TestChunk.h
#pragma once
#include "VoxelRenderer.h"
#include <random>

namespace gl3 {

    inline Chunk makeTestChunk() {
        Chunk chunk;

        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> heightDist(0, CHUNK_SIZE / 2);

        for (int x = 0; x < CHUNK_SIZE; ++x)
            for (int z = 0; z < CHUNK_SIZE; ++z) {
                int height = heightDist(rng);
                for (int y = 0; y < height; ++y) {
                    auto& voxel = chunk.voxels[x][y][z];
                    voxel.type = 1;
                    voxel.color = glm::vec3(float(x)/CHUNK_SIZE, float(y)/CHUNK_SIZE, float(z)/CHUNK_SIZE);

                }
            }

        return chunk;
    }

}
