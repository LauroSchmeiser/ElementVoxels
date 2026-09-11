#pragma once
#undef NEAR
#undef FAR
#include <vector>
#include <glad/glad.h>
#include "VoxelStructures.h"

namespace gl3 {
    struct Chunk {
        Voxel voxels[CHUNK_SIZE + 1][CHUNK_SIZE + 1][CHUNK_SIZE + 1];

        bool hasEmissive = false;
        bool hasFluid = false;
        bool hasGas = false;
        bool inEmissiveList = false;

        std::vector<VoxelLight> emissiveLights;
        bool lightingDirty = true;

        ChunkCoord coord;
        bool meshDirty = true;
        uint32_t gpuSlot = 0;

        struct BurnState {
            bool active = false;
            float t = 0.0f;
            float duration = 1.25f;
            glm::vec3 center = glm::vec3(0.0f);
            float radius = 0.01f;
            float noiseScale = 0.35f;
            float edgeWidth = 0.12f;
            float slowAccum = 0.0f;
        };
        BurnState burn;

        bool isCleared = false;

        struct GPUCache {
            GLuint vao = 0;
            GLuint vbo = 0;
            GLuint triangleSSBO = 0;
            uint32_t vertexCount = 0;
            bool isValid = false;
            uint64_t lastLightUpdateFrame = 15;
            std::vector<VoxelLight*> nearbyLights;
            GLuint counterReadbackBuffer = 0;
            GLsync counterFence = 0;
            uint32_t pendingVertexCount = 0;
            bool hasPendingCount = false;
        } gpuCache;

        Chunk() {
            resetVoxels();
        }

        void resetVoxels() {
            for (int x = 0; x <= CHUNK_SIZE; ++x) {
                for (int y = 0; y <= CHUNK_SIZE; ++y) {
                    for (int z = 0; z <= CHUNK_SIZE; ++z) {
                        voxels[x][y][z].type = 0;
                        voxels[x][y][z].material = 0;
                        voxels[x][y][z].density = -1000.0f;
                        voxels[x][y][z].fluidDensity = -1000.0f;
                        voxels[x][y][z].color = glm::vec3(0.0f);
                    }
                }
            }
        }

        void clear() {
            resetVoxels();

            if (gpuCache.vao != 0) {
                glDeleteVertexArrays(1, &gpuCache.vao);
                gpuCache.vao = 0;
            }

            if (gpuCache.vbo != 0) {
                glDeleteBuffers(1, &gpuCache.vbo);
                gpuCache.vbo = 0;
            }

            if (gpuCache.triangleSSBO != 0) {
                glDeleteBuffers(1, &gpuCache.triangleSSBO);
                gpuCache.triangleSSBO = 0;
            }

            if (gpuCache.counterReadbackBuffer != 0) {
                glDeleteBuffers(1, &gpuCache.counterReadbackBuffer);
                gpuCache.counterReadbackBuffer = 0;
            }

            if (gpuCache.counterFence != 0) {
                glDeleteSync(gpuCache.counterFence);
                gpuCache.counterFence = 0;
            }

            hasEmissive = false;
            hasFluid = false;
            hasGas = false;
            inEmissiveList = false;
            gpuCache.vertexCount = 0;
            gpuCache.isValid = false;
            gpuCache.hasPendingCount = false;
            gpuCache.nearbyLights.clear();
            emissiveLights.clear();
            meshDirty = true;
            lightingDirty = true;
        }

        void updateTypeFlags() {
            hasFluid = false;

            for (int x = 0; x < CHUNK_SIZE; ++x) {
                for (int y = 0; y < CHUNK_SIZE; ++y) {
                    for (int z = 0; z < CHUNK_SIZE; ++z) {
                        if (voxels[x][y][z].type == 3) {
                            hasFluid = true;
                            return;
                        }
                    }
                }
            }
        }
    };
}