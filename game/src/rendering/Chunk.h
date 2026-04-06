// In Chunk.h
#pragma once
#undef NEAR
#undef FAR
#include <vector>
#include <glad/glad.h>

namespace gl3 {
    struct Chunk {
        // Core voxel data
        Voxel voxels[CHUNK_SIZE + 1][CHUNK_SIZE + 1][CHUNK_SIZE + 1];

        // Lighting data
        std::vector<VoxelLight> emissiveLights;
        bool lightingDirty = true;

        // Mesh data
        ChunkCoord coord;
        bool meshDirty = true;
        uint32_t gpuSlot = 0;

        struct BurnState {
            bool active = false;
            float t = 0.0f;           // seconds since burn start
            float duration = 1.25f;   // seconds to fully disappear
            glm::vec3 center = glm::vec3(0.0f);
            float radius = 0.0f;      // 0 => disable distance term; else spherical propagation
            float noiseScale = 0.35f;
            float edgeWidth = 0.12f;
            float slowAccum=0.0f;
        };
        BurnState burn;

        bool isCleared = false;
        // GPU CACHE - stored with the chunk!
        struct GPUCache {
            GLuint vao = 0;
            GLuint vbo = 0;
            GLuint triangleSSBO = 0;
            uint32_t vertexCount = 0;
            bool isValid = false;
            uint64_t lastLightUpdateFrame = 15;
            std::vector<VoxelLight*> nearbyLights; // Pointers to other chunks' lights
            GLuint counterReadbackBuffer = 0;
            GLsync counterFence = 0;
            uint32_t pendingVertexCount = 0;
            bool hasPendingCount = false;
        } gpuCache;

        // Helper methods
        void clear() {
            for (int x = 0; x < CHUNK_SIZE + 1; ++x) {
                for (int y = 0; y < CHUNK_SIZE + 1; ++y) {
                    for (int z = 0; z < CHUNK_SIZE + 1; ++z) {
                        voxels[x][y][z].type = 0;
                        voxels[x][y][z].density = -1000.0f;
                        voxels[x][y][z].color = glm::vec3(0.0f);
                    }
                }
            }

            // Clean up GPU resources if they exist
            if (gpuCache.vao != 0) {
                glDeleteVertexArrays(1, &gpuCache.vao);
                glDeleteBuffers(1, &gpuCache.vbo);
                gpuCache.vao = 0;
                gpuCache.vbo = 0;
            }

            if (gpuCache.triangleSSBO != 0) {
                glDeleteBuffers(1, &gpuCache.triangleSSBO);
                gpuCache.triangleSSBO = 0;
            }

            gpuCache.vertexCount = 0;
            gpuCache.isValid = false;
            gpuCache.nearbyLights.clear();
            emissiveLights.clear();
            meshDirty = true;
            lightingDirty = true;
        }
    };
}