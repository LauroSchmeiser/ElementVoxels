#pragma once
#undef NEAR
#undef FAR
#include <vector>
#include <memory>
#include <glad/glad.h>
#include "VoxelStructures.h"

namespace gl3 {
    struct Chunk {
        // Heap-allocated now — nullptr when unloaded, allocated on demand.
        std::unique_ptr<Voxel[]> voxelData;

        static constexpr int DIM = CHUNK_SIZE + 1;

        inline Voxel& voxels(int x, int y, int z) {
            return voxelData[(x * DIM + y) * DIM + z];
        }
        inline const Voxel& voxels(int x, int y, int z) const {
            return voxelData[(x * DIM + y) * DIM + z];
        }

        bool hasEmissive = false;
        bool hasFluid = false;
        bool hasGas = false;
        bool inEmissiveList = false;

        std::vector<VoxelLight> emissiveLights;
        bool lightingDirty = true;

        ChunkCoord coord;
        bool meshDirty = true;
        uint32_t gpuSlot = 0;
        bool queuedForRebuild = false;

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
            allocateVoxels();
        }

        // Called when a chunk becomes active (new, or reactivated from the pool)
        void allocateVoxels() {
            if (!voxelData) {
                voxelData = std::make_unique<Voxel[]>(size_t(DIM) * DIM * DIM);
            }
            resetVoxels();
        }

        // Called when a chunk is fully unloaded — actually gives the ~135KB back
        void releaseVoxels() {
            voxelData.reset();
        }

        void resetVoxels() {
            if (!voxelData) return;
            for (int x = 0; x < DIM; ++x) {
                for (int y = 0; y < DIM; ++y) {
                    for (int z = 0; z < DIM; ++z) {
                        Voxel& v = voxels(x, y, z);
                        v.type = 0;
                        v.material = 0;
                        v.density = -1000.0f;
                        v.fluidDensity = -1000.0f;
                        v.color = glm::vec3(0.0f);
                    }
                }
            }
        }

        void clear() {
            if (gpuCache.vao != 0) { glDeleteVertexArrays(1, &gpuCache.vao); gpuCache.vao = 0; }
            if (gpuCache.vbo != 0) { glDeleteBuffers(1, &gpuCache.vbo); gpuCache.vbo = 0; }
            if (gpuCache.triangleSSBO != 0) { glDeleteBuffers(1, &gpuCache.triangleSSBO); gpuCache.triangleSSBO = 0; }
            if (gpuCache.counterReadbackBuffer != 0) { glDeleteBuffers(1, &gpuCache.counterReadbackBuffer); gpuCache.counterReadbackBuffer = 0; }
            if (gpuCache.counterFence != 0) { glDeleteSync(gpuCache.counterFence); gpuCache.counterFence = 0; }

            releaseVoxels(); // <-- actually free the big array now

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
            queuedForRebuild=false;
        }

        void updateTypeFlags() {
            hasFluid = false;
            if (!voxelData) return;
            for (int x = 0; x < CHUNK_SIZE; ++x)
                for (int y = 0; y < CHUNK_SIZE; ++y)
                    for (int z = 0; z < CHUNK_SIZE; ++z)
                        if (voxels(x, y, z).type == 3) { hasFluid = true; return; }
        }
    };
}