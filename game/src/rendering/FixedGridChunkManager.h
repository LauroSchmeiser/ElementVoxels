#pragma once
#include <vector>
#include <functional>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "VoxelStructures.h"
#include "Chunk.h"
#include "glm/glm.hpp"
#include "iostream"
#include <memory>

namespace gl3 {

    class FixedGridChunkManager {
    public:
        static constexpr uint32_t INVALID_GPU_SLOT = 0xFFFFFFFFu;
        static constexpr uint32_t MAX_GPU_SLOTS = 1850;

        explicit FixedGridChunkManager(int radiusChunks)
                : R(radiusChunks),
                  dim(2 * R + 1)
        {
        }

        int radius() const { return R; }
        int dimension() const { return dim; }
        uint32_t maxChunksGpu() const { return MAX_GPU_SLOTS; }
        size_t totalChunksInGrid() const {
            return chunks.size();
        }
        bool inBounds(const ChunkCoord& cc) const {
            return (cc.x >= -R && cc.x <= R &&
                    cc.y >= -R && cc.y <= R &&
                    cc.z >= -R && cc.z <= R);
        }

        Chunk* getChunk(const ChunkCoord& cc) {
            if (!inBounds(cc)) return nullptr;

            auto it = chunks.find(cc);
            return it == chunks.end() ? nullptr : it->second.get();
        }

        const Chunk* getChunk(const ChunkCoord& cc) const {
            if (!inBounds(cc)) return nullptr;

            auto it = chunks.find(cc);
            return it == chunks.end() ? nullptr : it->second.get();
        }

        Chunk* getOrCreateChunk(const ChunkCoord& cc) {
            if (!inBounds(cc)) return nullptr;

            auto it = chunks.find(cc);

            if (it != chunks.end()) {
                Chunk* chunk = it->second.get();

                if (chunk->isCleared) {
                    chunk->allocateVoxels();
                    chunk->coord = cc;
                    chunk->gpuSlot = INVALID_GPU_SLOT;
                    chunk->isCleared = false;
                    chunk->hasEmissive = false;
                    chunk->hasFluid = false;
                    chunk->hasGas = false;
                    chunk->inEmissiveList = false;
                    chunk->emissiveLights.clear();
                    chunk->gpuCache.vertexCount = 0;
                    chunk->gpuCache.isValid = false;
                    chunk->gpuCache.nearbyLights.clear();
                    chunk->meshDirty = true;
                    chunk->lightingDirty = true;
                    chunk->queuedForRebuild = false;
                }

                return chunk;
            }

            auto chunk = std::make_unique<Chunk>();
            chunk->coord = cc;
            chunk->gpuSlot = INVALID_GPU_SLOT;
            chunk->isCleared = false;

            Chunk* result = chunk.get();
            chunks.emplace(cc, std::move(chunk));
            return result;
        }

        uint32_t allocateGpuSlot(const ChunkCoord& coord) {
            Chunk* chunk = getChunk(coord);
            if (!chunk) return INVALID_GPU_SLOT;

            if (chunk->gpuSlot != INVALID_GPU_SLOT) {
                return chunk->gpuSlot;
            }

            uint32_t slot = INVALID_GPU_SLOT;

            if (!freeGpuSlots.empty()) {
                slot = freeGpuSlots.back();
                freeGpuSlots.pop_back();
            } else if (nextGpuSlot < MAX_GPU_SLOTS) {
                slot = nextGpuSlot++;
            } else {
                if (!evictFurthestChunk(coord) || freeGpuSlots.empty()) {
                    return INVALID_GPU_SLOT;
                }

                slot = freeGpuSlots.back();
                freeGpuSlots.pop_back();
            }

            chunk->gpuSlot = slot;
            slotToChunkCoord[slot] = coord;
            activeSlots.insert(slot);

            return slot;
        }

        bool hasDirtyChunks() const { return !dirtyChunks.empty(); }


        void freeGpuSlot(const ChunkCoord& coord) {
            Chunk* chunk = getChunk(coord);
            if (!chunk || chunk->gpuSlot == INVALID_GPU_SLOT) return;

            uint32_t slot = chunk->gpuSlot;

            chunk->gpuCache.isValid = false;
            chunk->gpuCache.vertexCount = 0;
            chunk->meshDirty = true;
            chunk->gpuSlot = INVALID_GPU_SLOT;

            slotToChunkCoord.erase(slot);
            activeSlots.erase(slot);
            freeGpuSlots.push_back(slot);
        }


        void cleanupDistantChunks(
                const glm::vec3& cameraPos,
                const glm::vec3& /* cameraForward */,
                int renderRadiusChunks)
        {
            const int camCX = worldToChunk(cameraPos.x);
            const int camCY = worldToChunk(cameraPos.y);
            const int camCZ = worldToChunk(cameraPos.z);

            const int keepRadius = renderRadiusChunks + 4;

            std::vector<ChunkCoord> slotsToFree;
            slotsToFree.reserve(slotToChunkCoord.size());

            for (const auto& [slot, coord] : slotToChunkCoord) {
                const int dx = std::abs(coord.x - camCX);
                const int dy = std::abs(coord.y - camCY);
                const int dz = std::abs(coord.z - camCZ);

                if (dx > keepRadius || dy > keepRadius || dz > keepRadius) {
                    slotsToFree.push_back(coord);
                }
            }

            for (const ChunkCoord& coord : slotsToFree) {
                freeGpuSlot(coord);
            }
        }

        inline int worldToChunk(float worldPos) {
            const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
            return (int)std::floor(worldPos / chunkWorldSize);
        }

        inline glm::vec3 getChunkMin(const ChunkCoord& coord) const {
            return glm::vec3(coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                             coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                             coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE);
        }

        inline glm::vec3 getChunkMax(const ChunkCoord& coord) const {
            return glm::vec3((coord.x + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                             (coord.y + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                             (coord.z + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE);
        }

        inline glm::vec3 calculateNormalAt(Chunk* chunk, const glm::ivec3& pos) {
            // Simple central differences normal calculation
            if (pos.x <= 0 || pos.x >= CHUNK_SIZE ||
                pos.y <= 0 || pos.y >= CHUNK_SIZE ||
                pos.z <= 0 || pos.z >= CHUNK_SIZE) {
                return glm::vec3(0, 1, 0); // Fallback
            }

            float dx = chunk->voxels(pos.x+1,pos.y,pos.z).density -
                       chunk->voxels(pos.x-1,pos.y,pos.z).density;
            float dy = chunk->voxels(pos.x,pos.y+1,pos.z).density -
                       chunk->voxels(pos.x,pos.y-1,pos.z).density;
            float dz = chunk->voxels(pos.x,pos.y,pos.z+1).density -
                       chunk->voxels(pos.x,pos.y,pos.z-1).density;

            glm::vec3 normal(dx, dy, dz);
            if (glm::length(normal) > 0.0001f) {
                return glm::normalize(normal);
            }
            return glm::vec3(0, 1, 0);
        }


        size_t getActiveSlotCount() const {
            return activeSlots.size();
        }

        bool hasGpuSlot(const ChunkCoord& coord) const {
            const Chunk* chunk = const_cast<FixedGridChunkManager*>(this)->getChunk(coord);
            return chunk && chunk->gpuSlot != INVALID_GPU_SLOT;
        }

        void forEachChunk(const std::function<void(Chunk*)>& fn) {
            for (auto& [coord, chunk] : chunks) {
                fn(chunk.get());
            }
        }

        void clearAll() {
            for (auto& [coord, chunk] : chunks) {
                chunk->clear();
                chunk->coord = coord;
                chunk->gpuSlot = INVALID_GPU_SLOT;
                chunk->isCleared = true;
                chunk->meshDirty = false;
                chunk->lightingDirty = false;
                chunk->queuedForRebuild=false;
            }

            dirtyChunks.clear();
            nextGpuSlot = 0;
            freeGpuSlots.clear();
            slotToChunkCoord.clear();
            activeSlots.clear();
        }


        void forEachEmissiveChunk(const std::function<void(Chunk*)>& fn) {
            for (auto& [coord, chunk] : chunks) {
                if (chunk->hasEmissive) {
                    fn(chunk.get());
                }
            }
        }

        void updateEmissiveMembership(Chunk& c) {
            const bool nowEmissive = !c.emissiveLights.empty();
            c.hasEmissive = nowEmissive;
            c.inEmissiveList = nowEmissive;
        }

        std::vector<std::pair<ChunkCoord, Chunk*>> getChunksInRadius(
                const glm::vec3& center, float radiusWorld) {
            std::vector<std::pair<ChunkCoord, Chunk*>> out;

            const float chunkWorld = (float)CHUNK_SIZE * (float)VOXEL_SIZE;

            const int minCX = (int)std::floor((center.x - radiusWorld) / chunkWorld);
            const int maxCX = (int)std::floor((center.x + radiusWorld) / chunkWorld);
            const int minCY = (int)std::floor((center.y - radiusWorld) / chunkWorld);
            const int maxCY = (int)std::floor((center.y + radiusWorld) / chunkWorld);
            const int minCZ = (int)std::floor((center.z - radiusWorld) / chunkWorld);
            const int maxCZ = (int)std::floor((center.z + radiusWorld) / chunkWorld);

            const int clampedMinCX = std::max(minCX, -R);
            const int clampedMaxCX = std::min(maxCX,  R);
            const int clampedMinCY = std::max(minCY, -R);
            const int clampedMaxCY = std::min(maxCY,  R);
            const int clampedMinCZ = std::max(minCZ, -R);
            const int clampedMaxCZ = std::min(maxCZ,  R);

            for (int cx = clampedMinCX; cx <= clampedMaxCX; ++cx)
                for (int cy = clampedMinCY; cy <= clampedMaxCY; ++cy)
                    for (int cz = clampedMinCZ; cz <= clampedMaxCZ; ++cz) {
                        ChunkCoord cc{cx, cy, cz};
                        Chunk* c = getOrCreateChunk(cc);
                        if (!c) continue;
                        out.emplace_back(cc, c);
                    }

            return out;
        }

        template<typename MeshFn>
        void rebuildDirtyChunks(
                MeshFn&& rebuildMeshFn,
                const glm::vec3& cameraPos,
                const glm::mat4& projectionView)
        {
            if (dirtyChunks.empty()) {
                return;
            }

            // Partition the queue in-place:
            //
            // [ invisible dirty chunks | visible dirty chunks ]
            //
            // Invisible chunks stay queued and retain queuedForRebuild == true.
            // They cost one visibility test but no lighting, GPU-slot, upload, or
            // marching-cubes work this frame.
            auto firstVisible = std::partition(
                    dirtyChunks.begin(),
                    dirtyChunks.end(),
                    [&](const ChunkCoord& coord) {
                        return !isChunkVisible(coord, projectionView);
                    }
            );

            const int visibleDirtyCount = static_cast<int>(
                    dirtyChunks.end() - firstVisible
            );

            if (visibleDirtyCount == 0) {
                return;
            }

            const int toProcess = std::min(
                    MAX_CALC_PER_FRAME,
                    visibleDirtyCount
            );

            const float chunkWorldSize = float(CHUNK_SIZE) * VOXEL_SIZE;

            auto distanceSqToCamera = [&](const ChunkCoord& coord) {
                const glm::vec3 chunkCenter =
                        (glm::vec3(
                                static_cast<float>(coord.x),
                                static_cast<float>(coord.y),
                                static_cast<float>(coord.z)
                        ) + glm::vec3(0.5f)) * chunkWorldSize;

                const glm::vec3 delta = chunkCenter - cameraPos;
                return glm::dot(delta, delta);
            };

            // Sort farthest-to-nearest because processing is done with pop_back().
            auto farthestFirst = [&](const ChunkCoord& a, const ChunkCoord& b) {
                return distanceSqToCamera(a) > distanceSqToCamera(b);
            };

            // Select only the nearest visible chunks. Off-screen entries before
            // firstVisible are untouched.
            auto selectedBegin = dirtyChunks.end() - toProcess;

            if (toProcess < visibleDirtyCount) {
                std::nth_element(
                        firstVisible,
                        selectedBegin,
                        dirtyChunks.end(),
                        farthestFirst
                );
            }

            // MAX_CALC_PER_FRAME is small, so sorting only this suffix is cheap.
            std::sort(
                    selectedBegin,
                    dirtyChunks.end(),
                    farthestFirst
            );

            for (int i = 0; i < toProcess; ++i) {
                const ChunkCoord coord = dirtyChunks.back();
                dirtyChunks.pop_back();

                Chunk* chunk = getChunk(coord);
                if (!chunk) {
                    continue;
                }

                // This queue entry is now being consumed.
                chunk->queuedForRebuild = false;

                if (chunk->isCleared || !chunk->voxelData) {
                    continue;
                }

                if (chunk->lightingDirty) {
                    rebuildChunkLighting(chunk);
                }

                if (!chunk->meshDirty && chunk->gpuCache.isValid) {
                    continue;
                }

                if (chunk->gpuSlot == INVALID_GPU_SLOT) {
                    const uint32_t slot = allocateGpuSlot(coord);

                    if (slot == INVALID_GPU_SLOT) {
                        // Keep the chunk queued so it retries once a slot is freed.
                        markChunkDirty(coord);
                        continue;
                    }
                }

                rebuildMeshFn(chunk);
            }
        }

        bool isChunkVisible(
                const ChunkCoord& coord,
                const glm::mat4& projectionView) const
        {
            // Extract the six OpenGL clip-space frustum planes from PV.
            // GLM matrices are column-major, so these expressions construct rows.
            glm::vec4 planes[6] = {
                    // left, right
                    glm::vec4(
                            projectionView[0][3] + projectionView[0][0],
                            projectionView[1][3] + projectionView[1][0],
                            projectionView[2][3] + projectionView[2][0],
                            projectionView[3][3] + projectionView[3][0]
                    ),
                    glm::vec4(
                            projectionView[0][3] - projectionView[0][0],
                            projectionView[1][3] - projectionView[1][0],
                            projectionView[2][3] - projectionView[2][0],
                            projectionView[3][3] - projectionView[3][0]
                    ),

                    // bottom, top
                    glm::vec4(
                            projectionView[0][3] + projectionView[0][1],
                            projectionView[1][3] + projectionView[1][1],
                            projectionView[2][3] + projectionView[2][1],
                            projectionView[3][3] + projectionView[3][1]
                    ),
                    glm::vec4(
                            projectionView[0][3] - projectionView[0][1],
                            projectionView[1][3] - projectionView[1][1],
                            projectionView[2][3] - projectionView[2][1],
                            projectionView[3][3] - projectionView[3][1]
                    ),

                    // near, far
                    glm::vec4(
                            projectionView[0][3] + projectionView[0][2],
                            projectionView[1][3] + projectionView[1][2],
                            projectionView[2][3] + projectionView[2][2],
                            projectionView[3][3] + projectionView[3][2]
                    ),
                    glm::vec4(
                            projectionView[0][3] - projectionView[0][2],
                            projectionView[1][3] - projectionView[1][2],
                            projectionView[2][3] - projectionView[2][2],
                            projectionView[3][3] - projectionView[3][2]
                    )
            };

            const glm::vec3 minBounds = getChunkMin(coord);
            const glm::vec3 maxBounds = getChunkMax(coord);

            for (glm::vec4& plane : planes) {
                const glm::vec3 normal(plane.x, plane.y, plane.z);
                const float normalLength = glm::length(normal);

                if (normalLength <= 0.00001f) {
                    continue;
                }

                plane /= normalLength;

                // Pick the AABB corner furthest in the plane normal direction.
                // If even that corner is outside, the entire chunk is outside.
                const glm::vec3 positiveVertex(
                        plane.x >= 0.0f ? maxBounds.x : minBounds.x,
                        plane.y >= 0.0f ? maxBounds.y : minBounds.y,
                        plane.z >= 0.0f ? maxBounds.z : minBounds.z
                );

                if (glm::dot(glm::vec3(plane), positiveVertex) + plane.w < 0.0f) {
                    return false;
                }
            }

            return true;
        }

        void markChunkDirty(const ChunkCoord& coord) {
            Chunk* chunk = getChunk(coord);

            // The caller should only queue chunks that already exist.
            if (!chunk || chunk->isCleared || !chunk->voxelData) {
                return;
            }

            if (chunk->queuedForRebuild) {
                return;
            }

            chunk->queuedForRebuild = true;
            dirtyChunks.push_back(coord);
        }

        void rebuildChunkLighting(Chunk* chunk) {
            if (!chunk || chunk->isCleared) return;
            chunk->emissiveLights.clear();

            // chunk origin in world units
            glm::vec3 chunkOrigin(
                    chunk->coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                    chunk->coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                    chunk->coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
            );

            // Cluster emissive voxels within this chunk
            glm::vec3 sumPos(0.0f);
            glm::vec3 sumColor(0.0f);
            int count = 0;

            for (int x = 0; x <= CHUNK_SIZE; ++x) {
                for (int y = 0; y <= CHUNK_SIZE; ++y) {
                    for (int z = 0; z <= CHUNK_SIZE; ++z) {
                        const auto &vox = chunk->voxels(x,y,z);
                        if (vox.type == 2) { // Fire / emissive voxel
                            glm::vec3 voxelWorldPos = chunkOrigin + glm::vec3((float)x, (float)y, (float)z) * gl3::VOXEL_SIZE;
                            sumPos += voxelWorldPos;
                            sumColor += vox.color;
                            ++count;
                        }
                    }
                }
            }

            if (count > 0) {
                VoxelLight light;
                light.pos = sumPos / float(count);
                light.color = sumColor / float(count);
                light.intensity = float(count) * 65.0f;
                light.id = makeLightID(chunk->coord.x, chunk->coord.y, chunk->coord.z);

                chunk->emissiveLights.push_back(light);
            }

            chunk->lightingDirty = false;
            updateEmissiveMembership(*chunk);
        }

        uint32_t makeLightID(int cx, int cy, int cz) {
            return ((cx & 0xFFF) << 20) | ((cy & 0xFFF) << 8) | (cz & 0xFF);
        }

    private:
        int R = 0;
        int dim = 0;
        const int MAX_CALC_PER_FRAME = 2;

        std::unordered_map<ChunkCoord, std::unique_ptr<Chunk>, ChunkCoordHash> chunks;
        std::vector<ChunkCoord> dirtyChunks;

        uint32_t nextGpuSlot = 0;
        std::vector<uint32_t> freeGpuSlots;
        std::unordered_map<uint32_t, ChunkCoord> slotToChunkCoord;
        std::unordered_set<uint32_t> activeSlots;

        uint32_t toIndex(const ChunkCoord& cc) const {
            const int ix = cc.x + R;
            const int iy = cc.y + R;
            const int iz = cc.z + R;
            return (uint32_t)(ix + iy*dim + iz*dim*dim);
        }

        bool evictFurthestChunk(const ChunkCoord& referenceCoord) {
            if (slotToChunkCoord.empty()) return false;

            uint32_t furthestSlot = INVALID_GPU_SLOT;
            int maxDistSq = -1;

            for (const auto& [slot, coord] : slotToChunkCoord) {
                int dx = coord.x - referenceCoord.x;
                int dy = coord.y - referenceCoord.y;
                int dz = coord.z - referenceCoord.z;
                int distSq = dx*dx + dy*dy + dz*dz;

                if (distSq > maxDistSq) {
                    maxDistSq = distSq;
                    furthestSlot = slot;
                }
            }

            if (furthestSlot != INVALID_GPU_SLOT) {
                auto it = slotToChunkCoord.find(furthestSlot);
                if (it != slotToChunkCoord.end()) {
                    freeGpuSlot(it->second);
                    return true;
                }
            }

            return false;
        }
    };

}