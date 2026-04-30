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


namespace gl3 {

    class FixedGridChunkManager {
    public:
        static constexpr uint32_t INVALID_GPU_SLOT = 0xFFFFFFFFu;
        static constexpr uint32_t MAX_GPU_SLOTS = 1350;

        explicit FixedGridChunkManager(int radiusChunks)
                : R(radiusChunks),
                  dim(2 * R + 1),
                  chunks((size_t)dim * dim * dim)
        {
            for (int z = -R; z <= R; ++z)
                for (int y = -R; y <= R; ++y)
                    for (int x = -R; x <= R; ++x)
                    {
                        const uint32_t index = toIndex({x,y,z});
                        assert(index < chunks.size());
                        Chunk& c = chunks[index];
                        c.coord = {x,y,z};
                        c.gpuSlot = INVALID_GPU_SLOT;
                        c.clear();
                        c.isCleared = false;
                    }
        }

        int radius() const { return R; }
        int dimension() const { return dim; }
        uint32_t maxChunksGpu() const { return MAX_GPU_SLOTS; }
        size_t totalChunksInGrid() const { return chunks.size(); }

        bool inBounds(const ChunkCoord& cc) const {
            return (cc.x >= -R && cc.x <= R &&
                    cc.y >= -R && cc.y <= R &&
                    cc.z >= -R && cc.z <= R);
        }

        Chunk* getChunk(const ChunkCoord& cc) {
            if (!inBounds(cc)) return nullptr;
            return &chunks[toIndex(cc)];
        }

        uint32_t allocateGpuSlot(const ChunkCoord& coord) {
            Chunk* chunk = getChunk(coord);
            if (!chunk) return INVALID_GPU_SLOT;

            if (chunk->gpuSlot != INVALID_GPU_SLOT) {
                return chunk->gpuSlot;
            }

            uint32_t slot;
            if (!freeGpuSlots.empty()) {
                slot = freeGpuSlots.back();
                freeGpuSlots.pop_back();
            } else if (nextGpuSlot < MAX_GPU_SLOTS) {
                slot = nextGpuSlot++;
            } else {
                if (!evictFurthestChunk(coord)) {
                    return INVALID_GPU_SLOT;
                }
                slot = nextGpuSlot - 1;
            }

            chunk->gpuSlot = slot;
            slotToChunkCoord[slot] = coord;
            activeSlots.insert(slot);

            return slot;
        }

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

        void cleanupDistantSlots(const glm::vec3& cameraPos, int renderRadiusChunks) {
            const int camCX = worldToChunk(cameraPos.x);
            const int camCY = worldToChunk(cameraPos.y);
            const int camCZ = worldToChunk(cameraPos.z);

            const int keepRadius = renderRadiusChunks + 3;

            std::vector<ChunkCoord> toFree;

            for (auto& [slot, coord] : slotToChunkCoord) {
                int dx = std::abs(coord.x - camCX);
                int dy = std::abs(coord.y - camCY);
                int dz = std::abs(coord.z - camCZ);

                if (dx > keepRadius || dy > keepRadius || dz > keepRadius) {
                    toFree.push_back(coord);
                }
            }

            for (const auto& coord : toFree) {
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

            float dx = chunk->voxels[pos.x+1][pos.y][pos.z].density -
                       chunk->voxels[pos.x-1][pos.y][pos.z].density;
            float dy = chunk->voxels[pos.x][pos.y+1][pos.z].density -
                       chunk->voxels[pos.x][pos.y-1][pos.z].density;
            float dz = chunk->voxels[pos.x][pos.y][pos.z+1].density -
                       chunk->voxels[pos.x][pos.y][pos.z-1].density;

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
            for (auto& c : chunks) fn(&c);
        }


        void clearAll() {
            for (auto& c : chunks) {
                c.clear();
                c.isCleared = false;
                c.gpuSlot = INVALID_GPU_SLOT;
                c.gpuCache.isValid = false;
            }

            nextGpuSlot = 0;
            freeGpuSlots.clear();
            slotToChunkCoord.clear();
            activeSlots.clear();
        }

        void forEachEmissiveChunk(const std::function<void(Chunk*)>& fn) {
            for (uint32_t idx : emissiveIndices) {
                Chunk& c = chunks[idx];
                if (!c.hasEmissive) continue;
                fn(&c);
            }
        }

        void updateEmissiveMembership(Chunk& c) {
            const bool nowEmissive = !c.emissiveLights.empty();

            if (nowEmissive && !c.inEmissiveList) {
                c.hasEmissive = true;
                c.inEmissiveList = true;
                emissiveIndices.push_back(toIndex(c.coord));
            } else if (!nowEmissive) {
                c.hasEmissive = false;
                removeEmissiveIndex(toIndex(c.coord));
                c.inEmissiveList = false;
            }
        }

        void removeEmissiveIndex(uint32_t idx) {
            auto it = std::find(emissiveIndices.begin(), emissiveIndices.end(), idx);
            if (it == emissiveIndices.end()) return;
            *it = emissiveIndices.back();
            emissiveIndices.pop_back();
            chunks[idx].inEmissiveList = false;
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
                        Chunk* c = getChunk(cc);
                        if (!c) continue;
                        out.emplace_back(cc, c);
                    }

            return out;
        }

    private:
        int R = 0;
        int dim = 0;
        std::vector<Chunk> chunks;
        std::vector<uint32_t> emissiveIndices;

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