#pragma once

#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <stdexcept>

#include "VoxelStructures.h"
#include "Chunk.h"
#include "..\robin_hood.h"

namespace gl3 {

    enum class VoxelCategory : uint8_t {
        STATIC = 0,
        DYNAMIC = 1,
        EMISSIVE = 2,
        FLUID = 3,
        INTERACTIVE = 4
    };

    class MultiGridChunkManager {
    private:
        // ----------------------------
        // Category grid (iteration + lookup)
        // ----------------------------
        struct CategoryGrid {
            robin_hood::unordered_map<ChunkCoord, Chunk*, ChunkCoordHash> chunkMap;
            std::vector<Chunk*> chunkPointers; // dense iteration list

            bool contains(const ChunkCoord& coord) const {
                return chunkMap.find(coord) != chunkMap.end();
            }

            Chunk* getChunk(const ChunkCoord& coord) const {
                auto it = chunkMap.find(coord);
                return (it != chunkMap.end()) ? it->second : nullptr;
            }

            void addChunk(const ChunkCoord& coord, Chunk* chunk) {
                // Prevent duplicate insertion (critical for correct iteration)
                if (contains(coord)) return;

                chunkMap[coord] = chunk;
                chunkPointers.push_back(chunk);
            }

            void removeChunk(const ChunkCoord& coord) {
                auto it = chunkMap.find(coord);
                if (it == chunkMap.end()) return;

                Chunk* ptr = it->second;
                chunkMap.erase(it);

                // swap-remove from chunkPointers
                for (size_t i = 0; i < chunkPointers.size(); ++i) {
                    if (chunkPointers[i] == ptr) {
                        chunkPointers[i] = chunkPointers.back();
                        chunkPointers.pop_back();
                        break;
                    }
                }
            }

            void clear() {
                chunkMap.clear();
                chunkPointers.clear();
            }
        };

        CategoryGrid staticGrid;
        CategoryGrid dynamicGrid;
        CategoryGrid emissiveGrid;
        CategoryGrid fluidGrid;
        CategoryGrid interactiveGrid;

        // Master storage for all chunks
        robin_hood::unordered_map<ChunkCoord, std::unique_ptr<Chunk>, ChunkCoordHash> allChunks;

        // Track which category each chunk is currently in (so we can remove correctly)
        robin_hood::unordered_map<ChunkCoord, VoxelCategory, ChunkCoordHash> chunkCategory;

        // ----------------------------
        // GLOBAL GPU slot allocator (the real fix)
        // ----------------------------
        uint32_t nextGpuSlot = 0;
        std::vector<uint32_t> freeGpuSlots;
        static constexpr uint32_t MAX_CHUNKS_GPU = 350;


        static constexpr uint32_t INVALID_SLOT = 0xFFFFFFFFu;

        uint32_t allocateGpuSlot() {
            if (!freeGpuSlots.empty()) {
                uint32_t s = freeGpuSlots.back();
                freeGpuSlots.pop_back();
                return s;
            }
            if (nextGpuSlot >= MAX_CHUNKS_GPU) {
                throw std::runtime_error("Out of GPU chunk slots (MAX_CHUNKS_GPU)");
            }
            return nextGpuSlot++;
        }

        void freeGpuSlot(uint32_t slot) {
            if (slot == INVALID_SLOT) return;
            if (slot < MAX_CHUNKS_GPU) freeGpuSlots.push_back(slot);
        }

        CategoryGrid& gridFor(VoxelCategory c) {
            switch (c) {
                case VoxelCategory::STATIC:       return staticGrid;
                case VoxelCategory::DYNAMIC:      return dynamicGrid;
                case VoxelCategory::EMISSIVE:     return emissiveGrid;
                case VoxelCategory::FLUID:        return fluidGrid;
                case VoxelCategory::INTERACTIVE:  return interactiveGrid;
            }
            return staticGrid; // unreachable, but keeps compiler happy
        }

        const CategoryGrid& gridFor(VoxelCategory c) const {
            switch (c) {
                case VoxelCategory::STATIC:       return staticGrid;
                case VoxelCategory::DYNAMIC:      return dynamicGrid;
                case VoxelCategory::EMISSIVE:     return emissiveGrid;
                case VoxelCategory::FLUID:        return fluidGrid;
                case VoxelCategory::INTERACTIVE:  return interactiveGrid;
            }
            return staticGrid;
        }

    public:
        void clear() {
            allChunks.clear();
            chunkCategory.clear();

            staticGrid.clear();
            dynamicGrid.clear();
            emissiveGrid.clear();
            fluidGrid.clear();
            interactiveGrid.clear();

            nextGpuSlot = 0;
            freeGpuSlots.clear();
        }

        // Add chunk (create if missing), assign a UNIQUE global gpuSlot, put it in correct grid
        void addChunk(const ChunkCoord& coord, VoxelCategory category) {
            // 1) Create if missing
            auto& uptr = allChunks[coord];
            if (!uptr) {
                uptr = std::make_unique<Chunk>();
                uptr->gpuSlot = INVALID_SLOT; // mark unassigned explicitly
            }

            Chunk* chunk = uptr.get();

            // 2) Remove from old category grid if it existed
            removeChunkFromAllCategories(coord);

            // 3) Ensure global unique gpuSlot assigned ONCE
            if (chunk->gpuSlot == INVALID_SLOT) {
                chunk->gpuSlot = allocateGpuSlot();
            }

            // 4) Mark dirty on add (as you had)
            chunk->gpuCache.isValid = false;
            chunk->meshDirty = true;

            // 5) Insert into new grid
            gridFor(category).addChunk(coord, chunk);
            chunkCategory[coord] = category;
        }

        // Optional: if you ever truly delete a chunk and want to reclaim slot
        void removeChunk(const ChunkCoord& coord) {
            auto it = allChunks.find(coord);
            if (it == allChunks.end()) return;

            removeChunkFromAllCategories(coord);

            // free GPU slot
            freeGpuSlot(it->second->gpuSlot);

            // erase storage
            allChunks.erase(it);
            chunkCategory.erase(coord);
        }

        Chunk* getChunk(const ChunkCoord& coord) {
            auto it = allChunks.find(coord);
            return (it != allChunks.end()) ? it->second.get() : nullptr;
        }

        Chunk* getChunkWithCategory(const ChunkCoord& coord, VoxelCategory expected) {
            // fast path: expected grid
            if (Chunk* c = gridFor(expected).getChunk(coord)) return c;

            // fallback: any
            return getChunk(coord);
        }

        // Iteration APIs (restored)
        void forEachChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* c : staticGrid.chunkPointers) callback(c);
            for (Chunk* c : emissiveGrid.chunkPointers) callback(c);
            for (Chunk* c : dynamicGrid.chunkPointers) callback(c);
            for (Chunk* c : fluidGrid.chunkPointers) callback(c);
            for (Chunk* c : interactiveGrid.chunkPointers) callback(c);
        }

        void forEachEmissiveChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* c : emissiveGrid.chunkPointers) callback(c);
        }

        void forEachFluidChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* c : fluidGrid.chunkPointers) callback(c);
        }

        void forEachDynamicChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* c : dynamicGrid.chunkPointers) callback(c);
        }

        // Get all chunk coordinates that exist
        std::vector<ChunkCoord> getAllChunkCoords() const {
            std::vector<ChunkCoord> coords;
            coords.reserve(allChunks.size());
            for (const auto& kv : allChunks) coords.push_back(kv.first);
            return coords;
        }

        // Get all chunks within a radius of a point (restored)
        std::vector<std::pair<ChunkCoord, Chunk*>> getChunksInRadius(
                const glm::vec3& center, float radius)
        {
            std::vector<std::pair<ChunkCoord, Chunk*>> result;

            const float chunkWorld = (float)CHUNK_SIZE * (float)VOXEL_SIZE;

            int minCX = (int)std::floor((center.x - radius) / chunkWorld);
            int maxCX = (int)std::floor((center.x + radius) / chunkWorld);
            int minCY = (int)std::floor((center.y - radius) / chunkWorld);
            int maxCY = (int)std::floor((center.y + radius) / chunkWorld);
            int minCZ = (int)std::floor((center.z - radius) / chunkWorld);
            int maxCZ = (int)std::floor((center.z + radius) / chunkWorld);

            for (int cx = minCX; cx <= maxCX; ++cx)
                for (int cy = minCY; cy <= maxCY; ++cy)
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord cc{cx, cy, cz};
                        if (Chunk* c = const_cast<MultiGridChunkManager*>(this)->getChunk(cc)) {
                            result.emplace_back(cc, c);
                        }
                    }

            return result;
        }

        // Change category (restored)
        void changeCategory(const ChunkCoord& coord, VoxelCategory newCategory) {
            addChunk(coord, newCategory);
        }

    private:
        void removeChunkFromAllCategories(const ChunkCoord& coord) {
            auto itCat = chunkCategory.find(coord);
            if (itCat == chunkCategory.end()) return;

            VoxelCategory old = itCat->second;
            gridFor(old).removeChunk(coord);
            chunkCategory.erase(itCat);
        }
    };

} // namespace gl3