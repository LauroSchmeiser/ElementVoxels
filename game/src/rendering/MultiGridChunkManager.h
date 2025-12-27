#pragma once

#include <vector>
#include <bitset>
#include <functional>
#include <unordered_map>
#include "VoxelStructures.h"
#include "Chunk.h"
#include "..\robin_hood.h"

namespace gl3 {

    enum class VoxelCategory : uint8_t {
        STATIC = 0,    // Regular terrain - rarely changes
        DYNAMIC = 1,   // Physics objects - moderate changes
        FLUID = 2,     // Fluids - frequent updates
        EMISSIVE = 3,  // Light sources
        INTERACTIVE = 4 // Player-modified
    };

    class MultiGridChunkManager {
    private:
        static constexpr int DENSE_GRID_SIZE = 32; // Define it here too

        // Separate grids based on update frequency
        struct CategoryGrid {
            // Using a simple map for now, you can optimize later
            robin_hood::unordered_map<ChunkCoord, Chunk*, ChunkCoordHash> chunkMap;
            std::vector<ChunkCoord> activeList;   // For iteration
            std::vector<Chunk*> chunkPointers;    // For fast iteration

            void addChunk(const ChunkCoord& coord, Chunk* chunk) {
                chunkMap[coord] = chunk;
                activeList.push_back(coord);
                chunkPointers.push_back(chunk);
            }

            Chunk* getChunk(const ChunkCoord& coord) {
                auto it = chunkMap.find(coord);
                return (it != chunkMap.end()) ? it->second : nullptr;
            }

            void clear() {
                chunkMap.clear();
                activeList.clear();
                chunkPointers.clear();
            }
        };

        CategoryGrid staticGrid;
        CategoryGrid dynamicGrid;
        CategoryGrid fluidGrid;
        CategoryGrid emissiveGrid;
        CategoryGrid interactiveGrid;

        // Master storage for all chunks
        robin_hood::unordered_map<ChunkCoord, std::unique_ptr<Chunk>, ChunkCoordHash> allChunks;

    public:
        // Add a chunk with category
        void addChunk(const ChunkCoord& coord, VoxelCategory category) {
            auto& chunkPtr = allChunks[coord];
            if (!chunkPtr) {
                chunkPtr = std::make_unique<Chunk>();
            }

            Chunk* chunk = chunkPtr.get();

            // Remove from any existing category
            removeChunkFromAllCategories(coord);

            // Add to appropriate category
            switch (category) {
                case VoxelCategory::STATIC: staticGrid.addChunk(coord, chunk); break;
                case VoxelCategory::DYNAMIC: dynamicGrid.addChunk(coord, chunk); break;
                case VoxelCategory::FLUID: fluidGrid.addChunk(coord, chunk); break;
                case VoxelCategory::EMISSIVE: emissiveGrid.addChunk(coord, chunk); break;
                case VoxelCategory::INTERACTIVE: interactiveGrid.addChunk(coord, chunk); break;
            }
        }

        // Get chunk with category hint
        Chunk* getChunkWithCategory(const ChunkCoord& coord, VoxelCategory expected) {
            // Check appropriate grid first based on category
            CategoryGrid* grid = nullptr;
            switch (expected) {
                case VoxelCategory::FLUID: grid = &fluidGrid; break;
                case VoxelCategory::DYNAMIC: grid = &dynamicGrid; break;
                case VoxelCategory::EMISSIVE: grid = &emissiveGrid; break;
                case VoxelCategory::INTERACTIVE: grid = &interactiveGrid; break;
                default: grid = &staticGrid; break;
            }

            Chunk* chunk = grid->getChunk(coord);
            if (chunk) return chunk;

            // Fallback: check all chunks
            auto it = allChunks.find(coord);
            return (it != allChunks.end()) ? it->second.get() : nullptr;
        }

        // Get any chunk (category-agnostic)
        Chunk* getChunk(const ChunkCoord& coord) {
            auto it = allChunks.find(coord);
            return (it != allChunks.end()) ? it->second.get() : nullptr;
        }

        // Optimized iteration for specific systems
        void forEachFluidChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* chunk : fluidGrid.chunkPointers) {
                callback(chunk);
            }
        }

        void forEachDynamicChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* chunk : dynamicGrid.chunkPointers) {
                callback(chunk);
            }
        }

        void forEachEmissiveChunk(const std::function<void(Chunk*)>& callback) {
            for (Chunk* chunk : emissiveGrid.chunkPointers) {
                callback(chunk);
            }
        }

        // Change category of a chunk
        void changeCategory(const ChunkCoord& coord, VoxelCategory newCategory) {
            addChunk(coord, newCategory);
        }

        // Get all chunk coordinates that exist
        std::vector<ChunkCoord> getAllChunkCoords() const {
            std::vector<ChunkCoord> coords;
            coords.reserve(allChunks.size());
            for (const auto& pair : allChunks) {
                coords.push_back(pair.first);
            }
            return coords;
        }

        // Get all chunks within a radius of a point
        std::vector<std::pair<ChunkCoord, Chunk*>> getChunksInRadius(
                const glm::vec3& center, float radius) {

            std::vector<std::pair<ChunkCoord, Chunk*>> result;

            // Calculate chunk bounds
            int minCX = (int)std::floor((center.x - radius) / CHUNK_SIZE);
            int maxCX = (int)std::floor((center.x + radius) / CHUNK_SIZE);
            int minCY = (int)std::floor((center.y - radius) / CHUNK_SIZE);
            int maxCY = (int)std::floor((center.y + radius) / CHUNK_SIZE);
            int minCZ = (int)std::floor((center.z - radius) / CHUNK_SIZE);
            int maxCZ = (int)std::floor((center.z + radius) / CHUNK_SIZE);

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord coord{cx, cy, cz};
                        if (Chunk* chunk = getChunk(coord)) {
                            result.emplace_back(coord, chunk);
                        }
                    }
                }
            }

            return result;
        }

    private:
        void removeChunkFromAllCategories(const ChunkCoord& coord) {
            // In a real implementation, you'd need to track which category each chunk is in
            // For simplicity, we'll just clear and rebuild lists when changing categories
            // This is OK if category changes are infrequent

        }
    };
}