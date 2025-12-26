#pragma once

#include <vector>
#include <array>
#include <memory>
#include "glm/glm.hpp"
#include "../robin_hood.h"
#include "VoxelStructures.h"
#include "Chunk.h"  // Now we include the full Chunk definition

namespace gl3 {

    class ChunkManager {
    public:
        static constexpr int DENSE_GRID_SIZE = 64; // Adjust based on memory

    private:
        // Dense storage for loaded/active chunks
        std::array<std::array<std::array<std::unique_ptr<Chunk>, DENSE_GRID_SIZE>, DENSE_GRID_SIZE>, DENSE_GRID_SIZE> denseGrid;
        glm::ivec3 gridOrigin; // World coordinates of denseGrid[0][0][0]

        // Sparse storage for everything else
        robin_hood::unordered_map<ChunkCoord, std::unique_ptr<Chunk>, ChunkCoordHash> sparseStorage;

        // Translation functions
        glm::ivec3 worldToGrid(const ChunkCoord& coord) const {
            return {
                    coord.x - gridOrigin.x,
                    coord.y - gridOrigin.y,
                    coord.z - gridOrigin.z
            };
        }

        bool isInDenseGrid(const glm::ivec3& gridPos) const {
            return gridPos.x >= 0 && gridPos.x < DENSE_GRID_SIZE &&
                   gridPos.y >= 0 && gridPos.y < DENSE_GRID_SIZE &&
                   gridPos.z >= 0 && gridPos.z < DENSE_GRID_SIZE;
        }

    public:
        // Constructor
        ChunkManager() {
            // Initialize all pointers to nullptr
            for (auto& layer : denseGrid) {
                for (auto& row : layer) {
                    for (auto& chunkPtr : row) {
                        chunkPtr.reset();
                    }
                }
            }
            gridOrigin = glm::ivec3(0, 0, 0);
        }

        // Move the dense grid center
        void setGridOrigin(const glm::ivec3& newOrigin) {
            // TODO: Implement proper chunk loading/unloading when origin changes
            gridOrigin = newOrigin;
        }

        // Get or create chunk
        Chunk* getOrCreateChunk(const ChunkCoord& coord) {
            auto gridPos = worldToGrid(coord);

            // Fast path: in dense grid
            if (isInDenseGrid(gridPos)) {
                auto& chunkPtr = denseGrid[gridPos.x][gridPos.y][gridPos.z];
                if (!chunkPtr) {
                    chunkPtr = std::make_unique<Chunk>();
                }
                return chunkPtr.get();
            }

            // Slow path: check/create in sparse storage
            auto it = sparseStorage.find(coord);
            if (it != sparseStorage.end()) {
                return it->second.get();
            }

            // Create new chunk
            auto newChunk = std::make_unique<Chunk>();
            Chunk* result = newChunk.get();
            sparseStorage[coord] = std::move(newChunk);
            return result;
        }

        // Get existing chunk (returns nullptr if doesn't exist)
        Chunk* getChunk(const ChunkCoord& coord) {
            auto gridPos = worldToGrid(coord);

            // Fast path: in dense grid
            if (isInDenseGrid(gridPos)) {
                return denseGrid[gridPos.x][gridPos.y][gridPos.z].get();
            }

            // Slow path: check sparse storage
            auto it = sparseStorage.find(coord);
            if (it != sparseStorage.end()) {
                return it->second.get();
            }

            return nullptr; // Chunk doesn't exist
        }

        // For physics/simulation that needs many sequential chunks
        void getChunkNeighbors(const ChunkCoord& center, std::array<Chunk*, 27>& neighbors) {
            int idx = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        ChunkCoord coord{center.x + dx, center.y + dy, center.z + dz};
                        neighbors[idx++] = getChunk(coord);
                    }
                }
            }
        }

        // Remove a chunk
        bool removeChunk(const ChunkCoord& coord) {
            auto gridPos = worldToGrid(coord);

            if (isInDenseGrid(gridPos)) {
                denseGrid[gridPos.x][gridPos.y][gridPos.z].reset();
                return true;
            }

            return sparseStorage.erase(coord) > 0;
        }

        // Get all active chunks (for rendering)
        std::vector<Chunk*> getAllActiveChunks() const {
            std::vector<Chunk*> result;
            result.reserve(DENSE_GRID_SIZE * DENSE_GRID_SIZE * DENSE_GRID_SIZE + sparseStorage.size());

            // Add dense grid chunks
            for (const auto& layer : denseGrid) {
                for (const auto& row : layer) {
                    for (const auto& chunkPtr : row) {
                        if (chunkPtr) {
                            result.push_back(chunkPtr.get());
                        }
                    }
                }
            }

            // Add sparse storage chunks
            for (const auto& pair : sparseStorage) {
                result.push_back(pair.second.get());
            }

            return result;
        }
    };
}