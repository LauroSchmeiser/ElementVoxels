#pragma once

#include <chrono>
#include <vector>
#include <cmath>
#include <cstdint>
#include <functional>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "../robin_hood.h"
#include "../rendering/Chunk.h"
#include "../rendering/MultiGridChunkManager.h"
#include "../rendering/VoxelStructures.h"

#include <tracy/Tracy.hpp>
#define TRACY_CPU_ZONE(nameStr) ZoneScopedN(nameStr)


namespace gl3 {

// Unified crater/formation carving using SDF with chunk AABB early rejection + voxel-sphere clipping.
    class VoxelCarver {
    public:
        struct CarveResult {
            std::vector<gl3::ChunkCoord> modifiedChunks;
            size_t voxelsModified = 0;
            float executionTimeMs = 0.0f;
        };

        struct VoxelUpdate {
            ChunkCoord coord;
            glm::ivec3 localPos;
            float newDensity;
            uint8_t newType;
            glm::vec3 color;
            uint64_t material;
        };

        struct CarveParams {
            float radius;                 // effect radius (world units)
            float strength = 1.0f;        // carving strength
            uint8_t targetType = 1;       // voxel type to set when solid
            glm::vec3 color{1.0f};        // voxel color
            uint64_t material = 0;        // material id
            bool autoCreate = true;       // create missing chunks
            bool additive = false;        // additive (union) vs subtractive
            float densityThreshold = -0.5f; // air threshold (matches your other code)
        };

        static CarveResult carveSDF(
                gl3::MultiGridChunkManager* chunkManager,
                const glm::vec3& center,
                const std::function<float(const glm::vec3&)>& sdf,
                const CarveParams& params
        ) {
            TRACY_CPU_ZONE("VoxelCarver::carveSDF");
            auto startTime = std::chrono::high_resolution_clock::now();

            CarveResult result;
            if (!chunkManager) return result;

            // Calculate affected chunk range (same math style as Game::worldToChunk)
            const int minCX = worldToChunk(center.x - params.radius);
            const int maxCX = worldToChunk(center.x + params.radius);
            const int minCY = worldToChunk(center.y - params.radius);
            const int maxCY = worldToChunk(center.y + params.radius);
            const int minCZ = worldToChunk(center.z - params.radius);
            const int maxCZ = worldToChunk(center.z + params.radius);

            std::vector<VoxelUpdate> pendingUpdates;
            pendingUpdates.reserve(estimateVoxelCount(params.radius));

            const float radiusSq = params.radius * params.radius;

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord coord{cx, cy, cz};

                        // Chunk AABB in world units (match Game::getChunkMin/getChunkMax behavior)
                        glm::vec3 chunkMin = getChunkMin(coord);
                        glm::vec3 chunkMax = chunkMin + glm::vec3(CHUNK_SIZE * VOXEL_SIZE);

                        // Quick reject if the sphere doesn't touch the chunk AABB
                        const float chunkDistSq = squaredDistanceToAABB(center, chunkMin, chunkMax);
                        if (chunkDistSq > radiusSq) continue;

                        Chunk* chunk = ensureChunkExists(chunkManager, coord, params.autoCreate);
                        if (!chunk) continue;

                        processChunkVoxelsSDF(
                                chunk, center, sdf, params,
                                chunkMin,
                                pendingUpdates,
                                result
                        );
                    }
                }
            }

            applyVoxelUpdates(chunkManager, pendingUpdates, result.modifiedChunks);

            auto endTime = std::chrono::high_resolution_clock::now();
            result.executionTimeMs =
                    std::chrono::duration<float, std::milli>(endTime - startTime).count();

            return result;
        }

    private:
        // ---- Glue helpers to match your existing coordinate system ----

        static int worldToChunk(float worldPos) {
            // Same as Game::worldToChunk
            const float chunkWorldSize = CHUNK_SIZE * gl3::VOXEL_SIZE;
            return (int)std::floor(worldPos / chunkWorldSize);
        }

        static glm::vec3 getChunkMin(const ChunkCoord& coord) {
            // Same as Game::getChunkMin
            return glm::vec3(
                    coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                    coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                    coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
            );
        }

        // Fast AABB squared distance
        static float squaredDistanceToAABB(const glm::vec3& point,
                                           const glm::vec3& bmin,
                                           const glm::vec3& bmax) {
            const float dx = glm::max(bmin.x - point.x,  point.x - bmax.x);
            const float dy = glm::max(bmin.y - point.y,  point.y - bmax.y);
            const float dz = glm::max(bmin.z - point.z, point.z - bmax.z);
            return dx * dx + dy * dy + dz * dz;
        }

        static size_t estimateVoxelCount(float radius) {
            // radius is world-units
            const float sphereVolume = (4.0f / 3.0f) * glm::pi<float>() * radius * radius * radius;
            const float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
            return (size_t)(sphereVolume / voxelVolume * 1.5f);
        }

        static void processChunkVoxelsSDF(
                Chunk* chunk,
                const glm::vec3& center,
                const std::function<float(const glm::vec3&)>& sdf,
                const CarveParams& params,
                const glm::vec3& chunkMin,
                std::vector<VoxelUpdate>& updates,
                CarveResult& result
        ) {
            TRACY_CPU_ZONE("VoxelCarver::processChunkVoxelsSDF");
            const float voxelSize = VOXEL_SIZE;
            const float radiusSq = params.radius * params.radius;

            // center in chunk-local voxel coordinates (float)
            const glm::vec3 localCenter = (center - chunkMin) / voxelSize;
            const float radiusVox = params.radius / voxelSize;

            const int minVx = glm::max(0, (int)std::floor(localCenter.x - radiusVox));
            const int maxVx = glm::min(CHUNK_SIZE, (int)std::ceil(localCenter.x + radiusVox));
            const int minVy = glm::max(0, (int)std::floor(localCenter.y - radiusVox));
            const int maxVy = glm::min(CHUNK_SIZE, (int)std::ceil(localCenter.y + radiusVox));
            const int minVz = glm::max(0, (int)std::floor(localCenter.z - radiusVox));
            const int maxVz = glm::min(CHUNK_SIZE, (int)std::ceil(localCenter.z + radiusVox));

            for (int vx = minVx; vx <= maxVx; ++vx) {
                const float worldX = chunkMin.x + vx * voxelSize;
                const float dx = worldX - center.x;
                const float dxSq = dx * dx;

                for (int vy = minVy; vy <= maxVy; ++vy) {
                    const float worldY = chunkMin.y + vy * voxelSize;
                    const float dy = worldY - center.y;
                    const float dySq = dy * dy;

                    if (dxSq + dySq > radiusSq) continue;

                    for (int vz = minVz; vz <= maxVz; ++vz) {
                        const float worldZ = chunkMin.z + vz * voxelSize;
                        const float dz = worldZ - center.z;

                        const float distSq = dxSq + dySq + dz * dz;
                        if (distSq > radiusSq) continue;

                        Voxel& voxel = chunk->voxels[vx][vy][vz];

                        // Skip already-air in subtractive mode (same idea as your code)
                        if (!params.additive && voxel.density < params.densityThreshold) {
                            continue;
                        }

                        const glm::vec3 worldPos(worldX, worldY, worldZ);
                        const float sdfValue = sdf(worldPos);

                        // IMPORTANT: your SDF conventions in Game.cpp are "positive inside"
                        // - For additive union, you want max(existing, sdf)
                        // - For subtractive carve, you typically want to *reduce* density where sdf indicates "inside crater"
                        //
                        // Here we assume sdfValue is a "density delta" style for subtractive:
                        // craterSDF in your Game.cpp returns negative values (because it returns -depth*falloff).
                        // So subtractive should be: density += sdfValue * strength (since sdfValue is negative).
                        float newDensity = voxel.density;
                        if (params.additive) {
                            newDensity = glm::max(voxel.density, sdfValue * params.strength);
                        } else {
                            newDensity = voxel.density + (sdfValue * params.strength);
                        }

                        if (std::abs(newDensity - voxel.density) > 0.001f) {
                            const uint8_t newType = (newDensity < params.densityThreshold) ? 0 : params.targetType;

                            updates.push_back(VoxelUpdate{
                                    chunk->coord,
                                    glm::ivec3(vx, vy, vz),
                                    newDensity,
                                    newType,
                                    params.color,
                                    params.material
                            });
                            result.voxelsModified++;
                        }
                    }
                }
            }
        }

        static void applyVoxelUpdates(
                MultiGridChunkManager* chunkManager,
                const std::vector<VoxelUpdate>& updates,
                std::vector<ChunkCoord>& modifiedChunks
        ) {
            TRACY_CPU_ZONE("VoxelCarver::applyVoxelUpdates");
            if (!chunkManager) return;

            // Use your existing ChunkCoordHash (same one used elsewhere in Game.cpp)
            robin_hood::unordered_flat_map<ChunkCoord, bool, ChunkCoordHash> chunkDirty;
            chunkDirty.reserve(updates.size() / 64 + 16);

            for (const auto& update : updates) {
                Chunk* chunk = chunkManager->getChunk(update.coord);
                if (!chunk) continue;

                Voxel& voxel = chunk->voxels[update.localPos.x][update.localPos.y][update.localPos.z];
                voxel.density = update.newDensity;
                voxel.type = update.newType;
                voxel.color = update.color;
                voxel.material = update.material;

                auto it = chunkDirty.find(update.coord);
                if (it == chunkDirty.end()) {
                    chunkDirty[update.coord] = true;

                    chunk->meshDirty = true;
                    chunk->lightingDirty = true;

                    modifiedChunks.push_back(update.coord);
                }
            }
        }

        static Chunk* ensureChunkExists(
                MultiGridChunkManager* chunkManager,
                const ChunkCoord& coord,
                bool autoCreate
        ) {
            Chunk* chunk = chunkManager->getChunk(coord);
            if (!chunk && autoCreate) {
                chunkManager->addChunk(coord, VoxelCategory::DYNAMIC);
                chunk = chunkManager->getChunk(coord);
                if (chunk) {
                    chunk->coord = coord;
                    chunk->clear();
                }
            }
            return chunk;
        }
    };

} // namespace gl3