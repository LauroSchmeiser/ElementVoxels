#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include "rendering/VoxelStructures.h"
#include "rendering/MultiGridChunkManager.h"
#include "rendering/Chunk.h"

namespace gl3 {

    struct FastCraterCarver {
        static float squaredDistanceToAABB(const glm::vec3& p, const glm::vec3& bmin, const glm::vec3& bmax) {
            float dx = 0.0f;
            if      (p.x < bmin.x) dx = bmin.x - p.x;
            else if (p.x > bmax.x) dx = p.x - bmax.x;

            float dy = 0.0f;
            if      (p.y < bmin.y) dy = bmin.y - p.y;
            else if (p.y > bmax.y) dy = p.y - bmax.y;

            float dz = 0.0f;
            if      (p.z < bmin.z) dz = bmin.z - p.z;
            else if (p.z > bmax.z) dz = p.z - bmax.z;

            return dx*dx + dy*dy + dz*dz;
        }

        static int worldToChunk(float worldPos) {
            float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
            return (int)std::floor(worldPos / chunkWorldSize);
        }

        static glm::vec3 chunkMinWorld(const ChunkCoord& c) {
            return glm::vec3(c.x * CHUNK_SIZE * VOXEL_SIZE,
                             c.y * CHUNK_SIZE * VOXEL_SIZE,
                             c.z * CHUNK_SIZE * VOXEL_SIZE);
        }

        static void carveCrater(
                MultiGridChunkManager* mgr,
                const glm::vec3& center,
                float radius,
                float maxDepth,
                float densityThreshold,
                std::vector<ChunkCoord>* outTouchedChunks = nullptr,
                bool autoCreateChunks = false
        ) {
            if (!mgr) return;
            if (radius <= 0.0f || maxDepth <= 0.0f) return;

            const float radiusSq = radius * radius;

            const int minCX = worldToChunk(center.x - radius);
            const int maxCX = worldToChunk(center.x + radius);
            const int minCY = worldToChunk(center.y - radius);
            const int maxCY = worldToChunk(center.y + radius);
            const int minCZ = worldToChunk(center.z - radius);
            const int maxCZ = worldToChunk(center.z + radius);

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord cc{cx,cy,cz};

                        Chunk* chunk = mgr->getChunk(cc);
                        if (!chunk && autoCreateChunks) {
                            mgr->addChunk(cc, VoxelCategory::DYNAMIC);
                            chunk = mgr->getChunk(cc);
                            if (chunk) { chunk->coord = cc; chunk->clear(); }
                        }
                        if (!chunk) continue;

                        const glm::vec3 cmin = chunkMinWorld(cc);
                        const glm::vec3 cmax = cmin + glm::vec3(CHUNK_SIZE * VOXEL_SIZE);

                        if (squaredDistanceToAABB(center, cmin, cmax) > radiusSq) {
                            continue;
                        }

                        glm::vec3 localCenterF = (center - cmin) / VOXEL_SIZE;
                        float rVox = radius / VOXEL_SIZE;

                        int minVx = std::max(0, (int)std::floor(localCenterF.x - rVox));
                        int maxVx = std::min(CHUNK_SIZE, (int)std::ceil (localCenterF.x + rVox));
                        int minVy = std::max(0, (int)std::floor(localCenterF.y - rVox));
                        int maxVy = std::min(CHUNK_SIZE, (int)std::ceil (localCenterF.y + rVox));
                        int minVz = std::max(0, (int)std::floor(localCenterF.z - rVox));
                        int maxVz = std::min(CHUNK_SIZE, (int)std::ceil (localCenterF.z + rVox));

                        bool touched = false;

                        for (int vx = minVx; vx <= maxVx; ++vx) {
                            float wx = cmin.x + vx * VOXEL_SIZE;
                            float dx = wx - center.x;
                            float dxSq = dx*dx;

                            for (int vy = minVy; vy <= maxVy; ++vy) {
                                float wy = cmin.y + vy * VOXEL_SIZE;
                                float dy = wy - center.y;
                                float dySq = dy*dy;

                                if (dxSq + dySq > radiusSq) continue;

                                for (int vz = minVz; vz <= maxVz; ++vz) {
                                    float wz = cmin.z + vz * VOXEL_SIZE;
                                    float dz = wz - center.z;

                                    float distSq = dxSq + dySq + dz*dz;
                                    if (distSq > radiusSq) continue;

                                    Voxel& v = chunk->voxels[vx][vy][vz];
                                    if (v.density < densityThreshold) continue;

                                    float dist = std::sqrt(distSq);
                                    float t = dist / radius;
                                    float craterShape = (1.0f - t*t);
                                    float densityReduction = maxDepth * craterShape;

                                    float newDensity = v.density - densityReduction;

                                    if (v.density > 2.0f) newDensity = std::max(newDensity, 0.1f);

                                    v.density = newDensity;
                                    v.type = (newDensity < densityThreshold) ? 0 : 1;

                                    touched = true;
                                }
                            }
                        }

                        if (touched) {
                            chunk->meshDirty = true;
                            chunk->lightingDirty = true;
                            if (outTouchedChunks) outTouchedChunks->push_back(cc);
                        }
                    }
                }
            }
        }
    };

}