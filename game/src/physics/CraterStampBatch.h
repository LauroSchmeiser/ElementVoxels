#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>
#include "../rendering/VoxelStructures.h"
#include "../rendering/FixedGridChunkManager.h"

#include "../robin_hood.h"

namespace gl3 {

    class FixedGridChunkManager;

    struct CraterStampBatch {
        struct Stamp {
            glm::vec3 center;
            float radius;
            float depth;
        };

        struct CellKey {
            int x,y,z;
            bool operator==(const CellKey& o) const { return x==o.x && y==o.y && z==o.z; }
        };
        struct CellKeyHash {
            size_t operator()(const CellKey& k) const noexcept {
                return (size_t)((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
            }
        };

        static int worldToChunk(float worldPos) {
            const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
            return (int)std::floor(worldPos / chunkWorldSize);
        }

        static glm::vec3 chunkMinWorld(const ChunkCoord& c) {
            return glm::vec3(c.x * CHUNK_SIZE * VOXEL_SIZE,
                             c.y * CHUNK_SIZE * VOXEL_SIZE,
                             c.z * CHUNK_SIZE * VOXEL_SIZE);
        }

        static void apply(FixedGridChunkManager* mgr,
                          const std::vector<Stamp>& stamps,
                          float densityThreshold = -0.5f)
        {
            if (!mgr || stamps.empty()) return;

            float maxR = stamps[0].radius;
            for (auto& s : stamps) maxR = std::max(maxR, s.radius);
            const float cellSize = std::max(maxR, VOXEL_SIZE);

            robin_hood::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
            grid.reserve(stamps.size() * 2);

            auto cellOf = [&](const glm::vec3& p)->CellKey{
                return CellKey{
                        (int)std::floor(p.x / cellSize),
                        (int)std::floor(p.y / cellSize),
                        (int)std::floor(p.z / cellSize)
                };
            };

            for (int i = 0; i < (int)stamps.size(); ++i) {
                grid[cellOf(stamps[i].center)].push_back(i);
            }

            glm::vec3 minP = stamps[0].center - glm::vec3(stamps[0].radius);
            glm::vec3 maxP = stamps[0].center + glm::vec3(stamps[0].radius);
            for (auto& s : stamps) {
                minP = glm::min(minP, s.center - glm::vec3(s.radius));
                maxP = glm::max(maxP, s.center + glm::vec3(s.radius));
            }

            int minCX = worldToChunk(minP.x);
            int maxCX = worldToChunk(maxP.x);
            int minCY = worldToChunk(minP.y);
            int maxCY = worldToChunk(maxP.y);
            int minCZ = worldToChunk(minP.z);
            int maxCZ = worldToChunk(maxP.z);

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord cc{cx,cy,cz};
                        Chunk* chunk = mgr->getChunk(cc);
                        if (!chunk) continue;

                        glm::vec3 cmin = chunkMinWorld(cc);
                        bool touched = false;

                        for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                            float wx = cmin.x + vx * VOXEL_SIZE;

                            for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                                float wy = cmin.y + vy * VOXEL_SIZE;

                                for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                    float wz = cmin.z + vz * VOXEL_SIZE;
                                    glm::vec3 p(wx,wy,wz);

                                    Voxel& v = chunk->voxels[vx][vy][vz];
                                    if (v.density < densityThreshold) continue;

                                    CellKey ck = cellOf(p);
                                    float totalReduction = 0.0f;

                                    for (int ox = -1; ox <= 1; ++ox) {
                                        for (int oy = -1; oy <= 1; ++oy) {
                                            for (int oz = -1; oz <= 1; ++oz) {
                                                CellKey nk{ck.x+ox, ck.y+oy, ck.z+oz};
                                                auto it = grid.find(nk);
                                                if (it == grid.end()) continue;

                                                for (int stampIndex : it->second) {
                                                    const Stamp& s = stamps[stampIndex];
                                                    glm::vec3 d = p - s.center;
                                                    float distSq = glm::dot(d,d);
                                                    if (distSq > s.radius*s.radius) continue;

                                                    float dist = std::sqrt(distSq);
                                                    float t = dist / s.radius;
                                                    float craterShape = (1.0f - t*t);
                                                    totalReduction = std::max(totalReduction, s.depth * craterShape);
                                                }
                                            }
                                        }
                                    }

                                    if (totalReduction > 0.0f) {
                                        float newD = v.density - totalReduction;
                                        if (v.density > 2.0f) newD = std::max(newD, 0.1f);
                                        v.density = newD;
                                        v.type = (newD < densityThreshold) ? 0 : v.type;
                                        touched = true;
                                    }
                                }
                            }
                        }

                        if (touched) {
                            chunk->meshDirty = true;
                            chunk->lightingDirty = true;
                        }
                    }
                }
            }
        }
    };

}