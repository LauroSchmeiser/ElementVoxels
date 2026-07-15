#include <array>
#include "ChunkRenderer.h"
#include "FixedGridChunkManager.h"
#include "SunBillboard.h"

namespace gl3 {
    ChunkRenderer::ChunkRenderer(FixedGridChunkManager* chunkMgr) {
        chunkManager=chunkMgr;
    }

    void ChunkRenderer::initialize() {
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");
        setupSSBOsAndTables();
        //MAX_CHUNKS_GPU = (int)chunkManager->maxChunksGpu();
        setupLightSSBOs();
        setupChunkBatchBuffers(MAX_CHUNKS_GPU);
        fluidMarchingCubesShader = std::make_unique<Shader>("shaders/fluid_marching_cubes.comp");
        setupFluidBatchBuffers(MAX_CHUNKS_GPU);
    }

    void ChunkRenderer::setupSSBOsAndTables() {
        // Prepare SSBOs and static tables

        // 0: voxels SSBO
        glGenBuffers(1, &ssboVoxels);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxelCount * sizeof(CpuVoxelStd430), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels); // bind to 0

        // 1: edge table SSBO
        glGenBuffers(1, &ssboEdgeTable);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboEdgeTable);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(edgeTableCPU), edgeTableCPU, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);

        // 2: tri table SSBO
        glGenBuffers(1, &ssboTriTable);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriTable);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(triTableCPU), triTableCPU, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);

        // 3: atomic counter (SSBO containing uint vertexCounter)
        glGenBuffers(1, &ssboCounter);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        unsigned int zero = 0;
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int), &zero, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);

        /*  // 4: triangles SSBO (output)
          glGenBuffers(1, &ssboTriangles);
          glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriangles);
          glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(OutVertex), nullptr, GL_DYNAMIC_DRAW);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);*/

        //5 particle ssbo
        //  glGenBuffers(1, &particleSSBO);

        //6 fieldbits ssbo
        //   glGenBuffers(1, &fieldBitsSSBO);

    }


    void ChunkRenderer::generateChunkMesh(Chunk* chunk)
    {
        if (ssboVoxels == 0 || ssboCounter == 0 || chunkIndirectBuffer == 0 || globalChunkVertexBuffer == 0) {
            std::cout << "ChunkRenderer not initialized properly\n";
            return;
        }

        if (!chunk) return;

        if (chunk->gpuSlot >= (uint32_t)MAX_CHUNKS_GPU) {
            std::cout << "THis is a BAD SLOT: " << chunk->gpuSlot << " MAX=" << MAX_CHUNKS_GPU << "\n";
            return;
        }

        if (chunk->isCleared) {
            DrawArraysIndirectCommand cmd{};
            cmd.count = 0;
            chunk->gpuCache.vertexCount = 0;
            cmd.instanceCount = 1;
            cmd.first = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
            cmd.baseInstance = chunk->gpuSlot;

            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
            glBufferSubData(GL_DRAW_INDIRECT_BUFFER,
                            chunk->gpuSlot * sizeof(DrawArraysIndirectCommand),
                            sizeof(DrawArraysIndirectCommand),
                            &cmd);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

            chunk->gpuCache.isValid = true;
            chunk->meshDirty = false;
            return;
        }

        // TODO: optional CPU early-out: if no solid, also set cmd.count=0 as above and return.

        glm::vec3 chunkOrigin(
                chunk->coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
        );

        marchingCubesShader->use();

        uploadVoxelChunk(*chunk, nullptr);

        resetAtomicCounter();
        setComputeUniforms(chunkOrigin, *marchingCubesShader);

        marchingCubesShader->setUInt("uChunkSlot", chunk->gpuSlot);
        marchingCubesShader->setUInt("uChunkMaxVerts", (uint32_t)CHUNK_MAX_VERTS);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, globalChunkVertexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, chunkIndirectBuffer);

        int cellsPerAxis = DIM - 1;
        int groups = (cellsPerAxis + 7) / 8;
        glDispatchCompute(groups, groups, groups);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
                        GL_COMMAND_BARRIER_BIT);

        // readback vertexCounter (4 bytes)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        uint32_t produced = 0;
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &produced);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        DrawArraysIndirectCommand cmd{};
        cmd.count = produced;
        chunk->gpuCache.vertexCount = produced;
        cmd.instanceCount = 1;
        cmd.first = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
        cmd.baseInstance = chunk->gpuSlot;

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
        glBufferSubData(GL_DRAW_INDIRECT_BUFFER,
                        chunk->gpuSlot * sizeof(DrawArraysIndirectCommand),
                        sizeof(DrawArraysIndirectCommand),
                        &cmd);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        chunk->gpuCache.isValid = true;
        chunk->meshDirty = false;
    }

    bool ChunkRenderer::tryResolveChunkVertexCount(Chunk* chunk)
    {
        if (!chunk->gpuCache.hasPendingCount || !chunk->gpuCache.counterFence)
            return false;

        // Non-blocking poll:
        GLenum res = glClientWaitSync(chunk->gpuCache.counterFence, 0, 0);
        if (res == GL_TIMEOUT_EXPIRED)
            return false;

        // Fence signaled (or already signaled). Read the 4 bytes.
        glDeleteSync(chunk->gpuCache.counterFence);
        chunk->gpuCache.counterFence = 0;
        chunk->gpuCache.hasPendingCount = false;

        glBindBuffer(GL_COPY_READ_BUFFER, chunk->gpuCache.counterReadbackBuffer);
        void* ptr = glMapBufferRange(GL_COPY_READ_BUFFER, 0, sizeof(uint32_t), GL_MAP_READ_BIT);
        if (ptr) {
            chunk->gpuCache.vertexCount = *reinterpret_cast<uint32_t*>(ptr);
            glUnmapBuffer(GL_COPY_READ_BUFFER);
        }
        glBindBuffer(GL_COPY_READ_BUFFER, 0);

        return true;
    }


    void ChunkRenderer::uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor) {
        if (!chunkManager) {
            std::cout << "chunkManager is null in ChunkRenderer\n";
            return;
        }
        const int localDIM = DIM; // Should be CHUNK_SIZE + 2 for padding
        const size_t total = size_t(localDIM) * localDIM * localDIM;
        std::vector<CpuVoxelStd430> voxels;
        voxels.resize(total);

        for (int x = -1; x <= CHUNK_SIZE; ++x) {
            for (int y = -1; y <= CHUNK_SIZE; ++y) {
                for (int z = -1; z <= CHUNK_SIZE; ++z) {
                    int idxX = x + 1;
                    int idxY = y + 1;
                    int idxZ = z + 1;
                    int idx = idxX + idxY * localDIM + idxZ * localDIM * localDIM;

                    const Voxel *srcVoxel = nullptr;

                    // Check if we need to sample from neighbor
                    if (x == -1 || x == CHUNK_SIZE ||
                        y == -1 || y == CHUNK_SIZE ||
                        z == -1 || z == CHUNK_SIZE) {

                        // Get neighbor chunk
                        ChunkCoord neighborCoord = chunk.coord;
                        int localX = x;
                        int localY = y;
                        int localZ = z;

                        // Adjust for each axis
                        if (x == -1) {
                            neighborCoord.x -= 1;
                            localX = CHUNK_SIZE - 1;
                        } else if (x == CHUNK_SIZE) {
                            neighborCoord.x += 1;
                            localX = 0;
                        }

                        if (y == -1) {
                            neighborCoord.y -= 1;
                            localY = CHUNK_SIZE - 1;
                        } else if (y == CHUNK_SIZE) {
                            neighborCoord.y += 1;
                            localY = 0;
                        }

                        if (z == -1) {
                            neighborCoord.z -= 1;
                            localZ = CHUNK_SIZE - 1;
                        } else if (z == CHUNK_SIZE) {
                            neighborCoord.z += 1;
                            localZ = 0;
                        }

                        Chunk *neighbor = chunkManager->getChunk(neighborCoord);
                        if (neighbor &&
                            localX >= 0 && localX <= CHUNK_SIZE &&
                            localY >= 0 && localY <= CHUNK_SIZE &&
                            localZ >= 0 && localZ <= CHUNK_SIZE) {
                            srcVoxel = &neighbor->voxels[localX][localY][localZ];
                        }
                    }

                    // If no neighbor data, use current chunk or default
                    if (!srcVoxel) {
                        // Clamp to valid range for current chunk
                        int clampX = glm::clamp(x, 0, CHUNK_SIZE);
                        int clampY = glm::clamp(y, 0, CHUNK_SIZE);
                        int clampZ = glm::clamp(z, 0, CHUNK_SIZE);
                        srcVoxel = &chunk.voxels[clampX][clampY][clampZ];
                    }

                    // Copy data
                    voxels[idx].density = srcVoxel->density;
                    voxels[idx].fluidDensity = srcVoxel->fluidDensity;
                    glm::vec3 col = overrideColor ? *overrideColor : srcVoxel->color;

                    voxels[idx].color[0] = col.r;
                    voxels[idx].color[1] = col.g;
                    voxels[idx].color[2] = col.b;
                    voxels[idx].color[3] = 1.0f;
                    voxels[idx].type = srcVoxel->type;
                    voxels[idx].material=srcVoxel->material;
                }
            }
        }

        // Upload to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxels.size() * sizeof(CpuVoxelStd430), voxels.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void ChunkRenderer::resetAtomicCounter() {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);
    }

    void ChunkRenderer::setComputeUniforms(const glm::vec3& chunkOrigin, Shader& computeShader) {
        computeShader.use();
        computeShader.setFloat("voxelSize", gl3::VOXEL_SIZE);
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));

        // IMPORTANT: padded voxel index (0,0,0) corresponds to world (chunkOrigin - voxelSize)
        computeShader.setVec3("gridOrigin", chunkOrigin - glm::vec3(gl3::VOXEL_SIZE));
    }
    void ChunkRenderer::setupChunkBatchBuffers(int maxChunksGpu)
    {
        MAX_CHUNKS_GPU = maxChunksGpu;
        CHUNK_MAX_VERTS = (DIM - 1) * (DIM - 1) * (DIM - 1) * 5 * 3;

        const int MAX_VERTS_PER_CHUNK = 10000;
        CHUNK_MAX_VERTS = std::min(
                (int)chunkMaxVertices(DIM),
                MAX_VERTS_PER_CHUNK
        );

        // Global vertex buffer
        glGenBuffers(1, &globalChunkVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, globalChunkVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
                     MAX_CHUNKS_GPU * CHUNK_MAX_VERTS * sizeof(OutVertexStd430),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Indirect command buffer
        glGenBuffers(1, &chunkIndirectBuffer);

        std::vector<DrawArraysIndirectCommand> cmds(MAX_CHUNKS_GPU);
        for (uint32_t s = 0; s < (uint32_t)MAX_CHUNKS_GPU; ++s) {
            cmds[s].count = 0;
            cmds[s].instanceCount = 1;
            cmds[s].first = s * (uint32_t)CHUNK_MAX_VERTS;
            cmds[s].baseInstance = s;
        }

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
        glBufferData(GL_DRAW_INDIRECT_BUFFER,
                     cmds.size() * sizeof(DrawArraysIndirectCommand),
                     cmds.data(),
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        // VAO setup
        glGenVertexArrays(1, &globalChunkVAO);
        glBindVertexArray(globalChunkVAO);

        glBindBuffer(GL_ARRAY_BUFFER, globalChunkVertexBuffer);

        constexpr GLsizei stride = sizeof(OutVertexStd430);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, color));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, uv));
        glEnableVertexAttribArray(4);
        glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, stride, (void*)offsetof(OutVertexStd430, flags));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void ChunkRenderer::buildAndUploadChunkLightIndexBuffer(int camCX, int camCY, int camCZ, int renderRadius)
    {
        static int lastUpdateFrame = -1;
        static int lastCamCX = camCX, lastCamCY = camCY, lastCamCZ = camCZ;
        static int lastRenderRadius = renderRadius;

        const int UPDATE_INTERVAL = 213;
        const int CAM_MOVE_THRESHOLD = CHUNK_SIZE;

        bool needsUpdate = false;

        if ((std::abs(camCX - lastCamCX) >= CAM_MOVE_THRESHOLD ||
             std::abs(camCY - lastCamCY) >= CAM_MOVE_THRESHOLD ||
             std::abs(camCZ - lastCamCZ) >= CAM_MOVE_THRESHOLD ||
             renderRadius != lastRenderRadius)||(frameCounter - lastUpdateFrame >= UPDATE_INTERVAL)) {
            needsUpdate = true;
        }

        if (!needsUpdate) {
            return; // Skip update this frame
        }

        lastUpdateFrame = frameCounter;
        lastCamCX = camCX;
        lastCamCY = camCY;
        lastCamCZ = camCZ;
        lastRenderRadius = renderRadius;

        std::vector<ChunkLightIndexGpu> chunkIdx(MAX_CHUNKS_GPU);
        for (auto& e : chunkIdx) {
            e.count = 0;
            for (int i = 0; i < 4; ++i) e.indices[i] = 0;
        }

        for (int cx = camCX - renderRadius; cx <= camCX + renderRadius; ++cx) {
            for (int cy = camCY - renderRadius; cy <= camCY + renderRadius; ++cy) {
                for (int cz = camCZ - renderRadius; cz <= camCZ + renderRadius; ++cz) {
                    Chunk* chunk = chunkManager->getChunk({cx,cy,cz});
                    if (!chunk) continue;
                    if (!chunk->gpuCache.isValid) continue;

                    if (frameCounter - chunk->gpuCache.lastLightUpdateFrame > LIGHT_UPDATE_INTERVAL ||
                        chunk->gpuCache.nearbyLights.empty()) {
                        updateChunkLights(chunk);
                    }

                    auto& dst = chunkIdx[chunk->gpuSlot];
                    int num = std::min((int)chunk->gpuCache.nearbyLights.size(), MAX_LIGHTS);
                    dst.count = (uint32_t)num;

                    for (int i = 0; i < num; ++i) {
                        const VoxelLight* L = chunk->gpuCache.nearbyLights[i];
                        dst.indices[i] = lightIndexFromPtr(L);
                    }
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboChunkLightIdx);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, chunkIdx.size() * sizeof(ChunkLightIndexGpu), chunkIdx.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    inline uint32_t ChunkRenderer::lightIndexFromPtr(const VoxelLight* ptr) const {
        const VoxelLight* base = mergedEmissiveLightPool.data();
        return (uint32_t)(ptr - base);
    }

    void ChunkRenderer::updateChunkLights(Chunk *chunk) {
        chunk->gpuCache.nearbyLights.clear();
        chunk->gpuCache.nearbyLights.reserve(MAX_LIGHTS);

        /*glm::vec3 chunkOrigin(
                chunk->coord.x * DIM,
                chunk->coord.y * DIM,
                chunk->coord.z * DIM
        );
        glm::vec3 chunkCenter = chunkOrigin + glm::vec3(DIM * 0.5f);*/

        glm::vec3 chunkOrigin(
                chunk->coord.x * CHUNK_SIZE * VOXEL_SIZE,
                chunk->coord.y * CHUNK_SIZE * VOXEL_SIZE,
                chunk->coord.z * CHUNK_SIZE * VOXEL_SIZE
        );
        glm::vec3 chunkCenter = chunkOrigin + glm::vec3(CHUNK_SIZE * VOXEL_SIZE * 0.5f);

        // Fast path
        if (flatEmissiveLightList.empty()) {
            chunk->gpuCache.lastLightUpdateFrame = frameCounter;
            return;
        }

        const float radiusSq = LIGHT_RADIUS_SQ;
        const int K = MAX_LIGHTS;

        // small stack arrays (K is tiny)
        std::array<const VoxelLight *, 8> bestPtrs{};   // pointer candidates
        std::array<float, 8> bestScore{};              // score = intensity / (distSq + 1)
        int bestCount = 0;

        // keep track of the current worst score index (min score)
        float worstScore = std::numeric_limits<float>::infinity();
        int worstIndex = -1;

        for (const VoxelLight *light: flatEmissiveLightList) {
            glm::vec3 d = light->pos - chunkCenter;
            float distSq = glm::dot(d, d);

            if (distSq > radiusSq) continue; // skip out-of-range lights

            // Score uses shader-like falloff (intensity divided by squared distance + 1)
            // +1 avoids division by zero and keeps near-zero distance finite
            float score = light->intensity / (distSq + 1.0f);

            if (bestCount < K) {
                // append
                bestPtrs[bestCount] = light;
                bestScore[bestCount] = score;
                ++bestCount;

                // find new worst
                worstScore = bestScore[0];
                worstIndex = 0;
                for (int i = 1; i < bestCount; ++i) {
                    if (bestScore[i] < worstScore) {
                        worstScore = bestScore[i];
                        worstIndex = i;
                    }
                }
            } else {
                // replace worst if this one has a higher score
                if (score > worstScore) {
                    bestPtrs[worstIndex] = light;
                    bestScore[worstIndex] = score;

                    // recompute worst (small K)
                    worstScore = bestScore[0];
                    worstIndex = 0;
                    for (int i = 1; i < K; ++i) {
                        if (bestScore[i] < worstScore) {
                            worstScore = bestScore[i];
                            worstIndex = i;
                        }
                    }
                }
            }
        }

        // Move found lights into chunk->gpuCache.nearbyLights sorted by descending score
        if (bestCount > 0) {
            std::vector<int> idx(bestCount);
            for (int i = 0; i < bestCount; ++i) idx[i] = i;
            // sort so highest score first
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return bestScore[a] > bestScore[b];
            });

            for (int i = 0; i < bestCount; ++i) {
                chunk->gpuCache.nearbyLights.push_back(const_cast<VoxelLight *>(bestPtrs[idx[i]]));
            }
        }

        chunk->gpuCache.lastLightUpdateFrame = frameCounter;
    }

    void ChunkRenderer::updateLightSpatialHash() {
        lightSpatialHash.clear();
        flatEmissiveLightList.clear();
        mergedEmissiveLightPool.clear();

        // 1) Gather raw pointers (lights stored inside chunks) and fill spatial-hash as before
        std::vector<const VoxelLight *> rawLights;
        chunkManager->forEachEmissiveChunk([this, &rawLights](Chunk *chunk) {
            for (auto &light: chunk->emissiveLights) {
                // coarse bucket size (same as before)
                ChunkCoord gridCell{
                        (int) std::floor(light.pos.x / (DIM * 2)),
                        (int) std::floor(light.pos.y / (DIM * 2)),
                        (int) std::floor(light.pos.z / (DIM * 2))
                };
                lightSpatialHash[gridCell].push_back(&light);
                rawLights.push_back(&light);
            }
        });

        // 2) If no lights, done
        if (rawLights.empty()) {
            //std::cout << "Light spatial hash updated: 0 grid cells; 0 emissive blobs\n";
            return;
        }

        // 3) Simple greedy clustering (merge lights that are spatially close)
        // Tune this merge radius. Using CHUNK_SIZE * 1.5 means lights that spill over
        // into adjacent chunks are folded into a single logical emitter.
        const float MERGE_RADIUS = DIM * VOXEL_SIZE* 1.5f;
        const float MERGE_RADIUS_SQ = MERGE_RADIUS * MERGE_RADIUS;

        std::vector<char> used(rawLights.size(), 0);
        for (size_t i = 0; i < rawLights.size(); ++i) {
            if (used[i]) continue;
            used[i] = 1;

            // accumulate weighted by intensity (so stronger lights dominate)
            float totalIntensity = 0.0f;
            glm::vec3 accumPos(0.0f);
            glm::vec3 accumColor(0.0f);
            uint32_t mergedId = rawLights[i]->id; // base id
            int amountMerged = 0;

            // merge any other lights that lie within MERGE_RADIUS of rawLights[i]
            for (size_t j = i; j < rawLights.size(); ++j) {
                if (used[j]) continue;
                float d2 = glm::dot(rawLights[i]->pos - rawLights[j]->pos, rawLights[i]->pos - rawLights[j]->pos);
                if (d2 <= MERGE_RADIUS_SQ) {
                    used[j] = 1;
                    const VoxelLight *L = rawLights[j];
                    float w = glm::max(1.0f, L->intensity); // weight (avoid zero)
                    accumPos += L->pos * w;
                    accumColor += L->color * w;
                    totalIntensity += L->intensity;
                    amountMerged++;
                }
            }

            // construct merged light (fallbacks)
            VoxelLight merged;
            if (totalIntensity > 0.0f) {
                merged.intensity = (totalIntensity / amountMerged);
                merged.pos = accumPos / (totalIntensity > 0.0f ? totalIntensity : 1.0f);
                merged.color = accumColor / (totalIntensity > 0.0f ? totalIntensity : 1.0f);
            } else {
                // fallback: single entry (should not typically occur)
                merged = *rawLights[i];
            }
            merged.id = mergedId;

            // store into pool and push pointer into flat list
            mergedEmissiveLightPool.push_back(merged);
        }
        //std::cout<<"mergedEmissiveLightPool: "<<mergedEmissiveLightPool.size()<<"\n";

        // 4) Build final flat list of pointers into mergedEmissiveLightPool
        flatEmissiveLightList.reserve(mergedEmissiveLightPool.size());
        for (auto &m: mergedEmissiveLightPool) {
            flatEmissiveLightList.push_back(&m);
        }

        /*std::cout << "Light spatial hash updated: " << lightSpatialHash.size()
                  << " grid cells; raw=" << rawLights.size()
                  << " merged=" << mergedEmissiveLightPool.size() << " emissive blobs\n";*/
    }
    void ChunkRenderer::uploadMergedLightsToGPU()
    {
        std::vector<VoxelLightGpu> gpu;
        gpu.resize(mergedEmissiveLightPool.size());

        for (size_t i = 0; i < mergedEmissiveLightPool.size(); ++i) {
            const auto& L = mergedEmissiveLightPool[i];
            gpu[i].posIntensity = glm::vec4(L.pos, L.intensity);
            gpu[i].color        = glm::vec4(L.color, 0.0f);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gpu.size() * sizeof(VoxelLightGpu), gpu.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void ChunkRenderer::collectEmissiveBillboards(std::vector<SunInstance>& out,
                                                  robin_hood::unordered_set<uint32_t>& usedIds, Chunk* chunk)
    {
        if (!chunk) return;
        if (chunk->isCleared) return;
        for (const auto& light : chunk->emissiveLights) {
            if (usedIds.insert(light.id).second) {
                SunInstance inst;
                inst.position = light.pos;
                inst.scale = std::sqrt(light.intensity) * 0.5f;
                inst.color = light.color * 1.0f;
                out.push_back(inst);
            }
        }
    }
    void ChunkRenderer::collectMergedEmissiveBillboards(std::vector<SunInstance>& out)
    {
        out.reserve(out.size() + mergedEmissiveLightPool.size());
        for (const auto& light : mergedEmissiveLightPool)
        {
            SunInstance inst;

            inst.position = light.pos;
            inst.color    = light.color;

            // Whatever scaling you like
            inst.scale = std::sqrt(light.intensity) * 1.0f;

            out.push_back(inst);
        }
    }

    void ChunkRenderer::setupLightSSBOs()
    {
        glGenBuffers(1, &ssboLights);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4096 * sizeof(VoxelLightGpu), nullptr, GL_DYNAMIC_DRAW); // capacity
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glGenBuffers(1, &ssboChunkLightIdx);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboChunkLightIdx);
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_CHUNKS_GPU * sizeof(ChunkLightIndexGpu), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void ChunkRenderer::setupFluidBatchBuffers(int maxChunksGpu)
    {
        FLUID_CHUNK_MAX_VERTS = 10000;

        glGenBuffers(1, &globalFluidVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, globalFluidVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
                     MAX_CHUNKS_GPU * FLUID_CHUNK_MAX_VERTS * sizeof(OutVertexStd430),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glGenBuffers(1, &fluidIndirectBuffer);

        std::vector<DrawArraysIndirectCommand> cmds(MAX_CHUNKS_GPU);
        for (uint32_t s = 0; s < (uint32_t)MAX_CHUNKS_GPU; ++s) {
            cmds[s].count = 0;
            cmds[s].instanceCount = 1;
            cmds[s].first = s * (uint32_t)FLUID_CHUNK_MAX_VERTS;
            cmds[s].baseInstance = s;
        }

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, fluidIndirectBuffer);
        glBufferData(GL_DRAW_INDIRECT_BUFFER,
                     cmds.size() * sizeof(DrawArraysIndirectCommand),
                     cmds.data(),
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        glGenVertexArrays(1, &globalFluidVAO);
        glBindVertexArray(globalFluidVAO);
        glBindBuffer(GL_ARRAY_BUFFER, globalFluidVertexBuffer);

        constexpr GLsizei stride = sizeof(OutVertexStd430);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, color));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, uv));
        glEnableVertexAttribArray(4);
        glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, stride, (void*)offsetof(OutVertexStd430, flags));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void ChunkRenderer::generateFluidMesh(Chunk* chunk)
    {
        if (!chunk) return;

        fluidMarchingCubesShader->use();

        uploadVoxelChunk(*chunk, nullptr);
        resetAtomicCounter();
        glm::vec3 chunkOrigin(
                chunk->coord.x * CHUNK_SIZE * VOXEL_SIZE,
                chunk->coord.y * CHUNK_SIZE * VOXEL_SIZE,
                chunk->coord.z * CHUNK_SIZE * VOXEL_SIZE
        );
        setComputeUniforms(chunkOrigin, *fluidMarchingCubesShader);

        fluidMarchingCubesShader->setUInt("uChunkSlot", chunk->gpuSlot);
        fluidMarchingCubesShader->setUInt("uChunkMaxVerts", (uint32_t)FLUID_CHUNK_MAX_VERTS);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, globalFluidVertexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, fluidIndirectBuffer);

        int cellsPerAxis = DIM - 1;
        int groups = (cellsPerAxis + 7) / 8;
        glDispatchCompute(groups, groups, groups);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
                        GL_COMMAND_BARRIER_BIT);

        uint32_t produced = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &produced);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        DrawArraysIndirectCommand cmd{};
        cmd.count = produced;
        cmd.instanceCount = 1;
        cmd.first = chunk->gpuSlot * (uint32_t)FLUID_CHUNK_MAX_VERTS;
        cmd.baseInstance = chunk->gpuSlot;

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, fluidIndirectBuffer);
        glBufferSubData(GL_DRAW_INDIRECT_BUFFER,
                        chunk->gpuSlot * sizeof(DrawArraysIndirectCommand),
                        sizeof(DrawArraysIndirectCommand),
                        &cmd);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    }
}