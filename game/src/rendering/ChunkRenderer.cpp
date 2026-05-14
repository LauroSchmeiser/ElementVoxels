#include "ChunkRenderer.h"
#include "FixedGridChunkManager.h"

namespace gl3 {
    ChunkRenderer::ChunkRenderer(FixedGridChunkManager* chunkMgr) {
        chunkManager=chunkMgr;
    }

    void ChunkRenderer::initialize() {
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");
        setupSSBOsAndTables();
        MAX_CHUNKS_GPU = (int)chunkManager->maxChunksGpu();
        setupChunkBatchBuffers(MAX_CHUNKS_GPU);
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
            std::cout << "BAD SLOT: " << chunk->gpuSlot << " MAX=" << MAX_CHUNKS_GPU << "\n";
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
                    voxels[idx].pad0 = voxels[idx].pad1 = voxels[idx].pad2 = 0.0f;
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

        const int MAX_VERTS_PER_CHUNK = 8000;
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
}