#pragma once
#include "Chunk.h"
#include "VoxelStructures.h"
#include "marchingTables.h"
#include "GpuStructsStd430.h"
#include "Shader.h"
#include <iostream>


namespace gl3 {
    class FixedGridChunkManager;
    class ChunkRenderer {

        FixedGridChunkManager* chunkManager = nullptr;
        std::unique_ptr<Shader> marchingCubesShader;

        const int DIM = CHUNK_SIZE+2; //Chunk Size with a bit off padding for marching cubes
        size_t voxelCount = DIM * DIM * DIM; //How many voxels can be in one Chunk

        size_t maxVerts = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 5 * 3;
        GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0,
                ssboCounter = 0, ssboTriangles = 0, particleSSBO = 0, fieldBitsSSBO = 0;

        GLuint globalChunkVertexBuffer = 0;
        GLuint chunkIndirectBuffer = 0;
        GLuint globalChunkVAO = 0;

        size_t CHUNK_MAX_VERTS = 0;
        int MAX_CHUNKS_GPU = 0;

        GLuint ssboLights = 0;
        GLuint ssboChunkLightIdx = 0;

        void setupSSBOsAndTables();
        bool tryResolveChunkVertexCount(Chunk* chunk);
        void uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor);
        void resetAtomicCounter();
        void setComputeUniforms(const glm::vec3& chunkOrigin, Shader& computeShader);
        void setupChunkBatchBuffers(int maxChunksGpu);

    public:
        void generateChunkMesh(Chunk* chunk);

        ChunkRenderer(FixedGridChunkManager* chunkMgr);

        void initialize();
    };

}
