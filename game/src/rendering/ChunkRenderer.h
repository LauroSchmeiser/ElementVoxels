#pragma once
#include "Chunk.h"
#include "VoxelStructures.h"
#include "marchingTables.h"
#include "GpuStructsStd430.h"
#include "Shader.h"
#include "../../../extern/robin_hood.h"
#include "SunBillboard.h"
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

        size_t CHUNK_MAX_VERTS = 0;

        void setupSSBOsAndTables();
        bool tryResolveChunkVertexCount(Chunk* chunk);
        void uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor);
        void resetAtomicCounter();
        void setComputeUniforms(const glm::vec3& chunkOrigin, Shader& computeShader);
        void setupChunkBatchBuffers(int maxChunksGpu);

    public:
        int MAX_CHUNKS_GPU = 1550;

        void generateChunkMesh(Chunk* chunk);

        ChunkRenderer(FixedGridChunkManager* chunkMgr);

        void initialize();

        GLuint globalChunkVAO = 0;
        GLuint chunkIndirectBuffer = 0;
        const int LIGHT_UPDATE_INTERVAL=301;
        std::vector<gl3::VoxelLight> mergedEmissiveLightPool;
        void buildAndUploadChunkLightIndexBuffer(int camCX, int camCY, int camCZ, int renderRadius);


        void updateChunkLights(Chunk *chunk);

        uint32_t lightIndexFromPtr(const VoxelLight *ptr) const;
        const int MAX_LIGHTS = 4; // has to match marching cubes shader
        const float LIGHT_RADIUS = 300.0f * CHUNK_SIZE*VOXEL_SIZE*2;
        uint64_t frameCounter = 180;
        const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;
        std::vector<const gl3::VoxelLight *> flatEmissiveLightList;

        robin_hood::unordered_map<ChunkCoord, std::vector<VoxelLight *>, ChunkCoordHash> lightSpatialHash;


        void updateLightSpatialHash();

        void uploadMergedLightsToGPU();

        void collectEmissiveBillboards(std::vector<SunInstance>& out, robin_hood::unordered_set<uint32_t>& usedIds,Chunk* chunk);
        void collectMergedEmissiveBillboards(std::vector<SunInstance>& out);

        GLuint ssboLights = 0;
        GLuint ssboChunkLightIdx = 0;

        void setupLightSSBOs();

        void setupFluidBatchBuffers(int maxChunksGpu);
        void ChunkRenderer::generateFluidMesh(Chunk* chunk);
        size_t FLUID_CHUNK_MAX_VERTS=0;
        std::unique_ptr<Shader> fluidMarchingCubesShader;
        GLuint globalFluidVertexBuffer = 0;
        GLuint fluidIndirectBuffer = 0;
        GLuint globalFluidVAO = 0;

    };

}
