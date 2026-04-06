#pragma once
#include <cstddef>
#include <cstdint>

struct alignas(16) CpuVoxelStd430 {
    float density;   // offset 0
    float pad0;      // offset 4
    float pad1;      // offset 8
    float pad2;      // offset 12
    float color[4];  // offset 16..31
    uint32_t type;
    uint32_t pad4, pad5, pad6;// pad to 16-byte multiple (4 bytes)
    uint32_t material;
    uint32_t pad7, pad8, pad9;// pad to 16-byte multiple (4 bytes)
};
static_assert(sizeof(CpuVoxelStd430) == 64, "CpuVoxelStd430 must be 64 bytes");
static_assert(alignof(CpuVoxelStd430) == 16, "CpuVoxelStd430 must be 16-byte aligned");

struct alignas(16) OutVertexStd430 {
    float pos[4];    // offset 0..15
    float normal[4]; // offset 16..31
    float color[4];  // offset 32..47
    float uv[2];      // 48..55
    float padUV[2];
    uint32_t flags[4];
};
static_assert(sizeof(OutVertexStd430) == 80, "OutVertexStd430 must be 80 bytes");
static_assert(alignof(OutVertexStd430) == 16, "OutVertexStd430 must be 16-byte aligned");

static_assert(offsetof(OutVertexStd430, pos) == 0, "pos offset must be 0");
static_assert(offsetof(OutVertexStd430, normal) == 16, "normal offset must be 16");
static_assert(offsetof(OutVertexStd430, color) == 32, "color offset must be 32");

struct DrawArraysIndirectCommand {
    uint32_t count;
    uint32_t instanceCount;
    uint32_t first;
    uint32_t baseInstance;
};


inline size_t chunkMaxVertices(int DIM) {
    const int cellsPerAxis = DIM - 1;
    return size_t(cellsPerAxis) * cellsPerAxis * cellsPerAxis * 5u * 3u;
}
struct alignas(16) VoxelLightGpu {
    glm::vec4 posIntensity; // xyz=pos, w=intensity
    glm::vec4 color;        // xyz=color, w=unused
};
static_assert(sizeof(VoxelLightGpu) == 32);

struct alignas(16) ChunkLightIndexGpu {
    uint32_t count;
    uint32_t indices[4];
    uint32_t pad[3];
};
static_assert(sizeof(ChunkLightIndexGpu) == 32);
