#pragma once
#include <cstddef>
#include <cstdint>

// std430-friendly voxel struct: matches GLSL
// struct Voxel { float density; vec4 color; };
struct alignas(16) CpuVoxelStd430 {
    float density;   // offset 0
    float pad0;      // offset 4
    float pad1;      // offset 8
    float pad2;      // offset 12
    float color[4];  // offset 16..31
};
static_assert(sizeof(CpuVoxelStd430) == 32, "CpuVoxelStd430 must be 32 bytes");
static_assert(alignof(CpuVoxelStd430) == 16, "CpuVoxelStd430 must be 16-byte aligned");

// std430-friendly OutVertex struct: matches GLSL
// struct OutVertex { vec4 pos; vec4 normal; vec4 color; };
struct alignas(16) OutVertexStd430 {
    float pos[4];    // offset 0..15
    float normal[4]; // offset 16..31
    float color[4];  // offset 32..47
};
static_assert(sizeof(OutVertexStd430) == 48, "OutVertexStd430 must be 48 bytes");
static_assert(alignof(OutVertexStd430) == 16, "OutVertexStd430 must be 16-byte aligned");

// Offsets used by glVertexAttribPointer
static_assert(offsetof(OutVertexStd430, pos) == 0, "pos offset must be 0");
static_assert(offsetof(OutVertexStd430, normal) == 16, "normal offset must be 16");
static_assert(offsetof(OutVertexStd430, color) == 32, "color offset must be 32");