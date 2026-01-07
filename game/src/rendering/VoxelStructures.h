#pragma once

#include <vector>
#include "glm/glm.hpp"

namespace gl3 {
    constexpr int CHUNK_SIZE = 12;

    struct Voxel {
        uint8_t type = 0;
        float density = 1.0f;
        glm::vec3 color = glm::vec3(1.0f);

        bool isSolid() const { return type != 0; }
        // 0==Empty, 1==base, 2==fire, 3==fluid
    };

    struct VoxelLight {
        glm::vec3 pos;
        float intensity;
        glm::vec3 color;
        uint32_t id;
    };

    struct EmissiveBlob {
        glm::vec3 sumPos;
        int count = 0;
        glm::vec3 sumColor;
    };

    struct OutVertex {
        glm::vec4 pos;    // xyz: position, w unused
        glm::vec4 normal; // xyz: normal, w unused
        glm::vec4 color;  // rgb: color, a unused
    };

    struct ChunkCoord {
        int x, y, z;

        bool operator==(const ChunkCoord &other) const {
            return x == other.x &&
                   y == other.y &&
                   z == other.z;
        }
    };

    // Better hash function for ChunkCoord
    struct ChunkCoordHash {
        // Using murmurhash-like mixing
        std::size_t operator()(const ChunkCoord &c) const noexcept {
            // FNV-1a hash
            constexpr uint64_t offset_basis = 0xcbf29ce484222325;
            constexpr uint64_t prime = 0x100000001b3;

            uint64_t hash = offset_basis;

            hash ^= static_cast<uint64_t>(c.x);
            hash *= prime;
            hash ^= static_cast<uint64_t>(c.y);
            hash *= prime;
            hash ^= static_cast<uint64_t>(c.z);
            hash *= prime;

            return static_cast<std::size_t>(hash);
        }
    };

    // Forward declare Chunk - full definition will be in Chunk.h
    struct Chunk;
}