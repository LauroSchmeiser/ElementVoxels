#pragma once

#include <cstdint>
#include <vector>
#include "glm/glm.hpp"

namespace gl3 {
    constexpr int CHUNK_SIZE = 8;

    struct Voxel {
        uint8_t type = 0;
        uint64_t material=0;
        float density = 1.0f;
        glm::vec3 color = glm::vec3(1.0f);

        bool isSolid() const { return type != 0; }
        // 0==Empty, 1==base, 2==fire, 3==fluid
        // 0==stone, 1==earth, 2==ice etc....
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

        bool operator!=(const ChunkCoord &other) const {
            return !(*this == other);
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

    struct SpellEffect {
        enum class Type {
            GRAVITY_WELL,
            CONSTRUCT,
            TELEKINESIS
        };

        Type type;
        glm::vec3 center;
        float radius;
        float strength;
        bool geometryCreated = false;
        uint64_t targetMaterial;
        glm::vec3 formationColor;
        float formationRadius = 0.0f;

        // store stable IDs (not vector indices)
        std::vector<uint64_t> animatedVoxelIDs;
    };

    struct AnimatedVoxel {
        uint64_t id = 0; // unique stable id
        glm::vec3 currentPos;
        glm::vec3 targetPos;
        glm::vec3 velocity;
        glm::vec3 color;
        glm::vec3 originalVoxelPos;
        glm::vec3 normal;
        float animationSpeed = 3.0f;
        bool isAnimating = false;
        bool hasArrived = false; // track if arrived at target
    };

    struct SpellFormation {
        glm::vec3 center;
        float radius;
        glm::vec3 color;
        uint64_t material;

        // SDF function for this formation
        float evaluate(const glm::vec3& p) const {
            float dist = glm::distance(p, center);
            // Smooth sphere SDF - same as your planet generation
            return radius - dist; // Positive inside, negative outside
        }
    };

    // Forward declare Chunk - full definition will be in Chunk.h
    struct Chunk;
}