#pragma once

#include <cstdint>
#include <vector>
#include "glm/glm.hpp"
#include <glad/glad.h>

namespace gl3 {
    constexpr int CHUNK_SIZE = 16;
    constexpr float VOXEL_SIZE = 3.0f;
    static constexpr int WORLD_RADIUS_CHUNKS = 5;

    struct Voxel {
        uint8_t type = 0;
        uint32_t material = 0;
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

    struct ChunkCoordHash {
        std::size_t operator()(const ChunkCoord &c) const noexcept {
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

    struct PhysicsMeshData {
        GLuint vao = 0;
        GLuint vbo = 0;
        uint32_t vertexCount = 0;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> colors;
        bool isValid = false;
    };

}