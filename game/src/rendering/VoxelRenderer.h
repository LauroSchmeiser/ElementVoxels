#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "glad/glad.h"

namespace gl3 {

    struct Voxel {
        uint8_t type = 0;
        float density=1.0f;
        glm::vec3 color = glm::vec3(1.0f);
        bool isSolid() const { return type != 0; }
        //0==Empty, 1==base, 2==fire, 3==fluid,
    };

    struct VoxelLight {
        glm::vec3 pos;
        float intensity;
        glm::vec3 color;
        uint32_t id;
    };


    struct EmissiveBlob {
        glm::vec3 sumPos;
        int count=0;
        glm::vec3 sumColor;
    };


    struct OutVertex {
        glm::vec4 pos;    // xyz: position, w unused
        glm::vec4 normal; // xyz: normal, w unused
        glm::vec4 color;  // rgb: color, a unused
    };


    constexpr int CHUNK_SIZE =12;


    struct Chunk {
        Voxel voxels[CHUNK_SIZE + 1][CHUNK_SIZE + 1][CHUNK_SIZE + 1];
        GLuint vao = 0;
        GLuint vbo = 0;
        uint32_t vertexCount = 0;

        bool meshDirty = true;
        bool lightingDirty = true;

        std::vector<VoxelLight> emissiveLights;
    };

    struct ChunkCoord {
        int x, y, z;

        bool operator==(const ChunkCoord& other) const {
            return x == other.x &&
                   y == other.y &&
                   z == other.z;
        }
    };

    struct ChunkCoordHash {
        std::size_t operator()(const ChunkCoord& c) const noexcept {
            std::size_t h1 = std::hash<int>{}(c.x);
            std::size_t h2 = std::hash<int>{}(c.y);
            std::size_t h3 = std::hash<int>{}(c.z);

            // Good hash mixing
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };


}
