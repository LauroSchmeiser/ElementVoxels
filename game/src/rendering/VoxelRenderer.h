#pragma once
#include "glm/glm.hpp"

namespace gl3 {

    struct Voxel {
        uint8_t type = 0;
        float density=1.0f;
        glm::vec3 color = glm::vec3(1.0f);
        bool isSolid() const { return type != 0; }
    };

    struct OutVertex {
        glm::vec4 pos;    // xyz: position, w unused
        glm::vec4 normal; // xyz: normal, w unused
        glm::vec4 color;  // rgb: color, a unused
    };


    constexpr int CHUNK_SIZE =12;

    struct Chunk {
        Voxel voxels[CHUNK_SIZE][CHUNK_SIZE][CHUNK_SIZE];
    };

}
