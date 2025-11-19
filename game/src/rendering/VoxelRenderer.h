#pragma once
#include "glm/glm.hpp"

namespace gl3 {

    struct Voxel {
        uint8_t type = 0;
        glm::vec3 color = glm::vec3(1.0f);
        float emission=0.0f;
        bool isSolid() const { return type != 0; }
    };

    constexpr int CHUNK_SIZE = 7;

    struct Chunk {
        Voxel voxels[CHUNK_SIZE][CHUNK_SIZE][CHUNK_SIZE];
    };

}
