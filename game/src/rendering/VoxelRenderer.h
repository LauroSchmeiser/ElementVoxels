#pragma once
#include "glm/glm.hpp"

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
    };

    struct EmissiveBlob {
        glm::vec3 sumPos;
        int count = 0;
        glm::vec3 color;
    };

    struct OutVertex {
        glm::vec4 pos;    // xyz: position, w unused
        glm::vec4 normal; // xyz: normal, w unused
        glm::vec4 color;  // rgb: color, a unused
    };


    constexpr int CHUNK_SIZE =13;


    struct Chunk {
        Voxel voxels[CHUNK_SIZE + 1][CHUNK_SIZE + 1][CHUNK_SIZE + 1];
    };

}
