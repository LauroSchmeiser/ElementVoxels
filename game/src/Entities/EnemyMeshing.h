#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "EnemyVoxelVolume.h"

namespace gl3 {

    struct EnemyMeshBuildResult {
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> colors;
    };

    inline EnemyMeshBuildResult     buildEnemyMeshMarchingCubesLocal(const LocalVoxelVolume& vol) {
        EnemyMeshBuildResult out;
        return out;
    }
}