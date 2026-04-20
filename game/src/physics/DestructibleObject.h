#pragma once

#include "../rendering/VoxelStructures.h"
#include "../entities/EnemyVoxelVolume.h"
#include <glm/glm.hpp>

namespace gl3 {

    struct DestructibleObject {
        LocalVoxelVolume volume;
        PhysicsMeshData  mesh;
        bool meshDirty = true;

        glm::vec3 localCenterOffsetWorld{0};
        float voxelSize = VOXEL_SIZE;

        glm::vec3 worldToLocal(const glm::vec3& worldPos, const glm::vec3& bodyPos) const {
            glm::vec3 originWorld = bodyPos - localCenterOffsetWorld;
            return (worldPos - originWorld);
        }
    };

    inline glm::ivec3 dimsFromHalfExtents(const glm::vec3& halfExtWorld, float voxelSize) {
        glm::ivec3 cells = glm::ivec3(glm::ceil((halfExtWorld * 2.0f) / voxelSize)) + glm::ivec3(3);
        glm::ivec3 corners = cells + glm::ivec3(1);
        corners.x = std::max(corners.x, 5);
        corners.y = std::max(corners.y, 5);
        corners.z = std::max(corners.z, 5);
        return corners;
    }

    inline glm::vec3 halfExtentsFromVolumeDims(const glm::ivec3& cornerDims, float voxelSize) {
        glm::vec3 sizeWorld = glm::vec3(cornerDims - glm::ivec3(1)) * voxelSize;
        return sizeWorld * 0.5f;
    }

    inline float sdfBox(const glm::vec3& p, const glm::vec3& c, const glm::vec3& b) {
        glm::vec3 q = glm::abs(p - c) - b;
        float outside = glm::length(glm::max(q, glm::vec3(0)));
        float inside  = std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
        return outside + inside;
    }

    inline void fillBox(LocalVoxelVolume& vol,
                        const glm::vec3& centerLocal,
                        const glm::vec3& halfExtWorld,
                        glm::vec3 color,
                        uint32_t material=0, uint8_t type=1)
    {
        for (int z=0; z<vol.dims.z; ++z)
            for (int y=0; y<vol.dims.y; ++y)
                for (int x=0; x<vol.dims.x; ++x) {
                    glm::vec3 p = glm::vec3(x,y,z) * vol.voxelSize;
                    float d = -sdfBox(p, centerLocal, halfExtWorld);
                    auto& c = vol.at(x,y,z);
                    c.density = d;
                    if (d >= -1.0f) {
                        c.color = color;
                        c.material = material;
                        c.type = type;
                    }
                }
    }

}