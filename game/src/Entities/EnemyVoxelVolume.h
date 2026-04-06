// EnemyVoxelVolume.h (reuse for spells too)
#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../rendering/VoxelStructures.h"

namespace gl3 {

    struct LocalVoxelVolume {
        glm::ivec3 dims = {33,33,33};  // CORNERS count, not cells
        float voxelSize = VOXEL_SIZE;

        struct Corner {
            float density = -1000.0f;
            glm::vec3 color = glm::vec3(1);
            uint32_t material = 0;
            uint8_t type = 0;
        };

        std::vector<Corner> corners;

        void init(glm::ivec3 cornerDims, float vs) {
            dims = cornerDims;
            voxelSize = vs;
            corners.assign((size_t)dims.x * dims.y * dims.z, {});
        }

        inline size_t idx(int x,int y,int z) const {
            return (size_t)x + (size_t)y * dims.x + (size_t)z * dims.x * dims.y;
        }

        Corner& at(int x,int y,int z) { return corners[idx(x,y,z)]; }
        const Corner& at(int x,int y,int z) const { return corners[idx(x,y,z)]; }

        // Simple “fill a sphere” in local space (for initial enemy body)
        void fillSphere(glm::vec3 centerLocal, float radiusWorld, glm::vec3 col, uint32_t material=0, uint8_t type=1) {
            const float r2 = radiusWorld * radiusWorld;
            for (int z=0; z<dims.z; ++z)
                for (int y=0; y<dims.y; ++y)
                    for (int x=0; x<dims.x; ++x) {
                        glm::vec3 p = glm::vec3(x,y,z) * voxelSize;
                        float d2 = glm::dot(p-centerLocal, p-centerLocal);
                        Corner& c = at(x,y,z);

                        // “density” convention: positive inside
                        float s = radiusWorld - std::sqrt(std::max(0.0f, d2));
                        c.density = s;
                        if (s >= -1.0f) {
                            c.color = col;
                            c.material = material;
                            c.type = type;
                        }
                    }
        }

        // Damage carve: subtract density inside sphere => “removes” matter
        void carveSphere(glm::vec3 centerLocal, float radiusWorld, float strength) {
            const float r2 = radiusWorld * radiusWorld;
            for (int z=0; z<dims.z; ++z)
                for (int y=0; y<dims.y; ++y)
                    for (int x=0; x<dims.x; ++x) {
                        glm::vec3 p = glm::vec3(x,y,z) * voxelSize;
                        float d2 = glm::dot(p-centerLocal, p-centerLocal);
                        if (d2 > r2) continue;
                        at(x,y,z).density -= strength;
                    }
        }
    };

} // namespace gl3