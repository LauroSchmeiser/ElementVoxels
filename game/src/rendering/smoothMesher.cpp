#include "SmoothMesher.h"
#include <vector>
#include <glm/glm.hpp>

namespace gl3 {

    static inline float sample(const Chunk &c, int x, int y, int z) {
        return (c.voxels[x][y][z].type > 0) ? 1.0f : 0.0f;
    }

    Mesh generateSmoothPlanetMesh(const Chunk &chunk) {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;

        const int size = CHUNK_SIZE;

        // index3D must cover the full chunk (size³)
        std::vector<int> index3D(size * size * size, -1);

        unsigned int idxCounter = 0;

        glm::vec3 chunkCenter(size * 0.5f);

        // 1) Create smooth “surface samples”
        for (int x = 0; x < size - 1; ++x) {
            for (int y = 0; y < size - 1; ++y) {
                for (int z = 0; z < size - 1; ++z) {

                    float v000 = chunk.voxels[x][y][z].type > 0 ? 1.0f : 0.0f;
                    float v111 = chunk.voxels[x+1][y+1][z+1].type > 0 ? 1.0f : 0.0f;

                    // Skip uniform cube cell
                    if (v000 == v111)
                        continue;

                    // Center of cell
                    glm::vec3 p(x + 0.5f, y + 0.5f, z + 0.5f);

                    // Outward normal from sphere center (approx.)
                    glm::vec3 n = glm::normalize(p - chunkCenter);

                    // Store index into flattened array
                    int flat = x + size * (y + size * z);
                    index3D[flat] = idxCounter++;

                    // Push vertex: position + color + normal
                    // For color, use voxel color or some constant
                    Voxel v = chunk.voxels[x][y][z];
                    glm::vec3 c = v.color;

                    vertices.push_back(p.x);
                    vertices.push_back(p.y);
                    vertices.push_back(p.z);

                    vertices.push_back(c.r);
                    vertices.push_back(c.g);
                    vertices.push_back(c.b);

                    vertices.push_back(n.x);
                    vertices.push_back(n.y);
                    vertices.push_back(n.z);
                }
            }
        }

        // 2) Connect surface samples with triangles
        for (int x = 0; x < size - 2; ++x) {
            for (int y = 0; y < size - 2; ++y) {
                for (int z = 0; z < size - 2; ++z) {

                    int i0 = index3D[x + size * (y + size * z)];
                    int i1 = index3D[(x+1) + size * (y + size * z)];
                    int i2 = index3D[x + size * ((y+1) + size * z)];
                    int i3 = index3D[(x+1) + size * ((y+1) + size * z)];

                    if (i0 >= 0 && i1 >= 0 && i2 >= 0 && i3 >= 0) {
                        indices.push_back(i0);
                        indices.push_back(i1);
                        indices.push_back(i2);

                        indices.push_back(i1);
                        indices.push_back(i3);
                        indices.push_back(i2);
                    }
                }
            }
        }

        return Mesh(vertices, indices);
    }

}
