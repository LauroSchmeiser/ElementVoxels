#include "SmoothMesher.h"
#include <vector>
#include <glm/glm.hpp>

namespace gl3 {
    static inline float sample(const Chunk &c, int x, int y, int z) {
        return (c.voxels[x][y][z].type > 0) ? 1.0f : 0.0f;
    }

    Mesh generateSmoothPlanetMesh(const Chunk &chunk) {
        std::vector<glm::vec3> verts;
        std::vector<glm::vec3> norms;
        std::vector<unsigned int> idx;

        int index3D[64];

        int idxCounter = 0;

        for (int x = 0; x < CHUNK_SIZE - 1; ++x) {
            for (int y = 0; y < CHUNK_SIZE - 1; ++y) {
                for (int z = 0; z < CHUNK_SIZE - 1; ++z) {
                    float v000 = sample(chunk, x, y, z);
                    float v111 = sample(chunk, x + 1, y + 1, z + 1);

                    // Skip if uniform
                    if (v000 == v111) continue;

                    // Compute the cell center
                    glm::vec3 center(x + 0.5f, y + 0.5f, z + 0.5f);

                    // Add vertex
                    index3D[x + CHUNK_SIZE * (y + CHUNK_SIZE * z)] = idxCounter++;
                    verts.push_back(center);
                    norms.push_back(glm::normalize(center - glm::vec3(CHUNK_SIZE / 2)));
                }
            }
        }

        // Build faces (simple grid stitching)
        for (int x = 0; x < CHUNK_SIZE - 2; ++x) {
            for (int y = 0; y < CHUNK_SIZE - 2; ++y) {
                for (int z = 0; z < CHUNK_SIZE - 2; ++z) {
                    int a = index3D[x + CHUNK_SIZE * (y + CHUNK_SIZE * z)];
                    int b = index3D[(x + 1) + CHUNK_SIZE * (y + CHUNK_SIZE * z)];
                    int c = index3D[x + CHUNK_SIZE * ((y + 1) + CHUNK_SIZE * z)];
                    int d = index3D[(x + 1) + CHUNK_SIZE * ((y + 1) + CHUNK_SIZE * z)];

                    if (a >= 0 && b >= 0 && c >= 0 && d >= 0) {
                        idx.push_back(a);
                        idx.push_back(b);
                        idx.push_back(c);

                        idx.push_back(b);
                        idx.push_back(d);
                        idx.push_back(c);
                    }
                }
            }
        }

        // Construct mesh
        Mesh mesh;
        mesh.loadData(verts, norms, idx);
        return mesh;
    }
}
