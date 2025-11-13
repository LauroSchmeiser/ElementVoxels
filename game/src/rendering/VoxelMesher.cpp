#pragma once
#include "VoxelRenderer.h"
#include "Mesh.h"
#include <vector>
#include "glm/glm.hpp"

namespace gl3 {

    struct Face {
        glm::vec3 normal;
        glm::vec3 positions[4];
    };

    // Define the 6 faces of a cube
    static const Face cubeFaces[6] = {
            { glm::vec3(0, 0, 1), { glm::vec3(0,0,1), glm::vec3(1,0,1), glm::vec3(1,1,1), glm::vec3(0,1,1) } }, // front
            { glm::vec3(0, 0,-1), { glm::vec3(1,0,0), glm::vec3(0,0,0), glm::vec3(0,1,0), glm::vec3(1,1,0) } }, // back
            { glm::vec3(1, 0, 0), { glm::vec3(1,0,1), glm::vec3(1,0,0), glm::vec3(1,1,0), glm::vec3(1,1,1) } }, // right
            { glm::vec3(-1,0, 0), { glm::vec3(0,0,0), glm::vec3(0,0,1), glm::vec3(0,1,1), glm::vec3(0,1,0) } }, // left
            { glm::vec3(0, 1, 0), { glm::vec3(0,1,1), glm::vec3(1,1,1), glm::vec3(1,1,0), glm::vec3(0,1,0) } }, // top
            { glm::vec3(0,-1, 0), { glm::vec3(0,0,0), glm::vec3(1,0,0), glm::vec3(1,0,1), glm::vec3(0,0,1) } }  // bottom
    };


    Mesh generateVoxelChunkMesh(const Chunk& chunk) {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        unsigned int indexOffset = 0;

        for (int x = 0; x < CHUNK_SIZE; ++x)
            for (int y = 0; y < CHUNK_SIZE; ++y)
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    const Voxel& voxel = chunk.voxels[x][y][z];
                    if (!voxel.isSolid()) continue;

                    for (int f = 0; f < 6; ++f) {
                        glm::ivec3 n = glm::ivec3 {x, y, z} + glm::ivec3(cubeFaces[f].normal);

                        bool neighborSolid = false;
                        if (n.x >= 0 && n.x < CHUNK_SIZE &&
                            n.y >= 0 && n.y < CHUNK_SIZE &&
                            n.z >= 0 && n.z < CHUNK_SIZE) {
                            neighborSolid = chunk.voxels[n.x][n.y][n.z].isSolid();
                        }

                        if (!neighborSolid) {
                            // Each vertex has: position (x,y,z) + color (r,g,b)
                            for (const auto& pos : cubeFaces[f].positions) {
                                vertices.push_back(pos.x + x);
                                vertices.push_back(pos.y + y);
                                vertices.push_back(pos.z + z);
                                vertices.push_back(voxel.color.r);
                                vertices.push_back(voxel.color.g);
                                vertices.push_back(voxel.color.b);
                                vertices.push_back(cubeFaces[f].normal.x);
                                vertices.push_back(cubeFaces[f].normal.y);
                                vertices.push_back(cubeFaces[f].normal.z);

                            }

                            // 2 triangles per face
                            indices.insert(indices.end(), {
                                    indexOffset, indexOffset + 1, indexOffset + 2,
                                    indexOffset, indexOffset + 2, indexOffset + 3
                            });
                            indexOffset += 4;
                        }
                    }
                }

        return Mesh(vertices, indices);
    }

}

