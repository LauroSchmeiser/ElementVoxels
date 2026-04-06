#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../rendering/marchingTables.h"

#include "EnemyVoxelVolume.h"

namespace gl3 {

    struct LocalMesh {
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> colors;
    };

    static inline glm::vec3 lerp3(const glm::vec3& a, const glm::vec3& b, float t) { return a + (b-a)*t; }
    static inline float lerp1(float a, float b, float t) { return a + (b-a)*t; }

    inline LocalMesh buildMeshLocalMC(const LocalVoxelVolume& vol) {
        LocalMesh out;

        glm::ivec3 cells = vol.dims - glm::ivec3(1);
        out.vertices.reserve((size_t)cells.x * cells.y * cells.z); // rough

        auto sampleD = [&](int x,int y,int z){ return vol.at(x,y,z).density; };
        auto sampleC = [&](int x,int y,int z){ return vol.at(x,y,z).color;  };

        const int edgeToA[12] = {0,1,2,3,4,5,6,7,0,1,2,3};
        const int edgeToB[12] = {1,2,3,0,5,6,7,4,4,5,6,7};

        const glm::ivec3 cornerOfs[8] = {
                {0,0,0},{1,0,0},{1,1,0},{0,1,0},
                {0,0,1},{1,0,1},{1,1,1},{0,1,1},
        };

        for (int z=0; z<cells.z; ++z)
            for (int y=0; y<cells.y; ++y)
                for (int x=0; x<cells.x; ++x) {
                    float d[8];
                    glm::vec3 c[8];
                    glm::vec3 p[8];

                    for (int i=0;i<8;++i) {
                        int cx = x + cornerOfs[i].x;
                        int cy = y + cornerOfs[i].y;
                        int cz = z + cornerOfs[i].z;
                        d[i] = sampleD(cx,cy,cz);
                        c[i] = sampleC(cx,cy,cz);
                        p[i] = glm::vec3(cx,cy,cz) * vol.voxelSize; // local space
                    }

                    int caseIndex = 0;
                    for (int i=0;i<8;++i) if (d[i] >= 0.0f) caseIndex |= (1<<i);
                    int e = edgeTableCPU[caseIndex];
                    if (e == 0) continue;

                    glm::vec3 ev[12];
                    glm::vec3 ec[12];

                    for (int ei=0; ei<12; ++ei) {
                        if ((e & (1<<ei)) == 0) continue;
                        int a = edgeToA[ei], b = edgeToB[ei];
                        float denom = (d[a] - d[b]);
                        float t = (std::abs(denom) < 1e-6f) ? 0.5f : (d[a] / denom);
                        t = glm::clamp(t, 0.0f, 1.0f);
                        ev[ei] = lerp3(p[a], p[b], t);
                        ec[ei] = lerp3(c[a], c[b], t);
                    }

                    int base = caseIndex * 16;
                    for (int ti = 0; ti < 16; ti += 3) {
                        int i0 = triTableCPU[base + ti + 0];
                        if (i0 == -1) break;
                        int i1 = triTableCPU[base + ti + 1];
                        int i2 = triTableCPU[base + ti + 2];

                        glm::vec3 v0 = ev[i0], v1 = ev[i1], v2 = ev[i2];
                        glm::vec3 col0 = ec[i0], col1 = ec[i1], col2 = ec[i2];

                        glm::vec3 fn = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                        if (!glm::all((glm::isinf(fn)))) fn = glm::vec3(0,1,0);

                        out.vertices.push_back(v0); out.vertices.push_back(v1); out.vertices.push_back(v2);
                        out.normals.push_back(fn); out.normals.push_back(fn); out.normals.push_back(fn);
                        out.colors.push_back(col0); out.colors.push_back(col1); out.colors.push_back(col2);
                    }
                }

        return out;
    }

} // namespace gl3