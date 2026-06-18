#pragma once
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include "../rendering/marchingTables.h"
#include "EnemyVoxelVolume.h"

namespace gl3 {

    struct LocalSubmesh {
        uint32_t material = 0;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> colors;
        std::vector<glm::vec2> uvs;
        std::vector<uint32_t> flags;
    };

    struct LocalMesh {
        std::vector<LocalSubmesh> parts;
    };

    static inline glm::vec3 lerp3(const glm::vec3& a, const glm::vec3& b, float t) { return a + (b - a) * t; }

    inline LocalSubmesh& getOrCreateSubmesh(LocalMesh& mesh, uint32_t material) {
        for (auto& part : mesh.parts) {
            if (part.material == material) return part;
        }
        mesh.parts.push_back({});
        mesh.parts.back().material = material;
        return mesh.parts.back();
    }

    inline LocalMesh buildMeshLocalMC(const LocalVoxelVolume& vol) {
        LocalMesh out;

        glm::ivec3 cells = vol.dims - glm::ivec3(1);

        auto sampleD = [&](int x,int y,int z){ return vol.at(x,y,z).density; };
        auto sampleC = [&](int x,int y,int z){ return vol.at(x,y,z).color; };
        auto sampleM = [&](int x,int y,int z){ return vol.at(x,y,z).material; };

        const int edgeToA[12] = {0,1,2,3,4,5,6,7,0,1,2,3};
        const int edgeToB[12] = {1,2,3,0,5,6,7,4,4,5,6,7};

        const glm::ivec3 cornerOfs[8] = {
                {0,0,0},{1,0,0},{1,1,0},{0,1,0},
                {0,0,1},{1,0,1},{1,1,1},{0,1,1},
        };

        for (int z = 0; z < cells.z; ++z)
            for (int y = 0; y < cells.y; ++y)
                for (int x = 0; x < cells.x; ++x) {
                    float d[8];
                    glm::vec3 c[8];
                    glm::vec3 p[8];
                    uint32_t m[8];

                    for (int i = 0; i < 8; ++i) {
                        int cx = x + cornerOfs[i].x;
                        int cy = y + cornerOfs[i].y;
                        int cz = z + cornerOfs[i].z;
                        d[i] = sampleD(cx, cy, cz);
                        c[i] = sampleC(cx, cy, cz);
                        m[i] = sampleM(cx, cy, cz);
                        p[i] = glm::vec3(cx, cy, cz) * vol.voxelSize;
                    }

                    int caseIndex = 0;
                    for (int i = 0; i < 8; ++i) {
                        if (d[i] >= 0.0f) caseIndex |= (1 << i);
                    }

                    int e = edgeTableCPU[caseIndex];
                    if (e == 0) continue;

                    glm::vec3 ev[12];
                    glm::vec3 ec[12];
                    uint32_t em[12];

                    for (int ei = 0; ei < 12; ++ei) {
                        if ((e & (1 << ei)) == 0) continue;

                        int a = edgeToA[ei];
                        int b = edgeToB[ei];

                        float denom = d[a] - d[b];
                        float t = (std::abs(denom) < 1e-6f) ? 0.5f : (d[a] / denom);
                        t = glm::clamp(t, 0.0f, 1.0f);

                        ev[ei] = lerp3(p[a], p[b], t);
                        ec[ei] = lerp3(c[a], c[b], t);
                        em[ei] = (t < 0.5f) ? m[a] : m[b];
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

                        // Pick triangle material by majority / priority.
                        uint32_t triMat = em[i0];
                        if (em[i1] == em[i2]) triMat = em[i1];
                        else if (em[i1] == em[i0]) triMat = em[i0];
                        else if (em[i2] == em[i0]) triMat = em[i0];

                        LocalSubmesh& dst = getOrCreateSubmesh(out, triMat);
                        uint32_t packedFlags = (triMat << 1u);

                        dst.vertices.push_back(v0); dst.vertices.push_back(v1); dst.vertices.push_back(v2);
                        dst.normals.push_back(fn);  dst.normals.push_back(fn);  dst.normals.push_back(fn);
                        dst.colors.push_back(col0); dst.colors.push_back(col1); dst.colors.push_back(col2);

                        dst.uvs.push_back(glm::vec2(0.0f));
                        dst.uvs.push_back(glm::vec2(0.0f));
                        dst.uvs.push_back(glm::vec2(0.0f));

                        dst.flags.push_back(packedFlags);
                        dst.flags.push_back(packedFlags);
                        dst.flags.push_back(packedFlags);
                    }
                }

        return out;
    }

} // namespace gl3