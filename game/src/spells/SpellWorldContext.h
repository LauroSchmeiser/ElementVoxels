#pragma once
#include <functional>
#include <glm/glm.hpp>
#include "../rendering/VoxelStructures.h"

namespace gl3 {

    class FixedGridChunkManager;
    class VoxelPhysicsManager;
    struct Chunk;

    struct SpellWorldContext {
        FixedGridChunkManager* chunks = nullptr;
        VoxelPhysicsManager* physics = nullptr;

        // world helpers
        std::function<int(float)> worldToChunk;
        std::function<glm::vec3()> getCameraFront;
        std::function<glm::vec3(const ChunkCoord&)> getChunkMin;
        std::function<void(const ChunkCoord&)> markChunkModified;

        // sampling
        std::function<float(const glm::vec3&)> sampleDensityAtWorld;
        std::function<glm::vec3(const glm::vec3&)> sampleNormalAtWorld;

        std::function<void(Chunk*)> generateChunkMesh;

        // Main thread dispatcher
        std::function<void(std::function<void()>)> mainThreadDispatcher;
    };
    struct WorldPlanet {
        glm::vec3 worldPos;
        float radius;
        glm::vec3 color;
        int type=0; // 1=rock, 2=lava, 3=water, 4=gas/mist
        int material=0;
    };

}