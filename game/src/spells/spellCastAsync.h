#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <cstdint>
#include <glm/glm.hpp>

#include "../rendering/VoxelStructures.h"
#include "../physics/CraterStampBatch.h"
#include "SpellEffect.h"

namespace gl3 {

    struct SpellCastRequest
    {
        // Inputs
        glm::vec3 center;
        float searchRadius = 0.0f;
        uint64_t targetMaterial = 0;
        float strength = 1.0f;
        FormationParams baseFormationParams;

        // --- Physics snapshot (captured on main thread at cast time) ---
        bool physicsEnabled = false;
        glm::vec3 launchDir = glm::vec3(0, 0, -1);
        float launchSpeed = 0.0f;
        float lifetime = 0.0f;

        struct ChunkSnapshot
        {
            ChunkCoord coord;
            glm::vec3 chunkMinWorld;
            std::vector<Voxel> voxelsLinear;
        };

        std::vector<ChunkSnapshot> chunks;
    };

    struct SpellCastResult
    {
        bool ok = false;
        std::string debugMsg;

        SpellEffect spell;
        std::vector<AnimatedVoxel> visualVoxels;

        std::vector<CraterStampBatch::Stamp> craterStamps;
        std::vector<ChunkCoord> touchedChunks;
    };

    class SpellCastAsync
    {
    public:
        SpellCastAsync();
        ~SpellCastAsync();

        void enqueueOrReplaceQueued(SpellCastRequest req);

        bool tryPopCompleted(SpellCastResult& out);

        void stop();

    private:
        void workerLoop();
        SpellCastResult runJob(const SpellCastRequest& req);

    private:
        std::thread worker;
        std::mutex mtx;
        std::condition_variable cv;
        bool shouldStop = false;

        std::optional<SpellCastRequest> inFlight;
        std::optional<SpellCastRequest> queued;

        std::vector<SpellCastResult> completed;

        static glm::mat3 makeBasisFromNormalUp(glm::vec3 normal, glm::vec3 upHint);

        static glm::vec3 distSphere(size_t index, size_t total, const FormationParams &p);

        static glm::vec3 distWall(size_t index, size_t total, const FormationParams &p);

        static glm::vec3 distCubeSurface(size_t index, size_t total, const FormationParams &p);

        static glm::vec3 distCylinder(size_t index, size_t total, const FormationParams &p);

        static glm::vec3 distPyramid(size_t index, size_t total, const FormationParams &p);

        static glm::vec3 distPlatform(size_t index, size_t total, const FormationParams &p);

    };

}