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
        glm::vec3 launchDir = glm::vec3(0, 0, -1);      // camera front at cast time
        float launchSpeed = 0.0f;                        // world units/sec (already includes VOXEL_SIZE scaling if you want)
        float lifetime = 0.0f;                           // seconds; 0 = infinite (matches your cleanupExpiredSpells logic)
        // NOTE: creationTime is always set to 0 on spawn

        // Snapshot data (immutable, safe off-thread)
        struct ChunkSnapshot
        {
            ChunkCoord coord;
            glm::vec3 chunkMinWorld;
            // 17^3 voxels; same indexing as Chunk::voxels[x][y][z] for x,y,z in [0..CHUNK_SIZE]
            std::vector<Voxel> voxelsLinear;
        };

        std::vector<ChunkSnapshot> chunks;
    };

    struct SpellCastResult
    {
        bool ok = false;
        std::string debugMsg;

        // What to spawn on main thread
        SpellEffect spell;
        std::vector<AnimatedVoxel> visualVoxels;

        // Main-thread world edits to apply
        std::vector<CraterStampBatch::Stamp> craterStamps;
        std::vector<ChunkCoord> touchedChunks; // chunks containing collected voxels
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
        static SpellCastResult runJob(const SpellCastRequest& req);

    private:
        std::thread worker;
        std::mutex mtx;
        std::condition_variable cv;
        bool shouldStop = false;

        std::optional<SpellCastRequest> inFlight;
        std::optional<SpellCastRequest> queued;

        std::vector<SpellCastResult> completed;
    };

} // namespace gl3