#include "SpellCastAsync.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <tracy/Tracy.hpp>

namespace gl3 {

    static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

    SpellCastAsync::SpellCastAsync()
    {
        worker = std::thread([this] { workerLoop(); });
    }

    SpellCastAsync::~SpellCastAsync()
    {
        stop();
    }

    void SpellCastAsync::stop()
    {
        {
            std::lock_guard<std::mutex> lk(mtx);
            shouldStop = true;
            cv.notify_all();
        }
        if (worker.joinable())
            worker.join();
    }

    void SpellCastAsync::enqueueOrReplaceQueued(SpellCastRequest req)
    {
        ZoneScopedN("SpellCastAsync::enqueueOrReplaceQueued");
        std::lock_guard<std::mutex> lk(mtx);

        if (!inFlight.has_value())
        {
            inFlight = std::move(req);
            cv.notify_all();
            return;
        }

        queued = std::move(req);
        cv.notify_all();
    }

    bool SpellCastAsync::tryPopCompleted(SpellCastResult& out)
    {
        ZoneScopedN("SpellCastAsync::tryPopCompleted");
        std::lock_guard<std::mutex> lk(mtx);
        if (completed.empty()) return false;

        out = std::move(completed.front());
        completed.erase(completed.begin());
        return true;
    }

    void SpellCastAsync::workerLoop()
    {
        tracy::SetThreadName("SpellCastWorker");

        for (;;)
        {
            ZoneScopedN("SpellCastWorker::Loop");

            SpellCastRequest job;

            {
                ZoneScopedN("SpellCastWorker::WaitForJob");

                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return shouldStop || inFlight.has_value(); });

                if (shouldStop) return;

                job = std::move(*inFlight);
            }

            SpellCastResult result;
            {
                ZoneScopedN("SpellCastWorker::runJob");
                result = runJob(job);
            }
            {
                ZoneScopedN("SpellCastWorker::PublishResult");
                std::lock_guard<std::mutex> lk(mtx);
                completed.push_back(std::move(result));

                if (queued.has_value())
                {
                    inFlight = std::move(*queued);
                    queued.reset();
                }
                else
                {
                    inFlight.reset();
                }

                cv.notify_all();
            }
        }
    }

    static inline size_t linearIndex(int x, int y, int z)
    {
        const int dim = CHUNK_SIZE + 1;
        return (size_t)x + (size_t)y * dim + (size_t)z * dim * dim;
    }

    SpellCastResult SpellCastAsync::runJob(const SpellCastRequest& req)
    {
        ZoneScopedN("SpellCastAsync::runJob");
        SpellCastResult out;
        out.ok = false;

        const float radiusSq = req.searchRadius * req.searchRadius;

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        int maxVoxels = static_cast<int>((req.strength * 70.0f) / voxelVolume);
        maxVoxels = clampi(maxVoxels, 50, 200);

        struct Candidate
        {
            glm::vec3 worldPos;
            glm::vec3 color;
            ChunkCoord chunkCoord;
            glm::ivec3 localPos;
            float distSq;
            uint8_t type;
        };

        std::vector<Candidate> candidates;
        candidates.reserve((size_t)maxVoxels * 2);

        int typeCounts[8] = {0};


        {
            ZoneScopedN("SpellCastAsync::runJob::ScanCandidates");
            for (const auto &cs: req.chunks) {
                const glm::vec3 chunkMin = cs.chunkMinWorld;
                const glm::vec3 chunkCenter = chunkMin + glm::vec3(CHUNK_SIZE * 0.5f) * VOXEL_SIZE;

                float distToChunkCenter = glm::distance(chunkCenter, req.center);
                float maxChunkDist = std::sqrt(3.0f) * (CHUNK_SIZE * VOXEL_SIZE * 0.5f);
                if (distToChunkCenter > req.searchRadius + maxChunkDist) continue;

                int startX = std::max(0, (int) ((req.center.x - req.searchRadius - chunkMin.x) / VOXEL_SIZE));
                int endX = std::min(CHUNK_SIZE,
                                    (int) ((req.center.x + req.searchRadius - chunkMin.x) / VOXEL_SIZE) + 1);
                int startY = std::max(0, (int) ((req.center.y - req.searchRadius - chunkMin.y) / VOXEL_SIZE));
                int endY = std::min(CHUNK_SIZE,
                                    (int) ((req.center.y + req.searchRadius - chunkMin.y) / VOXEL_SIZE) + 1);
                int startZ = std::max(0, (int) ((req.center.z - req.searchRadius - chunkMin.z) / VOXEL_SIZE));
                int endZ = std::min(CHUNK_SIZE,
                                    (int) ((req.center.z + req.searchRadius - chunkMin.z) / VOXEL_SIZE) + 1);

                for (int x = startX; x <= endX; ++x)
                    for (int y = startY; y <= endY; ++y)
                        for (int z = startZ; z <= endZ; ++z) {
                            const Voxel &v = cs.voxelsLinear[linearIndex(x, y, z)];
                            if (!v.isSolid()) continue;
                            if (v.material != req.targetMaterial) continue;

                            glm::vec3 worldPos = chunkMin + glm::vec3((float) x, (float) y, (float) z) * VOXEL_SIZE;
                            glm::vec3 diff = worldPos - req.center;
                            float d2 = glm::dot(diff, diff);
                            if (d2 > radiusSq) continue;

                            candidates.push_back({
                                                         worldPos,
                                                         v.color,
                                                         cs.coord,
                                                         glm::ivec3(x, y, z),
                                                         d2,
                                                         v.type
                                                 });

                            if (v.type < 8) typeCounts[v.type]++;
                        }
            }

        }
        if (candidates.empty())
        {
            out.debugMsg = "No voxels found for spell (snapshot scan)";
            return out;
        }

        {
            ZoneScopedN("SpellCastAsync::runJob::SortAndTrim");
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate &a, const Candidate &b) { return a.distSq < b.distSq; });

            if ((int) candidates.size() > maxVoxels)
                candidates.resize((size_t) maxVoxels);

            std::memset(typeCounts, 0, sizeof(typeCounts));
            for (auto &c: candidates)
                if (c.type < 8) typeCounts[c.type]++;

        }
        int maxCount = 0;
        uint8_t dominantType = 1;
        for (int i = 0; i < 8; ++i)
        {
            if (typeCounts[i] > maxCount)
            {
                maxCount = typeCounts[i];
                dominantType = (uint8_t)i;
            }
        }

        std::vector<AnimatedVoxel> visual;
        visual.reserve(candidates.size());

        glm::vec3 avgColor(0.0f);
        for (const auto& c : candidates)
        {
            AnimatedVoxel av;
            av.currentPos = c.worldPos;
            av.originalVoxelPos = c.worldPos;
            av.color = c.color;
            av.normal = glm::vec3(0,1,0);
            av.isAnimating = true;
            av.hasArrived = false;
            av.animationSpeed = req.strength * 1.5f;
            visual.push_back(av);

            avgColor += av.color;
        }
        avgColor /= (float)visual.size();

        const float voxelVolumeWorld = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        float desiredVolumeWorld = (float)visual.size() * voxelVolumeWorld;

        FormationParams adjusted = req.baseFormationParams;
        const float packingEfficiency = 0.3f;
        constexpr float PI = 3.14159265358979323846f;
        const float minWorldDim = VOXEL_SIZE * 0.15f;

        switch(adjusted.type)
        {
            case FormationType::SPHERE:
            {
                float computedRadius = std::cbrt((3.0f / (4.0f * PI)) * ((desiredVolumeWorld/2) / packingEfficiency));
                float maxRadius = std::max(minWorldDim, adjusted.radius * 0.75f);
                float minRadius = minWorldDim;
                adjusted.radius = std::max(minRadius, std::min(computedRadius, maxRadius));
                break;
            }
            case FormationType::PLATFORM:
            {
                float area = desiredVolumeWorld / (adjusted.sizeY * packingEfficiency);
                float side = std::sqrt(std::max(0.0f, area));
                adjusted.sizeX = std::max(side, minWorldDim);
                adjusted.sizeZ = std::max(side, minWorldDim);
                break;
            }
            case FormationType::WALL:
            {
                float area = desiredVolumeWorld / (adjusted.sizeZ * packingEfficiency) * 20.0f;
                float side = std::sqrt(std::max(0.0f, area));
                adjusted.sizeX = std::max(side, minWorldDim);
                adjusted.sizeY = std::max(adjusted.sizeX * 0.75f, minWorldDim);
                break;
            }
            case FormationType::CUBE:
            {
                float side = std::cbrt(std::max(0.0f, desiredVolumeWorld / packingEfficiency));
                adjusted.sizeX = adjusted.sizeY = adjusted.sizeZ = std::max(side, minWorldDim);
                break;
            }
            case FormationType::CYLINDER:
            {
                float computedRadius = std::cbrt((2.0f / (3.0f * PI)) * (desiredVolumeWorld / packingEfficiency));
                adjusted.radius = std::max(computedRadius, minWorldDim);
                adjusted.sizeY = std::max(adjusted.radius * 2.0f, minWorldDim);
                break;
            }
            default:
                break;
        }

        adjusted.center = req.center;

        auto fibonacciSphere = [&](size_t index, size_t total)->glm::vec3 {
            float goldenAngle = glm::pi<float>() * (3.0f - std::sqrt(5.0f));
            float y = 1.0f - (float(index) / (float(total) - 1.0f)) * 2.0f;
            float rAtY = std::sqrt(std::max(0.0f, 1.0f - y*y));
            float theta = goldenAngle * float(index);
            float x = std::cos(theta) * rAtY;
            float z = std::sin(theta) * rAtY;
            glm::vec3 local(x,y,z);
            return adjusted.center + local * adjusted.radius;
        };

        for (size_t i = 0; i < visual.size(); ++i)
        {
            if (adjusted.type == FormationType::SPHERE)
                visual[i].targetPos = fibonacciSphere(i, visual.size());
            else
                visual[i].targetPos = fibonacciSphere(i, visual.size());

            visual[i].animationSpeed = req.strength * 7.5f;
        }

        std::vector<CraterStampBatch::Stamp> stamps;
        stamps.reserve(visual.size());
        std::vector<ChunkCoord> touched;
        touched.reserve(visual.size());

        for (const auto& c : candidates)
        {
            CraterStampBatch::Stamp s;
            s.center = c.worldPos;
            s.radius = 2.0f * VOXEL_SIZE;
            s.depth  = 5.5f;
            stamps.push_back(s);

            touched.push_back(c.chunkCoord);
        }

        SpellEffect spell;
        spell.type = SpellEffect::Type::CONSTRUCT;
        spell.center = req.center;
        spell.radius = adjusted.getBoundingRadius();
        spell.strength = req.strength;
        spell.targetMaterial = req.targetMaterial;
        spell.dominantType = dominantType;
        spell.formationParams = adjusted;
        spell.formationColor = avgColor;
        spell.formationRadius = adjusted.getBoundingRadius();

        // --- NEW: carry physics data through (visual behavior matches old path) ---
        spell.isPhysicsEnabled = req.physicsEnabled;
        if (req.physicsEnabled)
        {
            spell.creationTime = 0.0f;
            spell.lifetime = req.lifetime;

            glm::vec3 dir = req.launchDir;
            if (glm::length(dir) < 0.001f) dir = glm::vec3(0,0,-1);
            dir = glm::normalize(dir);

            spell.initialVelocity = dir * req.launchSpeed;
        }

        out.ok = true;
        out.spell = spell;
        out.visualVoxels = std::move(visual);
        out.craterStamps = std::move(stamps);
        out.touchedChunks = std::move(touched);
        out.debugMsg = "Spell async job completed";
        return out;
    }

} // namespace gl3