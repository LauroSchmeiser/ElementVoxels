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
        int maxVoxels = static_cast<int>((glm::pow(req.strength,4)) / voxelVolume);
        maxVoxels = clampi(maxVoxels, 10, 60);

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

        // -------------------------------
        // 1) Scan candidates (snapshot)
        // -------------------------------
        {
            ZoneScopedN("SpellCastAsync::runJob::ScanCandidates");

            for (const auto& cs : req.chunks)
            {
                const glm::vec3 chunkMin = cs.chunkMinWorld;
                const glm::vec3 chunkCenter = chunkMin + glm::vec3(CHUNK_SIZE * 0.5f) * VOXEL_SIZE;

                float distToChunkCenter = glm::distance(chunkCenter, req.center);
                float maxChunkDist = std::sqrt(3.0f) * (CHUNK_SIZE * VOXEL_SIZE * 0.5f);
                if (distToChunkCenter > req.searchRadius + maxChunkDist) continue;

                int startX = std::max(0, (int)((req.center.x - req.searchRadius - chunkMin.x) / VOXEL_SIZE));
                int endX   = std::min(CHUNK_SIZE, (int)((req.center.x + req.searchRadius - chunkMin.x) / VOXEL_SIZE) + 1);
                int startY = std::max(0, (int)((req.center.y - req.searchRadius - chunkMin.y) / VOXEL_SIZE));
                int endY   = std::min(CHUNK_SIZE, (int)((req.center.y + req.searchRadius - chunkMin.y) / VOXEL_SIZE) + 1);
                int startZ = std::max(0, (int)((req.center.z - req.searchRadius - chunkMin.z) / VOXEL_SIZE));
                int endZ   = std::min(CHUNK_SIZE, (int)((req.center.z + req.searchRadius - chunkMin.z) / VOXEL_SIZE) + 1);

                for (int x = startX; x <= endX; ++x)
                    for (int y = startY; y <= endY; ++y)
                        for (int z = startZ; z <= endZ; ++z)
                        {
                            const Voxel& v = cs.voxelsLinear[linearIndex(x,y,z)];
                            if (!v.isSolid()) continue;
                            if (v.material != req.targetMaterial) continue;

                            glm::vec3 worldPos = chunkMin + glm::vec3((float)x, (float)y, (float)z) * VOXEL_SIZE;
                            glm::vec3 diff = worldPos - req.center;
                            float d2 = glm::dot(diff, diff);
                            if (d2 > radiusSq) continue;

                            candidates.push_back({
                                                         worldPos,
                                                         v.color,
                                                         cs.coord,
                                                         glm::ivec3(x,y,z),
                                                         d2,
                                                         v.type
                                                 });

                            if (v.type < 8) typeCounts[v.type]++;
                        }
            }
        }

        constexpr int kMinVoxelsToCast = 6; // tweak

        if ((int)candidates.size() < kMinVoxelsToCast)
        {
            out.debugMsg = "Not enough voxels found to cast";
            return out; // ok stays false
        }

        // -------------------------------
        // 2) Sort / trim / dominant type
        // -------------------------------
        uint8_t dominantType = 1;
        {
            ZoneScopedN("SpellCastAsync::runJob::SortAndTrim");

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b){ return a.distSq < b.distSq; });

            if ((int)candidates.size() > maxVoxels)
                candidates.resize((size_t)maxVoxels);

            // Dominant type from selected set
            std::memset(typeCounts, 0, sizeof(typeCounts));
            for (auto& c : candidates)
                if (c.type < 8) typeCounts[c.type]++;

            int maxCount = 0;
            for (int i = 0; i < 8; ++i)
            {
                if (typeCounts[i] > maxCount)
                {
                    maxCount = typeCounts[i];
                    dominantType = (uint8_t)i;
                }
            }
        }

        // -------------------------------
        // 3) Build visual voxels + avgColor
        // -------------------------------
        std::vector<AnimatedVoxel> visual;
        glm::vec3 avgColor(0.0f);
        {
            ZoneScopedN("SpellCastAsync::runJob::BuildVisualVoxels");

            visual.reserve(candidates.size());

            for (const auto& c : candidates)
            {
                AnimatedVoxel av;
                av.currentPos = c.worldPos;
                av.originalVoxelPos = c.worldPos;
                av.color = c.color;
                av.normal = glm::vec3(0,1,0);
                av.isAnimating = true;
                av.hasArrived = false;
                av.animationSpeed = (glm::pow(req.strength,2)/glm::sqrt(req.baseFormationParams.radius))*2;
                av.animationSpeed=glm::min(av.animationSpeed,9.5f);
                av.animationSpeed=glm::max(av.animationSpeed,4.5f);
                visual.push_back(av);

                avgColor += av.color;
            }

            if (!visual.empty())
                avgColor /= (float)visual.size();
        }

        // -------------------------------
        // 4) Adjust formation + set targets
        // -------------------------------
        FormationParams adjusted = req.baseFormationParams;
        {
            ZoneScopedN("SpellCastAsync::runJob::AdjustFormationAndTargets");

            const float voxelVolumeWorld = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
            float desiredVolumeWorld = (float)visual.size() * voxelVolumeWorld;

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

            for (size_t i = 0; i < visual.size(); ++i)
            {
                switch (adjusted.type)
                {
                    case FormationType::SPHERE:
                        visual[i].targetPos = distSphere(i, visual.size(), adjusted);
                        break;
                    case FormationType::PLATFORM:
                        visual[i].targetPos = distPlatform(i, visual.size(), adjusted);
                        break;
                    case FormationType::WALL:
                        visual[i].targetPos = distWall(i, visual.size(), adjusted);
                        break;
                    case FormationType::CUBE:
                        visual[i].targetPos = distCubeSurface(i, visual.size(), adjusted);
                        break;
                    case FormationType::CYLINDER:
                        visual[i].targetPos = distCylinder(i, visual.size(), adjusted);
                        break;
                    case FormationType::PYRAMID:
                        visual[i].targetPos = distPyramid(i, visual.size(), adjusted);
                        break;
                    case FormationType::CUSTOM_SDF:
                    default:
                        visual[i].targetPos = distSphere(i, visual.size(), adjusted);
                        break;
                }

                visual[i].animationSpeed = (glm::pow(req.strength,2)/glm::sqrt(req.baseFormationParams.radius))*2;
                visual[i].animationSpeed=glm::min(visual[i].animationSpeed,9.5f);
                visual[i].animationSpeed=glm::max(visual[i].animationSpeed,4.5f);            }
        }

        // -------------------------------
        // 5) Build stamps + output spell
        // -------------------------------
        std::vector<CraterStampBatch::Stamp> stamps;
        std::vector<ChunkCoord> touched;
        {
            ZoneScopedN("SpellCastAsync::runJob::BuildStampsAndSpell");

            stamps.reserve(visual.size());
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
        }

        return out;
    }
     inline float halton(int index, int base)
    {
        float f = 1.0f;
        float r = 0.0f;
        int i = index;
        while (i > 0) {
            f /= (float)base;
            r += f * (float)(i % base);
            i /= base;
        }
        return r;
    }

    glm::mat3 SpellCastAsync::makeBasisFromNormalUp(glm::vec3 normal, glm::vec3 upHint)
    {
        if (glm::dot(normal, normal) < 1e-8f) normal = glm::vec3(0,1,0);
        normal = glm::normalize(normal);

        if (glm::dot(upHint,upHint) < 1e-8f) {
            upHint = (std::abs(normal.y) > 0.9f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
        } else {
            upHint = glm::normalize(upHint);
        }

        glm::vec3 right = glm::cross(normal, upHint);
        if (glm::dot(right,right) < 1e-8f) {
            upHint = (std::abs(normal.y) > 0.9f) ? glm::vec3(1,0,0) : glm::vec3(0,1,0);
            right = glm::cross(normal, upHint);
        }
        right = glm::normalize(right);

        glm::vec3 up = glm::normalize(glm::cross(right, normal));
        return glm::mat3(right, up, normal);
    }

    glm::vec3 SpellCastAsync::distSphere(size_t index, size_t total, const FormationParams& p)
    {
        if (total <= 1) return p.center;

        float goldenAngle = glm::pi<float>() * (3.0f - std::sqrt(5.0f));
        float t = (float)index / (float)(total - 1);
        float y = 1.0f - t * 2.0f;
        float rAtY = std::sqrt(std::max(0.0f, 1.0f - y*y));
        float theta = goldenAngle * (float)index;
        float x = std::cos(theta) * rAtY;
        float z = std::sin(theta) * rAtY;
        return p.center + glm::vec3(x,y,z) * p.radius;
    }

    glm::vec3 SpellCastAsync::distPlatform(size_t index, size_t total, const FormationParams& p)
    {
        (void)total;
        float u = halton((int)index + 1, 2) - 0.5f;
        float v = halton((int)index + 1, 3) - 0.5f;

        glm::vec3 local(u * p.sizeX,
                        p.sizeY * 0.5f,
                        v * p.sizeZ);

        glm::mat3 basis = makeBasisFromNormalUp(p.normal, p.up);
        return p.center + basis * local;
    }

    glm::vec3 SpellCastAsync::distWall(size_t index, size_t total, const FormationParams& p)
    {
        (void)total;
        float u = halton((int)index + 1, 2) - 0.5f;
        float v = halton((int)index + 1, 3) - 0.5f;

        glm::vec3 local(u * p.sizeX,
                        v * p.sizeY,
                        p.sizeZ * 0.5f);

        glm::mat3 basis = makeBasisFromNormalUp(p.normal, p.up);
        return p.center + basis * local;
    }

    glm::vec3 SpellCastAsync::distCubeSurface(size_t index, size_t total, const FormationParams& p)
    {
        (void)total;
        int face = (int)(index % 6);
        float u = halton((int)index + 1, 2) - 0.5f;
        float v = halton((int)index + 1, 3) - 0.5f;

        glm::vec3 local(0.0f);
        switch(face) {
            case 0: local = glm::vec3( p.sizeX * 0.5f, u * p.sizeY, v * p.sizeZ); break; // +X
            case 1: local = glm::vec3(-p.sizeX * 0.5f, u * p.sizeY, v * p.sizeZ); break; // -X
            case 2: local = glm::vec3(u * p.sizeX,  p.sizeY * 0.5f, v * p.sizeZ); break; // +Y
            case 3: local = glm::vec3(u * p.sizeX, -p.sizeY * 0.5f, v * p.sizeZ); break; // -Y
            case 4: local = glm::vec3(u * p.sizeX, v * p.sizeY,  p.sizeZ * 0.5f); break; // +Z
            default:local = glm::vec3(u * p.sizeX, v * p.sizeY, -p.sizeZ * 0.5f); break; // -Z
        }

        return p.center + local;
    }

    glm::vec3 SpellCastAsync::distCylinder(size_t index, size_t total, const FormationParams& p)
    {
        if (total <= 1) return p.center;

        float t = (float)index / (float)total;
        float angle = t * glm::two_pi<float>();
        float h = halton((int)index + 1, 2) - 0.5f;

        glm::vec3 local(std::cos(angle) * p.radius,
                        h * p.sizeY,
                        std::sin(angle) * p.radius);
        return p.center + local;
    }

    glm::vec3 SpellCastAsync::distPyramid(size_t index, size_t total, const FormationParams& p)
    {
        (void)total;
        int surface = (int)(index % 5);

        if (surface < 4) {
            float u = halton((int)index + 1, 2);
            float v = halton((int)index + 1, 3);

            float baseX = (u - 0.5f) * p.sizeX;
            float baseZ = (v - 0.5f) * p.sizeZ;

            glm::vec3 local(0.0f);
            switch(surface) {
                case 0: local = glm::vec3(baseX, 0.0f,  p.sizeZ * 0.5f); break;
                case 1: local = glm::vec3(baseX, 0.0f, -p.sizeZ * 0.5f); break;
                case 2: local = glm::vec3( p.sizeX * 0.5f, 0.0f, baseZ); break;
                case 3: local = glm::vec3(-p.sizeX * 0.5f, 0.0f, baseZ); break;
            }

            float heightRatio = halton((int)index + 1, 5);
            local.y = heightRatio * p.sizeY;
            local.x *= (1.0f - heightRatio);
            local.z *= (1.0f - heightRatio);

            return p.center + local;
        } else {
            float u = halton((int)index + 1, 2) - 0.5f;
            float v = halton((int)index + 1, 3) - 0.5f;
            glm::vec3 local(u * p.sizeX, 0.0f, v * p.sizeZ);
            return p.center + local;
        }
    }
}