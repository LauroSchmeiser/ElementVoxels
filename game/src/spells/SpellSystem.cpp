#include "SpellSystem.h"

#include "../rendering/FixedGridChunkManager.h"
#include "../physics/CraterStampBatch.h"
#include "../physics/VoxelPhysicsManager.h"
#include "../physics/DestructibleObject.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../Entities/LocalMarchingCubes.h"
#include <amp_short_vectors.h>
#include <tracy/Tracy.hpp>
#define TRACY_CPU_ZONE(nameStr) ZoneScopedN(nameStr)

namespace gl3 {

    SpellSystem::SpellSystem(SpellWorldContext c)
            : ctx(std::move(c))
    {
        spellCastAsync = std::make_unique<SpellCastAsync>();
        initSphereMeshCache();

    }

    SpellSystem::~SpellSystem()
    {
        shuttingDown.store(true, std::memory_order_relaxed);

        if (spellCastAsync) {
            spellCastAsync->stop();
            spellCastAsync.reset();
        }

        for (auto& t : workerThreads) {
            if (t.joinable()) t.join();
        }
    }

    void SpellSystem::initSphereMeshCache() {
        const std::vector<float> commonRadii = {
                2.0f * VOXEL_SIZE,
                4.0f * VOXEL_SIZE,
                6.0f * VOXEL_SIZE,
                8.0f * VOXEL_SIZE,
                10.0f * VOXEL_SIZE
        };

        for (float radius : commonRadii) {
            int key = static_cast<int>(radius / VOXEL_SIZE);
            sphereMeshCache[key] = generateIcosphere(radius, 2);
        }

        std::cout << "Initialized " << sphereMeshCache.size() << " sphere meshes\n";
    }

    SpellSystem::SphereMesh SpellSystem::generateIcosphere(float radius, int subdivisions) {
        SphereMesh mesh;
        mesh.radius = radius;

        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<glm::vec3> positions = {
                {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
                { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
                { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
        };

        for (auto& p : positions) {
            p = glm::normalize(p) * radius;
        }

        std::vector<uint32_t> indices = {
                0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
                1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
                3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
                4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1
        };

        for (int sub = 0; sub < subdivisions; ++sub) {
            std::vector<uint32_t> newIndices;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpointCache;

            auto getMidpoint = [&](uint32_t i0, uint32_t i1) -> uint32_t {
                auto key = std::make_pair(std::min(i0, i1), std::max(i0, i1));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) {
                    return it->second;
                }

                glm::vec3 mid = glm::normalize((positions[i0] + positions[i1]) * 0.5f) * radius;
                uint32_t newIdx = positions.size();
                positions.push_back(mid);
                midpointCache[key] = newIdx;
                return newIdx;
            };

            for (size_t i = 0; i < indices.size(); i += 3) {
                uint32_t v0 = indices[i];
                uint32_t v1 = indices[i + 1];
                uint32_t v2 = indices[i + 2];

                uint32_t m01 = getMidpoint(v0, v1);
                uint32_t m12 = getMidpoint(v1, v2);
                uint32_t m20 = getMidpoint(v2, v0);

                newIndices.insert(newIndices.end(), {
                        v0, m01, m20,
                        v1, m12, m01,
                        v2, m20, m12,
                        m01, m12, m20
                });
            }

            indices = newIndices;
        }

        mesh.vertices = positions;
        mesh.normals.reserve(positions.size());
        for (const auto& p : positions) {
            mesh.normals.push_back(glm::normalize(p));
        }
        mesh.indices = indices;

        return mesh;
    }


    void SpellSystem::clear()
    {
        std::lock_guard<std::mutex> lk(spellApplyMutex);
        activeSpells.clear();
        animatedVoxels.clear();
        animatedVoxelIndexMap.clear();
        nextAnimatedVoxelID = 1;
    }

    void SpellSystem::update(float dt)
    {
        TRACY_CPU_ZONE("SpellSystem:update()");
        pumpAsyncResults();
        updateSpells(dt);
    }

    void SpellSystem::castSphere(const glm::vec3& center, float radius, uint64_t material, float strength)
    {
        if (!spellCastAsync || !ctx.chunks) return;
        float searchRadius = radius * 1.5f;

        FormationParams params = FormationParams::Sphere(center, radius);

        SpellCastRequest req = buildSpellCastRequestSnapshot(center, searchRadius, material, strength, params);
        req.physicsEnabled = true;
        if (ctx.getCameraFront) req.launchDir = ctx.getCameraFront();
        req.launchSpeed = strength * 20.0f * VOXEL_SIZE;
        req.lifetime = 20.0f;

        spellCastAsync->enqueueOrReplaceQueued(std::move(req));
    }

    void SpellSystem::castWall(const glm::vec3& center, const glm::vec3& normal,
                               float width, float height, float thickness,
                               uint64_t material, float strength)
    {
        TRACY_CPU_ZONE("SpellSystem:CastWall()");

        if (!spellCastAsync || !ctx.chunks) return;


        FormationParams params = FormationParams::Wall(center, normal, width, height, thickness);

        float axis=glm::max(width, height);
        float searchRadiusWorld = glm::pow(axis,3);

        SpellCastRequest req = buildSpellCastRequestSnapshot(center, searchRadiusWorld, material, strength, params);

        req.physicsEnabled = false;
        if (ctx.getCameraFront) req.launchDir = ctx.getCameraFront();
        req.launchSpeed = strength * 20.0f * VOXEL_SIZE;
        req.lifetime = 20.0f;

        spellCastAsync->enqueueOrReplaceQueued(std::move(req));
    }

    SpellCastRequest SpellSystem::buildSpellCastRequestSnapshot(
            const glm::vec3& center,
            float searchRadius,
            uint64_t targetMaterial,
            float strength,
            const FormationParams& baseFormationParams
    )
    {
        TRACY_CPU_ZONE("buildSnapshot");

        SpellCastRequest req;
        req.center = center;
        req.searchRadius = searchRadius;
        req.targetMaterial = targetMaterial;
        req.strength = strength;
        req.baseFormationParams = baseFormationParams;

        auto chunks = ctx.chunks->getChunksInRadius(center, searchRadius);
        req.chunks.reserve(chunks.size());

        for (const auto& [coord, chunk] : chunks)
        {
            if (!chunk) continue;

            SpellCastRequest::ChunkSnapshot snap;
            snap.coord = coord;
            snap.chunkMinWorld = ctx.getChunkMin ? ctx.getChunkMin(coord) : glm::vec3(0);

            const size_t count = (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1);
            snap.voxelsLinear.resize(count);

            for (int x = 0; x <= CHUNK_SIZE; ++x)
                for (int y = 0; y <= CHUNK_SIZE; ++y)
                    for (int z = 0; z <= CHUNK_SIZE; ++z)
                    {
                        const size_t idx =
                                (size_t)x +
                                (size_t)y * (CHUNK_SIZE + 1) +
                                (size_t)z * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1);

                        snap.voxelsLinear[idx] = chunk->voxels[x][y][z];
                    }

            req.chunks.push_back(std::move(snap));
        }

        return req;
    }

    void SpellSystem::pumpAsyncResults()
    {
        TRACY_CPU_ZONE("SpellSystem:pumpResults()");

        if (!spellCastAsync) return;

        SpellCastResult r;
        while (spellCastAsync->tryPopCompleted(r))
        {
            if (!r.ok) {
                std::cout << "[SpellAsync] failed: " << r.debugMsg << "\n";
                continue;
            }

            std::lock_guard<std::mutex> lk(spellApplyMutex);

            // 1) Apply crater stamps (world mutation)
            if (!r.craterStamps.empty() && ctx.chunks) {
                CraterStampBatch::apply(ctx.chunks, r.craterStamps, -0.5f);
            }

            // 2) Mark touched chunks dirty
            if (ctx.markChunkModified) {
                for (const auto& c : r.touchedChunks) ctx.markChunkModified(c);
            }

            // 3) Spawn animated voxels + spell (assign IDs on main thread)
            SpellEffect spell = r.spell;
            spell.ID = nextSpellID++;   // ADD THIS

            for (auto& v : r.visualVoxels)
            {
                v.id = nextAnimatedVoxelID++;
                if (ctx.sampleNormalAtWorld) v.normal = ctx.sampleNormalAtWorld(v.currentPos);

                animatedVoxels.push_back(v);
                animatedVoxelIndexMap[v.id] = animatedVoxels.size() - 1;
                spell.animatedVoxelIDs.push_back(v.id);
            }

            activeSpells.push_back(std::move(spell));
        }
    }

    SpellEffect* SpellSystem::findSpellById(uint64_t id)
    {
        for (auto& s : activeSpells)
            if (s.ID == id) return &s;
        return nullptr;
    }

    const SpellEffect* SpellSystem::findSpellById(uint64_t id) const
    {
        for (const auto& s : activeSpells)
            if (s.ID == id) return &s;
        return nullptr;
    }

    void SpellSystem::updateSpells(float dt)
    {
        TRACY_CPU_ZONE("SpellSystem:LoopUpdate()");

        // Process async results first
        processAsyncFormations();
        processAsyncPhysics();

        const float kSlowSpeedThreshold = 0.0001f * VOXEL_SIZE;
        const float kSlowTimeToBurn     = 0.75f;
        const float kBurnDuration       = 3.0f;

        // Process spells - but we need to be careful with parallel processing
        // since spells might be removed

        // First pass: update all spells (can be parallel)
        #pragma omp parallel for schedule(static) if(activeSpells.size() > 50)
        for (int i = 0; i < (int)activeSpells.size(); ++i) {
            processSingleSpell(activeSpells[i], dt, kSlowSpeedThreshold, kSlowTimeToBurn, kBurnDuration);
        }

        // Second pass: remove marked spells (must be single-threaded)
        for (size_t i = 0; i < activeSpells.size(); ) {
            if (activeSpells[i].markForRemoval) {
                forceCleanupSpellAnimatedVoxels(activeSpells[i]);
                destroyPhysicsBodyForSpell(activeSpells[i]);
                activeSpells.erase(activeSpells.begin() + (ptrdiff_t)i);
            } else {
                ++i;
            }
        }

        // Clean up non-animating voxels
        cleanupNonAnimatingVoxels();
        cleanupExpiredSpells();
    }

    void scheduleSpellRemoval(SpellEffect &effect);

    void SpellSystem::processSingleSpell(SpellEffect& s, float dt,
                                         float speedThreshold, float timeToBurn, float burnDuration)
    {
        TRACY_CPU_ZONE("SpellSystem:Process One Spell()");

        if (s.lifetime > 0.0f) s.creationTime += dt;

        // Update physics body reference (read-only for physics, safe)
        if (s.physicsBodyId != 0 && ctx.physics) {
            s.physicsBody = ctx.physics->getBodyById(s.physicsBodyId);
            if (s.physicsBody && s.isPhysicsEnabled) {
                s.center = s.physicsBody->position;
                s.formationParams.center = s.physicsBody->position;
            }
        } else {
            s.physicsBody = nullptr;
        }

        // Mark for removal if physics body disappeared
        if (s.physicsBodyId != 0 && !s.physicsBody) {
            s.markForRemoval = true;
            return;
        }

        // Handle burning spells
        if (s.burn.active) {
            s.burn.center = s.center;
            s.burn.t += dt;

            if (burn01(s.burn.t, s.burn.duration) >= 1.0f) {
                s.markForRemoval = true;
                return;
            }
            return;
        }

        // Check if spell should start burning
        const bool tooSmall = isSpellTooSmall(s);
        const bool tooSlowNow = isSpellTooSlowNow(s, speedThreshold);
        s.burn.slowAccum = tooSlowNow ? (s.burn.slowAccum + dt) : 0.0f;
        const bool tooSlowLong = (s.burn.slowAccum >= timeToBurn);

        if (tooSmall || tooSlowLong) {
            const float r = glm::max(s.formationParams.getBoundingRadius(), 1.0f * VOXEL_SIZE);
            startSpellBurn(s, r, burnDuration);
            return;
        }

        // Process animated voxels (the heavy part)
        processAnimatedVoxelsForSpell(s, dt);
    }

    void scheduleSpellRemoval(SpellEffect &effect) {

    }

    void SpellSystem::processAnimatedVoxelsForSpell(SpellEffect& s, float dt)
    {
        TRACY_CPU_ZONE("SpellSystem:AnimVoxelProcess()");

        // Batch process voxel arrivals
        std::vector<uint64_t> newlyArrivedIDs;
        newlyArrivedIDs.reserve(s.animatedVoxelIDs.size() / 4); // Estimate

        // Use local cache for hot path
        auto& localVoxelMap = animatedVoxelIndexMap;
        auto& localVoxels = animatedVoxels;

        for (uint64_t id : s.animatedVoxelIDs) {
            auto itIndex = localVoxelMap.find(id);

            if (itIndex == localVoxelMap.end()) {
                newlyArrivedIDs.push_back(id);
                continue;
            }

            size_t idx = itIndex->second;
            if (idx >= localVoxels.size()) {
                newlyArrivedIDs.push_back(id);
                continue;
            }

            AnimatedVoxel &voxel = localVoxels[idx];

            if (voxel.isAnimating) {
                glm::vec3 toTarget = voxel.targetPos - voxel.currentPos;
                float distanceSq = glm::dot(toTarget, toTarget);
                float threshold = VOXEL_SIZE * VOXEL_SIZE;

                if (distanceSq < threshold) {
                    voxel.isAnimating = false;
                    voxel.hasArrived = true;
                    newlyArrivedIDs.push_back(id);
                } else {
                    float distance = std::sqrt(distanceSq);
                    float speed = voxel.animationSpeed;
                    float slowdown = glm::clamp(distance*distance / 2.0f, 0.75f, 3.0f);
                    voxel.velocity = (toTarget / (VOXEL_SIZE * CHUNK_SIZE)) * speed * slowdown * 4.0f;
                    voxel.currentPos += voxel.velocity * dt;
                }
            } else {
                newlyArrivedIDs.push_back(id);
            }
        }

        // Create geometry when enough voxels have arrived
        if (!newlyArrivedIDs.empty() && !s.geometryCreated) {
            // Cache arrival count to avoid multiple loops
            static thread_local std::vector<bool> arrivalCache;
            arrivalCache.assign(s.animatedVoxelIDs.size(), false);

            int arrivedCount = 0;
            int total = (int)s.animatedVoxelIDs.size();

            for (size_t i = 0; i < s.animatedVoxelIDs.size(); ++i) {
                uint64_t id = s.animatedVoxelIDs[i];
                auto jt = animatedVoxelIndexMap.find(id);
                if (jt == animatedVoxelIndexMap.end()) {
                    ++arrivedCount;
                    arrivalCache[i] = true;
                } else {
                    AnimatedVoxel &v = animatedVoxels[jt->second];
                    if (!v.isAnimating) {
                        ++arrivedCount;
                        arrivalCache[i] = true;
                    }
                }
            }

            float arrivalRatio = total ? (float)arrivedCount / (float)total : 1.0f;

            if (arrivalRatio >= 0.8f) {
                // Queue async formation creation
                queueAsyncFormationCreation(s, arrivalRatio);
                s.geometryCreated = true;
            } else if (arrivalRatio > 0.0005f) {
                // Fast partial formation (can be async too)
                createPartialFormation(s, arrivalRatio);
            }
        }

        // Queue physics creation
        if (s.geometryCreated && s.isPhysicsEnabled && s.physicsBody == nullptr) {
            queueAsyncPhysicsCreation(s);
        }

        // Cleanup voxels if all arrived (deferred)
        if (s.geometryCreated && s.isPhysicsEnabled && !s.voxelsCleaned) {
            int stillAnimating = 0;
            for (uint64_t id : s.animatedVoxelIDs) {
                auto jt = animatedVoxelIndexMap.find(id);
                if (jt != animatedVoxelIndexMap.end()) {
                    AnimatedVoxel &v = animatedVoxels[jt->second];
                    if (v.isAnimating) ++stillAnimating;
                    if (stillAnimating > 100) break; // Early exit
                }
            }

            if (stillAnimating == 0) {
                std::cout << "All voxels arrived. Cleaning voxel data.\n";
                s.voxelsCleaned = true;

                // Clear animated voxel IDs (they're no longer needed)
                s.animatedVoxelIDs.clear();
            }
        }
    }

    void SpellSystem::cleanupExpiredSpells() {
        for (auto spellIt = activeSpells.begin(); spellIt != activeSpells.end(); ) {
            gl3::VoxelPhysicsBody* body = nullptr;
            if (spellIt->physicsBodyId != 0 && ctx.physics) {
                body = ctx.physics->getBodyById(spellIt->physicsBodyId);
            }

            if (spellIt->lifetime > 0) {
                float age = spellIt->creationTime;
                if (age > spellIt->lifetime) {
                    spellIt->markForRemoval = true;
                }
            }

            if (spellIt->physicsBody != nullptr && glm::length(spellIt->physicsBody->velocity) < 0.5f) {
                spellIt->markForRemoval = true;
            }

            if (spellIt->isPhysicsEnabled && spellIt->voxelsCleaned) {
                float distanceToMap = glm::distance(spellIt->center, glm::vec3(0)); // Placeholder
                if (distanceToMap > 1000.0f * VOXEL_SIZE) {
                    spellIt->markForRemoval = true;
                }
            }

            ++spellIt;
        }
    }

    void SpellSystem::destroyPhysicsBodyForSpell(gl3::SpellEffect& spell) {
        if (spell.physicsBodyId != 0 && ctx.physics) {
            gl3::VoxelPhysicsBody* body = ctx.physics->getBodyById(spell.physicsBodyId);

            if (body) {
                body->userData = nullptr;
            }

            // Create formation before removing body
            const float safeCollectedProxy = (float)spell.physicsMesh.vertexCount;
            createSpellFormation(
                    spell.center,
                    spell.formationParams,
                    spell.strength,
                    spell.targetMaterial,
                    spell.formationColor,
                    (size_t)safeCollectedProxy,
                    spell.dominantType
            );

            ctx.physics->removeBody(spell.physicsBodyId);
            spell.physicsBodyId = 0;
        }

        spell.physicsBody = nullptr;
        spell.isPhysicsEnabled = false;

        // Clean up mesh data
        if (spell.physicsMesh.vao) {
            glDeleteVertexArrays(1, &spell.physicsMesh.vao);
            glDeleteBuffers(1, &spell.physicsMesh.vbo);
            spell.physicsMesh.vao = 0;
            spell.physicsMesh.vbo = 0;
        }
        spell.physicsMesh.isValid = false;
        spell.physicsMesh.vertexCount = 0;
    }

    void SpellSystem::forceCleanupSpellAnimatedVoxels(gl3::SpellEffect& s) {
        for (uint64_t id : s.animatedVoxelIDs) {
            auto it = animatedVoxelIndexMap.find(id);
            if (it == animatedVoxelIndexMap.end()) continue;
            size_t idx = it->second;
            if (idx >= animatedVoxels.size()) continue;
            animatedVoxels[idx].isAnimating = false;
            animatedVoxels[idx].hasArrived = true;
        }
        s.animatedVoxelIDs.clear();
    }

    void SpellSystem::queueAsyncFormationCreation(const SpellEffect& s, float arrivalRatio)
    {
        // Copy what we need
        const glm::vec3 center = s.center;
        const uint64_t material = s.targetMaterial;
        const glm::vec3 color = s.formationColor;
        const uint8_t dominantType = s.dominantType;

        FormationParams paramsCopy = s.formationParams;
        paramsCopy.center = center;

        switch(paramsCopy.type) {
            case FormationType::SPHERE:   paramsCopy.radius *= (arrivalRatio * 0.7f + 0.3f); break;
            case FormationType::PLATFORM:
            case FormationType::WALL:
            case FormationType::CUBE:
                paramsCopy.sizeX *= (arrivalRatio * 0.7f + 0.3f);
                paramsCopy.sizeY *= (arrivalRatio * 0.7f + 0.3f);
                paramsCopy.sizeZ *= (arrivalRatio * 0.7f + 0.3f);
                break;
            case FormationType::CYLINDER:
                paramsCopy.radius *= (arrivalRatio * 0.7f + 0.3f);
                paramsCopy.sizeY *= (arrivalRatio * 0.7f + 0.3f);
                break;
            default: break;
        }

        std::weak_ptr<SpellSystem> weakSelf = shared_from_this();

        std::async(std::launch::async,
                   [weakSelf, center, material, color, dominantType, paramsCopy]() mutable
                   {
                       // If SpellSystem was destroyed/reset -> do nothing
                       auto self = weakSelf.lock();
                       if (!self) return;

                       WorldPlanet formation;
                       formation.worldPos = center;
                       formation.color = color;
                       formation.type = dominantType;
                       formation.radius = paramsCopy.getBoundingRadius();

                       if (self->ctx.mainThreadDispatcher) {
                           // IMPORTANT: also avoid capturing raw `this` here
                           self->ctx.mainThreadDispatcher(
                                   [weakSelf, formation, material, paramsCopy]() mutable
                                   {
                                       auto self2 = weakSelf.lock();
                                       if (!self2) return;
                                       self2->carveFormationWithSDF(formation, material, paramsCopy);
                                   }
                           );
                       }
                   }
        );
    }

    void SpellSystem::queueAsyncPhysicsCreation(SpellEffect& s)
    {
        TRACY_CPU_ZONE("SpellSystem:Async Physics()");

        if (!ctx.mainThreadDispatcher) {
            // If you don't have a dispatcher, safest is: do it synchronously on main thread.
            // (Or just return.)
            return;
        }

        // Capture only stable data for identifying the spell later
        const uint64_t spellId = s.ID;

        ctx.mainThreadDispatcher([this, spellId]() {

            // Find the live spell object
            SpellEffect* live = nullptr;
            for (auto& sp : activeSpells) {
                if (sp.ID == spellId) { live = &sp; break; }
            }
            if (!live) return;

            // MAIN THREAD: read radius from the live spell (real data)
            const float effectiveRadius = live->formationParams.getBoundingRadius();

            // MAIN THREAD: OpenGL readback
            GpuTrianglesReadback tri = readbackTrianglesMainThread(*live);

            // BACKGROUND THREAD: CPU-only heavy work
            std::async(std::launch::async,
                       [this, spellId, tri = std::move(tri), effectiveRadius]() mutable {

                           glm::vec3 extents;

                           if (!tri.vertsLocal.empty()) {
                               glm::vec3 mn = tri.vertsLocal[0];
                               glm::vec3 mx = tri.vertsLocal[0];
                               for (const auto& v : tri.vertsLocal) {
                                   mn = glm::min(mn, v);
                                   mx = glm::max(mx, v);
                               }
                               extents = (mx - mn) * 0.5f + glm::vec3(VOXEL_SIZE);
                           } else {
                               // Fallback if no triangles were read back:
                               extents = glm::vec3(effectiveRadius) + glm::vec3(VOXEL_SIZE);
                           }

                           // Schedule creation/merge back on MAIN THREAD (safe)

                           if (ctx.mainThreadDispatcher) {
                               ctx.mainThreadDispatcher([this, spellId, extents]() {
                                   SpellEffect* live2 = nullptr;
                                   for (auto& sp : activeSpells) {
                                       if (sp.ID == spellId) { live2 = &sp; break; }
                                   }
                                   if (!live2) return;

                        if (!ctx.physics || live2->physicsBody != nullptr) return;

                        // Create body on main thread (safe)
                        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
                        float mass = (float)live2->animatedVoxelIDs.size() * voxelVolume * 0.5f;
                        mass = glm::clamp(mass, 1.0f, 1000.0f);

                        live2->physicsBody = ctx.physics->createBody(
                                live2->center,
                                mass,
                                VoxelPhysicsBody::ShapeType::BOX,
                                extents
                        );
                        live2->physicsBodyId = live2->physicsBody ? live2->physicsBody->id : 0;

                        if (live2->physicsBody) {
                            // IMPORTANT: do NOT store &spell if you ever erase/move spells.
                            // Better: store spellId instead of pointer.
                            live2->physicsBody->userData =
                                    reinterpret_cast<void*>((uintptr_t)live2->ID);

                            live2->physicsBody->velocity = live2->initialVelocity;
                            live2->isPhysicsEnabled = true;

                            initSpellDestructibleVolume(*live2);
                            rebuildDestructibleMeshIfNeeded(live2->destruct);
                            live2->physicsBody->renderMesh = &live2->destruct.mesh;

                            removeFormationVoxels(*live2);
                        }
                    });
                }
            });
        });
    }

    void SpellSystem::processAsyncFormations()
    {
        TRACY_CPU_ZONE("SpellSystem:process Async Formations()");

        std::lock_guard<std::mutex> lock(queueMutex);

        // We need to process the queue differently since AsyncFormationRequest is now movable only
        std::queue<AsyncFormationRequest> completed;

        while (!formationQueue.empty()) {
            auto& request = formationQueue.front();
            if (request.result.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                try {
                    request.result.get(); // Check for exceptions
                    completed.push(std::move(request));
                } catch (const std::exception& e) {
                    std::cout << "Async formation failed: " << e.what() << "\n";
                }
                formationQueue.pop();
            } else {
                break; // Don't block main thread
            }
        }
    }

    bool SpellSystem::isSpellTooSmall(const gl3::SpellEffect& s) {
        float r = s.formationParams.getBoundingRadius();
        return (r < (0.05f * gl3::VOXEL_SIZE));
    }

    bool SpellSystem::isSpellTooSlowNow(const gl3::SpellEffect& s, float speedThreshold) {
        if (!s.physicsBody) return false;
        return glm::length(s.physicsBody->velocity) < speedThreshold;
    }

    float SpellSystem::burn01(float t, float duration) {
        return glm::clamp(t / duration, 0.0f, 1.0f);
    }

    void SpellSystem::startSpellBurn(SpellEffect& s, float radius, float duration) {
        s.burn.active = true;
        s.burn.t = 0.0f;
        s.burn.duration = duration;
        s.burn.radius = radius;
        s.burn.center = s.center;
    }

    void SpellSystem::cleanupNonAnimatingVoxels()
    {
        TRACY_CPU_ZONE("SpellSystem:CleanupNonAnimatingVoxels");

        for (size_t vi = 0; vi < animatedVoxels.size(); ) {
            if (!animatedVoxels[vi].isAnimating) {
                uint64_t removedID = animatedVoxels[vi].id;
                size_t last = animatedVoxels.size() - 1;
                if (vi != last) {
                    animatedVoxels[vi] = animatedVoxels[last];
                    animatedVoxelIndexMap[animatedVoxels[vi].id] = vi;
                }
                animatedVoxels.pop_back();
                animatedVoxelIndexMap.erase(removedID);
            } else {
                ++vi;
            }
        }
    }

    void SpellSystem::processAsyncPhysics()
    {
        TRACY_CPU_ZONE("SpellSystem: Process Async Physics()");

        std::lock_guard<std::mutex> lock(physicsResultMutex);

        std::vector<SpellEffect> resultsToMerge;
        resultsToMerge.swap(pendingPhysicsResults);

        for (auto& result : resultsToMerge) {
            bool found = false;
            for (auto& spell : activeSpells) {
                if (spell.ID == result.ID) {
                    // Safely merge data
                    spell.physicsBody = result.physicsBody;
                    spell.physicsBodyId = result.physicsBodyId;
                    spell.isPhysicsEnabled = result.isPhysicsEnabled;
                    spell.voxelsCleaned = result.voxelsCleaned;

                    // Copy mesh data carefully
                    if (result.physicsMesh.isValid) {
                        spell.physicsMesh = result.physicsMesh;
                    }

                    if (result.destruct.mesh.isValid) {
                        spell.destruct = result.destruct;
                    }

                    found = true;
                    break;
                }
            }

            if (!found) {
                // Spell no longer exists, clean up physics body
                if (result.physicsBodyId != 0 && ctx.physics) {
                    ctx.physics->removeBody(result.physicsBodyId);
                }
            }
        }
    }

    void SpellSystem::createPartialFormation(const SpellEffect& spell, float completionRatio)
    {
        queueAsyncFormationCreation(spell, completionRatio);
    }

    SpellSystem::GpuTrianglesReadback SpellSystem::readbackTrianglesMainThread(const SpellEffect& spell)
    {
        TRACY_CPU_ZONE("ReadbackTriangles");

        GpuTrianglesReadback out;

        if (!ctx.worldToChunk || !ctx.chunks) return out;

        const float effectiveRadius = spell.formationParams.getBoundingRadius();

        int minCX = ctx.worldToChunk(spell.center.x - effectiveRadius);
        int maxCX = ctx.worldToChunk(spell.center.x + effectiveRadius);
        int minCY = ctx.worldToChunk(spell.center.y - effectiveRadius);
        int maxCY = ctx.worldToChunk(spell.center.y + effectiveRadius);
        int minCZ = ctx.worldToChunk(spell.center.z - effectiveRadius);
        int maxCZ = ctx.worldToChunk(spell.center.z + effectiveRadius);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = ctx.chunks->getChunk(coord);
                    if (!chunk || !chunk->gpuCache.isValid) continue;
                    if (chunk->gpuCache.vertexCount == 0) continue;

                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);

                    const size_t vcount   = chunk->gpuCache.vertexCount;
                    const size_t byteSize = vcount * sizeof(OutVertex);

                    void* mapPtr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, byteSize, GL_MAP_READ_BIT);
                    if (!mapPtr) {
                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                        continue;
                    }

                    OutVertex* ov = reinterpret_cast<OutVertex*>(mapPtr);

                    // Copy *all* triangles (or you can keep your radius checks here).
                    // IMPORTANT: we only COPY data; no heavy processing here.
                    out.vertsLocal.reserve(out.vertsLocal.size() + vcount);
                    out.normals.reserve(out.normals.size() + vcount);
                    out.colors.reserve(out.colors.size() + vcount);

                    for (size_t i = 0; i < vcount; ++i)
                    {
                        glm::vec3 wpos(ov[i].pos.x, ov[i].pos.y, ov[i].pos.z);
                        glm::vec3 local = wpos - spell.center;

                        out.vertsLocal.push_back(local);
                        out.normals.push_back(glm::vec3(ov[i].normal.x, ov[i].normal.y, ov[i].normal.z));
                        out.colors.push_back(glm::vec3(ov[i].color.x,  ov[i].color.y,  ov[i].color.z));
                    }

                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                }

        return out;
    }

    void SpellSystem::createPhysicsBodyForSpell(SpellEffect& spell) {
        if (!ctx.physics || spell.physicsBody != nullptr) return;

        if (spell.formationParams.type == FormationType::SPHERE) {
            float radius = spell.formationParams.radius;

            float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
            float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
            mass = glm::clamp(mass, 1.0f, 1000.0f);

            spell.physicsBody = ctx.physics->createBody(
                    spell.center,
                    mass,
                    VoxelPhysicsBody::ShapeType::SPHERE,
                    glm::vec3(radius)
            );
            spell.physicsBodyId = spell.physicsBody ? spell.physicsBody->id : 0;

            if (spell.physicsBody) {
                spell.physicsBody->userData = reinterpret_cast<void*>((uintptr_t)spell.ID);
                spell.physicsBody->velocity = spell.initialVelocity;
                spell.physicsBody->orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

                initSpellDestructibleVolume(spell);
                rebuildDestructibleMeshIfNeeded(spell.destruct);
                spell.physicsBody->renderMesh = &spell.destruct.mesh;
                spell.isPhysicsEnabled = true;
            }

            removeFormationVoxels(spell);
            return;
        }

        float effectiveRadius = spell.formationParams.getBoundingRadius();

        struct TargetKey {
            int64_t x, y, z;
            bool operator==(const TargetKey& other) const {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct TargetKeyHash {
            std::size_t operator()(const TargetKey& k) const {
                return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
            }
        };

        std::unordered_map<TargetKey, bool, TargetKeyHash> expectedVoxels;
        std::unordered_map<TargetKey, bool, TargetKeyHash> boundaryVoxels;

        for (uint64_t id : spell.animatedVoxelIDs) {
            auto it = animatedVoxelIndexMap.find(id);
            if (it != animatedVoxelIndexMap.end()) {
                const AnimatedVoxel& voxel = animatedVoxels[it->second];
                TargetKey key{
                        static_cast<int64_t>(std::round(voxel.targetPos.x / VOXEL_SIZE)),
                        static_cast<int64_t>(std::round(voxel.targetPos.y / VOXEL_SIZE)),
                        static_cast<int64_t>(std::round(voxel.targetPos.z / VOXEL_SIZE))
                };
                expectedVoxels[key] = true;
            }
        }

        if (expectedVoxels.empty()) {
            std::cout << "No expected voxels found, generating from formation SDF\n";

            float sampleRadius = effectiveRadius;
            float step = VOXEL_SIZE;

            for (float x = -sampleRadius; x <= sampleRadius; x += step) {
                for (float y = -sampleRadius; y <= sampleRadius; y += step) {
                    for (float z = -sampleRadius; z <= sampleRadius; z += step) {
                        glm::vec3 samplePos = spell.center + glm::vec3(x, y, z);
                        float sdf = spell.formationParams.evaluate(samplePos);

                        if (sdf >= 0.0f) {
                            TargetKey key{
                                    static_cast<int64_t>(std::round(samplePos.x / VOXEL_SIZE)),
                                    static_cast<int64_t>(std::round(samplePos.y / VOXEL_SIZE)),
                                    static_cast<int64_t>(std::round(samplePos.z / VOXEL_SIZE))
                            };
                            expectedVoxels[key] = true;
                        }
                    }
                }
            }

            std::cout << "Generated " << expectedVoxels.size() << " expected voxels from SDF\n";
        }

        for (const auto& [key, _] : expectedVoxels) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        TargetKey neighbor{key.x + dx, key.y + dy, key.z + dz};
                        boundaryVoxels[neighbor] = true;
                    }
                }
            }
        }

        std::cout << "Expected formation has " << expectedVoxels.size()
                  << " voxels, with " << boundaryVoxels.size() << " boundary voxels\n";

        std::vector<glm::vec3> triangleVerts;
        std::vector<glm::vec3> triangleNormals;
        std::vector<glm::vec3> triangleColors;

        if (!ctx.worldToChunk || !ctx.chunks) return;

        int minCX = ctx.worldToChunk(spell.center.x - effectiveRadius);
        int maxCX = ctx.worldToChunk(spell.center.x + effectiveRadius);
        int minCY = ctx.worldToChunk(spell.center.y - effectiveRadius);
        int maxCY = ctx.worldToChunk(spell.center.y + effectiveRadius);
        int minCZ = ctx.worldToChunk(spell.center.z - effectiveRadius);
        int maxCZ = ctx.worldToChunk(spell.center.z + effectiveRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = ctx.chunks->getChunk(coord);
                    if (!chunk || !chunk->gpuCache.isValid) continue;
                    if (chunk->gpuCache.vertexCount == 0) continue;

                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);
                    size_t vcount = chunk->gpuCache.vertexCount;
                    size_t byteSize = vcount * sizeof(OutVertex);

                    if (byteSize > 0) {
                        void* mapPtr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                        byteSize, GL_MAP_READ_BIT);
                        if (mapPtr) {
                            OutVertex* ov = reinterpret_cast<OutVertex*>(mapPtr);

                            for (size_t vi = 0; vi < vcount; vi += 3) {
                                glm::vec3 v0(ov[vi].pos.x, ov[vi].pos.y, ov[vi].pos.z);
                                glm::vec3 v1(ov[vi + 1].pos.x, ov[vi + 1].pos.y, ov[vi + 1].pos.z);
                                glm::vec3 v2(ov[vi + 2].pos.x, ov[vi + 2].pos.y, ov[vi + 2].pos.z);

                                glm::vec3 center = (v0 + v1 + v2) / 3.0f;

                                float distToSpell = glm::length(center - spell.center);
                                if (distToSpell > effectiveRadius + VOXEL_SIZE * 2.0f) {
                                    continue;
                                }

                                bool shouldKeep = false;

                                std::vector<glm::vec3> samplePoints = {v0, v1, v2, center};

                                for (const auto& point : samplePoints) {
                                    TargetKey pointKey{
                                            static_cast<int64_t>(std::round(point.x / VOXEL_SIZE)),
                                            static_cast<int64_t>(std::round(point.y / VOXEL_SIZE)),
                                            static_cast<int64_t>(std::round(point.z / VOXEL_SIZE))
                                    };

                                    if (expectedVoxels.find(pointKey) != expectedVoxels.end() ||
                                        boundaryVoxels.find(pointKey) != boundaryVoxels.end()) {
                                        shouldKeep = true;
                                        break;
                                    }

                                    for (int dx = -1; dx <= 1 && !shouldKeep; ++dx) {
                                        for (int dy = -1; dy <= 1 && !shouldKeep; ++dy) {
                                            for (int dz = -1; dz <= 1 && !shouldKeep; ++dz) {
                                                TargetKey neighbor{pointKey.x + dx, pointKey.y + dy, pointKey.z + dz};
                                                if (expectedVoxels.find(neighbor) != expectedVoxels.end()) {
                                                    shouldKeep = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (!shouldKeep) {
                                    float sdf = spell.formationParams.evaluate(center);
                                    if (std::abs(sdf) < VOXEL_SIZE * 1.5f) {
                                        shouldKeep = true;
                                    }
                                }

                                if (shouldKeep) {
                                    for (int j = 0; j < 3; ++j) {
                                        glm::vec3 worldVert(ov[vi + j].pos.x, ov[vi + j].pos.y, ov[vi + j].pos.z);
                                        glm::vec3 localVert = worldVert - spell.center;
                                        triangleVerts.push_back(localVert);
                                        triangleNormals.push_back(glm::vec3(ov[vi + j].normal.x, ov[vi + j].normal.y, ov[vi + j].normal.z));
                                        triangleColors.push_back(glm::vec3(ov[vi + j].color.x, ov[vi + j].color.y, ov[vi + j].color.z));
                                    }
                                }
                            }

                            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                        }
                    }
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                }
            }
        }

        if (triangleVerts.empty()) {
            std::cout << "createPhysicsBody: No triangles found for spell, falling back to bounding box\n";

            glm::vec3 extents;
            switch(spell.formationParams.type) {
                case FormationType::CUBE:
                    extents = glm::vec3(
                            spell.formationParams.sizeX * 0.5f,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.sizeZ * 0.5f
                    );
                    break;
                case FormationType::PLATFORM:
                case FormationType::WALL:
                    extents = glm::vec3(
                            spell.formationParams.sizeX * 0.5f,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.sizeZ * 0.5f
                    );
                    break;
                case FormationType::CYLINDER:
                    extents = glm::vec3(
                            spell.formationParams.radius,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.radius
                    );
                    break;
                default:
                    extents = glm::vec3(effectiveRadius);
                    break;
            }

            extents += glm::vec3(VOXEL_SIZE);

            float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
            float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
            mass = glm::clamp(mass, 1.0f, 1000.0f);

            spell.physicsBody = ctx.physics->createBody(
                    spell.center,
                    mass,
                    VoxelPhysicsBody::ShapeType::BOX,
                    extents
            );
            spell.physicsBodyId = spell.physicsBody ? spell.physicsBody->id : 0;

            if (spell.physicsBody) {
                spell.physicsBody->userData = reinterpret_cast<void*>((uintptr_t)spell.ID);
                spell.physicsBody->velocity = spell.initialVelocity;
                spell.physicsBody->orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

                initSpellDestructibleVolume(spell);
                rebuildDestructibleMeshIfNeeded(spell.destruct);
                spell.physicsBody->renderMesh = &spell.destruct.mesh;
                spell.isPhysicsEnabled = true;
            }

            removeFormationVoxels(spell);
            return;
        }

        std::cout << "Found " << triangleVerts.size() / 3 << " triangles for physics body\n";

        glm::vec3 minBound = triangleVerts[0];
        glm::vec3 maxBound = triangleVerts[0];
        for (const auto& v : triangleVerts) {
            minBound = glm::min(minBound, v);
            maxBound = glm::max(maxBound, v);
        }

        glm::vec3 geomExtents = (maxBound - minBound) * 0.5f;
        geomExtents += glm::vec3(VOXEL_SIZE);

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
        mass = glm::clamp(mass, 1.0f, 1000.0f);

        spell.physicsBody = ctx.physics->createBody(
                spell.center,
                mass,
                VoxelPhysicsBody::ShapeType::BOX,
                geomExtents
        );
        spell.physicsBodyId = spell.physicsBody ? spell.physicsBody->id : 0;

        if (spell.physicsBody) {
            spell.physicsBody->userData = reinterpret_cast<void*>((uintptr_t)spell.ID);
            spell.physicsBody->velocity = spell.initialVelocity;

            spell.physicsBody->orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

            initSpellDestructibleVolume(spell);
            rebuildDestructibleMeshIfNeeded(spell.destruct);
            spell.physicsBody->renderMesh = &spell.destruct.mesh;
            spell.isPhysicsEnabled = true;

            std::cout << "Physics body created at spell.center: ("
                      << spell.center.x << "," << spell.center.y << "," << spell.center.z << ")\n";
            std::cout << "Extents: (" << geomExtents.x << "," << geomExtents.y << "," << geomExtents.z << ")\n";
        }

        removeFormationVoxels(spell);
    }

    void SpellSystem::generateMeshAsync(Chunk* chunk)
    {
        if (!chunk || !ctx.generateChunkMesh) return;

        // Delegate to main thread dispatcher if available
        if (ctx.mainThreadDispatcher) {
            ctx.mainThreadDispatcher([this, chunk]() {
                if (chunk && chunk->meshDirty && ctx.generateChunkMesh) {
                    ctx.generateChunkMesh(chunk);
                }
            });
        } else if (chunk->meshDirty && ctx.generateChunkMesh) {
            // Fallback to direct call (will block)
            ctx.generateChunkMesh(chunk);
        }
    }

    void SpellSystem::scheduleSpellRemoval(SpellEffect& effect)
    {
        // Defer removal to avoid iterator invalidation
        effect.markForRemoval = true;
    }

    void SpellSystem::mergePhysicsBodyResult(const SpellEffect& result)
    {
        std::lock_guard<std::mutex> lock(physicsResultMutex);

        // Store the result for processing on main thread
        pendingPhysicsResults.push_back(result);
    }

    void SpellSystem::carveFormationWithSDF(const WorldPlanet& formation, uint64_t material,
                                            const FormationParams& params) {
        if (!ctx.chunks || !ctx.worldToChunk || !ctx.getChunkMin || !ctx.markChunkModified) return;
        TRACY_CPU_ZONE("Carve");

        glm::vec3 center = formation.worldPos;
        float boundingRadius = params.getBoundingRadius();

        int minCX = ctx.worldToChunk(center.x - boundingRadius);
        int maxCX = ctx.worldToChunk(center.x + boundingRadius);
        int minCY = ctx.worldToChunk(center.y - boundingRadius);
        int maxCY = ctx.worldToChunk(center.y + boundingRadius);
        int minCZ = ctx.worldToChunk(center.z - boundingRadius);
        int maxCZ = ctx.worldToChunk(center.z + boundingRadius);

        const float voxelSize = VOXEL_SIZE;
        const int chunkSize = CHUNK_SIZE;
        const float voxelSizeInv = 1.0f / voxelSize;

        // Collect chunks for parallel processing
        struct ChunkWork {
            Chunk* chunk;
            ChunkCoord coord;
            glm::vec3 origin;
        };
        std::vector<ChunkWork> chunksToProcess;
        chunksToProcess.reserve((maxCX - minCX + 1) * (maxCY - minCY + 1) * (maxCZ - minCZ + 1));

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = ctx.chunks->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkOrigin = ctx.getChunkMin(coord);
                    chunksToProcess.push_back({chunk, coord, chunkOrigin});
                }
            }
        }

        // Process based on formation type
        if (params.type == FormationType::SPHERE) {
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)chunksToProcess.size(); ++i) {
                auto& work = chunksToProcess[i];
                carveSphereInChunk(work.chunk, work.origin, center, params.radius,
                                   formation, material);
            }
        } else if (params.type == FormationType::CUBE ||
                   params.type == FormationType::PLATFORM ||
                   params.type == FormationType::WALL) {
            // Calculate box bounds
            glm::vec3 halfSize(params.sizeX * 0.5f, params.sizeY * 0.5f, params.sizeZ * 0.5f);
            glm::vec3 minBounds = center - halfSize;
            glm::vec3 maxBounds = center + halfSize;

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)chunksToProcess.size(); ++i) {
                auto& work = chunksToProcess[i];
                carveBoxInChunk(work.chunk, work.origin, minBounds, maxBounds,
                                formation, material);
            }
        }

        // Mark chunks dirty on main thread
        for (auto& work : chunksToProcess) {
            if (work.chunk->meshDirty) {
                work.chunk->lightingDirty = true;
                if (ctx.markChunkModified) {
                    ctx.markChunkModified(work.coord);
                }
            }
        }
    }

// Optimized sphere carving
    void SpellSystem::carveSphereInChunk(Chunk* chunk, const glm::vec3& chunkOrigin,
                                         const glm::vec3& center, float radius,
                                         const WorldPlanet& formation, uint64_t material) {
        const float voxelSize = VOXEL_SIZE;
        const int chunkSize = CHUNK_SIZE;
        const float radiusSq = radius * radius;

        // Transform chunk origin to local space
        glm::vec3 localOrigin = chunkOrigin - center;

        // Pre-calculate bounds in local space
        int minX = std::max(0, (int)((-radius - localOrigin.x) / voxelSize));
        int maxX = std::min(chunkSize, (int)((radius - localOrigin.x) / voxelSize) + 1);
        int minY = std::max(0, (int)((-radius - localOrigin.y) / voxelSize));
        int maxY = std::min(chunkSize, (int)((radius - localOrigin.y) / voxelSize) + 1);
        int minZ = std::max(0, (int)((-radius - localOrigin.z) / voxelSize));
        int maxZ = std::min(chunkSize, (int)((radius - localOrigin.z) / voxelSize) + 1);

        bool chunkTouched = false;

        // Fast sphere carving with bounds checking
        for (int lz = minZ; lz <= maxZ; ++lz) {
            float z = localOrigin.z + (float)lz * voxelSize;
            float zSq = z * z;

            for (int ly = minY; ly <= maxY; ++ly) {
                float y = localOrigin.y + (float)ly * voxelSize;
                float ySq = y * y;
                float yzSq = ySq + zSq;
                if (yzSq > radiusSq) continue;

                for (int lx = minX; lx <= maxX; ++lx) {
                    float x = localOrigin.x + (float)lx * voxelSize;
                    float distSq = yzSq + x * x;

                    if (distSq <= radiusSq) {
                        // Calculate density (falloff from center)
                        float density = 1.0f - std::sqrt(distSq) / radius;
                        // Optional: add some noise for natural look
                        // density = std::max(0.0f, std::min(1.0f, density));

                        auto& voxel = chunk->voxels[lx][ly][lz];
                        if (density > voxel.density) {
                            voxel.density = density;
                            voxel.type = formation.type;
                            voxel.color = formation.color;
                            voxel.material = material;
                            chunkTouched = true;
                        }
                    }
                }
            }
        }

        if (chunkTouched) {
            chunk->meshDirty = true;
        }
    }

// Optimized box carving
    void SpellSystem::carveBoxInChunk(Chunk* chunk, const glm::vec3& chunkOrigin,
                                      const glm::vec3& minBounds, const glm::vec3& maxBounds,
                                      const WorldPlanet& formation, uint64_t material) {
        const float voxelSize = VOXEL_SIZE;
        const int chunkSize = CHUNK_SIZE;

        // Calculate chunk-local bounds
        int minX = std::max(0, (int)((minBounds.x - chunkOrigin.x) / voxelSize));
        int maxX = std::min(chunkSize, (int)((maxBounds.x - chunkOrigin.x) / voxelSize) + 1);
        int minY = std::max(0, (int)((minBounds.y - chunkOrigin.y) / voxelSize));
        int maxY = std::min(chunkSize, (int)((maxBounds.y - chunkOrigin.y) / voxelSize) + 1);
        int minZ = std::max(0, (int)((minBounds.z - chunkOrigin.z) / voxelSize));
        int maxZ = std::min(chunkSize, (int)((maxBounds.z - chunkOrigin.z) / voxelSize) + 1);

        // Early out if box doesn't intersect this chunk
        if (minX > maxX || minY > maxY || minZ > maxZ) return;

        bool chunkTouched = false;

        // Calculate center for distance-based density (optional)
        glm::vec3 boxCenter = (minBounds + maxBounds) * 0.5f;
        glm::vec3 halfSize = (maxBounds - minBounds) * 0.5f;

        for (int lz = minZ; lz <= maxZ; ++lz) {
            float worldZ = chunkOrigin.z + (float)lz * voxelSize;
            if (worldZ < minBounds.z || worldZ > maxBounds.z) continue;

            for (int ly = minY; ly <= maxY; ++ly) {
                float worldY = chunkOrigin.y + (float)ly * voxelSize;
                if (worldY < minBounds.y || worldY > maxBounds.y) continue;

                for (int lx = minX; lx <= maxX; ++lx) {
                    float worldX = chunkOrigin.x + (float)lx * voxelSize;
                    if (worldX < minBounds.x || worldX > maxBounds.x) continue;

                    // Calculate density based on distance to box surface
                    glm::vec3 pos(worldX, worldY, worldZ);
                    glm::vec3 distToCenter = pos - boxCenter;
                    glm::vec3 distToEdge = glm::abs(distToCenter) - halfSize;

                    float maxDistToEdge = std::max({distToEdge.x, distToEdge.y, distToEdge.z});
                    float density;

                    if (maxDistToEdge <= 0) {
                        // Inside box: solid density
                        density = 1.0f;
                    } else {
                        // Outside box: falloff
                        density = std::max(0.0f, 1.0f - (maxDistToEdge / (voxelSize * 2.0f)));
                    }

                    auto& voxel = chunk->voxels[lx][ly][lz];
                    if (density > voxel.density) {
                        voxel.density = density;
                        voxel.type = formation.type;
                        voxel.color = formation.color;
                        voxel.material = material;
                        chunkTouched = true;
                    }
                }
            }
        }

        if (chunkTouched) {
            chunk->meshDirty = true;
        }
    }

    void SpellSystem::createSpellFormation(const glm::vec3& center,
                                           const FormationParams& formationParams,
                                           float strength, uint64_t material,
                                           const glm::vec3& color, size_t collectedVoxels,
                                           uint8_t dominantType) {
        if (!ctx.chunks || !ctx.worldToChunk || !ctx.getChunkMin || !ctx.markChunkModified || !ctx.generateChunkMesh) return;
        TRACY_CPU_ZONE("CreateFormation");

        WorldPlanet newFormation;
        newFormation.worldPos = center;
        newFormation.color = color;
        newFormation.type = dominantType;

        float effectiveRadius = formationParams.getBoundingRadius();
        newFormation.radius = effectiveRadius;

        float preloadRadius = effectiveRadius;
        int minCX = ctx.worldToChunk(center.x - preloadRadius);
        int maxCX = ctx.worldToChunk(center.x + preloadRadius);
        int minCY = ctx.worldToChunk(center.y - preloadRadius);
        int maxCY = ctx.worldToChunk(center.y + preloadRadius);
        int minCZ = ctx.worldToChunk(center.z - preloadRadius);
        int maxCZ = ctx.worldToChunk(center.z + preloadRadius);

        FormationParams paramsCopy = formationParams;
        paramsCopy.center = center;

        carveFormationWithSDF(newFormation, material, paramsCopy);

        int regenMinCX = ctx.worldToChunk(center.x - effectiveRadius);
        int regenMaxCX = ctx.worldToChunk(center.x + effectiveRadius);
        int regenMinCY = ctx.worldToChunk(center.y - effectiveRadius);
        int regenMaxCY = ctx.worldToChunk(center.y + effectiveRadius);
        int regenMinCZ = ctx.worldToChunk(center.z - effectiveRadius);
        int regenMaxCZ = ctx.worldToChunk(center.z + effectiveRadius);

        for (int cx = regenMinCX; cx <= regenMaxCX; ++cx) {
            for (int cy = regenMinCY; cy <= regenMaxCY; ++cy) {
                for (int cz = regenMinCZ; cz <= regenMaxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = ctx.chunks->getChunk(coord);
                    if (chunk && chunk->meshDirty && ctx.generateChunkMesh) {
                        ctx.generateChunkMesh(chunk);
                    }
                }
            }
        }
    }

    void SpellSystem::initSpellDestructibleVolume(SpellEffect& spell) {
        TRACY_CPU_ZONE("InitVolume");

        auto& d = spell.destruct;
        d.voxelSize = VOXEL_SIZE;

        glm::vec3 halfExtWorld(2.0f * VOXEL_SIZE);

        if (spell.formationParams.type == FormationType::SPHERE) {
            halfExtWorld = glm::vec3(spell.formationParams.radius * 2.0f);
        } else {
            float r = spell.formationParams.getBoundingRadius();
            halfExtWorld = glm::vec3(r);
        }

        glm::ivec3 cornerDims = dimsFromHalfExtents(halfExtWorld, VOXEL_SIZE);
        d.volume.init(cornerDims, VOXEL_SIZE);

        glm::vec3 localHalf = halfExtentsFromVolumeDims(cornerDims, VOXEL_SIZE);
        glm::vec3 centerLocal = localHalf;
        d.localCenterOffsetWorld = centerLocal;

        if (spell.formationParams.type == FormationType::SPHERE) {
            d.volume.fillSphere(centerLocal, spell.formationParams.radius,
                                spell.formationColor, (uint32_t)spell.targetMaterial,
                                spell.dominantType);
        } else {
            glm::vec3 halfBox = halfExtWorld;

            switch(spell.formationParams.type) {
                case FormationType::CUBE:
                    halfBox = glm::vec3(
                            spell.formationParams.sizeX * 0.5f,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.sizeZ * 0.5f
                    );
                    break;
                case FormationType::PLATFORM:
                case FormationType::WALL:
                    halfBox = glm::vec3(
                            spell.formationParams.sizeX * 0.5f,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.sizeZ * 0.5f
                    );
                    break;
                case FormationType::CYLINDER:
                    halfBox = glm::vec3(
                            spell.formationParams.radius,
                            spell.formationParams.sizeY * 0.5f,
                            spell.formationParams.radius
                    );
                    break;
                default:
                    break;
            }

            fillBox(d.volume, centerLocal, halfBox, spell.formationColor,
                    (uint32_t)spell.targetMaterial, spell.dominantType);
        }

        d.meshDirty = true;
    }

    void SpellSystem::rebuildDestructibleMeshIfNeeded(DestructibleObject& d) {
        if (!d.meshDirty) return;
        TRACY_CPU_ZONE("RebuildMesh");

        gl3::LocalMesh mesh = gl3::buildMeshLocalMC(d.volume);

        const glm::vec3 centerLocal = d.localCenterOffsetWorld;

        for (auto& v : mesh.vertices) {
            v -= centerLocal;
        }

        createPhysicsMeshData(d.mesh, mesh.vertices, mesh.normals, mesh.colors);
        d.meshDirty = false;
    }

    void SpellSystem::createPhysicsMeshData(PhysicsMeshData& mesh,
                               const std::vector<glm::vec3>& vertices,
                               const std::vector<glm::vec3>& normals,
                               const std::vector<glm::vec3>& colors) {
        TRACY_CPU_ZONE("CreateMeshData");

        if (mesh.vao) {
            glDeleteVertexArrays(1, &mesh.vao);
            glDeleteBuffers(1, &mesh.vbo);
        }

        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);

        glBindVertexArray(mesh.vao);
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);

        struct InterleavedVertex {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec3 color;
        };

        std::vector<InterleavedVertex> interleaved;
        interleaved.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            interleaved.push_back({
                                          vertices[i],
                                          normals[i],
                                          colors[i]
                                  });
        }

        glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(InterleavedVertex),
                     interleaved.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(InterleavedVertex),
                              (void*)offsetof(InterleavedVertex, position));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(InterleavedVertex),
                              (void*)offsetof(InterleavedVertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InterleavedVertex),
                              (void*)offsetof(InterleavedVertex, color));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        mesh.isValid = true;
        mesh.vertexCount = (int)vertices.size();
    }

    void SpellSystem::removeFormationVoxels(SpellEffect& spell) {
        TRACY_CPU_ZONE("RemoveVoxelsFromForm");

        if (!ctx.chunks || !ctx.worldToChunk || !ctx.markChunkModified) return;

        float effectiveRadius = spell.formationParams.getBoundingRadius();

        int minCX = ctx.worldToChunk(spell.center.x - effectiveRadius);
        int maxCX = ctx.worldToChunk(spell.center.x + effectiveRadius);
        int minCY = ctx.worldToChunk(spell.center.y - effectiveRadius);
        int maxCY = ctx.worldToChunk(spell.center.y + effectiveRadius);
        int minCZ = ctx.worldToChunk(spell.center.z - effectiveRadius);
        int maxCZ = ctx.worldToChunk(spell.center.z + effectiveRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = ctx.chunks->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkOrigin = ctx.getChunkMin(coord);
                    bool chunkModified = false;

                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz) * VOXEL_SIZE;
                                float sdfValue = spell.formationParams.evaluate(worldPos);

                                if (sdfValue >= 0.0f) {
                                    chunk->voxels[lx][ly][lz].density = -1.0f;
                                    chunk->voxels[lx][ly][lz].type = 0;
                                    chunkModified = true;
                                }
                            }
                        }
                    }

                    if (chunkModified) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        if (ctx.markChunkModified) {
                            ctx.markChunkModified(coord);
                        }
                    }
                }
            }
        }
    }

    glm::vec3 SpellSystem::calculateSphereDistribution(size_t index, size_t total, const FormationParams& params) {
        if (total <= 1) return params.center;

        float goldenAngle = glm::pi<float>() * (3.0f - glm::sqrt(5.0f));
        float y = 1.0f - (static_cast<float>(index) / (static_cast<float>(total) - 1.0f)) * 2.0f;
        float radiusAtY = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float theta = goldenAngle * static_cast<float>(index);

        float x = std::cos(theta) * radiusAtY;
        float z = std::sin(theta) * radiusAtY;

        glm::vec3 localPos(x, y, z);
        return params.center + localPos * params.radius;
    }

    glm::vec3 SpellSystem::calculatePlatformDistribution(size_t index, size_t total,
                                                  const FormationParams& params) {
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos(
                u * params.sizeX,
                params.sizeY * 0.5f,
                v * params.sizeZ
        );

        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 SpellSystem::calculateWallDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos(
                u * params.sizeX,
                v * params.sizeY,
                params.sizeZ * 0.5f
        );

        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 SpellSystem::calculateCubeDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        int faceIndex = index % 6;
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos;
        switch(faceIndex) {
            case 0:
                localPos = glm::vec3(params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 1:
                localPos = glm::vec3(-params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 2:
                localPos = glm::vec3(u * params.sizeX, params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 3:
                localPos = glm::vec3(u * params.sizeX, -params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 4:
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, params.sizeZ * 0.5f);
                break;
            case 5:
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, -params.sizeZ * 0.5f);
                break;
        }

        glm::vec3 worldPos = params.center + localPos;
        return worldPos;
    }

    glm::vec3 SpellSystem::calculateCylinderDistribution(size_t index, size_t total,
                                                  const FormationParams& params) {
        float angle = static_cast<float>(index) / static_cast<float>(total) * glm::two_pi<float>();
        float height = haltonSequence(index, 2) - 0.5f;

        glm::vec3 localPos(
                std::cos(angle) * params.radius,
                height * params.sizeY,
                std::sin(angle) * params.radius
        );

        glm::vec3 worldPos = params.center + localPos;
        return worldPos;
    }

    glm::vec3 SpellSystem::calculatePyramidDistribution(size_t index, size_t total,
                                                 const FormationParams& params) {
        int surface = index % 5;

        if (surface < 4) {
            float u = haltonSequence(index, 2);
            float v = haltonSequence(index, 3);

            float baseX = (u - 0.5f) * params.sizeX;
            float baseZ = (v - 0.5f) * params.sizeZ;

            glm::vec3 localPos;
            switch(surface) {
                case 0:
                    localPos = glm::vec3(baseX, 0.0f, params.sizeZ * 0.5f);
                    break;
                case 1:
                    localPos = glm::vec3(baseX, 0.0f, -params.sizeZ * 0.5f);
                    break;
                case 2:
                    localPos = glm::vec3(params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
                case 3:
                    localPos = glm::vec3(-params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
            }

            float heightRatio = haltonSequence(index, 5);
            localPos.y = heightRatio * params.sizeY;
            localPos.x *= (1.0f - heightRatio);
            localPos.z *= (1.0f - heightRatio);

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        } else {
            float u = haltonSequence(index, 2) - 0.5f;
            float v = haltonSequence(index, 3) - 0.5f;

            glm::vec3 localPos(
                    u * params.sizeX,
                    0.0f,
                    v * params.sizeZ
            );

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        }
    }

    float SpellSystem::haltonSequence(int index, int base) {
        float result = 0.0f;
        float f = 1.0f;
        int i = index;

        while (i > 0) {
            f /= static_cast<float>(base);
            result += f * static_cast<float>(i % base);
            i = i / base;
        }

        return result;
    }


}