#include "EnemyManager.h"
#include "EnemyMeshing.h"
#include <stdexcept>
#include <random>
#include "LocalMarchingCubes.h"

namespace gl3 {

    void EnemyManager::init(VoxelPhysicsManager* physics,FixedGridChunkManager* chunkManager, Game* g) {
        physicsMgr = physics;
        chunkMgr=chunkManager;
        game = g;
    }

    EnemyRuntime& EnemyManager::spawn(const EnemyArchetype& type, const glm::vec3& pos) {
        EnemyRuntime e;
        e.inst.id = nextId++;
        e.inst.type = type;
        e.inst.position = pos;
        e.inst.hp = type.maxHP;
        e.inst.baseRadius = type.radius;
        e.inst.currentRadius = type.radius;

        e.volume.init(glm::ivec3(33,33,33), VOXEL_SIZE);

        glm::vec3 centerLocal = glm::vec3(16,16,16) * VOXEL_SIZE;

// Main body
        e.volume.unionSphere(centerLocal, type.radius, glm::vec3(0.72f, 0.58f, 0.52f), 7, 1);

// MUCH bigger front eye bulge
        e.volume.unionSphere(centerLocal + glm::vec3(type.radius * 0.65f, 0, 0),
                             type.radius * 0.65f,  // Larger bulge
                             glm::vec3(0.75f, 0.70f, 0.0f), 8, 1);

/*// More prominent brow ridge
        e.volume.unionSphere(centerLocal + glm::vec3(0, type.radius * 0.25f, type.radius * 0.55f),
                             type.radius * 0.28f,
                             glm::vec3(0.40f, 0.25f, 0.28f), 7, 1);

// Lower eyelid more pronounced
        e.volume.unionSphere(centerLocal + glm::vec3(0, -type.radius * 0.22f, type.radius * 0.52f),
                             type.radius * 0.24f,
                             glm::vec3(0.40f, 0.25f, 0.28f), 7, 1);*/

// Small stalk/tendril roots
        glm::vec3 eyeDir = glm::vec3(1.0f, 0.0f, 0.0f);
        int rerollCount=0;
        for (int i = 0; i < 6; ++i) {
            std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> distDir(-1.0f, 1.0f);
            glm::vec3 dir = glm::normalize(glm::vec3(distDir(rng),distDir(rng),distDir(rng)));

            glm::vec3 root = centerLocal + dir * (type.radius * 0.95f);

            if (rerollCount<6&&glm::dot(dir, eyeDir) > 0.3f)
            {
                i--;
                rerollCount++;
                continue;
            }

            const int segments = 20;
            for (int s = 0; s < segments; ++s) {
                float t = float(s) / float(segments - 1);

                float dist = glm::mix(type.radius * 0.10f, type.radius * 1.15f, t);
                float segRadius = glm::mix(type.radius * 0.30f, type.radius * 0.20f, t);

                glm::vec3 segPos = root + dir * dist;
                segPos.y += sin(t * 3.14159f) * type.radius * 0.08f; // subtle curl

                e.volume.unionSphere(
                        segPos,
                        segRadius,
                        glm::vec3(0.36f, 0.22f, 0.26f),
                        7, 1
                );
            }
        }

        e.inst.meshDirty = true;

        enemies.push_back(std::move(e));
        EnemyRuntime& ref = enemies.back();

        ensurePhysicsBody(ref);
        rebuildMeshIfNeeded(ref);

        return ref;
    }

    void EnemyManager::ensurePhysicsBody(EnemyRuntime& e) {
        if (!physicsMgr) return;
        if (e.inst.body) return;

        const float r = e.inst.currentRadius;

        e.inst.body = physicsMgr->createBody(
                e.inst.position,
                e.inst.type.mass,
                e.inst.type.shapeType,
                glm::vec3(r+0.5f)
        );

        if (!e.inst.body) return;

        e.inst.bodyId = e.inst.body->id;
        e.inst.body->orientation = e.inst.rotation;
        e.inst.body->velocity = glm::vec3(0);
        e.inst.body->angularVelocity = glm::vec3(0);
        e.inst.body->material=7;


        //e.inst.body->userData = nullptr;
    }

    void EnemyManager::update(float dt, const glm::vec3& playerPos) {
        for (size_t i = 0; i < enemies.size(); /*manual*/) {
            EnemyRuntime& e = enemies[i];

            if (e.inst.body && e.inst.hp > 0.0f) {
                glm::vec3 toP = playerPos - e.inst.body->position;
                float d = glm::length(toP);
                if (d > 0.001f) {
                    glm::vec3 dir = toP / d;
                    e.inst.body->velocity = dir * e.inst.type.moveSpeed;

                    glm::quat targetRot = rotateFromTo(glm::vec3(1,0,0), dir);
                    e.inst.rotation = glm::slerp(e.inst.rotation, targetRot, 0.08f);
                    e.inst.body->orientation = e.inst.rotation;
                }            }

            if (e.inst.body) e.inst.position = e.inst.body->position;

            e.inst.cdRemaining[0]-=0.5f*dt;
            glm::vec3 dir = glm::normalize(playerPos - e.inst.position);
            RayCastResult hit= rayCastFromPosition(e.inst.position,dir,500);
            float radius= e.inst.type.radius * 1.25;
            int material = 0;
            if(e.inst.type.radius==8.0f*VOXEL_SIZE)
            {
                radius= 10.0f * VOXEL_SIZE;
                material=9;
            }
            if (game && e.inst.cdRemaining[0] <= 0.0f&&glm::distance(hit.hitPosition,playerPos)<=10*CHUNK_SIZE*VOXEL_SIZE) {
                    glm::vec3 start = e.inst.position + dir * (e.inst.currentRadius + 1.0f * VOXEL_SIZE);
                    game->spawnEnemyLaunchSphere(start, playerPos,
                                                 radius,
                                                 10.0f/glm::sqrt(radius),
                                                 glm::vec3(1.0f, 0.2f, 0.2f),material);
                    e.inst.cdRemaining[0] = e.inst.type.cooldownsSec[0];
            }

            rebuildMeshIfNeeded(e);

            const uint32_t kTooSmallVtx = 10;
            size_t totalVerts = 0;
            for (const auto& part : e.renderParts) {
                totalVerts += part.mesh.vertexCount;
            }
            if (e.inst.hp <= 0.0f || totalVerts < kTooSmallVtx) {
                destroyEnemy(i);
                continue;
            }

            ++i;
        }
    }

    glm::quat EnemyManager::rotateFromTo(const glm::vec3& from, const glm::vec3& to)
    {
        glm::vec3 f = glm::normalize(from);
        glm::vec3 t = glm::normalize(to);

        float cosTheta = glm::dot(f, t);

        if (cosTheta > 0.9999f) return glm::quat(1,0,0,0);

        if (cosTheta < -0.9999f) {
            glm::vec3 axis = glm::cross(glm::vec3(0,1,0), f);
            if (glm::length(axis) < 1e-4f)
                axis = glm::cross(glm::vec3(0,0,1), f);
            axis = glm::normalize(axis);
            return glm::angleAxis(glm::pi<float>(), axis);
        }

        glm::vec3 axis = glm::normalize(glm::cross(f, t));
        float angle = acos(glm::clamp(cosTheta, -1.0f, 1.0f));
        return glm::angleAxis(angle, axis);
    }

    void EnemyManager::applyDamageSphere(uint64_t enemyId,
                                         const glm::vec3& hitWorldPos,
                                         float radiusWorld,
                                         float strength)
    {
        EnemyRuntime* e = find(enemyId);
        if (!e) return;

        glm::vec3 hitLocal = (hitWorldPos - e->inst.position);


        glm::vec3 volumeCenterLocal = glm::vec3(16,16,16) * VOXEL_SIZE;
        glm::vec3 carveCenter = hitLocal + volumeCenterLocal;

        e->volume.carveSphere(carveCenter, radiusWorld, strength);

        e->inst.hp -= strength * 1.0f;
        std::cout<<"Enemy Damage taken\n";


        e->inst.meshDirty = true;
    }

    EnemyRuntime* EnemyManager::find(uint64_t id) {
        for (auto& e : enemies) if (e.inst.bodyId == id) return &e;
        return nullptr;
    }

    void EnemyManager::destroyRenderMesh(PhysicsMeshData& mesh) {
        if (mesh.vao) {
            glDeleteVertexArrays(1, &mesh.vao);
            mesh.vao = 0;
        }
        if (mesh.vbo) {
            glDeleteBuffers(1, &mesh.vbo);
            mesh.vbo = 0;
        }
        mesh.isValid = false;
        mesh.vertexCount = 0;
    }

    void EnemyManager::rebuildMeshIfNeeded(EnemyRuntime& e) {
        if (!e.inst.meshDirty) return;
        if (!game) return;

        for (auto& part : e.renderParts) {
            destroyRenderMesh(part.mesh);
        }
        e.renderParts.clear();

        gl3::LocalMesh mesh = gl3::buildMeshLocalMC(e.volume);

        e.renderParts.reserve(mesh.parts.size());

        for (const auto& src : mesh.parts) {
            EnemyRenderPart part;
            part.material = src.material;

            game->createPhysicsMeshData(
                    part.mesh,
                    src.vertices,
                    src.normals,
                    src.colors,
                    src.uvs,
                    src.flags
            );

            e.renderParts.push_back(std::move(part));
        }

        e.inst.meshDirty = false;
    }

    void EnemyManager::destroyEnemy(size_t index) {
        EnemyRuntime& e = enemies[index];

        if (physicsMgr && e.inst.bodyId != 0) {
            physicsMgr->removeBody(e.inst.bodyId);
            e.inst.bodyId = 0;
            e.inst.body = nullptr;
        }

        for(int i=0;i<e.renderParts.size();i++)
        {
            destroyRenderMesh(e.renderParts.at(i).mesh);
        }

        enemies[index] = std::move(enemies.back());
        enemies.pop_back();
    }

    RayCastResult EnemyManager::rayCastFromPosition(glm::vec3 position, glm::vec3 direction, float maxDistance) {
        RayCastResult result;
        result.hit = false;

        glm::vec3 rayDir = direction;
        glm::vec3 rayOrigin = position;

        float stepSize = 0.25f;
        float currentDist = 0.0f;

        while (currentDist < maxDistance) {
            glm::vec3 samplePos = rayOrigin + rayDir * currentDist;

            ChunkCoord coord;
            coord.x = chunkMgr->worldToChunk(samplePos.x);
            coord.y = chunkMgr->worldToChunk(samplePos.y);
            coord.z = chunkMgr->worldToChunk(samplePos.z);

            Chunk* chunk = chunkMgr->getChunk(coord);
            if (chunk) {
                glm::vec3 chunkMin = chunkMgr->getChunkMin(coord);
                glm::ivec3 localPos = glm::ivec3(
                        samplePos.x - chunkMin.x,
                        samplePos.y - chunkMin.y,
                        samplePos.z - chunkMin.z
                );

                if (localPos.x >= 0 && localPos.x <= CHUNK_SIZE &&
                    localPos.y >= 0 && localPos.y <= CHUNK_SIZE &&
                    localPos.z >= 0 && localPos.z <= CHUNK_SIZE) {

                    if (chunk->voxels[localPos.x][localPos.y][localPos.z].isSolid()) {
                        result.hitPosition = samplePos;
                        result.hitNormal = chunkMgr->calculateNormalAt(chunk, localPos);
                        result.distance = currentDist;
                        result.hit = true;
                        return result;
                    }
                }
            }

            currentDist += stepSize;
        }

        result.hitPosition = rayOrigin + rayDir * maxDistance;
        result.hitNormal = -rayDir;
        result.distance = maxDistance;
        result.hit = false;
        return result;
    }

}