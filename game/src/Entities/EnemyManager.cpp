#include "EnemyManager.h"
#include "EnemyMeshing.h"
#include <stdexcept>
#include <iostream>
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
        e.inst.type = &type;
        e.inst.position = pos;
        e.inst.hp = type.maxHP;

        e.volume.init(glm::ivec3(33,33,33), VOXEL_SIZE);

        glm::vec3 centerLocal = glm::vec3(16,16,16) * VOXEL_SIZE;
        e.volume.fillSphere(centerLocal, type.radius, glm::vec3(0.9f,0.15f,0.15f));

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
                e.inst.type->mass,
                e.inst.type->shapeType,
                glm::vec3(r)
        );

        if (!e.inst.body) return;

        e.inst.bodyId = e.inst.body->id;
        e.inst.body->orientation = e.inst.rotation;
        e.inst.body->velocity = glm::vec3(0);
        e.inst.body->angularVelocity = glm::vec3(0);

        //e.inst.body->userData = nullptr;
    }

    void EnemyManager::update(float dt, const glm::vec3& playerPos) {
        for (size_t i = 0; i < enemies.size(); /*manual*/) {
            EnemyRuntime& e = enemies[i];

            if (e.inst.body && e.inst.hp > 0.0f) {
                glm::vec3 toP = playerPos - e.inst.body->position;
                float d = glm::length(toP);
                if (d > 0.001f) e.inst.body->velocity = (toP / d) * e.inst.type->moveSpeed;
            }

            if (e.inst.body) e.inst.position = e.inst.body->position;

            e.inst.cdRemaining[0]-=0.5f*dt;
            glm::vec3 dir = glm::normalize(playerPos - e.inst.position);
            RayCastResult hit= rayCastFromPosition(e.inst.position,dir,500);
            if (game && e.inst.cdRemaining[0] <= 0.0f&&glm::distance(hit.hitPosition,playerPos)<=10*CHUNK_SIZE*VOXEL_SIZE) {
                    glm::vec3 start = e.inst.position + dir * (e.inst.currentRadius + 1.0f * VOXEL_SIZE);
                    game->spawnEnemyLaunchSphere(start, playerPos,
                                                 5.0f * VOXEL_SIZE,
                                                 15.0f,
                                                 glm::vec3(1.0f, 0.2f, 0.2f));
                    e.inst.cdRemaining[0] = e.inst.type->cooldownsSec[0];
            }

            rebuildMeshIfNeeded(e);

            const uint32_t kTooSmallVtx = 10;
            if (e.inst.hp <= 0.0f || (e.renderMesh.isValid && e.renderMesh.vertexCount < kTooSmallVtx)) {
                destroyEnemy(i);
                continue;
            }

            ++i;
        }
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

        e->inst.meshDirty = true;
    }

    EnemyRuntime* EnemyManager::find(uint64_t id) {
        for (auto& e : enemies) if (e.inst.id == id) return &e;
        return nullptr;
    }

    void EnemyManager::rebuildMeshIfNeeded(EnemyRuntime& e) {
        if (!e.inst.meshDirty) return;
        if (!game) return;

        gl3::LocalMesh mesh = gl3::buildMeshLocalMC(e.volume);
        //game->createPhysicsMeshData(e.renderMesh, mesh.vertices, mesh.normals, mesh.colors);

        e.inst.meshDirty = false;
    }

    void EnemyManager::destroyEnemy(size_t index) {
        EnemyRuntime& e = enemies[index];

        if (physicsMgr && e.inst.bodyId != 0) {
            physicsMgr->removeBody(e.inst.bodyId);
            e.inst.bodyId = 0;
            e.inst.body = nullptr;
        }

        if (e.renderMesh.vao) {
            glDeleteVertexArrays(1, &e.renderMesh.vao);
            glDeleteBuffers(1, &e.renderMesh.vbo);
            e.renderMesh.vao = 0;
            e.renderMesh.vbo = 0;
        }
        e.renderMesh.isValid = false;
        e.renderMesh.vertexCount = 0;

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