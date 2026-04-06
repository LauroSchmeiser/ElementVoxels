#include "EnemyManager.h"
#include "EnemyMeshing.h"
#include <stdexcept>
#include <iostream>
#include "LocalMarchingCubes.h"

namespace gl3 {

    void EnemyManager::init(VoxelPhysicsManager* physics, Game* g) {
        physicsMgr = physics;
        game = g;
    }

    EnemyRuntime& EnemyManager::spawn(const EnemyArchetype& type, const glm::vec3& pos) {
        EnemyRuntime e;
        e.inst.id = nextId++;
        e.inst.type = &type;
        e.inst.position = pos;
        e.inst.hp = type.maxHP;

        // local volume: 33³ corners => 32³ cells
        e.volume.init(glm::ivec3(33,33,33), VOXEL_SIZE);

        // fill a sphere in LOCAL space (centered in volume)
        glm::vec3 centerLocal = glm::vec3(16,16,16) * VOXEL_SIZE;
        e.volume.fillSphere(centerLocal, type.radius, glm::vec3(0.9f,0.15f,0.15f));

        e.inst.meshDirty = true;

        enemies.push_back(std::move(e));
        EnemyRuntime& ref = enemies.back();

        ensurePhysicsBody(ref);      // proxy collision body (AABB/sphere/capsule)
        rebuildMeshIfNeeded(ref);    // build once

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
                glm::vec3(r) // for SPHERE: extents.x = radius
        );

        if (!e.inst.body) return;

        e.inst.bodyId = e.inst.body->id;
        e.inst.body->orientation = e.inst.rotation;
        e.inst.body->velocity = glm::vec3(0);
        e.inst.body->angularVelocity = glm::vec3(0);

        // IMPORTANT: don't store &e in userData (vector realloc)
        //e.inst.body->userData = nullptr;
    }

    void EnemyManager::update(float dt, const glm::vec3& playerPos) {
        for (size_t i = 0; i < enemies.size(); /*manual*/) {
            EnemyRuntime& e = enemies[i];

            // AI chase (physics proxy)
            if (e.inst.body && e.inst.hp > 0.0f) {
                glm::vec3 toP = playerPos - e.inst.body->position;
                float d = glm::length(toP);
                if (d > 0.001f) e.inst.body->velocity = (toP / d) * e.inst.type->moveSpeed;
            }

            // sync from physics
            if (e.inst.body) e.inst.position = e.inst.body->position;

            e.inst.cdRemaining[0]-=0.5f*dt;
            if (game && e.inst.cdRemaining[0] <= 0.0f) {
                glm::vec3 dir = glm::normalize(playerPos - e.inst.position);
                glm::vec3 start = e.inst.position + dir * (e.inst.currentRadius + 1.5f * VOXEL_SIZE);
                game->spawnEnemyLaunchSphere(start, playerPos,
                       2.0f * VOXEL_SIZE,
                       40.0f * VOXEL_SIZE,
                       glm::vec3(1.0f, 0.2f, 0.2f));
                e.inst.cdRemaining[0] = e.inst.type->cooldownsSec[0];
            }

            // rebuild mesh only if destroyed
            rebuildMeshIfNeeded(e);

            // remove policy: dead OR too small mesh
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

        // convert to local (ignore rotation for now)
        glm::vec3 hitLocal = (hitWorldPos - e->inst.position);

        // if your local volume is centered, you likely want an offset:
        // i.e. localOrigin at (0,0,0) and sphere centered at half-extents.
        glm::vec3 volumeCenterLocal = glm::vec3(16,16,16) * VOXEL_SIZE;
        glm::vec3 carveCenter = hitLocal + volumeCenterLocal;

        e->volume.carveSphere(carveCenter, radiusWorld, strength);

        // HP can be reduced independently
        e->inst.hp -= strength * 1.0f; // tune

        e->inst.meshDirty = true;
    }

    EnemyRuntime* EnemyManager::find(uint64_t id) {
        for (auto& e : enemies) if (e.inst.id == id) return &e;
        return nullptr;
    }

    void EnemyManager::rebuildMeshIfNeeded(EnemyRuntime& e) {
        if (!e.inst.meshDirty) return;
        if (!game) return;

        std::cout<<"corners size: "<<e.volume.corners.size()<<"\n";

        gl3::LocalMesh mesh = gl3::buildMeshLocalMC(e.volume);
        std::cout<<"vertices size: "<<mesh.vertices.size()<<"\n";
        game->createPhysicsMeshData(e.renderMesh, mesh.vertices, mesh.normals, mesh.colors);

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

} // namespace gl3