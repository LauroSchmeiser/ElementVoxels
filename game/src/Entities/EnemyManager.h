#pragma once
#include <vector>
#include <memory>

#include "Enemy.h"
#include "EnemyVoxelVolume.h"
#include "../Game.h"

namespace gl3 {

    struct PhysicsMeshData;

    struct EnemyRenderPart {
        uint32_t material = 0;
        PhysicsMeshData mesh;
    };

    struct EnemyRuntime {
        EnemyInstance inst;
        gl3::LocalVoxelVolume volume;
        std::vector<EnemyRenderPart> renderParts;
    };

    struct RayCastResult {
        glm::vec3 hitPosition;
        glm::vec3 hitNormal;
        float distance;
        bool hit;
    };

    class VoxelPhysicsManager;

    class EnemyManager {
    public:
        void init(VoxelPhysicsManager* physics,FixedGridChunkManager* chunkManager, Game* game);
        EnemyRuntime& spawn(const EnemyArchetype& type, const glm::vec3& pos);

        void update(float dt, const glm::vec3& playerPos);

        // damage API
        void applyDamageSphere(uint64_t enemyId, const glm::vec3& hitWorldPos, float radiusWorld, float strength);

        std::vector<EnemyRuntime>& all() { return enemies; }

        void destroyEnemy(size_t index);

        uint32_t getEnemiesAlive() const { return enemies.size();}

        glm::vec3 getEnemyPos(int idx) { return find(idx)->inst.position;}

        float getEnemyHP(int idx)  { return find(idx)->inst.hp;}
        void setEnemyHP(int idx, float hp) { find(idx)->inst.hp=hp;}

        float getEnemyMat(int idx) { return find(idx)->inst.body->material;}
        void setEnemyMat(int idx, int mat)  { find(idx)->inst.body->material=mat;}

        float getEnemyRadius(int idx) { return find(idx)->inst.body->radius;}
        void setEnemyRadius(int idx, float r)
        {
            EnemyRuntime* enemy = find(idx);
            enemy->inst.body->radius=r;
            rebuildMeshIfNeeded(*enemy);
        }

        EnemyRuntime* find(uint64_t id);


    private:
        Game* game = nullptr;
        void ensurePhysicsBody(EnemyRuntime& e);
        void rebuildMeshIfNeeded(EnemyRuntime& e);
        RayCastResult rayCastFromPosition(glm::vec3 position,  glm::vec3 direction, float maxDistance = 1000.0f);

        VoxelPhysicsManager* physicsMgr = nullptr;
        FixedGridChunkManager* chunkMgr = nullptr;
        std::vector<EnemyRuntime> enemies;
        uint64_t nextId = 1;

        void destroyRenderMesh(PhysicsMeshData &mesh);

        glm::quat rotateFromTo(const glm::vec3 &from, const glm::vec3 &to);
    };

}