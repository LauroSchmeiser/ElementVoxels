#pragma once
#include <vector>
#include <memory>

#include "Enemy.h"
#include "EnemyVoxelVolume.h"
#include "../Game.h"

namespace gl3 {

    struct PhysicsMeshData;

    struct EnemyRuntime {
        EnemyInstance inst;
        gl3::LocalVoxelVolume volume;

        PhysicsMeshData renderMesh;
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

    private:
        Game* game = nullptr;
        EnemyRuntime* find(uint64_t id);

        void ensurePhysicsBody(EnemyRuntime& e);
        void rebuildMeshIfNeeded(EnemyRuntime& e);
        RayCastResult rayCastFromPosition(glm::vec3 position,  glm::vec3 direction, float maxDistance = 1000.0f);


    private:
        VoxelPhysicsManager* physicsMgr = nullptr;
        FixedGridChunkManager* chunkMgr = nullptr;
        std::vector<EnemyRuntime> enemies;
        uint64_t nextId = 1;
    };

} // namespace gl3