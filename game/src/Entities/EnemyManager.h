#pragma once
#include <vector>
#include <memory>

#include "Enemy.h"
#include "EnemyVoxelVolume.h"
#include "../Game.h"

namespace gl3 {

    struct PhysicsMeshData; // from VoxelStructures.h

    struct EnemyRuntime {
        EnemyInstance inst;
        gl3::LocalVoxelVolume volume;

        // You can reuse PhysicsMeshData for rendering (same struct SpellEffect uses) citeturn2search0
        PhysicsMeshData renderMesh;
    };

    class VoxelPhysicsManager;

    class EnemyManager {
    public:
        void init(VoxelPhysicsManager* physics, Game* game);
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

    private:
        VoxelPhysicsManager* physicsMgr = nullptr;
        std::vector<EnemyRuntime> enemies;
        uint64_t nextId = 1;
    };

} // namespace gl3