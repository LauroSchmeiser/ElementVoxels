#pragma once
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/detail/type_quat.hpp>

#include "../physics/VoxelPhysicsBody.h"
#include "../rendering/VoxelStructures.h"

namespace gl3 {

    enum class EnemySpellId : uint8_t {
        NONE = 0,
        FIREBALL,
        GRAVITY_WELL,
        // add more
    };

    struct EnemyArchetype {
        const char* name = "Enemy";

        // stats
        float maxHP = 100.0f;
        float moveSpeed = 25.0f;

        VoxelPhysicsBody::ShapeType shapeType = VoxelPhysicsBody::ShapeType::CAPSULE;
        glm::vec3 shapeExtents = glm::vec3(1.0f);
        float mass = 10.0f;

        glm::ivec3 voxelDims = {24, 32, 24}; // local grid
        float voxelSize = VOXEL_SIZE;

        std::array<EnemySpellId, 3> spells { EnemySpellId::FIREBALL, EnemySpellId::NONE, EnemySpellId::NONE };
        std::array<float, 3> cooldownsSec { 1.5f, 4.0f, 8.0f };

        float radius = 2.5f * VOXEL_SIZE;

    };

    struct EnemyInstance {
        uint64_t id = 0;

        // gameplay transform
        glm::vec3 position {0};
        glm::quat rotation {1,0,0,0};

        // health
        float hp = 100.0f;

        // AI/combat
        glm::vec3 targetPos {0};
        std::array<float, 3> cdRemaining {0,0,0};

        // collision/physics handle
        VoxelPhysicsBody* body = nullptr;
        uint64_t bodyId = 0;

        // destructible geometry flags
        bool meshDirty = true;
        bool voxelsDirty = true;

        const EnemyArchetype* type = nullptr;

        float baseRadius = 2.5f * VOXEL_SIZE;
        float currentRadius = 2.5f * VOXEL_SIZE;

        bool pendingRemoval = false;
    };

}