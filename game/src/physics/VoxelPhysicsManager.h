// VoxelPhysicsManager.h
#pragma once
#include "VoxelPhysicsBody.h"
#include "../rendering/MultiGridChunkManager.h"
#include <vector>
#include <functional>

namespace gl3 {

    class VoxelPhysicsManager {
    public:
        using CollisionCallback = std::function<void(
                VoxelPhysicsBody* body,
                const glm::vec3& hitPos,
                const glm::vec3& hitNormal,
                float impactSpeed
        )>;

        VoxelPhysicsManager(MultiGridChunkManager* chunkMgr)
                : chunkManager(chunkMgr) {
                gravity = glm::vec3(0, -3.81f, 0);
        }

        VoxelPhysicsBody* createBody(
                const glm::vec3& position,
                float mass,
                VoxelPhysicsBody::ShapeType shape,
                const glm::vec3& extents
        );

        void update(float dt, std::vector<uint64_t>& removedBodies);
        void setCollisionCallback(CollisionCallback cb) { collisionCallback = cb; }
        void removeBody(uint64_t id);
        void removeBody(VoxelPhysicsBody* body);

    private:
        MultiGridChunkManager* chunkManager;
        std::vector<VoxelPhysicsBody> bodies;
        CollisionCallback collisionCallback;
        glm::vec3 gravity;

        bool checkCapsuleCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );
        glm::vec3 sampleNormal(const glm::vec3& worldPos);
        void resolveCollision(
                VoxelPhysicsBody& body,
                const glm::vec3& normal,
                float penetration,
                float impactSpeed
        );
        float sampleDensityTrilinear(const glm::vec3& worldPos);

        float getDensityAtWorldCorner(const glm::vec3& worldPos);
        bool checkBoxCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );
        glm::vec3 sampleGradient(const glm::vec3& worldPos);
        float sampleDensity(const glm::vec3& worldPos);
        bool checkVoxelCollision(
                VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );
    };
}