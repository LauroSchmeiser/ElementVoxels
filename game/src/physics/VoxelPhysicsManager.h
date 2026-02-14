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
                gravity = glm::vec3(0, 0.2f, 0);
        }

        // Add a physics body
        VoxelPhysicsBody* createBody(
                const glm::vec3& position,
                float mass,
                VoxelPhysicsBody::ShapeType shape,
                const glm::vec3& extents
        );

        // Update all bodies
        void update(float dt, std::vector<uint64_t>& removedBodies);
        // Set collision callback
        void setCollisionCallback(CollisionCallback cb) {
            collisionCallback = cb;
        }

        // Cleanup
        void removeBody(uint64_t id);
        void removeBody(VoxelPhysicsBody* body);
        static std::vector<glm::vec3> buildUniqueVertexList(const std::vector<glm::vec3>& triangleVerts);

    private:
        MultiGridChunkManager* chunkManager;
        std::vector<VoxelPhysicsBody> bodies;
        CollisionCallback collisionCallback;
        glm::vec3 gravity = glm::vec3(0, 0, 0);

        // Collision detection
        bool checkVoxelCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );

        float sampleSDF(const glm::vec3& worldPos);
        glm::vec3 sampleNormal(const glm::vec3& worldPos);

        // Collision response
        void resolveCollision(
                VoxelPhysicsBody& body,
                const glm::vec3& normal,
                float penetration,
                float impactSpeed
        );
    };

}