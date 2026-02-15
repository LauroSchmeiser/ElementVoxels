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
                gravity = glm::vec3(0, -2.81f, 0);  // Fixed gravity direction
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
        static std::vector<glm::vec3> buildUniqueVertexList(const std::vector<glm::vec3>& triangleVerts);

    private:
        MultiGridChunkManager* chunkManager;
        std::vector<VoxelPhysicsBody> bodies;
        CollisionCallback collisionCallback;
        glm::vec3 gravity = glm::vec3(0, -9.81f, 0);

        // Core collision functions
        bool checkVoxelCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );

        // Shape-specific collision
        bool checkSphereCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );

        bool checkBoxCollision(
                const VoxelPhysicsBody& body,
                glm::vec3& outNormal,
                float& outPenetration
        );

        // Helper functions
        float sampleDensity(const glm::vec3& worldPos);
        glm::vec3 sampleGradient(const glm::vec3& worldPos);
        glm::vec3 sampleNormal(const glm::vec3& worldPos);

        void resolveCollision(
                VoxelPhysicsBody& body,
                const glm::vec3& normal,
                float penetration,
                float impactSpeed
        );

        // World coordinate helpers (now using chunkManager)
        int worldToChunk(float worldPos) const {
                float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
                return (int)std::floor(worldPos / chunkWorldSize);
        }

        glm::vec3 getChunkMin(const ChunkCoord& coord) const {
                return glm::vec3(coord.x * CHUNK_SIZE * VOXEL_SIZE,
                                 coord.y * CHUNK_SIZE * VOXEL_SIZE,
                                 coord.z * CHUNK_SIZE * VOXEL_SIZE);
        }
    };

}