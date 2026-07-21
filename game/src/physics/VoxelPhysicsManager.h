#pragma once
#include "VoxelPhysicsBody.h"
#include "../rendering/FixedGridChunkManager.h"
#include <vector>
#include <functional>
#include <deque>

namespace gl3 {

    // Callback when body hits voxel world
    class VoxelPhysicsManager {
    public:
        using VoxelCollisionCallback = std::function<void(
                VoxelPhysicsBody* body,
                const glm::vec3& hitPos,
                const glm::vec3& hitNormal,
                float impactSpeed
        )>;

        //Callback when two bodies collide
        using BodyBodyCollisionCallback = std::function<void(
                VoxelPhysicsBody* bodyA,
                VoxelPhysicsBody* bodyB,
                const glm::vec3& hitPos,
                const glm::vec3& hitNormal,
                float impactSpeed
        )>;
        using BodyBodyPreSolveCallback = std::function<bool(
                VoxelPhysicsBody* bodyA,
                VoxelPhysicsBody* bodyB,
                const glm::vec3& hitPos,
                const glm::vec3& hitNormal,
                float impactSpeed
        )>;
        using BodyStepCallback = std::function<void(
                VoxelPhysicsBody* body,
                const glm::vec3& from,
                const glm::vec3& to
        )>;


        VoxelPhysicsManager(FixedGridChunkManager* chunkMgr)
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
        void setVoxelCollisionCallback(VoxelCollisionCallback cb) { voxelCollisionCallback = cb; }
        void setBodyBodyCollisionCallback(BodyBodyCollisionCallback cb) { bodyBodyCollisionCallback = cb; }
        void setBodyBodyPreSolveCallback(BodyBodyPreSolveCallback cb) { bodyBodyPreSolveCallback = cb; }
        void setBodyStepCallback(BodyStepCallback cb) { bodyStepCallback = cb; }
        void removeBody(uint64_t id);
        const std::vector<std::unique_ptr<VoxelPhysicsBody>>& getBodies() const { return bodies; }

        VoxelPhysicsBody* getBodyById(uint64_t id) const;

    private:
        FixedGridChunkManager* chunkManager;
        std::vector<std::unique_ptr<VoxelPhysicsBody>> bodies;
        VoxelCollisionCallback voxelCollisionCallback;
        BodyBodyCollisionCallback bodyBodyCollisionCallback;
        BodyBodyPreSolveCallback bodyBodyPreSolveCallback;
        BodyStepCallback bodyStepCallback;


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
        void resolveBodyBodyCollision(
                VoxelPhysicsBody& bodyA,
                VoxelPhysicsBody& bodyB,
                const glm::vec3& normal,
                float penetration
        );
        bool checkBodyBodyCollision(
                VoxelPhysicsBody& bodyA,
                VoxelPhysicsBody& bodyB,
                glm::vec3& outNormal,
                float& outPenetration
        );

        bool sphereIntersectsWorld(const VoxelPhysicsBody &body, const glm::vec3 &center, glm::vec3 &outNormal,
                                   float &outPenetration);

        float sampleFluidDensityTrilinear(const glm::vec3& worldPos);
        bool isPointInFluid(const glm::vec3& worldPos);

        float fluidDrag = 0.85f;
        float fluidBuoyancy = 3.0f;
        float fluidDensityScale = 0.2f;
    };
}