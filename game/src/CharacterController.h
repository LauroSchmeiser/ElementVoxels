#pragma once
#include "glm/glm.hpp"
#include "rendering/VoxelStructures.h"
#include "rendering/Chunk.h"
#include "rendering/MultiGridChunkManager.h"
#include <memory>

namespace gl3 {
    class CharacterController {
    public:
        struct CharacterState {
            glm::vec3 position = glm::vec3(0.0f, 10.0f, 0.0f);
            glm::vec3 velocity = glm::vec3(0.0f);
            bool isGrounded = false;
            bool isCrouching = false;
            bool isSprinting = false;
            float jumpBuffer = 0.0f;
            float coyoteTime = 0.0f;
        };

        CharacterController(MultiGridChunkManager* chunkManager, float height = 1.8f, float radius = 0.4f);

        void update(float deltaTime, const glm::vec3& moveInput,
                    bool jumpInput, bool sprintInput,
                    bool crouchInput, const glm::vec2& mouseDelta,
                    const glm::vec3& cameraForward, const glm::vec3& cameraRight);

        const CharacterState& getState() const { return state; }
        glm::vec3 getCameraPosition() const;

        // SDF-based collision query
        float getDistanceToSurface(const glm::vec3& position) const;
        glm::vec3 getSurfaceNormal(const glm::vec3& position, float epsilon = 0.1f) const;

        // Movement settings
        struct MovementSettings {
            float walkSpeed = 2.0f;
            float sprintSpeed = 5.0f;
            float crouchSpeed = 4.0f;
            float acceleration = 0.5f;
            float airControl = 1.0f;
            float friction = 8.0f;
            float airFriction = 2.0f;

            float jumpForce = 7.5f;
            float gravity = 0.0f;
            float terminalVelocity = 20.0f;

            float coyoteTimeDuration = 0.15f;
            float jumpBufferDuration = 0.1f;

            float crouchHeightMultiplier = 0.6f;

            // Collision settings
            float collisionRadius = 0.4f;
            float collisionHeight = 1.8f;
            float skinWidth = 0.05f;  // Small gap to prevent sticking
            float slopeLimit = 45.0f; // Max slope angle in degrees
        } settings;

    private:
        CharacterState state;
        MultiGridChunkManager* chunkManager;

        // Capsule collision (better for characters)
        struct Capsule {
            glm::vec3 position;
            float height;
            float radius;

            glm::vec3 getTop() const { return position + glm::vec3(0, height * 0.5f, 0); }
            glm::vec3 getBottom() const { return position - glm::vec3(0, height * 0.5f, 0); }
        };

        // SDF sampling from voxel density field
        float sampleDensity(const glm::vec3& worldPos) const;
        float interpolateDensity(float x, float y, float z, Chunk* chunk) const;

        // Collision resolution using sphere tracing
        bool sphereTrace(const glm::vec3& start, const glm::vec3& direction, float maxDistance,
                         glm::vec3& hitPos, glm::vec3& hitNormal, float& hitDistance) const;

        void resolveCapsuleCollision();
        void moveCapsule(glm::vec3& position, const glm::vec3& velocity, float deltaTime);

        // Helper methods
        glm::vec3 calculateWishVelocity(const glm::vec3& moveInput,
                                        const glm::vec3& forward, const glm::vec3& right) const;
        void applyFriction(float deltaTime);
        void accelerate(const glm::vec3& wishDir, float wishSpeed, float accel, float deltaTime);

        Capsule getCapsule() const;
        void updateGroundState();
    };
}