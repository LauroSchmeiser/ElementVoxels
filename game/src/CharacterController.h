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
            glm::vec3 acceleration = glm::vec3(0.0f);
            bool isGrounded = false;
            bool isCrouching = false;
            bool isSprinting = false;
            float jumpBuffer = 0.0f;
            float coyoteTime = 0.0f;
        };

        CharacterController(MultiGridChunkManager* chunkManager, float height = 1.8f, float radius = 0.4f);

        void CharacterController::update(float deltaTime, const glm::vec3& moveInput,
                                         bool jumpInput, bool sprintInput,
                                         bool crouchInput, const glm::vec2& mouseDelta,
                                         const glm::vec3& cameraForward, const glm::vec3& cameraRight);

        const CharacterState &getState() const { return state; }

        glm::vec3 getCameraPosition() const;

        // Collision methods
        bool checkCollision(const glm::vec3 &testPosition, glm::vec3 &outNormal, float &outPenetration) const;

        void resolveCollisions();

        // Getters
        float getHeight() const { return height; }

        float getEyeHeight() const { return height - 0.2f; }

        float getRadius() const { return radius; }

        glm::vec3 getFeetPosition() const { return state.position - glm::vec3(0.0f, height * 0.5f, 0.0f); }

        glm::vec3 getHeadPosition() const { return state.position + glm::vec3(0.0f, height * 0.5f, 0.0f); }

        // Movement settings
        struct MovementSettings {
            float walkSpeed = 20.0f;
            float sprintSpeed = 50.0f;
            float crouchSpeed = 3.0f;
            float acceleration = 50.0f;
            float airControl = 0.9f;
            float friction = 8.0f;
            float airFriction = 0.9f;

            float jumpForce = 45.5f;
            float gravity = 20.0f;
            float terminalVelocity = 100.0f;

            float coyoteTimeDuration = 1.5f;
            float jumpBufferDuration = 0.25f;

            float crouchHeightMultiplier = 0.6f;
        } settings;

    private:
        CharacterState state;
        MultiGridChunkManager* chunkManager;
        // Character dimensions
        float height;
        float radius;
        float currentHeight;
        float defaultHeight;

        // Helper methods
        glm::vec3
        calculateWishVelocity(const glm::vec3 &moveInput, const glm::vec3 &forward, const glm::vec3 &right) const;

        void applyFriction(float deltaTime);

        void accelerate(const glm::vec3 &wishDir, float wishSpeed, float accel, float deltaTime);

        void move(float deltaTime);

        // Collision helpers
        bool testVoxelCollision(const glm::vec3 &voxelPos) const;

        glm::vec3 getAABBMin(const glm::vec3 &center) const;

        glm::vec3 getAABBMax(const glm::vec3 &center) const;
    };
}