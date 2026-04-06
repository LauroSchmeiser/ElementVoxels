#pragma once

#include <glm/glm.hpp>
#include "rendering//MultiGridChunkManager.h"
#include "physics/VoxelPhysicsManager.h"

namespace gl3 {

    struct CharacterSettings {
        // Movement
        float walkSpeed = 40.0f;
        float sprintSpeed = 80.0f;
        float crouchSpeed = 20.5f;
        float acceleration = 50.0f;
        float friction = 6.0f;
        float airFriction = 0.2f;
        float airControl = 0.3f;

        // Jumping
        float jumpForce = 55.0f;
        float gravity = 25.0f;
        float terminalVelocity = 50.0f;
        float coyoteTimeDuration = 1.5f;
        float jumpBufferDuration = 0.1f;

        // Crouching
        float crouchHeightMultiplier = 0.5f;

        // NEW: Air slam settings
        float airSlamGravityMultiplier = 1.75f;    // How much faster you fall during slam
        float fallingGravityMultiplier = 1.5f;    // Normal falling gravity multiplier
        float airSlamInitialVelocity = 10.0f;     // Initial downward velocity when starting slam
        float airSlamDuration = 2.0f;             // How long slam effect lasts
        float airSlamImpactThreshold = 15.0f;     // Velocity threshold for impact effects

        // Ground detection
        float groundCheckDistance = 0.2f;         // How far below character to check for ground
    };

    struct CharacterState {
        glm::vec3 position = glm::vec3(0.0f, 10.0f, 0.0f);
        glm::vec3 velocity = glm::vec3(0.0f);

        bool isGrounded = false;
        bool isCrouching = false;
        bool isSprinting = false;
        bool isAirSlamming = false;      // NEW: Air slam state
        bool airSlamAvailable = true;    // NEW: Can perform air slam

        float coyoteTime = 0.0f;
        float jumpBuffer = 0.0f;
        float airSlamTimer = 0.0f;       // NEW: Air slam duration timer
    };

    class CharacterController {
    private:
        float height;
        float radius;
        float currentHeight;
        float defaultHeight;

        MultiGridChunkManager* chunkManager;
        VoxelPhysicsManager* physicsManager;

        CharacterState state;
        CharacterSettings settings;

        // Collision detection
        bool checkCollision(const glm::vec3& testPosition,
                            glm::vec3& outNormal,
                            float& outPenetration) const;

        // NEW: Improved ground detection
        bool checkIfGrounded() const;

        // Movement helpers
        void resolveCollisions();
        glm::vec3 calculateWishVelocity(const glm::vec3& moveInput,
                                        const glm::vec3& forward,
                                        const glm::vec3& right) const;
        void applyFriction(float deltaTime);
        void accelerate(const glm::vec3& wishDir, float wishSpeed,
                        float accel, float deltaTime);
        void move(float deltaTime);

        // NEW: Air slam functions
        void performAirSlam();
        void updateAirSlam(float deltaTime);

    public:
        CharacterController(MultiGridChunkManager* chunkMgr, VoxelPhysicsManager *physicsMgr,
                            float height = 2.0f,
                            float radius = 0.5f);


        //Callback when player collides with a physics body**
        using PlayerBodyCollisionCallback = std::function<void(
                VoxelPhysicsBody* body,
                const glm::vec3& hitPos,
                const glm::vec3& hitNormal,
                float playerSpeed
        )>;

        void setPlayerBodyCollisionCallback(PlayerBodyCollisionCallback cb) {
            playerBodyCollisionCallback = cb;
        }

        // Update with all inputs including air reset
        void update(float deltaTime,
                    const glm::vec3& moveInput,
                    bool jumpInput,
                    bool sprintInput,
                    bool crouchInput,
                    const glm::vec2& mouseDelta,
                    const glm::vec3& cameraForward,
                    const glm::vec3& cameraRight,
                    bool airResetInput);  // ADDED: air reset input

        // Getters
        glm::vec3 getPosition() const { return state.position; }
        glm::vec3 getVelocity() const { return state.velocity; }
        CharacterState getState() const { return state; }
        CharacterSettings getSettings() const { return settings; }

        // Setters
        void setPosition(const glm::vec3& pos) { state.position = pos; }
        void setSettings(const CharacterSettings& newSettings) { settings = newSettings; }

        // Camera
        glm::vec3 getCameraPosition() const;
        float getEyeHeight() const { return currentHeight * 0.85f; }

        // Debug
        bool checkPhysicsBodyCollision(const glm::vec3& testPosition,
                                       glm::vec3& outNormal,
                                       float& outPenetration,VoxelPhysicsBody** outBody) const;

    private:
        PlayerBodyCollisionCallback playerBodyCollisionCallback;

    };
}