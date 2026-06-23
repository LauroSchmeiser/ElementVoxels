#pragma once

#include <glm/glm.hpp>
#include "physics/VoxelPhysicsManager.h"
#include "rendering/FixedGridChunkManager.h"

namespace gl3 {

    struct CharacterSettings {
        // Movement
        float walkSpeed = 50.0f;
        float sprintSpeed = 150.0f;
        float crouchSpeed = 20.5f;
        float acceleration = 5.0f;
        float friction = 6.0f;
        float airFriction = 0.2f;
        float airControl = 0.3f;

        // Jumping
        float jumpForce = 55.0f;
        glm::vec3 gravityDir = glm::vec3(0.0f, -1.0f, 0.0f);
        glm::vec3 lastGravPoint = glm::vec3(0.0f, 0.0f, 0.0f);;
        float gravity= 10.0f;
        float gravityMaxIntensity = 50.0f;
        float gravityMinIntensity = 0.1f;

        float terminalVelocity = 750.0f;
        float coyoteTimeDuration = 3.5f;
        float jumpBufferDuration = 0.75f;

        // Crouching
        float crouchHeightMultiplier = 0.5f;

        // NEW: Air slam settings
        float airSlamGravityMultiplier = 1.75f;    // How much faster you fall during slam
        float fallingGravityMultiplier = 1.0f;    // Normal falling gravity multiplier
        float airSlamInitialVelocity = 10.0f;     // Initial downward velocity when starting slam
        float airSlamDuration = 2.0f;             // How long slam effect lasts
        float airSlamImpactThreshold = 15.0f;     // Velocity threshold for impact effects

        // Ground detection
        float groundCheckDistance = 0.2f;         // How far below character to check for ground

        // Surface landing / adhesion
        float landingMinApproachSpeed = 0.05f;    // Low threshold for gentle landings
        float adhesionDuration = 0.5f;
        float adhesionMaxDistance = 20.0f;
        float adhesionSnapDistance = 1.0f;        // Can afford larger since detection is tighter
        float adhesionAcceleration = 0.5f;
        float minGroundNormalDot = 0.15f;          // More lenient for angled surfaces

        // Camera / orientation smoothing
        float upLerpSpeed = 1.5f;                 // Medium speed (adaptive logic handles extremes)
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

        // Adhesion / landing state
        bool isSurfaceAdhered = false;
        glm::vec3 adheredNormal = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 lastGroundPoint = glm::vec3(0.0f);
        float adhesionTimer = 0.0f;

        // Smoothed orientation up used by movement/camera
        glm::vec3 currentUp = glm::vec3(0.0f, 1.0f, 0.0f);
    };

    class CharacterController {
    private:
        float height;
        float radius;
        float currentHeight;
        float defaultHeight;

        FixedGridChunkManager* chunkManager;
        VoxelPhysicsManager* physicsManager;
        CharacterState state;
    public:
        CharacterSettings settings;

        bool isSurfaceAdhered() const { return state.isSurfaceAdhered; }

        void resolveCameraCollision(glm::vec3& cameraPos, float eyeRadius) const;

    private:
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

        bool findGroundContact(glm::vec3& outPoint, glm::vec3& outNormal, float& outDistance) const;
        bool shouldLandOnContact(const glm::vec3& contactNormal) const;
        void beginSurfaceAdhesion(const glm::vec3& contactPoint, const glm::vec3& contactNormal);
        void updateSurfaceAdhesion(float deltaTime);
        void updateOrientation(float deltaTime);

    public:
        CharacterController(FixedGridChunkManager *chunkMgr, VoxelPhysicsManager *physicsMgr,
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

        void setGravityDirection(const glm::vec3& g);
        glm::vec3 getGravityDirection() const { return settings.gravityDir; }
        glm::vec3 getUpDirection() const;

        float getRadius() const { return radius; }
    private:
        PlayerBodyCollisionCallback playerBodyCollisionCallback;
        bool depenetrateSphere(glm::vec3& center, float sphereRadius, int maxIterations = 8) const;
        void enforceCameraClearance();
    private:
        bool checkCameraCollision(const glm::vec3& cameraPos, float eyeRadius,
                                  glm::vec3& outNormal, float& outPenetration) const;

        void enforceCameraClearanceAggressive();
    };
}