#include <algorithm>
#include "Game.h"

namespace gl3 {
    CharacterController::CharacterController(MultiGridChunkManager* chunkMgr, float height, float radius)
            : height(height), radius(radius), currentHeight(height), defaultHeight(height),
              chunkManager(chunkMgr) {
    }


    glm::vec3 CharacterController::getCameraPosition() const {
        // Camera is slightly above character center (eye level)
        return state.position + glm::vec3(0.0f, getEyeHeight(), 0.0f);
    }

    bool CharacterController::testVoxelCollision(const glm::vec3& voxelPos) const {
        if (!chunkManager) return false; // Safety check

        // Convert world position to chunk coordinates
        int chunkX = static_cast<int>(glm::floor(voxelPos.x / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkY = static_cast<int>(glm::floor(voxelPos.y / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkZ = static_cast<int>(glm::floor(voxelPos.z / (CHUNK_SIZE * VOXEL_SIZE)));

        ChunkCoord coord{chunkX, chunkY, chunkZ};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) return false;

        // Convert to local voxel coordinates within chunk
        float localX = voxelPos.x - (coord.x * CHUNK_SIZE * VOXEL_SIZE);
        float localY = voxelPos.y - (coord.y * CHUNK_SIZE * VOXEL_SIZE);
        float localZ = voxelPos.z - (coord.z * CHUNK_SIZE * VOXEL_SIZE);

        // Convert to voxel grid coordinates
        int voxelX = static_cast<int>(glm::floor(localX / VOXEL_SIZE));
        int voxelY = static_cast<int>(glm::floor(localY / VOXEL_SIZE));
        int voxelZ = static_cast<int>(glm::floor(localZ / VOXEL_SIZE));

        // Clamp to valid range
        voxelX = glm::clamp(voxelX, 0,  CHUNK_SIZE * (int)std::ceil(VOXEL_SIZE));
        voxelY = glm::clamp(voxelY, 0, CHUNK_SIZE * (int)std::ceil(VOXEL_SIZE));
        voxelZ = glm::clamp(voxelZ, 0, CHUNK_SIZE * (int)std::ceil(VOXEL_SIZE));

        // Check if voxel is solid
        return chunk->voxels[voxelX][voxelY][voxelZ].isSolid();
    }

    glm::vec3 CharacterController::getAABBMin(const glm::vec3 &center) const {
        return center - glm::vec3(radius, currentHeight * 0.5f, radius);
    }

    glm::vec3 CharacterController::getAABBMax(const glm::vec3 &center) const {
        return center + glm::vec3(radius, currentHeight * 0.5f, radius);
    }

    bool CharacterController::checkCollision(const glm::vec3 &testPosition,
                                             glm::vec3 &outNormal,
                                             float &outPenetration) const {
        glm::vec3 aabbMin = getAABBMin(testPosition);
        glm::vec3 aabbMax = getAABBMax(testPosition);

        // Expand AABB slightly for collision detection
        const float skinWidth = 0.05f;
        aabbMin -= glm::vec3(skinWidth);
        aabbMax += glm::vec3(skinWidth);

        // Track deepest collision
        bool collisionFound = false;
        float deepestPenetration = 0.0f;
        glm::vec3 collisionNormal(0.0f);

        // Check voxels in AABB region
        for (float x = aabbMin.x; x <= aabbMax.x; x += std::ceil(VOXEL_SIZE)) {
            for (float y = aabbMin.y; y <= aabbMax.y; y += std::ceil(VOXEL_SIZE)) {
                for (float z = aabbMin.z; z <= aabbMax.z; z += std::ceil(VOXEL_SIZE)) {
                    glm::vec3 voxelPos(glm::floor(x / VOXEL_SIZE) * VOXEL_SIZE,
                                       glm::floor(y / VOXEL_SIZE) * VOXEL_SIZE,
                                       glm::floor(z / VOXEL_SIZE) * VOXEL_SIZE);

                    if (testVoxelCollision(voxelPos)) {
                        // Voxel AABB
                        glm::vec3 voxelMin = voxelPos;
                        glm::vec3 voxelMax = voxelPos + glm::vec3(VOXEL_SIZE);

                        // Calculate overlap
                        glm::vec3 overlap1 = aabbMax - voxelMin;
                        glm::vec3 overlap2 = voxelMax - aabbMin;

                        glm::vec3 overlap = glm::min(overlap1, overlap2);

                        // Find axis with smallest penetration
                        float minOverlap = glm::min(glm::min(overlap.x, overlap.y), overlap.z);

                        if (minOverlap > deepestPenetration) {
                            deepestPenetration = minOverlap;

                            // Determine collision normal
                            if (minOverlap == overlap.x) {
                                collisionNormal = glm::vec3(overlap1.x < overlap2.x ? 1.0f : -1.0f, 0.0f, 0.0f);
                            } else if (minOverlap == overlap.y) {
                                collisionNormal = glm::vec3(0.0f, overlap1.y < overlap2.y ? 1.0f : -1.0f, 0.0f);
                            } else {
                                collisionNormal = glm::vec3(0.0f, 0.0f, overlap1.z < overlap2.z ? 1.0f : -1.0f);
                            }

                            collisionFound = true;
                        }
                    }
                }
            }
        }

        if (collisionFound) {
            outNormal = collisionNormal;
            outPenetration = deepestPenetration;
        }

        return collisionFound;
    }

    void CharacterController::resolveCollisions() {
        const int maxIterations = 8;

        for (int i = 0; i < maxIterations; i++) {
            glm::vec3 normal;
            float penetration;

            if (checkCollision(state.position, normal, penetration)) {
                // Push character out of collision
                state.position += normal * (penetration + 0.0001f);

                // Remove velocity in collision direction
                float velDotNormal = glm::dot(state.velocity, normal);
                if (velDotNormal < 0) {
                    state.velocity -= normal * velDotNormal;
                }

                // Check if we're on ground
                if (normal.y > 0.5f) { // Mostly upward normal
                    state.isGrounded = true;
                    state.coyoteTime = settings.coyoteTimeDuration;
                }
            } else {
                break;
            }
        }
    }

    glm::vec3 CharacterController::calculateWishVelocity(const glm::vec3 &moveInput,
                                                         const glm::vec3 &forward,
                                                         const glm::vec3 &right) const {
        // Calculate movement direction relative to camera
        glm::vec3 wishDir = (forward * moveInput.z) + (right * moveInput.x);

        // Flatten to horizontal plane
        wishDir.y = 0.0f;

        if (glm::length(wishDir) > 0.001f) {
            wishDir = glm::normalize(wishDir);
        }

        return wishDir;
    }

    void CharacterController::applyFriction(float deltaTime) {
        if (state.isGrounded) {
            float speed = glm::length(state.velocity);

            if (speed > 0.01f) {
                float drop = speed * settings.friction * deltaTime;
                float newSpeed = glm::max(speed - drop, 0.0f);
                state.velocity *= newSpeed / speed;
            }
        } else {
            // Air friction (much lower)
            float speed = glm::length(state.velocity);

            if (speed > 0.01f) {
                float drop = speed * settings.airFriction * deltaTime;
                float newSpeed = glm::max(speed - drop, 0.0f);
                state.velocity *= newSpeed / speed;
            }
        }
    }

    void CharacterController::accelerate(const glm::vec3 &wishDir, float wishSpeed,
                                         float accel, float deltaTime) {
        float currentSpeed = glm::dot(state.velocity, wishDir);
        float addSpeed = wishSpeed - currentSpeed;

        if (addSpeed <= 0) return;

        float accelSpeed = accel * wishSpeed * deltaTime;
        accelSpeed = glm::min(accelSpeed, addSpeed);

        state.velocity += wishDir * accelSpeed;
    }

    void CharacterController::move(float deltaTime) {
        // Apply gravity
        if (!state.isGrounded) {
            state.velocity.y -= settings.gravity * deltaTime;
            state.velocity.y = glm::max(state.velocity.y, -settings.terminalVelocity);
        }

        // Update position
        glm::vec3 moveStep = state.velocity * deltaTime;

        // Check for collisions during movement
        glm::vec3 newPosition = state.position;

        // Move in small steps for more accurate collision
        const float stepSize = 0.5f;
        int steps = glm::max(1, static_cast<int>(glm::length(moveStep) / stepSize));
        glm::vec3 step = moveStep / static_cast<float>(steps);

        for (int i = 0; i < steps; i++) {
            newPosition += step;

            glm::vec3 normal;
            float penetration;
            if (checkCollision(newPosition, normal, penetration)) {
                // Slide along collision plane
                glm::vec3 remainingStep = step * (static_cast<float>(steps - i - 1) / static_cast<float>(steps));

                // Remove component into collision
                float intoCollision = glm::dot(remainingStep, normal);
                remainingStep -= normal * intoCollision;

                // Try sliding
                glm::vec3 slidePosition = newPosition + remainingStep;
                if (!checkCollision(slidePosition, normal, penetration)) {
                    newPosition = slidePosition;
                }
                break;
            }
        }

        state.position = newPosition;
    }

    void CharacterController::update(float deltaTime, const glm::vec3& moveInput,
                                     bool jumpInput, bool sprintInput,
                                     bool crouchInput, const glm::vec2& mouseDelta,
                                     const glm::vec3& cameraForward, const glm::vec3& cameraRight) {
        // Update coyote time and jump buffer
        if (state.isGrounded) {
            state.coyoteTime = settings.coyoteTimeDuration;
        } else {
            state.coyoteTime -= deltaTime;
        }

        if (jumpInput) {
            state.jumpBuffer = settings.jumpBufferDuration;
        } else {
            state.jumpBuffer = glm::max(0.0f, state.jumpBuffer - deltaTime);
        }

        // Handle crouching
        bool wantsCrouch = crouchInput;
        if (wantsCrouch != state.isCrouching) {
            state.isCrouching = wantsCrouch;
            currentHeight = state.isCrouching ?
                            defaultHeight * settings.crouchHeightMultiplier : defaultHeight;
        }

        // Handle sprinting (can't sprint while crouching)
        state.isSprinting = sprintInput && !state.isCrouching && state.isGrounded;

        // Calculate movement speed
        float targetSpeed = state.isCrouching ? settings.crouchSpeed :
                            (state.isSprinting ? settings.sprintSpeed : settings.walkSpeed);

        // Get camera-relative movement vectors (you'll need to pass these or calculate from camera)
        // For now, assume forward/right are global axes
        glm::vec3 forward = glm::normalize(glm::vec3(cameraForward.x, 0.0f, cameraForward.z));
        glm::vec3 right = glm::normalize(glm::vec3(cameraRight.x, 0.0f, cameraRight.z));

        // Calculate wish velocity with camera-relative vectors
        glm::vec3 wishDir = calculateWishVelocity(moveInput, forward, right);

        // Apply friction
        applyFriction(deltaTime);

        // Apply acceleration
        float accel = state.isGrounded ? settings.acceleration :
                      settings.acceleration * settings.airControl;
        accelerate(wishDir, targetSpeed, accel, deltaTime);

        // Handle jumping
        if (state.jumpBuffer > 0.0f && state.coyoteTime > 0.0f) {
            state.velocity.y = settings.jumpForce;
            state.isGrounded = false;
            state.jumpBuffer = 0.0f;
            state.coyoteTime = 0.0f;
        }

        // Move character
        move(deltaTime);

        // Resolve collisions
        resolveCollisions();

        // Update ground state
        if (!state.isGrounded) {
            // Raycast down to check if we've landed
            glm::vec3 groundCheckPos = state.position - glm::vec3(0.0f, currentHeight * 0.5f + 0.1f, 0.0f);
            glm::vec3 normal;
            float penetration;
            if (checkCollision(groundCheckPos, normal, penetration)) {
                if (normal.y > 0.5f) {
                    state.isGrounded = true;
                    state.velocity.y = glm::max(state.velocity.y, 0.0f); // Stop downward velocity
                }
            }
        }
    }
}