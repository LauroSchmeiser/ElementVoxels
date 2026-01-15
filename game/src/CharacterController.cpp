#include <algorithm>
#include "Game.h"

namespace gl3 {
    CharacterController::CharacterController(MultiGridChunkManager* chunkMgr, float height, float radius)
            : chunkManager(chunkMgr) {
        // Use settings instead of separate height/radius variables
        settings.collisionHeight = height;
        settings.collisionRadius = radius;
    }

    glm::vec3 CharacterController::getCameraPosition() const {
        return state.position + glm::vec3(0.0f, settings.collisionHeight * 0.5f, 0.0f);
    }

    CharacterController::Capsule CharacterController::getCapsule() const {
        return Capsule{state.position, settings.collisionHeight, settings.collisionRadius};
    }

    float CharacterController::sampleDensity(const glm::vec3& worldPos) const {
        if (!chunkManager) return -1000.0f;

        // Convert to chunk coordinates
        int chunkX = static_cast<int>(glm::floor(worldPos.x / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkY = static_cast<int>(glm::floor(worldPos.y / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkZ = static_cast<int>(glm::floor(worldPos.z / (CHUNK_SIZE * VOXEL_SIZE)));

        ChunkCoord coord{chunkX, chunkY, chunkZ};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) return -1000.0f;

        // Convert to local coordinates within chunk
        float localX = worldPos.x - (coord.x * CHUNK_SIZE * VOXEL_SIZE);
        float localY = worldPos.y - (coord.y * CHUNK_SIZE * VOXEL_SIZE);
        float localZ = worldPos.z - (coord.z * CHUNK_SIZE * VOXEL_SIZE);

        // Convert to voxel grid coordinates (floating point for interpolation)
        float voxelX = localX / VOXEL_SIZE;
        float voxelY = localY / VOXEL_SIZE;
        float voxelZ = localZ / VOXEL_SIZE;

        // Clamp to valid range
        voxelX = glm::clamp(voxelX, 0.0f, static_cast<float>(CHUNK_SIZE));
        voxelY = glm::clamp(voxelY, 0.0f, static_cast<float>(CHUNK_SIZE));
        voxelZ = glm::clamp(voxelZ, 0.0f, static_cast<float>(CHUNK_SIZE));

        return interpolateDensity(voxelX, voxelY, voxelZ, chunk);
    }

    float CharacterController::interpolateDensity(float x, float y, float z, Chunk* chunk) const {
        int x0 = static_cast<int>(glm::floor(x));
        int y0 = static_cast<int>(glm::floor(y));
        int z0 = static_cast<int>(glm::floor(z));

        int x1 = glm::min(x0 + 1, CHUNK_SIZE);
        int y1 = glm::min(y0 + 1, CHUNK_SIZE);
        int z1 = glm::min(z0 + 1, CHUNK_SIZE);

        float tx = x - x0;
        float ty = y - y0;
        float tz = z - z0;

        // Trilinear interpolation
        float c00 = chunk->voxels[x0][y0][z0].density * (1 - tz) + chunk->voxels[x0][y0][z1].density * tz;
        float c01 = chunk->voxels[x0][y1][z0].density * (1 - tz) + chunk->voxels[x0][y1][z1].density * tz;
        float c10 = chunk->voxels[x1][y0][z0].density * (1 - tz) + chunk->voxels[x1][y0][z1].density * tz;
        float c11 = chunk->voxels[x1][y1][z0].density * (1 - tz) + chunk->voxels[x1][y1][z1].density * tz;

        float c0 = c00 * (1 - ty) + c01 * ty;
        float c1 = c10 * (1 - ty) + c11 * ty;

        return c0 * (1 - tx) + c1 * tx;
    }

    float CharacterController::getDistanceToSurface(const glm::vec3& position) const {
        // Positive = outside, Negative = inside, Zero = on surface
        return sampleDensity(position);
    }

    glm::vec3 CharacterController::getSurfaceNormal(const glm::vec3& position, float epsilon) const {
        // Calculate normal using central differences
        float d = getDistanceToSurface(position);
        float dx = getDistanceToSurface(position + glm::vec3(epsilon, 0, 0)) - d;
        float dy = getDistanceToSurface(position + glm::vec3(0, epsilon, 0)) - d;
        float dz = getDistanceToSurface(position + glm::vec3(0, 0, epsilon)) - d;

        return glm::normalize(glm::vec3(dx, dy, dz));
    }

    bool CharacterController::sphereTrace(const glm::vec3& start, const glm::vec3& direction,
                                          float maxDistance, glm::vec3& hitPos,
                                          glm::vec3& hitNormal, float& hitDistance) const {
        float totalDistance = 0.0f;
        glm::vec3 currentPos = start;

        for (int i = 0; i < 32; i++) { // Max iterations
            float distance = getDistanceToSurface(currentPos);

            if (distance < 0) {
                // We're inside the surface
                hitPos = currentPos;
                hitNormal = getSurfaceNormal(currentPos);
                hitDistance = totalDistance;
                return true;
            }

            if (distance < 0.001f) { // Close enough to surface
                hitPos = currentPos;
                hitNormal = getSurfaceNormal(currentPos);
                hitDistance = totalDistance;
                return true;
            }

            if (totalDistance > maxDistance) {
                return false;
            }

            // March forward
            float step = glm::max(distance * 0.5f, 0.01f); // Adaptive step size
            currentPos += direction * step;
            totalDistance += step;
        }

        return false;
    }

    void CharacterController::resolveCapsuleCollision() {
        Capsule capsule = getCapsule();
        const int maxIterations = 4;

        for (int iter = 0; iter < maxIterations; iter++) {
            // Test multiple points along the capsule
            const int numSamples = 8;
            bool collisionFound = false;
            glm::vec3 averageNormal(0.0f);
            float totalPenetration = 0.0f;
            int numCollisions = 0;

            // Sample points along the capsule
            for (int i = 0; i < numSamples; i++) {
                float t = static_cast<float>(i) / (numSamples - 1);
                glm::vec3 samplePoint = capsule.getBottom() +
                                        glm::vec3(0, t * capsule.height, 0);

                // Expand sample sphere by capsule radius
                float distance = getDistanceToSurface(samplePoint);
                float penetration = capsule.radius - distance;

                if (penetration > 0) {
                    collisionFound = true;
                    numCollisions++;
                    totalPenetration += penetration;

                    // Get normal at surface
                    glm::vec3 normal = getSurfaceNormal(samplePoint);
                    averageNormal += normal;
                }
            }

            if (!collisionFound) break;

            if (numCollisions > 0) {
                averageNormal = glm::normalize(averageNormal);
                float avgPenetration = totalPenetration / numCollisions;

                // Push out of collision
                state.position += averageNormal * (avgPenetration + settings.skinWidth);

                // Remove velocity in collision direction
                float velDotNormal = glm::dot(state.velocity, averageNormal);
                if (velDotNormal < 0) {
                    state.velocity -= averageNormal * velDotNormal;
                }

                // Check if ground (normal mostly upward)
                if (averageNormal.y > 0.7f) {
                    state.isGrounded = true;
                    state.coyoteTime = settings.coyoteTimeDuration;
                }
            }
        }
    }

    void CharacterController::moveCapsule(glm::vec3& position, const glm::vec3& velocity, float deltaTime) {
        if (glm::length(velocity) < 0.001f) return;

        glm::vec3 direction = glm::normalize(velocity);
        float remainingDistance = glm::length(velocity) * deltaTime;
        const float minStep = 0.01f;

        // Adaptive stepping based on distance to surface
        while (remainingDistance > minStep) {
            Capsule capsule{position, settings.collisionHeight, settings.collisionRadius};

            // Find safe step distance
            float safeDistance = remainingDistance;

            // Test multiple points along capsule in movement direction
            for (int i = 0; i < 5; i++) {
                float t = static_cast<float>(i) / 4.0f;
                glm::vec3 testPoint = capsule.getBottom() +
                                      glm::vec3(0, t * capsule.height, 0);

                glm::vec3 hitPos, hitNormal;
                float hitDist;

                if (sphereTrace(testPoint, direction, remainingDistance + capsule.radius,
                                hitPos, hitNormal, hitDist)) {
                    safeDistance = glm::min(safeDistance, hitDist - capsule.radius - settings.skinWidth);
                }
            }

            safeDistance = glm::max(safeDistance, minStep);
            position += direction * safeDistance;
            remainingDistance -= safeDistance;

            // Resolve any collision from this step
            resolveCapsuleCollision();

            if (safeDistance < minStep) break; // Stuck
        }
    }

    glm::vec3 CharacterController::calculateWishVelocity(const glm::vec3& moveInput,
                                                         const glm::vec3& forward,
                                                         const glm::vec3& right) const {
        glm::vec3 wishDir = (forward * moveInput.z) + (right * moveInput.x);
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
                state.velocity *= glm::max(speed - drop, 0.0f) / speed;
            }
        } else {
            float speed = glm::length(state.velocity);
            if (speed > 0.01f) {
                float drop = speed * settings.airFriction * deltaTime;
                state.velocity *= glm::max(speed - drop, 0.0f) / speed;
            }
        }
    }

    void CharacterController::accelerate(const glm::vec3& wishDir, float wishSpeed,
                                         float accel, float deltaTime) {
        float currentSpeed = glm::dot(state.velocity, wishDir);
        float addSpeed = wishSpeed - currentSpeed;

        if (addSpeed <= 0) return;

        float accelSpeed = accel * deltaTime;
        accelSpeed = glm::min(accelSpeed, addSpeed);

        state.velocity += wishDir * accelSpeed;
    }

    void CharacterController::updateGroundState() {
        // Cast multiple rays downward to detect ground
        Capsule capsule = getCapsule();
        const int numRays = 5;
        int groundHits = 0;

        for (int i = 0; i < numRays; i++) {
            float angle = (static_cast<float>(i) / numRays) * glm::two_pi<float>();
            glm::vec3 rayStart = capsule.getBottom() +
                                 glm::vec3(cos(angle) * capsule.radius * 0.7f, 0, sin(angle) * capsule.radius * 0.7f);

            glm::vec3 hitPos, hitNormal;
            float hitDist;

            if (sphereTrace(rayStart, glm::vec3(0, -1, 0), 0.2f, hitPos, hitNormal, hitDist)) {
                // Check slope angle
                float slopeAngle = glm::degrees(glm::acos(glm::dot(hitNormal, glm::vec3(0, 1, 0))));
                if (slopeAngle < settings.slopeLimit) {
                    groundHits++;
                }
            }
        }

        state.isGrounded = (groundHits > numRays / 2);
        if (state.isGrounded) {
            state.coyoteTime = settings.coyoteTimeDuration;
        }
    }

    void CharacterController::update(float deltaTime, const glm::vec3& moveInput,
                                     bool jumpInput, bool sprintInput,
                                     bool crouchInput, const glm::vec2& mouseDelta,
                                     const glm::vec3& cameraForward, const glm::vec3& cameraRight) {
        // Update timing states
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
            settings.collisionHeight = state.isCrouching ?
                                       settings.collisionHeight * settings.crouchHeightMultiplier :
                                       1.8f; // Reset to default
        }

        // Handle sprinting
        state.isSprinting = sprintInput && !state.isCrouching && state.isGrounded;

        // Calculate target speed
        float targetSpeed = state.isCrouching ? settings.crouchSpeed :
                            (state.isSprinting ? settings.sprintSpeed : settings.walkSpeed);

        // Get camera-relative movement
        glm::vec3 forward = glm::normalize(glm::vec3(cameraForward.x, 0.0f, cameraForward.z));
        glm::vec3 right = glm::normalize(glm::vec3(cameraRight.x, 0.0f, cameraRight.z));

        glm::vec3 wishDir = calculateWishVelocity(moveInput, forward, right);

        // Apply friction
        applyFriction(deltaTime);

        // Apply acceleration
        float accel = state.isGrounded ? settings.acceleration :
                      settings.acceleration * settings.airControl;
        accelerate(wishDir, targetSpeed, accel, deltaTime);

        // Apply gravity
        if (!state.isGrounded) {
            state.velocity.y -= settings.gravity * deltaTime;
            state.velocity.y = glm::max(state.velocity.y, -settings.terminalVelocity);
        }

        // Handle jumping
        if (state.jumpBuffer > 0.0f && state.coyoteTime > 0.0f) {
            state.velocity.y = settings.jumpForce;
            state.isGrounded = false;
            state.jumpBuffer = 0.0f;
            state.coyoteTime = 0.0f;
        }

        // Move with collision
        moveCapsule(state.position, state.velocity, deltaTime);

        // Update ground state
        updateGroundState();
    }
}