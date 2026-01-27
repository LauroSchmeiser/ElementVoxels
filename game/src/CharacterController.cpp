#include <algorithm>
#include "Game.h"

namespace gl3 {
    CharacterController::CharacterController(MultiGridChunkManager *chunkMgr, float height, float radius)
            : height(height), radius(radius), currentHeight(height), defaultHeight(height),
              chunkManager(chunkMgr) {
    }

    static float getDensityAtWorld(MultiGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        if (!chunkManager) return -10000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);
        if (!chunk) return -10000.0f;

        glm::vec3 chunkMin = glm::vec3(coord.x * chunkWorldSize,
                                       coord.y * chunkWorldSize,
                                       coord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        // Round to nearest integer sample (we expect caller to provide exact voxel corner positions)
        int ix = static_cast<int>(std::lround(local.x));
        int iy = static_cast<int>(std::lround(local.y));
        int iz = static_cast<int>(std::lround(local.z));

        // Clamp to available sample range [0 .. CHUNK_SIZE]
        ix = glm::clamp(ix, 0, CHUNK_SIZE);
        iy = glm::clamp(iy, 0, CHUNK_SIZE);
        iz = glm::clamp(iz, 0, CHUNK_SIZE);

        return chunk->voxels[ix][iy][iz].density;
    }

    // Trilinear sample of the density field at arbitrary world position.
    // This handles chunk boundaries by evaluating the 8 corner sample WORLD positions
    // and asking getDensityAtWorld() for each corner (which finds the correct chunk).
    static float sampleDensityAtWorld(MultiGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        if (!chunkManager) return -10000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int baseCX = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int baseCY = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int baseCZ = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord baseCoord{baseCX, baseCY, baseCZ};
        glm::vec3 chunkMin = glm::vec3(baseCoord.x * chunkWorldSize,
                                       baseCoord.y * chunkWorldSize,
                                       baseCoord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        // fractional coordinates inside cell
        float fx = local.x - std::floor(local.x);
        float fy = local.y - std::floor(local.y);
        float fz = local.z - std::floor(local.z);

        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));

        // Build the eight corner sample world positions and query density for each.
        // Using explicit world positions avoids complex wrapping logic at chunk boundaries.
        float samples[2][2][2];
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    glm::vec3 cornerWorld =
                            chunkMin + glm::vec3((float)(ix + dx), (float)(iy + dy), (float)(iz + dz)) * VOXEL_SIZE;
                    samples[dx][dy][dz] = getDensityAtWorld(chunkManager, cornerWorld);
                }
            }
        }

        // trilinear interpolation
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

        float c00 = lerp(samples[0][0][0], samples[1][0][0], fx);
        float c10 = lerp(samples[0][1][0], samples[1][1][0], fx);
        float c01 = lerp(samples[0][0][1], samples[1][0][1], fx);
        float c11 = lerp(samples[0][1][1], samples[1][1][1], fx);

        float c0 = lerp(c00, c10, fy);
        float c1 = lerp(c01, c11, fy);

        return lerp(c0, c1, fz);
    }

    // Estimate normal from central differences of the sampled density field.
    // Return an OUTWARD normal (pointing from solid -> free space).
    static glm::vec3 sampleNormalAtWorld(MultiGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        const float e = VOXEL_SIZE * 0.5f; // small offset for gradients
        float dx = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(e, 0, 0)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(e, 0, 0));
        float dy = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(0, e, 0)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(0, e, 0));
        float dz = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(0, 0, e)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(0, 0, e));
        glm::vec3 grad(dx, dy, dz);

        float len = glm::length(grad);
        if (len < 1e-6f) {
            // fallback to upward normal to avoid NaNs
            return glm::vec3(0.0f, 1.0f, 0.0f);
        }

        // IMPORTANT: density increases into the solid, so gradient points INTO the solid.
        // We want the outward surface normal (from solid -> free space) so negate it.
        return -glm::normalize(grad);
    }



    // Capsule SDF probe: sample along capsule center-line and find deepest penetration.
// Returns true when penetration > skinWidth. outNormal is outward, outPenetration is positive depth.
    bool CharacterController::checkCollision(const glm::vec3 &testPosition,
                                             glm::vec3 &outNormal,
                                             float &outPenetration) const {
        if (!chunkManager) return false;

        // Capsule center segment
        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        glm::vec3 p0 = testPosition - up * halfSegment; // bottom
        glm::vec3 p1 = testPosition + up * halfSegment; // top
        float segmentLength = glm::length(p1 - p0);

        const float maxStep = VOXEL_SIZE * 0.5f;
        int steps = 1;
        if (segmentLength > 0.0f) steps = glm::clamp((int)std::ceil(segmentLength / maxStep), 2, 64);
        else steps = 2;

        float bestPen = -std::numeric_limits<float>::infinity();
        glm::vec3 bestSamplePos = testPosition;

        // find deepest penetration along center-line
        for (int i = 0; i <= steps; ++i) {
            float t = (steps > 0) ? (float)i / (float)steps : 0.0f;
            glm::vec3 samplePos = glm::mix(p0, p1, t);

            float sdf = sampleDensityAtWorld(chunkManager, samplePos); // positive = inside
            float s = sdf - radius; // positive => penetration amount for capsule surface

            if (s > bestPen) {
                bestPen = s;
                bestSamplePos = samplePos;
            }

            // early out if deep
            if (bestPen > VOXEL_SIZE * 2.0f) break;
        }

        // tuning: a small skin avoids jitter from float noise / marching-cubes isosurface fuzz
        const float skinWidth = 0.02f * VOXEL_SIZE; // ~2% of voxel size (tune)
        if (!(bestPen > skinWidth)) {
            // no meaningful collision
            return false;
        }

        // compute outward normal at the best sample position
        glm::vec3 normal = sampleNormalAtWorld(chunkManager, bestSamplePos);

        normal = glm::normalize(normal);

        // Optional: small smoothing to reduce jitter where normal flips quickly:
        // sample neighbor normals and average (cheap: 3 samples)
        const float smoothEps = VOXEL_SIZE * 0.25f;
        glm::vec3 n1 = sampleNormalAtWorld(chunkManager, bestSamplePos + normal * smoothEps);
        glm::vec3 n2 = sampleNormalAtWorld(chunkManager, bestSamplePos - normal * smoothEps);
        normal = glm::normalize(normal + 0.5f * n1 + 0.5f * n2);

        // Cap how much we correct in one solver step to avoid huge teleports
        const float maxPush = VOXEL_SIZE * 1.5f;
        float penetration = glm::min(bestPen, maxPush);

        outNormal = normal;
        outPenetration = penetration;

        return true;
    }


    glm::vec3 CharacterController::getCameraPosition() const {
        // Camera is slightly above character center (eye level)
        return state.position + glm::vec3(0.0f, getEyeHeight(), 0.0f);
    }

    bool CharacterController::testVoxelCollision(const glm::vec3 &voxelPos) const {
        if (!chunkManager) return false; // Safety check

        // Convert world position to chunk coordinates
        int chunkX = static_cast<int>(glm::floor(voxelPos.x / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkY = static_cast<int>(glm::floor(voxelPos.y / (CHUNK_SIZE * VOXEL_SIZE)));
        int chunkZ = static_cast<int>(glm::floor(voxelPos.z / (CHUNK_SIZE * VOXEL_SIZE)));

        ChunkCoord coord{chunkX, chunkY, chunkZ};
        Chunk *chunk = chunkManager->getChunk(coord);

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
        voxelX = glm::clamp(voxelX, 0, CHUNK_SIZE * (int) std::ceil(VOXEL_SIZE));
        voxelY = glm::clamp(voxelY, 0, CHUNK_SIZE * (int) std::ceil(VOXEL_SIZE));
        voxelZ = glm::clamp(voxelZ, 0, CHUNK_SIZE * (int) std::ceil(VOXEL_SIZE));

        // Check if voxel is solid
        return chunk->voxels[voxelX][voxelY][voxelZ].isSolid();
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

    void CharacterController::update(float deltaTime, const glm::vec3 &moveInput,
                                     bool jumpInput, bool sprintInput,
                                     bool crouchInput, const glm::vec2 &mouseDelta,
                                     const glm::vec3 &cameraForward, const glm::vec3 &cameraRight) {
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