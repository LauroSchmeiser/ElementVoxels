#include <algorithm>
#include "Game.h"

namespace gl3 {
    CharacterController::CharacterController(FixedGridChunkManager *chunkMgr, VoxelPhysicsManager *physicsMgr,
                                             float height, float radius)
            : height(height), radius(radius), currentHeight(height), defaultHeight(height),
              chunkManager(chunkMgr), physicsManager(physicsMgr) {
    }

    static float distancePointSegment(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b, glm::vec3 &outClosest) {
        glm::vec3 ab = b - a;
        float abLenSq = glm::dot(ab, ab);
        if (abLenSq < 1e-8f) {
            outClosest = a;
            return glm::length(p - a);
        }

        float t = glm::dot(p - a, ab) / abLenSq;
        t = glm::clamp(t, 0.0f, 1.0f);
        outClosest = a + t * ab;
        return glm::length(p - outClosest);
    }

    static float clamp01(float t) { return glm::clamp(t, 0.0f, 1.0f); }

    static float closestPointsSegmentSegment(
            const glm::vec3& p0, const glm::vec3& p1,
            const glm::vec3& q0, const glm::vec3& q1,
            glm::vec3& outP, glm::vec3& outQ
    ) {
        const glm::vec3 u = p1 - p0;
        const glm::vec3 v = q1 - q0;
        const glm::vec3 w = p0 - q0;

        const float a = glm::dot(u,u); // always >= 0
        const float b = glm::dot(u,v);
        const float c = glm::dot(v,v); // always >= 0
        const float d = glm::dot(u,w);
        const float e = glm::dot(v,w);

        const float D = a*c - b*b;
        float sc, sN, sD = D;
        float tc, tN, tD = D;

        const float EPS = 1e-8f;

        // compute the line parameters of the two closest points
        if (D < EPS) {
            // almost parallel
            sN = 0.0f; sD = 1.0f;
            tN = e;    tD = c;
        } else {
            sN = (b*e - c*d);
            tN = (a*e - b*d);
            if (sN < 0.0f) { sN = 0.0f; tN = e; tD = c; }
            else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
        }

        if (tN < 0.0f) {
            tN = 0.0f;
            if (-d < 0.0f) sN = 0.0f;
            else if (-d > a) sN = sD;
            else { sN = -d; sD = a; }
        } else if (tN > tD) {
            tN = tD;
            if ((-d + b) < 0.0f) sN = 0.0f;
            else if ((-d + b) > a) sN = sD;
            else { sN = (-d + b); sD = a; }
        }

        sc = (std::abs(sN) < EPS ? 0.0f : sN / sD);
        tc = (std::abs(tN) < EPS ? 0.0f : tN / tD);

        outP = p0 + sc * u;
        outQ = q0 + tc * v;

        return glm::length(outP - outQ);
    }

    static float getDensityAtWorld(FixedGridChunkManager *chunkManager, const glm::vec3 &worldPos) {
        if (!chunkManager) return -10000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk *chunk = chunkManager->getChunk(coord);
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
    static float sampleDensityAtWorld(FixedGridChunkManager *chunkManager, const glm::vec3 &worldPos) {
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

        // Build the eight corner sample world positions
        float samples[2][2][2];
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    glm::vec3 cornerWorld =
                            chunkMin + glm::vec3((float) (ix + dx), (float) (iy + dy), (float) (iz + dz)) * VOXEL_SIZE;
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

    static glm::vec3 sampleNormalAtWorld(FixedGridChunkManager *chunkManager, const glm::vec3 &worldPos) {
        const float e = VOXEL_SIZE * 0.5f;
        float dx = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(e, 0, 0)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(e, 0, 0));
        float dy = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(0, e, 0)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(0, e, 0));
        float dz = sampleDensityAtWorld(chunkManager, worldPos + glm::vec3(0, 0, e)) -
                   sampleDensityAtWorld(chunkManager, worldPos - glm::vec3(0, 0, e));
        glm::vec3 grad(dx, dy, dz);

        float len = glm::length(grad);
        if (len < 1e-6f) {
            return glm::vec3(0.0f, 1.0f, 0.0f);
        }

        return -glm::normalize(grad);
    }

    bool CharacterController::checkCollision(const glm::vec3 &testPosition,
                                             glm::vec3 &outNormal,
                                             float &outPenetration) const {
        if (!chunkManager) return false;

        // Capsule center segment
        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 up = getUpDirection();
        glm::vec3 p0 = testPosition - up * halfSegment; // bottom
        glm::vec3 p1 = testPosition + up * halfSegment; // top
        float segmentLength = glm::length(p1 - p0);

        const float maxStep = VOXEL_SIZE * 0.5f;
        int steps = 1;
        if (segmentLength > 0.0f) steps = glm::clamp((int) std::ceil(segmentLength / maxStep), 2, 64);
        else steps = 2;

        float bestPen = -std::numeric_limits<float>::infinity();
        glm::vec3 bestSamplePos = testPosition;

        // find deepest penetration along center-line
        for (int i = 0; i <= steps; ++i) {
            float t = (steps > 0) ? (float) i / (float) steps : 0.0f;
            glm::vec3 samplePos = glm::mix(p0, p1, t);

            float sdf = sampleDensityAtWorld(chunkManager, samplePos);
            float s = sdf - radius;

            if (s > bestPen) {
                bestPen = s;
                bestSamplePos = samplePos;
            }

            if (bestPen > VOXEL_SIZE * 2.0f) break;
        }

        const float skinWidth = 0.02f * VOXEL_SIZE;
        if (!(bestPen > skinWidth)) {
            return false;
        }

        // compute outward normal
        glm::vec3 normal = sampleNormalAtWorld(chunkManager, bestSamplePos);
        normal = glm::normalize(normal);

        // smoothing
        const float smoothEps = VOXEL_SIZE * 0.25f;
        glm::vec3 n1 = sampleNormalAtWorld(chunkManager, bestSamplePos + normal * smoothEps);
        glm::vec3 n2 = sampleNormalAtWorld(chunkManager, bestSamplePos - normal * smoothEps);
        normal = glm::normalize(normal + 0.5f * n1 + 0.5f * n2);

        const float maxPush = VOXEL_SIZE * 1.5f;
        float penetration = glm::min(bestPen, maxPush);

        outNormal = normal;
        outPenetration = penetration;

        return true;
    }

    bool CharacterController::checkIfGrounded() const {
        if (!chunkManager) return false;

        glm::vec3 up = state.isSurfaceAdhered ? glm::normalize(state.adheredNormal)
                                              : getUpDirection();
        glm::vec3 down = -up;

        // Calculate tangent vectors for radial checks
        glm::vec3 tangentA;
        if (std::abs(up.y) < 0.99f)
            tangentA = glm::normalize(glm::cross(up, glm::vec3(0,1,0)));
        else
            tangentA = glm::normalize(glm::cross(up, glm::vec3(1,0,0)));

        glm::vec3 tangentB = glm::normalize(glm::cross(up, tangentA));

        // Check from bottom sphere center
        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 bottomSphereCenter = state.position - up * halfSegment;

        const float groundCheckDistance = radius * 0.35f; // Check slightly below bottom
        const float sideOffset = radius * 0.6f;

        glm::vec3 testPoints[] = {
                bottomSphereCenter + down * groundCheckDistance,
                bottomSphereCenter + down * groundCheckDistance + tangentA * sideOffset,
                bottomSphereCenter + down * groundCheckDistance - tangentA * sideOffset,
                bottomSphereCenter + down * groundCheckDistance + tangentB * sideOffset,
                bottomSphereCenter + down * groundCheckDistance - tangentB * sideOffset
        };

        int groundHits = 0;
        const int requiredHits = 2;

        for (const glm::vec3& p : testPoints) {
            float sdf = sampleDensityAtWorld(chunkManager, p);

            // If the point is inside or very close to the surface
            if (sdf > -VOXEL_SIZE * 0.1f) {
                glm::vec3 normal = sampleNormalAtWorld(chunkManager, p);
                if (glm::dot(normal, up) > 0.25f) { // Slightly more lenient
                    ++groundHits;
                    if (groundHits >= requiredHits) return true;
                }
            }
        }

        return false;
    }


    glm::vec3 CharacterController::getCameraPosition() const {
        glm::vec3 up = getUpDirection();

        float eyeOffset = getEyeHeight();

        return state.position + up * eyeOffset;
    }

    void CharacterController::resolveCollisions() {
        const int maxIterations = 8;

        for (int i = 0; i < maxIterations; i++) {
            glm::vec3 normal;
            float penetration;
            bool collided = false;

            // Check voxel world collision
            if (checkCollision(state.position, normal, penetration)) {
                state.position += normal * (penetration + VOXEL_SIZE * 0.01f);

                float velDotNormal = glm::dot(state.velocity, normal);
                if (velDotNormal < 0) {
                    state.velocity -= normal * velDotNormal;
                }

                // FIXED: Calculate proper contact point on the capsule surface
                glm::vec3 up = getUpDirection();
                float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);

                // Project normal onto the capsule to find contact point
                glm::vec3 bottomSphereCenter = state.position - up * halfSegment;
                glm::vec3 topSphereCenter = state.position + up * halfSegment;

                // Find closest point on capsule segment to collision normal direction
                glm::vec3 capsuleDir = topSphereCenter - bottomSphereCenter;
                float capsuleLength = glm::length(capsuleDir);

                glm::vec3 contactPoint;
                if (capsuleLength > 1e-6f) {
                    capsuleDir /= capsuleLength;
                    float t = glm::clamp(glm::dot(-normal, capsuleDir), 0.0f, 1.0f);
                    glm::vec3 closestOnSegment = glm::mix(bottomSphereCenter, topSphereCenter, t);
                    contactPoint = closestOnSegment - normal * radius;
                } else {
                    contactPoint = state.position - normal * radius;
                }

                if (shouldLandOnContact(normal)) {
                    beginSurfaceAdhesion(contactPoint, normal);
                }
                collided = true;
            }

            VoxelPhysicsBody *collidingBody = nullptr;
            if (checkPhysicsBodyCollision(state.position, normal, penetration, &collidingBody)) {
                state.position += normal * (penetration + 0.0001f);

                float velDotNormal = glm::dot(state.velocity, normal);
                if (velDotNormal < 0) {
                    state.velocity -= normal * velDotNormal;
                }

                if (playerBodyCollisionCallback && collidingBody) {
                    float playerSpeed = glm::length(state.velocity);
                    glm::vec3 contactPoint = state.position - normal * radius;
                    playerBodyCollisionCallback(collidingBody, contactPoint, normal, playerSpeed);
                }

                collided = true;
            }

            if (!collided) break;
        }
    }

    glm::vec3 CharacterController::calculateWishVelocity(const glm::vec3 &moveInput,
                                                         const glm::vec3 &forward,
                                                         const glm::vec3 &right) const {
        glm::vec3 up = getUpDirection();
        glm::vec3 wishDir = (forward * moveInput.z) + (right * moveInput.x);

        wishDir -= up * glm::dot(wishDir, up);

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
        // Apply gravity (with air slam modifier)
        float gravityMultiplier = 1.0f;
        float velAlongGravity = glm::dot(state.velocity, settings.gravityDir);
        if (state.isAirSlamming) {
            gravityMultiplier = settings.airSlamGravityMultiplier;
        } else if (!state.isGrounded && velAlongGravity > 0.0f) {
            gravityMultiplier = settings.fallingGravityMultiplier;
        }

        if (!state.isGrounded) {
            state.velocity += settings.gravityDir * settings.gravity * gravityMultiplier * deltaTime;

            float alongGrav = glm::dot(state.velocity, settings.gravityDir);
            if (alongGrav > settings.terminalVelocity) {
                glm::vec3 lateral = state.velocity - settings.gravityDir * alongGrav;
                state.velocity = lateral + settings.gravityDir * settings.terminalVelocity;
            }
        }

        // Update position with collision sliding
        glm::vec3 moveStep = state.velocity * deltaTime;
        glm::vec3 newPosition = state.position;

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

                float intoCollision = glm::dot(remainingStep, normal);
                remainingStep -= normal * intoCollision;

                glm::vec3 slidePosition = newPosition + remainingStep;
                if (!checkCollision(slidePosition, normal, penetration)) {
                    newPosition = slidePosition;
                }
                break;
            }
        }

        // ONLY snap to ground if moving fast enough (prevents standing jank)
        if (state.isGrounded && glm::length(state.velocity) > 1.0f) {
            glm::vec3 gp, gn;
            float gd = 0.0f;
            if (findGroundContact(gp, gn, gd) && gd > 0.001f && gd <= settings.adhesionSnapDistance) {
                newPosition -= getUpDirection() * gd;
            }
        }

        state.position = newPosition;
    }

    void CharacterController::performAirSlam() {
        if (!state.isGrounded && state.airSlamAvailable && !state.isAirSlamming) {
            state.isAirSlamming = true;
            state.airSlamAvailable = false;

            glm::vec3 down = settings.gravityDir;
            float velDown = glm::dot(state.velocity, down);

            if (velDown < settings.airSlamInitialVelocity) {
                state.velocity += down * (settings.airSlamInitialVelocity - velDown);
            }

            state.airSlamTimer = settings.airSlamDuration;
        }
    }

    // NEW: Update air slam state
    void CharacterController::updateAirSlam(float deltaTime) {
        if (state.isAirSlamming) {
            state.airSlamTimer -= deltaTime;

            // End air slam when timer expires or we hit ground
            if (state.airSlamTimer <= 0 || state.isGrounded) {
                state.isAirSlamming = false;

                // Apply impact effect when hitting ground during slam
                if (state.isGrounded && state.velocity.y < -settings.airSlamImpactThreshold) {
                    // Could add screen shake, particle effect, or damage here
                }
            }
        }
    }

    void CharacterController::update(float deltaTime, const glm::vec3 &moveInput,
                                     bool jumpInput, bool sprintInput,
                                     bool crouchInput, const glm::vec2 &mouseDelta,
                                     const glm::vec3 &cameraForward, const glm::vec3 &cameraRight,
                                     bool airResetInput) { // ADDED: airResetInput parameter
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

        // Handle air slam/air reset input
        if (airResetInput) {
            performAirSlam();
        }

        // Update air slam state
        updateAirSlam(deltaTime);

        updateSurfaceAdhesion(deltaTime);
        updateOrientation(deltaTime);

        // Handle crouching
        bool wantsCrouch = crouchInput;
        if (wantsCrouch != state.isCrouching) {
            state.isCrouching = wantsCrouch;
            currentHeight = state.isCrouching ?
                            defaultHeight * settings.crouchHeightMultiplier : defaultHeight;
        }

        // Handle sprinting
        state.isSprinting = sprintInput && !state.isCrouching && state.isGrounded;

        // Calculate movement speed
        float targetSpeed = state.isCrouching ? settings.crouchSpeed :
                            (state.isSprinting ? settings.sprintSpeed : settings.walkSpeed);

        glm::vec3 up = getUpDirection();

        glm::vec3 forward = cameraForward - up * glm::dot(cameraForward, up);
        glm::vec3 right = cameraRight - up * glm::dot(cameraRight, up);

        if (glm::length(forward) > 1e-6f) forward = glm::normalize(forward);
        if (glm::length(right) > 1e-6f) right = glm::normalize(right);

        // Calculate wish velocity
        glm::vec3 wishDir = calculateWishVelocity(moveInput, forward, right);

        // Apply friction
        applyFriction(deltaTime);

        // Apply acceleration
        float accel = state.isGrounded ? settings.acceleration :
                      settings.acceleration * settings.airControl;
        accelerate(wishDir, targetSpeed, accel, deltaTime);

        // Handle jumping (can't jump while air slamming)
        if (!state.isAirSlamming && state.jumpBuffer > 0.0f && state.coyoteTime > 0.0f) {
            glm::vec3 up = getUpDirection();

            float velUp = glm::dot(state.velocity, up);
            if (velUp < settings.jumpForce) {
                state.velocity += up * (settings.jumpForce - velUp);
            }

            state.isGrounded = false;
            state.isSurfaceAdhered = false;
            state.jumpBuffer = 0.0f;
            state.coyoteTime = 0.0f;
            state.airSlamAvailable = true;
        }

        // Move character
        move(deltaTime);

        // Resolve collisions
        resolveCollisions();

        updateOrientation(deltaTime);

        if (state.isSurfaceAdhered) {
            bool stillGrounded = checkIfGrounded();

            if (!stillGrounded) {
                state.isGrounded = false;
                state.isSurfaceAdhered = false;
                state.coyoteTime = settings.coyoteTimeDuration;
            } else {
                glm::vec3 gp, gn;
                float gd = 0.0f;
                if (findGroundContact(gp, gn, gd)) {
                    state.lastGroundPoint = gp;
                    state.adheredNormal = glm::normalize(gn);
                    state.isGrounded = true;
                    setGravityDirection(-state.adheredNormal);
                }
            }
        } else {
            float velDown = glm::dot(state.velocity, settings.gravityDir);

            if (checkIfGrounded() && velDown >= 0.0f) {
                glm::vec3 gp, gn;
                float gd = 0.0f;
                if (findGroundContact(gp, gn, gd)) {
                    beginSurfaceAdhesion(gp, gn);

                    glm::vec3 up = glm::normalize(gn);
                    float velUp = glm::dot(state.velocity, up);
                    if (velUp < 0.0f) {
                        state.velocity -= up * velUp;
                    }
                }
            }
        }
    }

    bool CharacterController::checkPhysicsBodyCollision(
            const glm::vec3 &testPosition,
            glm::vec3 &outNormal,
            float &outPenetration,
            VoxelPhysicsBody **outBody
    ) const {
        if (!physicsManager) return false;

        glm::vec3 up = getUpDirection();
        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 p0 = testPosition - up * halfSegment;
        glm::vec3 p1 = testPosition + up * halfSegment;
        float capR = radius;

        bool hitAny = false;
        float bestPen = 0.0f;
        glm::vec3 bestNormal(0, 1, 0);
        VoxelPhysicsBody *bestBody = nullptr;

        const auto &bodies = physicsManager->getBodies();
        for (const auto &upBody : bodies) {
            VoxelPhysicsBody *body = upBody.get();
            if (!body || !body->active) continue;

            const float sumR = capR + body->radius;

            glm::vec3 b0 = body->prevPosition;
            glm::vec3 b1 = body->position;

            if (!std::isfinite(b0.x) || !std::isfinite(b0.y) || !std::isfinite(b0.z)) {
                b0 = body->position;
            }

            glm::vec3 cpPlayer, cpBody;
            float dist = closestPointsSegmentSegment(p0, p1, b0, b1, cpPlayer, cpBody);

            if (dist < sumR) {
                hitAny = true;

                float pen = sumR - dist;

                glm::vec3 n;
                glm::vec3 dir = (cpPlayer - cpBody);
                float len = glm::length(dir);
                if (len > 1e-6f) n = dir / len;
                else n = glm::vec3(0, 1, 0);

                if (pen > bestPen) {
                    bestPen = pen;
                    bestNormal = n;
                    bestBody = body;
                }
            }
        }

        if (!hitAny) return false;

        outNormal = glm::normalize(bestNormal);
        outPenetration = bestPen;
        if (outBody) *outBody = bestBody;
        return true;
    }

    void CharacterController::setGravityDirection(const glm::vec3& g)
    {
        if (glm::length(g) < 1e-6f) return;
        settings.gravityDir = glm::normalize(g);
    }

    glm::vec3 CharacterController::getUpDirection() const
    {
        if (glm::length(state.currentUp) < 1e-5f) {
            return -settings.gravityDir;
        }
        return glm::normalize(state.currentUp);
    }


    bool CharacterController::findGroundContact(glm::vec3& outPoint, glm::vec3& outNormal, float& outDistance) const {
        if (!chunkManager) return false;

        glm::vec3 up = getUpDirection();
        glm::vec3 down = -up;

        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 bottomSphereCenter = state.position - up * halfSegment;

        const float maxDist = settings.adhesionSnapDistance;
        const int steps = 16; // Even more precision

        float bestDist = std::numeric_limits<float>::infinity();
        bool found = false;
        glm::vec3 bestPoint, bestNormal;

        for (int i = 0; i <= steps; ++i) {
            float t = (float)i / (float)steps;
            float d = t * maxDist;

            glm::vec3 samplePos = bottomSphereCenter + down * d;
            float sdf = sampleDensityAtWorld(chunkManager, samplePos);

            // Surface band: accept samples near zero crossing (sphere just touching surface)
            // sdf = distance to surface from sample point
            // We want: sdf ≈ radius (sphere touching surface from outside)
            float distToSurface = sdf - radius;

            // Accept if within tight band around surface
            if (distToSurface >= -VOXEL_SIZE * 0.15f && distToSurface <= VOXEL_SIZE * 0.25f) {
                glm::vec3 n = sampleNormalAtWorld(chunkManager, samplePos);

                if (glm::dot(n, up) >= settings.minGroundNormalDot) {
                    // Prefer the sample closest to ideal zero-crossing
                    float surfaceError = std::abs(distToSurface);
                    if (surfaceError < std::abs(bestDist)) {
                        bestDist = d;
                        bestPoint = samplePos;
                        bestNormal = n;
                        found = true;
                    }
                }
            }
        }

        if (found) {
            // Project bestPoint onto actual surface (improve accuracy)
            // The sample might be slightly off; trace to true surface
            float sdf = sampleDensityAtWorld(chunkManager, bestPoint);
            glm::vec3 normal = sampleNormalAtWorld(chunkManager, bestPoint);

            // Move sample to exact surface (where SDF = radius)
            float correction = sdf - radius;
            bestPoint -= normal * correction;

            outPoint = bestPoint;
            outNormal = bestNormal;
            outDistance = bestDist;
        }

        return found;
    }


    bool CharacterController::shouldLandOnContact(const glm::vec3& contactNormal) const {
        glm::vec3 n = glm::normalize(contactNormal);
        float speedIntoSurface = -glm::dot(state.velocity, n); // positive if moving into wall/floor

        if (speedIntoSurface < settings.landingMinApproachSpeed) {
            return false;
        }

        // Don't treat ceilings as landing surfaces relative to current body up
        glm::vec3 currentUp = getUpDirection();
        if (glm::dot(n, currentUp) < -0.35f) {
            return false;
        }

        return true;
    }

    void CharacterController::beginSurfaceAdhesion(const glm::vec3& contactPoint, const glm::vec3& contactNormal) {
        glm::vec3 n = glm::normalize(contactNormal);

        state.isGrounded = true;
        state.isSurfaceAdhered = true;
        state.adheredNormal = n;
        state.lastGroundPoint = contactPoint;
        state.adhesionTimer = settings.adhesionDuration;
        state.coyoteTime = settings.coyoteTimeDuration;
        state.airSlamAvailable = true;

        settings.gravityDir = -n;

        // Remove velocity into surface
        float vn = glm::dot(state.velocity, n);
        if (vn < 0.0f) {
            state.velocity -= n * vn;
        }
    }

    void CharacterController::updateSurfaceAdhesion(float deltaTime) {
        if (!state.isSurfaceAdhered) return;

        state.adhesionTimer -= deltaTime;

        glm::vec3 surfaceNormal = glm::normalize(state.adheredNormal);
        glm::vec3 down = -surfaceNormal;

        float distFromAnchor = glm::length(state.position - state.lastGroundPoint);
        if (distFromAnchor > settings.adhesionMaxDistance || state.adhesionTimer <= 0.0f) {
            state.isSurfaceAdhered = false;
            state.isGrounded = false;
            return;
        }

        // MUCH WEAKER pull to prevent fighting with player movement
        state.velocity += down * settings.adhesionAcceleration * deltaTime * 0.2f; // 20% strength
    }

    void CharacterController::updateOrientation(float deltaTime) {
        glm::vec3 targetUp;

        if (state.isSurfaceAdhered) {
            targetUp = glm::normalize(state.adheredNormal);
        } else {
            targetUp = glm::normalize(-settings.gravityDir);
        }

        glm::vec3 curUp = glm::normalize(state.currentUp);

        // Calculate angular difference
        float dot = glm::clamp(glm::dot(curUp, targetUp), -1.0f, 1.0f);
        float angleDiff = std::acos(dot);

        // STOP ROTATION when close enough (prevents micro-jitter)
        if (angleDiff < glm::radians(0.5f)) {
            state.currentUp = targetUp;
            return;
        }

        // Use FIXED slow speed for all rotations (no adaptive speed)
        float rotationSpeed = 1.5f; // Lower = smoother

        float t = glm::clamp(rotationSpeed * deltaTime, 0.0f, 1.0f);

        // Use spherical lerp (slerp) for smooth rotation
        glm::vec3 blended;
        if (angleDiff > 0.001f) {
            float sinAngle = std::sin(angleDiff);
            float a = std::sin((1.0f - t) * angleDiff) / sinAngle;
            float b = std::sin(t * angleDiff) / sinAngle;
            blended = glm::normalize(curUp * a + targetUp * b);
        } else {
            blended = targetUp;
        }

        state.currentUp = blended;
    }

}