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

    static float getSolidDensityAtWorld(FixedGridChunkManager *chunkManager, const glm::vec3 &worldPos) {
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

        int ix = static_cast<int>(std::lround(local.x));
        int iy = static_cast<int>(std::lround(local.y));
        int iz = static_cast<int>(std::lround(local.z));

        ix = glm::clamp(ix, 0, CHUNK_SIZE);
        iy = glm::clamp(iy, 0, CHUNK_SIZE);
        iz = glm::clamp(iz, 0, CHUNK_SIZE);

        const Voxel& v = chunk->voxels[ix][iy][iz];

        // Fluid no longer shares this field (see Voxel::fluidDensity), so the
        // solid density is already fluid-free by construction here.
        if (v.type == 0) return -1000.0f; // air
        return v.density;
    }

    // Trilinear sample of the FLUID SDF field - mirrors sampleDensityAtWorld
    // below but reads Voxel::fluidDensity, which is completely independent of
    // solid terrain. Used by isPointInFluid so fluid detection and solid
    // collision agree on the exact same sampling scheme.
    static float sampleFluidDensityAtWorld(FixedGridChunkManager *chunkManager, const glm::vec3 &worldPos) {
        if (!chunkManager) return -1000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int baseCX = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int baseCY = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int baseCZ = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord baseCoord{baseCX, baseCY, baseCZ};
        glm::vec3 chunkMin = glm::vec3(baseCoord.x * chunkWorldSize,
                                       baseCoord.y * chunkWorldSize,
                                       baseCoord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        float fx = local.x - std::floor(local.x);
        float fy = local.y - std::floor(local.y);
        float fz = local.z - std::floor(local.z);

        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));

        auto sampleCorner = [&](int dx, int dy, int dz) -> float {
            glm::vec3 cornerWorld = chunkMin + glm::vec3((float)(ix + dx), (float)(iy + dy), (float)(iz + dz)) * VOXEL_SIZE;
            int cx = static_cast<int>(std::floor(cornerWorld.x / chunkWorldSize));
            int cy = static_cast<int>(std::floor(cornerWorld.y / chunkWorldSize));
            int cz = static_cast<int>(std::floor(cornerWorld.z / chunkWorldSize));
            ChunkCoord coord{cx, cy, cz};
            Chunk *chunk = chunkManager->getChunk(coord);
            if (!chunk) return -1000.0f;
            glm::vec3 chunkOrigin = glm::vec3(coord.x * chunkWorldSize, coord.y * chunkWorldSize, coord.z * chunkWorldSize);
            glm::vec3 localCorner = (cornerWorld - chunkOrigin) / VOXEL_SIZE;
            int lx = glm::clamp((int)std::round(localCorner.x), 0, CHUNK_SIZE);
            int ly = glm::clamp((int)std::round(localCorner.y), 0, CHUNK_SIZE);
            int lz = glm::clamp((int)std::round(localCorner.z), 0, CHUNK_SIZE);
            return chunk->voxels[lx][ly][lz].fluidDensity;
        };

        float samples[2][2][2];
        for (int dx = 0; dx <= 1; ++dx)
            for (int dy = 0; dy <= 1; ++dy)
                for (int dz = 0; dz <= 1; ++dz)
                    samples[dx][dy][dz] = sampleCorner(dx, dy, dz);

        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

        float c00 = lerp(samples[0][0][0], samples[1][0][0], fx);
        float c10 = lerp(samples[0][1][0], samples[1][1][0], fx);
        float c01 = lerp(samples[0][0][1], samples[1][0][1], fx);
        float c11 = lerp(samples[0][1][1], samples[1][1][1], fx);

        float c0 = lerp(c00, c10, fy);
        float c1 = lerp(c01, c11, fy);

        return lerp(c0, c1, fz);
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

        float fx = local.x - std::floor(local.x);
        float fy = local.y - std::floor(local.y);
        float fz = local.z - std::floor(local.z);

        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));

        float samples[2][2][2];
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    glm::vec3 cornerWorld =
                            chunkMin + glm::vec3((float)(ix + dx), (float)(iy + dy), (float)(iz + dz)) * VOXEL_SIZE;
                    samples[dx][dy][dz] = getSolidDensityAtWorld(chunkManager, cornerWorld);
                }
            }
        }

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
        glm::vec3 p0 = testPosition - up * halfSegment;
        glm::vec3 p1 = testPosition + up * halfSegment;
        float segmentLength = glm::length(p1 - p0);

        const float maxStep = VOXEL_SIZE * 0.5f;
        int steps = 1;
        if (segmentLength > 0.0f) steps = glm::clamp((int) std::ceil(segmentLength / maxStep), 2, 64);
        else steps = 2;

        float bestPen = -std::numeric_limits<float>::infinity();
        glm::vec3 bestSamplePos = testPosition;
        bool foundValidSample = false;

        // find deepest penetration along center-line
        for (int i = 0; i <= steps; ++i) {
            float t = (steps > 0) ? (float) i / (float) steps : 0.0f;
            glm::vec3 samplePos = glm::mix(p0, p1, t);

            // No fluid special-casing needed here anymore: fluid lives in its
            // own independent field (Voxel::fluidDensity) and never touches
            // the solid density/type sampled below, so this is already
            // fluid-agnostic by construction.
            float sdf = sampleDensityAtWorld(chunkManager, samplePos);
            float s = sdf - radius;

            if (s > bestPen) {
                bestPen = s;
                bestSamplePos = samplePos;
                foundValidSample = true;
            }

            if (bestPen > VOXEL_SIZE * 2.0f) break;
        }

        const float skinWidth = 0.02f * VOXEL_SIZE;
        if (!foundValidSample || !(bestPen > skinWidth)) {
            return false;
        }

        // compute outward normal - but only use it if the sample point is NOT in fluid
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

        glm::vec3 tangentA;
        if (std::abs(up.y) < 0.99f)
            tangentA = glm::normalize(glm::cross(up, glm::vec3(0,1,0)));
        else
            tangentA = glm::normalize(glm::cross(up, glm::vec3(1,0,0)));

        glm::vec3 tangentB = glm::normalize(glm::cross(up, tangentA));

        float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);
        glm::vec3 bottomSphereCenter = state.position - up * halfSegment;

        const float groundCheckDistance = radius * 0.35f;
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
            // Solid density is fluid-free by construction now, so no fluid
            // special-casing is needed for ground detection.
            float sdf = sampleDensityAtWorld(chunkManager, p);

            if (sdf > -VOXEL_SIZE * 0.1f) {
                glm::vec3 normal = sampleNormalAtWorld(chunkManager, p);
                if (glm::dot(normal, up) > 0.25f) {
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
        state.hasWorldContact = false;
        state.currentContactMaterial = 0;
        state.currentContactType = 0;

        const int maxIterations = 8;

        for (int i = 0; i < maxIterations; i++) {
            glm::vec3 normal;
            float penetration;
            bool collided = false;

            // Check voxel world collision - this now skips fluid entirely
            if (checkCollision(state.position, normal, penetration)) {
                // We know checkCollision already skipped fluid, so this is a solid collision
                state.position += normal * (penetration + VOXEL_SIZE * 0.01f);

                float velDotNormal = glm::dot(state.velocity, normal);
                if (velDotNormal < 0) {
                    state.velocity -= normal * velDotNormal;
                }

                glm::vec3 up = getUpDirection();
                float halfSegment = glm::max(0.0f, (currentHeight * 0.5f) - radius);

                glm::vec3 bottomSphereCenter = state.position - up * halfSegment;
                glm::vec3 topSphereCenter = state.position + up * halfSegment;

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
                state.currentContactPoint = contactPoint;
                state.currentContactNormal = normal;
                state.currentContactMaterial = Game::sampleMaterialAtWorld(chunkManager, contactPoint);
                state.currentContactType = Game::sampleTypeAtWorld(chunkManager, contactPoint);
                state.hasWorldContact = true;

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
        float speed = glm::length(state.velocity);
        if (speed <= 0.01f) return;

        float drop = 0.0f;
        if (state.isInFluid) {
            // Fluid friction - stronger drag
            drop = speed * (settings.fluidResistance * 0.5f) * deltaTime;
            // Additional resistance at higher speeds
            drop += speed * speed * 0.01f * deltaTime;
        } else if (state.isGrounded) {
            drop = speed * settings.friction * deltaTime;
        } else {
            drop = speed * settings.airFriction * deltaTime;
        }

        float newSpeed = glm::max(speed - drop, 0.0f);
        state.velocity *= newSpeed / speed;
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
        // Check if player is in fluid
        bool wasInFluid = state.isInFluid;
        state.isInFluid = isPointInFluid(chunkManager, state.position);

        // Update fluid depth and normal
        if (state.isInFluid) {
            // Sample fluid density at several points to determine depth
            glm::vec3 up = getUpDirection();
            float depthSamples = 0.0f;
            float totalDensity = 0.0f;

            // Sample along a line going upward from the player
            for (int i = 0; i <= 10; ++i) {
                float t = (float)i / 10.0f;
                glm::vec3 samplePos = state.position + up * t * radius * 2.0f;
                float d = getFluidDensityAtWorld(chunkManager, samplePos);
                if (d > 0.0f) {
                    totalDensity += d;
                    depthSamples += 1.0f;
                }
            }
            state.fluidDepth = depthSamples > 0.0f ? totalDensity / depthSamples : 0.0f;

            // Get fluid surface normal (where density goes from positive to zero)
            state.fluidNormal = getFluidNormalAtWorld(chunkManager, state.position);
        } else {
            state.fluidDepth = 0.0f;
        }

        // Apply gravity with fluid buoyancy
        float gravityMultiplier = 1.0f;
        float velAlongGravity = glm::dot(state.velocity, settings.gravityDir);

        if (state.isInFluid) {
            // Apply buoyancy (upward force) - only if not swimming manually
            float buoyancyForce = settings.fluidBuoyancy * state.fluidDepth * settings.fluidDensity;
            state.velocity += -settings.gravityDir * buoyancyForce * deltaTime;

            // Reduce gravity in fluid
            gravityMultiplier = 0.2f; // 20% gravity in water
        } else if (state.isAirSlamming) {
            gravityMultiplier = settings.airSlamGravityMultiplier;
        } else if (!state.isGrounded && velAlongGravity > 0.0f) {
            gravityMultiplier = settings.fallingGravityMultiplier;
        }

        if (!state.isGrounded && !state.isInFluid) {
            state.velocity += settings.gravityDir * settings.gravity * gravityMultiplier * deltaTime;

            float alongGrav = glm::dot(state.velocity, settings.gravityDir);
            if (alongGrav > settings.terminalVelocity) {
                glm::vec3 lateral = state.velocity - settings.gravityDir * alongGrav;
                state.velocity = lateral + settings.gravityDir * settings.terminalVelocity;
            }
        }

        // Apply fluid resistance (drag) - MUCH stronger in fluid
        if (state.isInFluid) {
            float speed = glm::length(state.velocity);
            if (speed > 0.001f) {
                // Stronger drag in fluid
                float dragFactor = 1.0f - settings.fluidResistance * deltaTime * 3.0f;
                dragFactor = glm::max(dragFactor, 0.1f);
                state.velocity *= dragFactor;
            }
        }

        // Update position with collision sliding (same as before)
        glm::vec3 moveStep = state.velocity * deltaTime;
        glm::vec3 newPosition = state.position;

        const float stepSize = 0.5f;
        int steps = glm::max(1, static_cast<int>(glm::length(moveStep) / stepSize));
        glm::vec3 step = moveStep / static_cast<float>(steps);

        for (int i = 0; i < steps; i++) {
            newPosition += step;

            // Check solid collision (fluid is ignored for solid collision)
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

        // Snap to ground - BUT NOT if in fluid!
        if (!state.isInFluid && state.isGrounded && glm::length(state.velocity) > 1.0f) {
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
                                     bool airResetInput) {
        // Update coyote time and jump buffer
        if (state.isGrounded && !state.isInFluid) {
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

        // Handle crouching
        bool wantsCrouch = crouchInput;
        if (wantsCrouch != state.isCrouching) {
            state.isCrouching = wantsCrouch;
            currentHeight = state.isCrouching ?
                            defaultHeight * settings.crouchHeightMultiplier : defaultHeight;
        }

        // Handle sprinting - can't sprint in fluid
        state.isSprinting = sprintInput && !state.isCrouching && state.isGrounded && !state.isInFluid;

        // Calculate movement speed
        float targetSpeed;
        if (state.isInFluid) {
            targetSpeed = settings.fluidSwimSpeed;
        } else {
            targetSpeed = state.isCrouching ? settings.crouchSpeed :
                          (state.isSprinting ? settings.sprintSpeed : settings.walkSpeed);
        }

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
        float accel;
        if (state.isInFluid) {
            accel = settings.fluidSwimAcceleration;
        } else {
            accel = state.isGrounded ? settings.acceleration :
                    settings.acceleration * settings.airControl;
        }
        accelerate(wishDir, targetSpeed, accel, deltaTime);

        // Handle jumping/swimming up
        if (state.isInFluid) {
            // In fluid, jump input makes you swim up
            if (state.jumpBuffer > 0.0f) {
                // Apply upward swim force
                float swimUpForce = settings.jumpForce * 0.5f; // 50% of normal jump
                float velUp = glm::dot(state.velocity, -settings.gravityDir);
                if (velUp < swimUpForce) {
                    state.velocity += -settings.gravityDir * (swimUpForce - velUp);
                }
                state.jumpBuffer = 0.0f;
            }
            // Also allow swimming down with crouch
            if (state.isCrouching) {
                float swimDownForce = settings.jumpForce * 0.2f;
                float velDown = glm::dot(state.velocity, settings.gravityDir);
                if (velDown < swimDownForce) {
                    state.velocity += settings.gravityDir * (swimDownForce - velDown);
                }
            }
        } else if (!state.isAirSlamming && state.jumpBuffer > 0.0f && state.coyoteTime > 0.0f) {
            // Normal jump
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

        enforceCameraClearance();
        enforceCameraClearanceAggressive();

        updateOrientation(deltaTime);

        // Ground detection - only when NOT in fluid
        if (!state.isInFluid) {
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

                        state.currentContactPoint = gp;
                        state.currentContactNormal = gn;
                        state.currentContactMaterial = Game::sampleMaterialAtWorld(chunkManager, gp);
                        state.currentContactType = Game::sampleTypeAtWorld(chunkManager, gp);
                        state.hasWorldContact = true;
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
        } else {
            // In fluid - can't be grounded
            state.isGrounded = false;
            state.isSurfaceAdhered = false;
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
        const int steps = 16;

        float bestDist = std::numeric_limits<float>::infinity();
        bool found = false;
        glm::vec3 bestPoint, bestNormal;

        for (int i = 0; i <= steps; ++i) {
            float t = (float)i / (float)steps;
            float d = t * maxDist;

            glm::vec3 samplePos = bottomSphereCenter + down * d;

            // Solid density is fluid-free by construction now, so no fluid
            // special-casing is needed for ground contact detection.
            float sdf = sampleDensityAtWorld(chunkManager, samplePos);

            // Surface band: accept samples near zero crossing (sphere just touching surface)
            float distToSurface = sdf - radius;

            // Accept if within tight band around surface
            if (distToSurface >= -VOXEL_SIZE * 0.15f && distToSurface <= VOXEL_SIZE * 0.25f) {
                glm::vec3 n = sampleNormalAtWorld(chunkManager, samplePos);

                if (glm::dot(n, up) >= settings.minGroundNormalDot) {
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

        state.velocity += down * settings.adhesionAcceleration * deltaTime * 0.25f; // 25% strength
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

    bool CharacterController::depenetrateSphere(glm::vec3& center, float sphereRadius, int maxIterations) const {
        if (!chunkManager) return false;

        bool movedAny = false;
        const float skin = VOXEL_SIZE * 0.02f;
        const float maxPushPerIter = VOXEL_SIZE * 0.75f;

        for (int i = 0; i < maxIterations; ++i) {
            // Skip if in fluid
            if (isPointInFluid(chunkManager, center)) break;

            float sdf = sampleDensityAtWorld(chunkManager, center);
            float pen = sdf - sphereRadius;

            if (pen <= skin) break;

            glm::vec3 n = sampleNormalAtWorld(chunkManager, center);
            float nLen = glm::length(n);
            if (nLen < 1e-6f) n = getUpDirection();
            else n /= nLen;

            float push = glm::min(pen + skin, maxPushPerIter);
            center += n * push;
            movedAny = true;
        }

        return movedAny;
    }

    void CharacterController::enforceCameraClearance() {
        glm::vec3 up = getUpDirection();
        float eyeOffset = getEyeHeight();

        // Use smaller radius for head to allow closer to surfaces
        const float eyeRadius = glm::max(VOXEL_SIZE * 0.12f, radius * 0.35f);

        // Get current camera position
        glm::vec3 eye = state.position + up * eyeOffset;
        glm::vec3 originalEye = eye;

        // Resolve camera collision (now handles fluid)
        resolveCameraCollision(eye, eyeRadius);

        // Only adjust character position if camera was pushed significantly
        float pushDist = glm::length(eye - originalEye);
        if (pushDist > VOXEL_SIZE * 0.02f) {
            // Check if we're in fluid - if so, don't push the character
            if (isPointInFluid(chunkManager, eye)) {
                // We're in fluid, don't push the character
                return;
            }

            glm::vec3 newPosition = eye - up * eyeOffset;

            // Check if new position is valid (not inside geometry)
            float sdf = sampleDensityAtWorld(chunkManager, newPosition);
            bool inFluid2 = isPointInFluid(chunkManager, newPosition);
            uint8_t t2 = Game::sampleTypeAtWorld(chunkManager, newPosition);
            if ((sdf < radius * 0.5f || inFluid2) && t2 != 1u) {
                state.position = newPosition;
            } else {
                // Option 2: Only move partially
                glm::vec3 partialMove = (newPosition - state.position) * 0.3f;
                state.position += partialMove;
            }
        }
    }

    bool CharacterController::checkCameraCollision(glm::vec3& cameraPos, float eyeRadius,
                                                   glm::vec3& outNormal, float& outPenetration) const {
        if (!chunkManager) return false;

        // Check if we're in fluid - if so, no collision
        if (isPointInFluid(chunkManager, cameraPos)) {
            // In fluid - allow free movement
            outNormal = glm::vec3(0.0f, 1.0f, 0.0f);
            outPenetration = 0.0f;
            return false; // No collision
        }

        float sdf = sampleDensityAtWorld(chunkManager, cameraPos);

        // Push camera out of solid geometry
        if (sdf > 0.0f) {
            glm::vec3 normal = sampleNormalAtWorld(chunkManager, cameraPos);
            cameraPos += normal * (sdf + VOXEL_SIZE * 0.05f);
        }

        float pen = sdf - eyeRadius;
        if (pen <= 0.0f) return false;

        glm::vec3 normal = sampleNormalAtWorld(chunkManager, cameraPos);
        float nLen = glm::length(normal);
        if (nLen < 1e-6f) {
            normal = getUpDirection();
            nLen = 1.0f;
        }
        normal /= nLen;

        outNormal = normal;
        outPenetration = pen;
        return true;
    }

    void CharacterController::resolveCameraCollision(glm::vec3& cameraPos, float eyeRadius) const {
        const int maxIterations = 20;
        const float skinWidth = VOXEL_SIZE * 0.04f;

        for (int i = 0; i < maxIterations; i++) {
            glm::vec3 normal;
            float penetration;

            if (!checkCameraCollision(cameraPos, eyeRadius, normal, penetration)) {
                break;
            }

            // If penetration is 0, we're in fluid - no push needed
            if (penetration <= 0.0f) {
                break;
            }

            // Push camera out of solid geometry
            float push = penetration + skinWidth;
            cameraPos += normal * push;

            // Prevent infinite loops
            if (i == maxIterations - 1) break;
        }
    }

    void CharacterController::enforceCameraClearanceAggressive() {
        glm::vec3 up = getUpDirection();
        float eyeOffset = getEyeHeight();

        const float eyeRadius = glm::max(VOXEL_SIZE * 0.12f, radius * 0.35f);
        glm::vec3 eye = state.position + up * eyeOffset;

        // Check if camera is deeply inside geometry
        float sdf = sampleDensityAtWorld(chunkManager, eye);

        // Don't push out of fluid
        if (isPointInFluid(chunkManager, eye)) return;

        if (sdf > eyeRadius * 0.8f) {
            // Deep penetration - push camera and character
            glm::vec3 normal = sampleNormalAtWorld(chunkManager, eye);
            float penetration = sdf - eyeRadius;

            // Push camera
            eye += normal * (penetration + VOXEL_SIZE * 0.05f);

            // Also push character to maintain relationship
            glm::vec3 newPos = eye - up * eyeOffset;

            // Only push character if it's safe
            float charSdf = sampleDensityAtWorld(chunkManager, newPos);
            bool charInFluid = isPointInFluid(chunkManager, newPos);
            if (charSdf < radius * 0.5f || charInFluid) {
                state.position = newPos;
            } else {
                // Partial push
                state.position += (newPos - state.position) * 0.2f;
            }
        }
    }
    bool CharacterController::isPointInFluid(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) const {
        if (!chunkManager) return false;
        // Uses the same trilinear sampling scheme as solid collision, against
        // the independent fluid field, so "am I in fluid?" and "am I colliding
        // with solid terrain?" never disagree at cell boundaries.
        return sampleFluidDensityAtWorld(chunkManager, worldPos) >= 0.0f;
    }

    float CharacterController::getFluidDensityAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) const {
        if (!chunkManager) return 0.0f;
        float d = sampleFluidDensityAtWorld(chunkManager, worldPos);
        return d >= 0.0f ? d : 0.0f;
    }
    // Get fluid surface normal at a point (uses gradient of fluid density)
    glm::vec3 CharacterController::getFluidNormalAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) const {
        const float e = VOXEL_SIZE * 0.5f;
        float dx = getFluidDensityAtWorld(chunkManager, worldPos + glm::vec3(e, 0, 0)) -
                   getFluidDensityAtWorld(chunkManager, worldPos - glm::vec3(e, 0, 0));
        float dy = getFluidDensityAtWorld(chunkManager, worldPos + glm::vec3(0, e, 0)) -
                   getFluidDensityAtWorld(chunkManager, worldPos - glm::vec3(0, e, 0));
        float dz = getFluidDensityAtWorld(chunkManager, worldPos + glm::vec3(0, 0, e)) -
                   getFluidDensityAtWorld(chunkManager, worldPos - glm::vec3(0, 0, e));
        glm::vec3 grad(dx, dy, dz);
        float len = glm::length(grad);
        if (len < 1e-6f) return glm::vec3(0.0f, 1.0f, 0.0f);
        return -glm::normalize(grad); // Points toward fluid interior
    }
}