// VoxelPhysicsManager.cpp
#include "VoxelPhysicsManager.h"
#include <glm/gtc/quaternion.hpp>
#include <algorithm>

namespace gl3 {

    VoxelPhysicsBody* VoxelPhysicsManager::createBody(
            const glm::vec3& position,
            float mass,
            VoxelPhysicsBody::ShapeType shape,
            const glm::vec3& extents
    ) {
        static uint64_t nextId = 1;

        auto body = std::make_unique<VoxelPhysicsBody>();
        body->id = nextId++;
        body->position = position;
        body->velocity = glm::vec3(0.0f);
        body->mass = mass;
        body->shapeType = shape;
        body->shapeExtents = extents;
        body->orientation = glm::quat(1,0,0,0);

        switch(shape) {
            case VoxelPhysicsBody::ShapeType::SPHERE: body->radius = extents.x; break;
            case VoxelPhysicsBody::ShapeType::BOX:    body->radius = glm::length(extents); break;
            default:                                  body->radius = glm::length(extents); break;
        }

        VoxelPhysicsBody* ptr = body.get();
        bodies.push_back(std::move(body));
        return ptr;
    }

    VoxelPhysicsBody* VoxelPhysicsManager::getBodyById(uint64_t id) const {
        for (auto& b : bodies) {
            if (b->id == id) return b.get();
        }
        return nullptr;
    }

    void VoxelPhysicsManager::removeBody(uint64_t id) {
        bodies.erase(
                std::remove_if(bodies.begin(), bodies.end(),
                               [id](const std::unique_ptr<VoxelPhysicsBody>& b){ return b->id == id; }),
                bodies.end()
        );
    }

    float VoxelPhysicsManager::sampleDensity(const glm::vec3& worldPos) {
        if (!chunkManager) return -10000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);
        if (!chunk) return -10000.0f;

        glm::vec3 chunkMin(cx * chunkWorldSize, cy * chunkWorldSize, cz * chunkWorldSize);
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        // Trilinear interpolation (same as player)
        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));

        // Clamp to valid range
        ix = glm::clamp(ix, 0, CHUNK_SIZE - 1);
        iy = glm::clamp(iy, 0, CHUNK_SIZE - 1);
        iz = glm::clamp(iz, 0, CHUNK_SIZE - 1);

        float fx = local.x - ix;
        float fy = local.y - iy;
        float fz = local.z - iz;

        fx = glm::clamp(fx, 0.0f, 1.0f);
        fy = glm::clamp(fy, 0.0f, 1.0f);
        fz = glm::clamp(fz, 0.0f, 1.0f);

        int ix1 = glm::min(ix + 1, CHUNK_SIZE);
        int iy1 = glm::min(iy + 1, CHUNK_SIZE);
        int iz1 = glm::min(iz + 1, CHUNK_SIZE);

        // Sample 8 corners
        float s000 = chunk->voxels[ix][iy][iz].density;
        float s100 = chunk->voxels[ix1][iy][iz].density;
        float s010 = chunk->voxels[ix][iy1][iz].density;
        float s110 = chunk->voxels[ix1][iy1][iz].density;
        float s001 = chunk->voxels[ix][iy][iz1].density;
        float s101 = chunk->voxels[ix1][iy][iz1].density;
        float s011 = chunk->voxels[ix][iy1][iz1].density;
        float s111 = chunk->voxels[ix1][iy1][iz1].density;

        // Trilinear interpolation
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

        float c00 = lerp(s000, s100, fx);
        float c10 = lerp(s010, s110, fx);
        float c01 = lerp(s001, s101, fx);
        float c11 = lerp(s011, s111, fx);

        float c0 = lerp(c00, c10, fy);
        float c1 = lerp(c01, c11, fy);

        return lerp(c0, c1, fz);
    }


    glm::vec3 VoxelPhysicsManager::sampleGradient(const glm::vec3& worldPos) {
        const float e = VOXEL_SIZE * 0.5f;

        float dx = sampleDensity(worldPos + glm::vec3(e, 0, 0)) -
                   sampleDensity(worldPos - glm::vec3(e, 0, 0));
        float dy = sampleDensity(worldPos + glm::vec3(0, e, 0)) -
                   sampleDensity(worldPos - glm::vec3(0, e, 0));
        float dz = sampleDensity(worldPos + glm::vec3(0, 0, e)) -
                   sampleDensity(worldPos - glm::vec3(0, 0, e));

        return glm::vec3(dx, dy, dz);
    }


    bool VoxelPhysicsManager::checkBoxCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        glm::vec3 halfExtents = body.shapeExtents;
        glm::mat3 rot = glm::mat3_cast(body.orientation);

        // Generate 8 corners of the box
        glm::vec3 corners[8];
        for (int i = 0; i < 8; ++i) {
            glm::vec3 localCorner(
                    (i & 1) ? halfExtents.x : -halfExtents.x,
                    (i & 2) ? halfExtents.y : -halfExtents.y,
                    (i & 4) ? halfExtents.z : -halfExtents.z
            );
            corners[i] = body.position + rot * localCorner;
        }

        float maxPenetration = 0.0f;
        glm::vec3 collisionNormal(0.0f);
        int collisionCount = 0;

        // Check each corner
        for (int i = 0; i < 8; ++i) {
            float density = sampleDensity(corners[i]);

            if (density > 0) {
                float penetration = density;
                glm::vec3 normal = sampleNormal(corners[i]);

                collisionNormal += normal;
                maxPenetration = std::max(maxPenetration, penetration);
                collisionCount++;
            }
        }

        // Also check edge centers for better accuracy
        const int edges[12][2] = {
                {0,1}, {1,3}, {3,2}, {2,0},
                {4,5}, {5,7}, {7,6}, {6,4},
                {0,4}, {1,5}, {3,7}, {2,6}
        };

        for (int e = 0; e < 12; ++e) {
            glm::vec3 edgeCenter = (corners[edges[e][0]] + corners[edges[e][1]]) * 0.5f;
            float density = sampleDensity(edgeCenter);

            if (density > 0) {
                float penetration = density;
                glm::vec3 normal = sampleNormal(edgeCenter);

                collisionNormal += normal;
                maxPenetration = std::max(maxPenetration, penetration);
                collisionCount++;
            }
        }

        if (collisionCount > 0) {
            outNormal = glm::normalize(collisionNormal);
            outPenetration = maxPenetration;
            return true;
        }

        return false;
    }

    void VoxelPhysicsManager::removeBody(VoxelPhysicsBody* body) {
        if (body) removeBody(body->id);
    }

    // VoxelPhysicsManager.cpp - Replace the collision detection

    float VoxelPhysicsManager::getDensityAtWorldCorner(const glm::vec3& worldPos) {
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

        // Round to nearest voxel corner
        int ix = static_cast<int>(std::lround(local.x));
        int iy = static_cast<int>(std::lround(local.y));
        int iz = static_cast<int>(std::lround(local.z));

        // Clamp to valid range
        ix = glm::clamp(ix, 0, CHUNK_SIZE);
        iy = glm::clamp(iy, 0, CHUNK_SIZE);
        iz = glm::clamp(iz, 0, CHUNK_SIZE);

        return chunk->voxels[ix][iy][iz].density;
    }

    float VoxelPhysicsManager::sampleDensityTrilinear(const glm::vec3& worldPos) {
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

        // Get fractional coordinates
        float fx = local.x - std::floor(local.x);
        float fy = local.y - std::floor(local.y);
        float fz = local.z - std::floor(local.z);

        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));

        // Sample 8 corners (handles chunk boundaries properly)
        float samples[2][2][2];
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    glm::vec3 cornerWorld =
                            chunkMin + glm::vec3((float)(ix + dx), (float)(iy + dy), (float)(iz + dz)) * VOXEL_SIZE;
                    samples[dx][dy][dz] = getDensityAtWorldCorner(cornerWorld);
                }
            }
        }

        // Trilinear interpolation
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };

        float c00 = lerp(samples[0][0][0], samples[1][0][0], fx);
        float c10 = lerp(samples[0][1][0], samples[1][1][0], fx);
        float c01 = lerp(samples[0][0][1], samples[1][0][1], fx);
        float c11 = lerp(samples[0][1][1], samples[1][1][1], fx);

        float c0 = lerp(c00, c10, fy);
        float c1 = lerp(c01, c11, fy);

        return lerp(c0, c1, fz);
    }


    bool VoxelPhysicsManager::checkVoxelCollision(
            VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // For spheres: sample along the surface
        if (body.shapeType == VoxelPhysicsBody::ShapeType::SPHERE) {
            const int numSamples = 26; // More samples for better detection

            // Sample directions (cube corners + face centers + edge centers)
            glm::vec3 sampleDirs[26];
            int idx = 0;

            // Face centers (6)
            sampleDirs[idx++] = glm::vec3(1, 0, 0);
            sampleDirs[idx++] = glm::vec3(-1, 0, 0);
            sampleDirs[idx++] = glm::vec3(0, 1, 0);
            sampleDirs[idx++] = glm::vec3(0, -1, 0);
            sampleDirs[idx++] = glm::vec3(0, 0, 1);
            sampleDirs[idx++] = glm::vec3(0, 0, -1);

            // Edge centers (12)
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, 1, 0));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, -1, 0));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, 1, 0));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, -1, 0));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, 0, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, 0, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, 0, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, 0, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(0, 1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(0, 1, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(0, -1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(0, -1, -1));

            // Cube corners (8)
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, 1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, 1, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, -1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(1, -1, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, 1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, 1, -1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, -1, 1));
            sampleDirs[idx++] = glm::normalize(glm::vec3(-1, -1, -1));

            float maxPenetration = -std::numeric_limits<float>::infinity();
            glm::vec3 bestNormal(0, 1, 0);
            glm::vec3 bestSamplePos = body.position;
            bool hit = false;

            for (int i = 0; i < numSamples; ++i) {
                glm::vec3 samplePos = body.position + sampleDirs[i] * body.radius;
                float density = sampleDensity(samplePos);

                // Positive density = inside solid
                if (density > 0.0f) {
                    hit = true;

                    // Calculate signed distance (negative = penetrating)
                    float signedDist = density - body.radius;

                    if (signedDist > maxPenetration) {
                        maxPenetration = signedDist;
                        bestSamplePos = samplePos;
                        bestNormal = sampleNormal(samplePos);
                    }
                }
            }

            const float skinWidth = 0.02f * VOXEL_SIZE;
            if (!hit || maxPenetration <= skinWidth) {
                return false;
            }

            outNormal = glm::normalize(bestNormal);
            outPenetration = maxPenetration;

            return true;
        }

        // For boxes: check corners and edges
        if (body.shapeType == VoxelPhysicsBody::ShapeType::BOX) {
            return checkBoxCollision(body, outNormal, outPenetration);
        }

        return false;
    }

    void VoxelPhysicsManager::resolveCollision(
            VoxelPhysicsBody& body,
            const glm::vec3& normal,
            float penetration,
            float impactSpeed
    ) {
        // Push body out of collision
        const float extraMargin = 0.01f * VOXEL_SIZE;
        body.position += normal * (penetration/10 + extraMargin);

        // Reflect velocity along normal
        float velDotNormal = glm::dot(body.velocity, normal);

        if (velDotNormal < 0) {
            // Separate into normal and tangential components
            glm::vec3 normalVel = normal * velDotNormal;
            glm::vec3 tangentVel = body.velocity - normalVel;

            // Apply restitution (bounce)
            glm::vec3 newNormalVel = -normalVel * body.restitution;

            // Apply friction (reduce tangential velocity)
            glm::vec3 newTangentVel = tangentVel * (1.0f - body.friction);

            // Combine
            body.velocity = newNormalVel + newTangentVel;

            // Add angular velocity from impact
            if (glm::length(tangentVel) > 0.1f) {
                glm::vec3 rotationAxis = glm::cross(normal, tangentVel);
                float rotSpeed = glm::length(tangentVel) * body.friction * 0.5f / glm::max(body.radius, 0.1f);
                body.angularVelocity += glm::normalize(rotationAxis) * rotSpeed;
            }
        }

        // Dampen angular velocity
        body.angularVelocity *= 0.95f;
    }

    void VoxelPhysicsManager::update(float dt, std::vector<uint64_t>& removedBodies) {
        removedBodies.clear();

        // ---- STEP 1: integrate + voxel collisions + lifetime ----
        // Use index iteration so we can swap-remove safely.
        size_t i = 0;
        while (i < bodies.size()) {
            VoxelPhysicsBody* body = bodies[i].get();

            // Lifetime check
            if (body->lifetime > 0.0f) {
                body->lifetime -= dt;
                if (body->lifetime <= 0.0f) {
                    removedBodies.push_back(body->id);

                    // swap-remove (O(1), invalidates pointer to THIS body, but we are deleting it anyway)
                    bodies[i] = std::move(bodies.back());
                    bodies.pop_back();
                    continue; // don't increment i; process swapped-in element
                }
            }

            if (!body->active) {
                ++i;
                continue;
            }

            body->velocity += gravity * dt;

            glm::vec3 oldVel = body->velocity;

            body->position += body->velocity * dt;

            if (glm::length(body->angularVelocity) > 0.001f) {
                glm::quat spin = glm::quat(
                        0,
                        body->angularVelocity.x * dt * 0.5f,
                        body->angularVelocity.y * dt * 0.5f,
                        body->angularVelocity.z * dt * 0.5f
                );
                body->orientation += spin * body->orientation;
                body->orientation = glm::normalize(body->orientation);
            }

            glm::vec3 normal;
            float penetration;

            if (checkVoxelCollision(*body, normal, penetration)) {
                float impactSpeed = glm::length(oldVel) * VOXEL_SIZE;
                resolveCollision(*body, normal, penetration, impactSpeed);

                if (voxelCollisionCallback && impactSpeed > 1.0f * VOXEL_SIZE) {
                    voxelCollisionCallback(body, body->position, normal, impactSpeed);
                }
            }

            ++i;
        }

        // ---- STEP 2: body-body collisions ----
        // Build a raw pointer list of ACTIVE bodies for pairwise checks.
        std::vector<VoxelPhysicsBody*> activeBodies;
        activeBodies.reserve(bodies.size());
        for (auto& up : bodies) {
            if (up && up->active) activeBodies.push_back(up.get());
        }

        for (size_t a = 0; a < activeBodies.size(); ++a) {
            for (size_t b = a + 1; b < activeBodies.size(); ++b) {
                VoxelPhysicsBody* bodyA = activeBodies[a];
                VoxelPhysicsBody* bodyB = activeBodies[b];

                float maxDist = bodyA->radius + bodyB->radius;
                glm::vec3 delta = bodyB->position - bodyA->position;
                float distSq = glm::dot(delta, delta);

                if (distSq > maxDist * maxDist) continue;

                glm::vec3 normal;
                float penetration;
                if (checkBodyBodyCollision(*bodyA, *bodyB, normal, penetration)) {
                    resolveBodyBodyCollision(*bodyA, *bodyB, normal, penetration);
                }
            }
        }
    }
    
    glm::vec3 VoxelPhysicsManager::sampleNormal(const glm::vec3& worldPos) {
        const float e = VOXEL_SIZE * 0.5f;

        float dx = sampleDensityTrilinear(worldPos + glm::vec3(e, 0, 0)) -
                   sampleDensityTrilinear(worldPos - glm::vec3(e, 0, 0));
        float dy = sampleDensityTrilinear(worldPos + glm::vec3(0, e, 0)) -
                   sampleDensityTrilinear(worldPos - glm::vec3(0, e, 0));
        float dz = sampleDensityTrilinear(worldPos + glm::vec3(0, 0, e)) -
                   sampleDensityTrilinear(worldPos - glm::vec3(0, 0, e));

        glm::vec3 grad(dx, dy, dz);
        float len = glm::length(grad);

        if (len < 1e-6f) {
            return glm::vec3(0, 1, 0);
        }

        // Gradient points inward, we want outward normal
        return -glm::normalize(grad);
    }

    bool VoxelPhysicsManager::checkCapsuleCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // Treat the body as a capsule along its primary axis
        // For spheres, this degenerates to a point capsule

        float halfHeight = 0.0f;
        glm::vec3 axis = glm::vec3(0, 1, 0);

        if (body.shapeType == VoxelPhysicsBody::ShapeType::BOX) {
            // Use longest axis as capsule direction
            if (body.shapeExtents.y > body.shapeExtents.x && body.shapeExtents.y > body.shapeExtents.z) {
                halfHeight = body.shapeExtents.y;
                axis = glm::vec3(0, 1, 0);
            } else if (body.shapeExtents.x > body.shapeExtents.z) {
                halfHeight = body.shapeExtents.x;
                axis = glm::vec3(1, 0, 0);
            } else {
                halfHeight = body.shapeExtents.z;
                axis = glm::vec3(0, 0, 1);
            }
        }

        // Rotate axis by body orientation
        axis = glm::normalize(glm::mat3_cast(body.orientation) * axis);

        // Capsule endpoints
        glm::vec3 p0 = body.position - axis * halfHeight;
        glm::vec3 p1 = body.position + axis * halfHeight;

        float segmentLength = glm::length(p1 - p0);

        // Sample along the capsule center line
        const float maxStep = VOXEL_SIZE * 1.75f;
        int steps = glm::max(4, (int)std::ceil(segmentLength / maxStep + 1));

        float bestPenetration = -std::numeric_limits<float>::infinity();
        glm::vec3 bestSamplePos = body.position;

        // Find deepest penetration point along capsule
        for (int i = 0; i <= steps; ++i) {
            float t = (steps > 0) ? (float)i / (float)steps : 0.5f;
            glm::vec3 samplePos = glm::mix(p0, p1, t);

            float sdf = sampleDensityTrilinear(samplePos);
            float signedDist = sdf - body.radius;

            if (signedDist > bestPenetration) {
                bestPenetration = signedDist;
                bestSamplePos = samplePos;
            }

            // Early exit if deep penetration
            if (bestPenetration > VOXEL_SIZE * .0f) break;
        }

        const float skinWidth = 1.00f * VOXEL_SIZE;
        if (bestPenetration <= skinWidth) {
            return false; // No collision
        }

        // Calculate normal at collision point
        glm::vec3 normal = sampleNormal(bestSamplePos);

        // Smooth the normal by sampling nearby
        const float smoothEps = VOXEL_SIZE * 0.75f;
        glm::vec3 n1 = sampleNormal(bestSamplePos + normal * smoothEps);
        glm::vec3 n2 = sampleNormal(bestSamplePos - normal * smoothEps);
        normal = glm::normalize(normal + 0.5f * n1 + 0.5f * n2);

        // Clamp penetration to reasonable value
        const float maxPush = VOXEL_SIZE * 2.0f;
        float penetration = glm::min(bestPenetration, maxPush);

        outNormal = normal;
        outPenetration = penetration;

        return true;
    }

    bool VoxelPhysicsManager::checkBodyBodyCollision(
            VoxelPhysicsBody& bodyA,
            VoxelPhysicsBody& bodyB,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // Get bounding sphere radii (already calculated in bodies)
        float radiusA = bodyA.radius;
        float radiusB = bodyB.radius;

        // Vector from A to B
        glm::vec3 AtoB = bodyB.position - bodyA.position;
        float distance = glm::length(AtoB);

        // Sum of radii
        float minDist = radiusA + radiusB;

        // Check if spheres intersect
        if (distance >= minDist) {
            return false; // No collision
        }

        // Collision detected
        outPenetration = minDist - distance;

        // Normal points from A to B
        if (distance > 0.0001f) {
            outNormal = AtoB / distance; // Normalized
        } else {
            // Bodies are at exactly the same position - push apart in arbitrary direction
            outNormal = glm::vec3(0, 1, 0);
        }

        return true;
    }

    void VoxelPhysicsManager::resolveBodyBodyCollision(
            VoxelPhysicsBody& bodyA,
            VoxelPhysicsBody& bodyB,
            const glm::vec3& normal,
            float penetration
    ) {
        // Calculate relative mass
        float totalMass = bodyA.mass + bodyB.mass;
        float massRatioA = bodyB.mass / totalMass; // How much A moves
        float massRatioB = bodyA.mass / totalMass; // How much B moves

        // Separate bodies (push them apart based on mass)
        bodyA.position -= normal * penetration * massRatioA;
        bodyB.position += normal * penetration * massRatioB;

        // Calculate relative velocity
        glm::vec3 relativeVel = bodyB.velocity - bodyA.velocity;
        float velAlongNormal = glm::dot(relativeVel, normal);

        // Don't resolve if bodies are separating
        if (velAlongNormal > 0) {
            return;
        }

        // Calculate restitution (average of both bodies)
        float restitution = (bodyA.restitution + bodyB.restitution) * 0.5f;

        // Calculate impulse scalar
        float j = -(1.0f + restitution) * velAlongNormal;
        j /= (1.0f / bodyA.mass + 1.0f / bodyB.mass);

        // Apply impulse
        glm::vec3 impulse = j * normal;
        bodyA.velocity -= impulse / bodyA.mass;
        bodyB.velocity += impulse / bodyB.mass;

        // Apply friction (tangent to collision normal)
        glm::vec3 tangent = relativeVel - normal * velAlongNormal;
        if (glm::length(tangent) > 0.001f) {
            tangent = glm::normalize(tangent);

            float friction = (bodyA.friction + bodyB.friction) * 0.5f;
            float jt = -glm::dot(relativeVel, tangent);
            jt /= (1.0f / bodyA.mass + 1.0f / bodyB.mass);

            // Clamp friction to Coulomb's law
            jt = glm::clamp(jt, -j * friction, j * friction);

            glm::vec3 frictionImpulse = jt * tangent;
            bodyA.velocity -= frictionImpulse / bodyA.mass;
            bodyB.velocity += frictionImpulse / bodyB.mass;
        }

        // Add some angular velocity from collision
        glm::vec3 contactPoint = (bodyA.position + bodyB.position) * 0.5f;
        glm::vec3 rA = contactPoint - bodyA.position;
        glm::vec3 rB = contactPoint - bodyB.position;

        bodyA.angularVelocity += glm::cross(rA, impulse) / (bodyA.mass * bodyA.radius);
        bodyB.angularVelocity -= glm::cross(rB, impulse) / (bodyB.mass * bodyB.radius);

        if (bodyBodyCollisionCallback) {
            float impactSpeed = glm::length(relativeVel);
            bodyBodyCollisionCallback(&bodyA, &bodyB, contactPoint, normal, impactSpeed);
        }
    }
}