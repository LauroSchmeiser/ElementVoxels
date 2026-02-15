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

        VoxelPhysicsBody body;
        body.id = nextId++;
        body.position = position;
        body.velocity = glm::vec3(0.0f);
        body.mass = mass;
        body.shapeType = shape;
        body.shapeExtents = extents;
        body.orientation = glm::quat(1, 0, 0, 0);

        // Calculate bounding radius
        switch(shape) {
            case VoxelPhysicsBody::ShapeType::SPHERE:
                body.radius = extents.x;
                break;
            case VoxelPhysicsBody::ShapeType::BOX:
                body.radius = glm::length(extents);
                break;
            default:
                body.radius = glm::length(extents);
        }

        bodies.push_back(body);
        return &bodies.back();
    }

    void VoxelPhysicsManager::update(float dt, std::vector<uint64_t>& removedBodies) {
        removedBodies.clear();

        for (auto it = bodies.begin(); it != bodies.end(); ) {
            VoxelPhysicsBody& body = *it;

            // Lifetime check
            if (body.lifetime > 0) {
                body.lifetime -= dt;
                if (body.lifetime <= 0) {
                    removedBodies.push_back(body.id);
                    it = bodies.erase(it);
                    continue;
                }
            }

            if (!body.active) {
                ++it;
                continue;
            }

            // Apply gravity
            body.velocity += gravity * dt;

            // Store old position for collision response
            glm::vec3 oldVel = body.velocity;

            // Integrate position
            body.position += body.velocity * dt;

            // Check collision with voxel world
            glm::vec3 normal;
            float penetration;

            if (checkVoxelCollision(body, normal, penetration)) {
                std::cout<<"collision detected\n";
                float impactSpeed = glm::length(oldVel)*10;

                // Resolve collision
                resolveCollision(body, normal, penetration, impactSpeed);

                // Call callback
                if (collisionCallback && impactSpeed > 1.0f) {
                    collisionCallback(&body, body.position, normal, impactSpeed);
                }
            }

            ++it;
        }
    }

    float VoxelPhysicsManager::sampleDensity(const glm::vec3& worldPos) {
        if (!chunkManager) return -1000.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = (int)std::floor(worldPos.x / chunkWorldSize);
        int cy = (int)std::floor(worldPos.y / chunkWorldSize);
        int cz = (int)std::floor(worldPos.z / chunkWorldSize);

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) return -1000.0f;

        glm::vec3 chunkMin(cx * chunkWorldSize, cy * chunkWorldSize, cz * chunkWorldSize);
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        // Clamp to valid range
        int ix = glm::clamp((int)std::floor(local.x), 0, CHUNK_SIZE - 1);
        int iy = glm::clamp((int)std::floor(local.y), 0, CHUNK_SIZE - 1);
        int iz = glm::clamp((int)std::floor(local.z), 0, CHUNK_SIZE - 1);

        float fx = local.x - ix;
        float fy = local.y - iy;
        float fz = local.z - iz;

        // Clamp fractional parts
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

    glm::vec3 VoxelPhysicsManager::sampleNormal(const glm::vec3& worldPos) {
        glm::vec3 grad = sampleGradient(worldPos);
        float len = glm::length(grad);

        if (len < 1e-6f) {
            return glm::vec3(0, 1, 0);
        }

        // Gradient points from low to high density (inward into solid)
        // We want outward normal from solid, so use normalized gradient
        return glm::normalize(grad);
    }

    bool VoxelPhysicsManager::checkVoxelCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        switch(body.shapeType) {
            case VoxelPhysicsBody::ShapeType::SPHERE:
                return checkSphereCollision(body, outNormal, outPenetration);
            case VoxelPhysicsBody::ShapeType::BOX:
                std::cout<<"entered Collision check, type of:Box\n";
                return checkBoxCollision(body, outNormal, outPenetration);
            default:
                std::cout<<"entered Collision check, type of:default\n";
                return checkSphereCollision(body, outNormal, outPenetration);
        }
    }

    bool VoxelPhysicsManager::checkSphereCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // Sample density at sphere center
        float centerDensity = sampleDensity(body.position);

        // Quick reject if far outside
        if (centerDensity < -body.radius - VOXEL_SIZE) {
            return false;
        }

        // Sample directions (more thorough than just surface points)
        const int numSamples = 26; // 6 axes + 20 diagonal directions
        glm::vec3 sampleDirs[26];

        // Cardinal directions
        sampleDirs[0] = glm::vec3(1, 0, 0);
        sampleDirs[1] = glm::vec3(-1, 0, 0);
        sampleDirs[2] = glm::vec3(0, 1, 0);
        sampleDirs[3] = glm::vec3(0, -1, 0);
        sampleDirs[4] = glm::vec3(0, 0, 1);
        sampleDirs[5] = glm::vec3(0, 0, -1);

        // Face diagonals
        int idx = 6;
        for (int x = -1; x <= 1; x += 2) {
            for (int y = -1; y <= 1; y += 2) {
                for (int z = -1; z <= 1; z += 2) {
                    if (x != 0 && y != 0 && z != 0) {
                        sampleDirs[idx++] = glm::normalize(glm::vec3(x, y, z));
                    }
                }
            }
        }

        // Edge diagonals
        for (int i = 0; i < 3; ++i) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    glm::vec3 dir(0);
                    dir[i] = 0;
                    dir[(i+1)%3] = s1;
                    dir[(i+2)%3] = s2;
                    sampleDirs[idx++] = glm::normalize(dir);
                }
            }
        }

        float bestPen = -std::numeric_limits<float>::infinity();
        glm::vec3 bestSamplePos = body.position;

        // Sample along each direction to find penetration
        for (int i = 0; i < numSamples; ++i) {
            // Sample along this direction from center outward
            for (float r = 0; r <= body.radius; r += VOXEL_SIZE * 0.5f) {
                glm::vec3 samplePos = body.position + sampleDirs[i] * r;
                float density = sampleDensity(samplePos);

                if (density > 0) {
                    // We hit terrain - calculate penetration
                    // Penetration = density (how deep) + (body.radius - r) (how much sphere extends beyond this point)
                    float penetration = density + (body.radius - r);

                    if (penetration > bestPen) {
                        bestPen = penetration;
                        bestSamplePos = samplePos;
                    }
                    break; // Stop sampling along this direction after first hit
                }
            }
        }

        const float skinWidth = 0.01f * VOXEL_SIZE;
        if (bestPen <= skinWidth) {
            return false;
        }

        // Calculate normal at collision point
        glm::vec3 normal = sampleNormal(bestSamplePos);

        if (glm::length(normal) > 0.001f) {
            normal = glm::normalize(normal);
        } else {
            normal = glm::vec3(0, 1, 0);
        }

        // Smooth normal
        const float smoothEps = VOXEL_SIZE * 0.25f;
        glm::vec3 n1 = sampleNormal(bestSamplePos + normal * smoothEps);
        glm::vec3 n2 = sampleNormal(bestSamplePos - normal * smoothEps);

        if (glm::length(n1) > 0.001f && glm::length(n2) > 0.001f) {
            normal = glm::normalize(normal + 0.5f * glm::normalize(n1) + 0.5f * glm::normalize(n2));
        }

        // Limit penetration
        const float maxPush = VOXEL_SIZE * 2.0f;
        float penetration = glm::min(bestPen, maxPush);

        outNormal = normal;
        outPenetration = penetration;

        return true;
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

    void VoxelPhysicsManager::resolveCollision(
            VoxelPhysicsBody& body,
            const glm::vec3& normal,
            float penetration,
            float impactSpeed
    ) {
        // Push body out of collision (minimum translation)
        const float extraMargin = VOXEL_SIZE * 0.01f;
        body.position += normal * (penetration/10 + extraMargin);

        // Reflect velocity with proper physics
        float velDotNormal = glm::dot(body.velocity, normal);

        if (velDotNormal < 0) {
            // Remove velocity component along normal
            glm::vec3 normalVel = normal * velDotNormal;
            glm::vec3 tangentVel = body.velocity - normalVel;

            // Apply restitution (bounce)
            float restitution = body.restitution;
            glm::vec3 newNormalVel = -normalVel * restitution;

            // Apply friction (reduce tangent velocity)
            float friction = body.friction;
            glm::vec3 newTangentVel = tangentVel * (1.0f - friction * 0.5f);

            // Combine
            body.velocity = newNormalVel + newTangentVel;
        }

        // Add slight rotation from collision (simplified)
        if (glm::length(body.velocity) > 0.1f) {
            glm::vec3 rotationAxis = glm::cross(normal, body.velocity);
            if (glm::length(rotationAxis) > 0.001f) {
                rotationAxis = glm::normalize(rotationAxis);
                body.angularVelocity += rotationAxis * impactSpeed * 0.1f;
            }
        }

        // Dampen angular velocity
        body.angularVelocity *= 0.95f;
    }

    void VoxelPhysicsManager::removeBody(uint64_t id) {
        bodies.erase(
                std::remove_if(bodies.begin(), bodies.end(),
                               [id](const VoxelPhysicsBody& b) { return b.id == id; }),
                bodies.end()
        );
    }

    void VoxelPhysicsManager::removeBody(VoxelPhysicsBody* body) {
        if (body) removeBody(body->id);
    }

    std::vector<glm::vec3> VoxelPhysicsManager::buildUniqueVertexList(const std::vector<glm::vec3>& triangleVerts) {
        std::vector<glm::vec3> unique;
        unique.reserve(triangleVerts.size());

        std::unordered_set<size_t> seen;
        auto quantize = [](const glm::vec3 &v) -> glm::uvec3 {
            const float Q = 1000.0f;
            return glm::uvec3((unsigned int)std::round(v.x * Q),
                              (unsigned int)std::round(v.y * Q),
                              (unsigned int)std::round(v.z * Q));
        };

        for (const auto &v : triangleVerts) {
            glm::uvec3 q = quantize(v);
            size_t h = ((size_t)q.x << 42) ^ ((size_t)q.y << 21) ^ (size_t)q.z;
            if (seen.insert(h).second) {
                unique.push_back(v);
            }
        }

        return unique;
    }

}