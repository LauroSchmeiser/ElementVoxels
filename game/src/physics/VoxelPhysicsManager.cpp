// VoxelPhysicsManager.cpp
#include "VoxelPhysicsManager.h"
#include "glm/gtc/quaternion.hpp"

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
        /*for(auto it = bodies.begin(); it != bodies.end();)
        {
            std::cout<<"Start Pos of:<<body.id<<"<<" , Start Pos:"<<body.position.x<<" , "<<body.position.y<<" , "<<body.position.z<<"\n";
        }*/
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
                    removedBodies.push_back(body.id);  // Record removed ID
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
            glm::vec3 oldPos = body.position;
            glm::vec3 oldVel = body.velocity;

            // Integrate position
            body.position += body.velocity * dt;

            // Check collision with voxel world
            glm::vec3 normal;
            float penetration;

            if (checkVoxelCollision(body, normal, penetration)) {
                float impactSpeed = glm::length(oldVel);
                resolveCollision(body, normal, penetration, impactSpeed);

                // Call callback
                if (collisionCallback && impactSpeed > 1.0f) {
                    collisionCallback(&body, body.position, normal, impactSpeed);
                }
            }

            ++it;
        }
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
                return checkBoxCollision(body, outNormal, outPenetration);
            default:
                return checkSphereCollision(body, outNormal, outPenetration);
        }
    }

    bool VoxelPhysicsManager::checkSphereCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // For sphere, we can use a more accurate method: sample in a spiral pattern
        const int numRings = 5;
        const int samplesPerRing = 8;

        float maxPenetration = 0.0f;
        glm::vec3 collisionNormal(0.0f);
        int collisionCount = 0;

        for (int ring = 0; ring < numRings; ++ring) {
            float phi = glm::pi<float>() * float(ring) / (numRings - 1); // 0 to PI
            float y = cos(phi);
            float radiusAtY = sin(phi);

            for (int s = 0; s < samplesPerRing; ++s) {
                float theta = 2.0f * glm::pi<float>() * float(s) / samplesPerRing;

                glm::vec3 dir(
                        radiusAtY * cos(theta),
                        y,
                        radiusAtY * sin(theta)
                );

                glm::vec3 samplePos = body.position + dir * body.radius;
                float sdf = sampleSDF(samplePos);

                if (sdf < 0) {
                    float penetration = -sdf;
                    glm::vec3 normal = sampleNormal(samplePos);

                    collisionNormal += normal;
                    maxPenetration = std::max(maxPenetration, penetration);
                    collisionCount++;
                }
            }
        }

        if (collisionCount > 0) {
            outNormal = glm::normalize(collisionNormal);
            outPenetration = maxPenetration;
            return true;
        }

        return false;
    }

    bool VoxelPhysicsManager::checkBoxCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // For box, we need to check all 8 corners and edges
        glm::vec3 halfExtents = body.shapeExtents;

        // Generate all 8 corners of the box in world space
        glm::vec3 corners[8];
        glm::mat3 rot = glm::mat3_cast(body.orientation);

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
            float sdf = sampleSDF(corners[i]);

            if (sdf < 0) {
                float penetration = -sdf;
                glm::vec3 normal = sampleNormal(corners[i]);

                collisionNormal += normal;
                maxPenetration = std::max(maxPenetration, penetration);
                collisionCount++;
            }
        }

        // Also check edge midpoints for better accuracy
        const int edges[12][2] = {
                {0,1}, {1,3}, {3,2}, {2,0}, // bottom face
                {4,5}, {5,7}, {7,6}, {6,4}, // top face
                {0,4}, {1,5}, {3,7}, {2,6}  // vertical edges
        };

        for (int e = 0; e < 12; ++e) {
            glm::vec3 edgeMid = (corners[edges[e][0]] + corners[edges[e][1]]) * 0.5f;
            float sdf = sampleSDF(edgeMid);

            if (sdf < 0) {
                float penetration = -sdf;
                glm::vec3 normal = sampleNormal(edgeMid);

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

    float VoxelPhysicsManager::sampleSDF(const glm::vec3& worldPos) {
        if (!chunkManager) return -10000.0f;
        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int baseCX = worldToChunk(worldPos.x);
        int baseCY = worldToChunk(worldPos.y);
        int baseCZ = worldToChunk(worldPos.z);
        glm::vec3 chunkMin = glm::vec3(baseCX * chunkWorldSize,baseCY * chunkWorldSize,baseCZ * chunkWorldSize);
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;
        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));
        float fx = local.x - ix;
        float fy = local.y - iy;
        float fz = local.z - iz;

        // gather 8 corner samples by querying the chunk manager (handles chunk boundaries)
        auto sampleCorner = [&](int sx, int sy, int sz)->float {
            // corner world position:
            glm::vec3 cornerWorld = chunkMin + glm::vec3((float)(ix + sx), (float)(iy + sy), (float)(iz + sz)) * VOXEL_SIZE;
            int cx = worldToChunk(cornerWorld.x);
            int cy = worldToChunk(cornerWorld.y);
            int cz = worldToChunk(cornerWorld.z);
            ChunkCoord coord{cx, cy, cz};
            Chunk* chunk = chunkManager->getChunk(coord);
            if (!chunk) return -1000.0f;
            // local index inside that chunk (0..CHUNK_SIZE)
            glm::vec3 localCorner = (cornerWorld - getChunkMin(coord)) / VOXEL_SIZE;
            int lx = glm::clamp((int)std::round(localCorner.x), 0, CHUNK_SIZE);
            int ly = glm::clamp((int)std::round(localCorner.y), 0, CHUNK_SIZE);
            int lz = glm::clamp((int)std::round(localCorner.z), 0, CHUNK_SIZE);
            return chunk->voxels[lx][ly][lz].density;
        };

        float s000 = sampleCorner(0,0,0);
        float s100 = sampleCorner(1,0,0);
        float s010 = sampleCorner(0,1,0);
        float s110 = sampleCorner(1,1,0);
        float s001 = sampleCorner(0,0,1);
        float s101 = sampleCorner(1,0,1);
        float s011 = sampleCorner(0,1,1);
        float s111 = sampleCorner(1,1,1);

        auto lerp = [](float a, float b, float t){ return a + (b - a) * t; };
        float c00 = lerp(s000, s100, fx);
        float c10 = lerp(s010, s110, fx);
        float c01 = lerp(s001, s101, fx);
        float c11 = lerp(s011, s111, fx);

        float c0 = lerp(c00, c10, fy);
        float c1 = lerp(c01, c11, fy);
        return lerp(c0, c1, fz);
    }

    int VoxelPhysicsManager::worldToChunk(float worldPos){
        // Each chunk covers CHUNK_SIZE voxels, each voxel is VOXEL_SIZE world units.
        // So chunk world width = CHUNK_SIZE * VOXEL_SIZE.
        float chunkWorldSize = CHUNK_SIZE * gl3::VOXEL_SIZE;
        return (int) std::floor(worldPos / chunkWorldSize);
    }

    glm::vec3 VoxelPhysicsManager::getChunkMin(const ChunkCoord& coord) const {
        // chunk origin in world units
        return glm::vec3(coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE);
    }

    glm::vec3 VoxelPhysicsManager::getChunkMax(const ChunkCoord& coord) const {
        return glm::vec3((coord.x + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         (coord.y + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         (coord.z + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE);
    }
    glm::vec3 VoxelPhysicsManager::sampleNormal(const glm::vec3& worldPos) {
        const float e = VOXEL_SIZE * 0.5f;
        float dx = sampleSDF(worldPos + glm::vec3(e,0,0)) - sampleSDF(worldPos - glm::vec3(e,0,0));
        float dy = sampleSDF(worldPos + glm::vec3(0,e,0)) - sampleSDF(worldPos - glm::vec3(0,e,0));
        float dz = sampleSDF(worldPos + glm::vec3(0,0,e)) - sampleSDF(worldPos - glm::vec3(0,0,e));
        glm::vec3 g(dx,dy,dz);
        float len = glm::length(g);
        if (len < 1e-6f) return glm::vec3(0.0f, 1.0f, 0.0f);
        // density increases INTO solid => gradient points inward; we want outward normal
        return -glm::normalize(g);
    }

    void VoxelPhysicsManager::resolveCollision(
            VoxelPhysicsBody& body,
            const glm::vec3& normal,
            float penetration,
            float impactSpeed
    ) {
/*
        // Push body out of collision
        body.position += normal * (penetration + 0.0001f * VOXEL_SIZE);

        // Reflect velocity
        float velDotNormal = glm::dot(body.velocity, normal);

        if (velDotNormal < 0) {
            // Remove velocity component along normal
            body.velocity -= normal * velDotNormal;

            // Apply restitution (bounciness)
            body.velocity -= normal * velDotNormal * body.restitution;

            // Apply friction (dampen tangential velocity)
            glm::vec3 tangent = body.velocity - normal * glm::dot(body.velocity, normal);
            body.velocity -= tangent * body.friction * 0.5f;
        }

        // Dampen angular velocity on impact
        body.angularVelocity *= 0.9f;
        */
    }

    void VoxelPhysicsManager::removeBody(uint64_t id) {
        // First, find the body and clear its userData to prevent dangling pointers
        for (auto& body : bodies) {
            if (body.id == id) {
                body.userData = nullptr;
                break;
            }
        }

        // Then erase it
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
        // Use quantized hashing to avoid tiny floating point dupes
        std::unordered_set<size_t> seen;
        auto quantize = [](const glm::vec3 &v)->glm::uvec3 {
            const float Q = 1000.0f; // quantization factor - tweak if needed
            return glm::uvec3((unsigned int)std::round(v.x * Q),
                              (unsigned int)std::round(v.y * Q),
                              (unsigned int)std::round(v.z * Q));
        };
        for (const auto &v : triangleVerts) {
            glm::uvec3 q = quantize(v);
            // hash the quantized coords to a size_t
            size_t h = ((size_t)q.x) << 42 ^ ((size_t)q.y << 21) ^ (size_t)q.z;
            if (seen.insert(h).second) unique.push_back(v);
        }
        return unique;
    }

}