// VoxelPhysicsManager.cpp
#include "VoxelPhysicsManager.h"

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

    void VoxelPhysicsManager::update(float dt) {
        for (auto it = bodies.begin(); it != bodies.end(); ) {
            VoxelPhysicsBody& body = *it;

            // Lifetime check
            if (body.lifetime > 0) {
                body.lifetime -= dt;
                if (body.lifetime <= 0) {
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
            std::cout<<"Velocity of: "<<body.id<<" , Velocity:"<<body.velocity.x<<" , "<<body.velocity.y<<" , "<<body.velocity.z<<"\n";

            // Store old position for collision response
            glm::vec3 oldPos = body.position;
            std::cout<<"OldPos of: "<<body.id<<" , Pos was:"<<oldPos.x<<" , "<<oldPos.y<<" , "<<oldPos.z<<"\n";

            glm::vec3 oldVel = body.velocity;

            // Integrate position
            body.position += body.velocity * dt;
            std::cout<<"Pos of: "<<body.id<<" , NewPos is:"<<body.position.x<<" , "<<body.position.y<<" , "<<body.position.z<<"\n";


            // Integrate rotation
            glm::quat spin = glm::quat(0,
                                       body.angularVelocity.x * dt * 0.5f,
                                       body.angularVelocity.y * dt * 0.5f,
                                       body.angularVelocity.z * dt * 0.5f
            );
            body.orientation += spin * body.orientation;
            body.orientation = glm::normalize(body.orientation);

            // Check collision with voxel world
            glm::vec3 normal;
            float penetration;

            if (checkVoxelCollision(body, normal, penetration)) {
                float impactSpeed = glm::length(oldVel);

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

    bool VoxelPhysicsManager::checkVoxelCollision(
            const VoxelPhysicsBody& body,
            glm::vec3& outNormal,
            float& outPenetration
    ) {
        // Sample SDF at body position
        float sdf = sampleSDF(body.position);

        // Signed distance from surface (accounting for body radius)
        float signedDist = sdf - body.radius;

        if (signedDist > 0.01f * VOXEL_SIZE) {
            return false; // No collision
        }

        // Calculate normal
        outNormal = sampleNormal(body.position);
        outPenetration = -signedDist; // Convert to penetration depth

        return true;
    }

    float VoxelPhysicsManager::sampleSDF(const glm::vec3& worldPos) {
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

        // Trilinear interpolation (same as your CharacterController)
        int ix = (int)std::floor(local.x);
        int iy = (int)std::floor(local.y);
        int iz = (int)std::floor(local.z);

        float fx = local.x - ix;
        float fy = local.y - iy;
        float fz = local.z - iz;

        // Clamp indices
        ix = glm::clamp(ix, 0, CHUNK_SIZE - 1);
        iy = glm::clamp(iy, 0, CHUNK_SIZE - 1);
        iz = glm::clamp(iz, 0, CHUNK_SIZE - 1);

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

    glm::vec3 VoxelPhysicsManager::sampleNormal(const glm::vec3& worldPos) {
        const float e = VOXEL_SIZE * 0.5f;

        float dx = sampleSDF(worldPos + glm::vec3(e, 0, 0)) -
                   sampleSDF(worldPos - glm::vec3(e, 0, 0));
        float dy = sampleSDF(worldPos + glm::vec3(0, e, 0)) -
                   sampleSDF(worldPos - glm::vec3(0, e, 0));
        float dz = sampleSDF(worldPos + glm::vec3(0, 0, e)) -
                   sampleSDF(worldPos - glm::vec3(0, 0, e));

        glm::vec3 grad(dx, dy, dz);
        float len = glm::length(grad);

        if (len < 1e-6f) {
            return glm::vec3(0, 1, 0);
        }

        // Gradient points inward (density increases), we want outward normal
        return -glm::normalize(grad);
    }

    void VoxelPhysicsManager::resolveCollision(
            VoxelPhysicsBody& body,
            const glm::vec3& normal,
            float penetration,
            float impactSpeed
    ) {

        // Push body out of collision
        body.position += normal * (penetration + 0.00001f * VOXEL_SIZE);

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