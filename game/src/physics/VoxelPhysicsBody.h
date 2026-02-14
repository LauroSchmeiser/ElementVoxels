
#pragma once
#include <glm/glm.hpp>
#include "glm/detail/type_quat.hpp"

namespace gl3 {
    struct PhysicsMeshData;

struct VoxelPhysicsBody {
    // Transform
    glm::vec3 position=glm::vec3(0,0,0);
    glm::vec3 velocity;
    glm::vec3 angularVelocity;
    glm::quat orientation;

    // Properties
    float mass;
    float radius; // Bounding sphere for broad-phase
    float restitution = 0.3f;
    float friction = 0.7f;

    // Collision shape (simplified)
    enum class ShapeType { SPHERE, CAPSULE, BOX } shapeType;
    glm::vec3 shapeExtents; // radius for sphere, halfExtents for box

    // Lifecycle
    bool active = true;
    float lifetime = -1.0f; // -1 = infinite

    // Rendering link
    PhysicsMeshData* renderMesh = nullptr;

    // User data
    void* userData = nullptr;
    uint64_t id = 0;
};
}