#pragma once

#include <cstdint>
#include <vector>
#include "glm/glm.hpp"
#include <glad/glad.h>
#include "../physics/VoxelPhysicsBody.h"

namespace gl3 {
    constexpr int CHUNK_SIZE = 16;
    constexpr float VOXEL_SIZE = 3.0f;

    struct Voxel {
        uint8_t type = 0;
        uint32_t material=0;
        float density = 1.0f;
        glm::vec3 color = glm::vec3(1.0f);

        bool isSolid() const { return type != 0; }
        // 0==Empty, 1==base, 2==fire, 3==fluid
        // 0==stone, 1==earth, 2==ice etc....
    };

    struct VoxelLight {
        glm::vec3 pos;
        float intensity;
        glm::vec3 color;
        uint32_t id;
    };

    struct EmissiveBlob {
        glm::vec3 sumPos;
        int count = 0;
        glm::vec3 sumColor;
    };

    struct OutVertex {
        glm::vec4 pos;    // xyz: position, w unused
        glm::vec4 normal; // xyz: normal, w unused
        glm::vec4 color;  // rgb: color, a unused
    };

    struct ChunkCoord {
        int x, y, z;

        bool operator==(const ChunkCoord &other) const {
            return x == other.x &&
                   y == other.y &&
                   z == other.z;
        }

        bool operator!=(const ChunkCoord &other) const {
            return !(*this == other);
        }
    };

    // Better hash function for ChunkCoord
    struct ChunkCoordHash {
        // Using murmurhash-like mixing
        std::size_t operator()(const ChunkCoord &c) const noexcept {
            // FNV-1a hash
            constexpr uint64_t offset_basis = 0xcbf29ce484222325;
            constexpr uint64_t prime = 0x100000001b3;

            uint64_t hash = offset_basis;

            hash ^= static_cast<uint64_t>(c.x);
            hash *= prime;
            hash ^= static_cast<uint64_t>(c.y);
            hash *= prime;
            hash ^= static_cast<uint64_t>(c.z);
            hash *= prime;

            return static_cast<std::size_t>(hash);
        }
    };


////------------Spell-Systen-------------------------------------------

    struct AnimatedVoxel {
        uint64_t id = 0; // unique stable id
        glm::vec3 currentPos;
        glm::vec3 targetPos;
        glm::vec3 velocity;
        glm::vec3 color;
        glm::vec3 originalVoxelPos;
        glm::vec3 normal;
        float animationSpeed = 3.0f;
        bool isAnimating = false;
        bool hasArrived = false; // track if arrived at target
    };

    enum class FormationType {
        SPHERE,
        PLATFORM,
        WALL,
        CUBE,
        PYRAMID,
        CYLINDER,
        CUSTOM_SDF
    };
    typedef float (*SDFFunction)(const glm::vec3& p, void* userData);

    struct FormationParams {
        FormationType type = FormationType::SPHERE;

        // Universal parameters
        glm::vec3 center;
        glm::vec3 normal; // For walls/platforms
        glm::vec3 up; // Orientation
        float sizeX = 1.0f;
        float sizeY = 1.0f;
        float sizeZ = 1.0f;
        float radius = 1.0f;

        // For custom SDF functions
        SDFFunction customSDF = nullptr;
        void *customUserData = nullptr; // Optional user data

        // Transformation
        glm::mat4 transform = glm::mat4(1.0f);

        // Helper constructors
        static FormationParams Sphere(const glm::vec3 &center, float radius) {
            FormationParams params;
            params.type = FormationType::SPHERE;
            params.center = center;
            params.radius = radius;
            return params;
        }

        static FormationParams Platform(const glm::vec3 &center, const glm::vec3 &normal,
                                        float width, float depth, float thickness = 0.5f) {
            FormationParams params;
            params.type = FormationType::PLATFORM;
            params.center = center;
            params.normal = glm::normalize(normal);
            params.sizeX = width;
            params.sizeY = thickness;
            params.sizeZ = depth;

            // Create orientation from normal
            glm::vec3 up = glm::abs(params.normal.y) > 0.9f ?
                           glm::vec3(0.0f, 0.0f, 1.0f) :
                           glm::vec3(0.0f, 1.0f, 0.0f);
            params.up = glm::normalize(glm::cross(params.normal, glm::cross(up, params.normal)));
            return params;
        }

        static FormationParams Wall(const glm::vec3 &center, const glm::vec3 &normal,
                                    float width, float height, float thickness = 0.5f) {
            FormationParams params;
            params.type = FormationType::WALL;
            params.center = center;
            params.normal = glm::normalize(normal);
            params.sizeX = width;
            params.sizeY = height;
            params.sizeZ = thickness;

            glm::vec3 up(0.0f, 1.0f, 0.0f);
            if (glm::abs(glm::dot(normal, up)) > 0.9f) {
                up = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            params.up = glm::normalize(up);
            return params;
        }

        static FormationParams Cube(const glm::vec3 &center, const glm::vec3 &size) {
            FormationParams params;
            params.type = FormationType::CUBE;
            params.center = center;
            params.sizeX = size.x;
            params.sizeY = size.y;
            params.sizeZ = size.z;
            return params;
        }

        static FormationParams Cylinder(const glm::vec3 &center, float radius, float height) {
            FormationParams params;
            params.type = FormationType::CYLINDER;
            params.center = center;
            params.radius = radius;
            params.sizeY = height;
            return params;
        }

        // Evaluate the SDF at a point
        float evaluate(const glm::vec3 &p) const {
            glm::vec3 localP = p - center;

            switch (type) {
                case FormationType::SPHERE:
                    return sphereSDF(localP);
                case FormationType::PLATFORM:
                    return platformSDF(localP);
                case FormationType::WALL:
                    return wallSDF(localP);
                case FormationType::CUBE:
                    return cubeSDF(localP);
                case FormationType::PYRAMID:
                    return pyramidSDF(localP);
                case FormationType::CYLINDER:
                    return cylinderSDF(localP);
                case FormationType::CUSTOM_SDF:
                    if (customSDF) {
                        return customSDF(p, customUserData);
                    }
                    return 1.0f;
                default:
                    return sphereSDF(localP);
            }
        }

        // Get bounding radius for culling
        float getBoundingRadius() const {
            switch (type) {
                case FormationType::SPHERE:
                    return radius;
                case FormationType::PLATFORM:
                    return glm::length(glm::vec3(sizeX, sizeY, sizeZ)) * 0.5f;
                case FormationType::WALL:
                    return glm::length(glm::vec3(sizeX, sizeY, sizeZ)) * 0.5f;
                case FormationType::CUBE:
                    return glm::length(glm::vec3(sizeX, sizeY, sizeZ)) * 0.5f;
                case FormationType::CYLINDER:
                    return glm::length(glm::vec3(radius, sizeY * 0.5f, radius));
                default:
                    return glm::max(sizeX, glm::max(sizeY, sizeZ)) * 0.5f;
            }
        }

    private:
        float sphereSDF(const glm::vec3 &p) const {
            return radius - glm::length(p);
        }

        // platformSDF (negated to be positive inside)
        float platformSDF(const glm::vec3 &p) const {
            glm::mat3 rotation = buildRotationMatrix(normal, up);
            glm::vec3 rotated = rotation * p;

            glm::vec3 halfSize = glm::vec3(sizeX, sizeY, sizeZ) * 0.5f;
            glm::vec3 q = glm::abs(rotated) - halfSize;
            float outside = glm::length(glm::max(q, glm::vec3(0.0f)));
            float inside = glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
            // NEGATE: want positive inside (consistent with sphereSDF)
            return -(outside + inside);
        }

        float wallSDF(const glm::vec3 &p) const {
            glm::mat3 rotation = buildRotationMatrix(normal, up);
            glm::vec3 rotated = rotation * p;

            glm::vec3 halfSize = glm::vec3(sizeX, sizeY, sizeZ) * 0.5f;
            glm::vec3 q = glm::abs(rotated) - halfSize;
            float outside = glm::length(glm::max(q, glm::vec3(0.0f)));
            float inside = glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
            return -(outside + inside);
        }

        float cubeSDF(const glm::vec3 &p) const {
            glm::vec3 halfSize = glm::vec3(sizeX, sizeY, sizeZ) * 0.5f;
            glm::vec3 q = glm::abs(p) - halfSize;
            float outside = glm::length(glm::max(q, glm::vec3(0.0f)));
            float inside = glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
            return -(outside + inside);
        }

        float pyramidSDF(const glm::vec3 &p) const {
            // Original implementation produced negative-inside convention — invert result
            float h = sizeY;
            float halfBase = sizeX * 0.5f;

            glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y);
            float a = q.x - glm::min(q.x, (h - q.y) * (halfBase / h));
            float b = glm::max(q.x + q.y - h, 0.0f);
            float val = glm::sqrt(a * a + b * b) * glm::sign(q.x - halfBase);
            return -val; // negate so inside -> positive
        }

        float cylinderSDF(const glm::vec3 &p) const {
            glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y)) -
                          glm::vec2(radius, sizeY * 0.5f);
            float outside = glm::length(glm::max(d, glm::vec2(0.0f)));
            float inside = glm::min(glm::max(d.x, d.y), 0.0f);
            return -(inside + outside);
        }

        glm::mat3 buildRotationMatrix(const glm::vec3 &forward, const glm::vec3 &up) const {
            glm::vec3 right = glm::normalize(glm::cross(forward, up));
            glm::vec3 correctedUp = glm::normalize(glm::cross(right, forward));
            return glm::mat3(right, correctedUp, forward);
        }
    };

    struct PhysicsMeshData {
        GLuint vao = 0;
        GLuint vbo = 0;
        uint32_t vertexCount = 0;
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> colors;
        bool isValid = false;
    };

    struct SpellEffect {
        enum class Type {
            GRAVITY_WELL,
            CONSTRUCT,
            TELEKINESIS
        };

        Type type;
        glm::vec3 center;
        float radius;
        float strength;
        bool geometryCreated = false;
        uint64_t targetMaterial;
        glm::vec3 formationColor;
        float formationRadius = 0.0f;
        uint8_t dominantType = 0;

        // NEW: Formation configuration
        FormationParams formationParams;

        // store stable IDs (not vector indices)
        std::vector<uint64_t> animatedVoxelIDs;

        bool voxelsCleaned = false;  // Add this - true when animated voxels are removed
        bool markForRemoval = false; // Add this - set when ready to delete

        VoxelPhysicsBody* physicsBody = nullptr; // optional cache
        uint64_t physicsBodyId = 0;              // authoritative identity#
        bool isPhysicsEnabled = false;     // Whether this formation has physics
        glm::vec3 initialVelocity = glm::vec3(0.0f);  // Velocity to apply when body is created

        // For tracking physics state
        bool physicsInitialized = false;
        float creationTime = 0.0f;
        float lifetime = 50.0f; // -1 = infinite
        PhysicsMeshData physicsMesh;

        // Add these fields to gl3::SpellEffect
        struct BurnState {
            bool active = false;
            float t = 0.0f;
            float duration = 1.0f;
            glm::vec3 center = glm::vec3(0.0f);
            float radius = 0.0f;
            float noiseScale = 0.45f;
            float edgeWidth = 0.12f;
            float slowAccum=0.0f;

        };
        BurnState burn;
    };



    // Forward declare Chunk - full definition will be in Chunk.h
    struct Chunk;
}