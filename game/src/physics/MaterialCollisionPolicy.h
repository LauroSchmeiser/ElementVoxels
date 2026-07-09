#pragma once

#include <glm/glm.hpp>
#include <cmath>
#include <vector>

enum class MaterialCollisionMode : uint8_t {
    DefaultBounce = 0,
    StickOnWorld,
    StickOnBody,
    StickOnAll,
    PassThroughFirstBody,
    CollidePlayerOnly,
    DestroyTargetKeepFlying
};

struct MaterialCollisionRule {
    MaterialCollisionMode mode = MaterialCollisionMode::DefaultBounce;
    int passThroughBodiesLeft = 0;   // runtime counter seed
    bool collideWorld = true;
    bool collideBodies = true;
    bool collidePlayer = true;
    float bounceMultiplier = 1.0f;   // for default mode

    bool convertOnStick = false;
    float convertRadius = 0.0f;      // world units
};

struct CollisionDecision {
    bool resolvePhysics = true;      // bounce/resolve
    bool consumeProjectile = false;  // remove self
    bool destroyOther = false;       // remove target
    bool keepFlying = false;         // don't damp velocity
    bool stick = false;              // attach at contact
    bool ignoreCollision = false;    // pass through

    bool convertWorld = false;       // convert world voxels at hit
    bool convertOtherBody = false;
};