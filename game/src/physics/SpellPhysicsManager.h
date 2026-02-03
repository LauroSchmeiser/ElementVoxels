/*
SpellPhysicsManager.cpp

Drop-in helper to:
- initialize a Bullet dynamics world
- create dynamic rigid bodies from a triangle/vertex list or voxel-snapshot vertices (convex-hull)
- detect collision impulses and call back into game logic so you can carve voxels on impact
- optionally schedule fragmentation (hook point)

Usage notes (integration sketch)
- Include this file in your build and call SpellPhysicsManager::init() at startup.
- Each frame call SpellPhysicsManager::stepSimulation(deltaTime).
- When a formation finishes and you have a triangle mesh (std::vector<glm::vec3> verts),
call createRigidBodyFromVertices(...) to spawn a rigidbody. Provide an impact callback
(std::function) which will be invoked with contact point, computed damage radius and impulse.
- The manager uses btConvexHullShape (fast, stable). For high fidelity you can replace
that with VHACD / compound shapes later.

This file depends on:
- Bullet (btBulletDynamicsCommon.h)
- glm
- STL
- Your game should provide an impact handler that applies voxel damage (e.g., carve voxels)

Author: Generated for ElementVoxels integration
*/

#pragma once
#include "../../../extern/bullet3-3.25/src/btBulletDynamicsCommon.h"
#include <vector>
#include <unordered_set>
#include <functional>
#include <memory>
#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>

namespace gl3 {
    // Forward declare minimal types used here for clarity; your project already provides these.
    struct OutVertex;
}

// Small POD that is attached to btRigidBody::setUserPointer()
// so the collision callback can identify which game entity / formation this body corresponds to.
struct RigidBodyPayload {
    uint64_t formationID = 0; // optional id you can assign
    void* userData = nullptr; // arbitrary pointer (e.g., pointer to your Game or entity)
};

class SpellPhysicsManager {
public:
    using ImpactCallback = std::function<void(const glm::vec3& worldPos, float damageRadius, float impulse, RigidBodyPayload*)>;

    SpellPhysicsManager()
            : collisionConfiguration(nullptr), dispatcher(nullptr),
              broadphase(nullptr), solver(nullptr), dynamicsWorld(nullptr) {}

    ~SpellPhysicsManager() {
        shutdown();
    }

    // Initialize Bullet world. Call once.
    void init(const ImpactCallback& cb) {
        impactCallback = cb;

        collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
        dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
        broadphase = std::make_unique<btDbvtBroadphase>();
        solver = std::make_unique<btSequentialImpulseConstraintSolver>();

        dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(
                dispatcher.get(), broadphase.get(), solver.get(), collisionConfiguration.get());
        dynamicsWorld->setGravity(btVector3(0.0f, -9.81f, 0.0f));

        // register tick callback to scan contacts each step
        dynamicsWorld->setInternalTickCallback(&SpellPhysicsManager::internalTickCallback, this, true);

        // allow callback to find this manager from world pointer
        dynamicsWorld->setWorldUserInfo(this);
    }

    // Step simulation each frame
    void stepSimulation(float dt, int maxSubSteps = 1, float fixedTimeStep = 1.0f / 60.0f) {
        if (dynamicsWorld) {
            dynamicsWorld->stepSimulation(dt, maxSubSteps, fixedTimeStep);

            // cleanup any bodies scheduled for removal
            if (!bodiesToRemove.empty()) {
                for (btRigidBody* b : bodiesToRemove) {
                    removeRigidBodyImmediate(b);
                }
                bodiesToRemove.clear();
            }
        }
    }

    // Shutdown and free resources (call on exit)
    void shutdown() {
        if (dynamicsWorld) {
            // remove bodies
            for (int i = dynamicsWorld->getNumCollisionObjects() - 1; i >= 0; --i) {
                btCollisionObject* obj = dynamicsWorld->getCollisionObjectArray()[i];
                btRigidBody* body = btRigidBody::upcast(obj);
                if (body) {
                    removeRigidBodyImmediate(body);
                } else {
                    dynamicsWorld->removeCollisionObject(obj);
                    delete obj;
                }
            }
        }

        // free shapes allocated in our heap
        for (btCollisionShape* s : ownedShapes) delete s;
        ownedShapes.clear();

        // free motionstates we kept
        for (btMotionState* m : ownedMotionStates) delete m;
        ownedMotionStates.clear();

        // free payloads
        for (RigidBodyPayload* p : payloadsOwned) delete p;
        payloadsOwned.clear();

        // smart pointers free the rest
        dynamicsWorld.reset();
        solver.reset();
        broadphase.reset();
        dispatcher.reset();
        collisionConfiguration.reset();
    }

    // Create a dynamic rigid body using a convex hull built from uniqueVertices.
    // uniqueVertices: world-space positions forming the hull.
    // massKg: mass in kilograms
    // startTransform: initial position/orientation (btTransform)
    // formationID + userData stored in payload and set as user pointer
    btRigidBody* createRigidBodyFromVertices(const std::vector<glm::vec3>& uniqueVertices,
                                             float massKg,
                                             const btTransform& startTransform,
                                             uint64_t formationID = 0,
                                             void* userData = nullptr,
                                             float contactResponseThreshold = 1.0f) {
        if (uniqueVertices.empty() || !dynamicsWorld) return nullptr;

        // Build convex hull
        btConvexHullShape* hull = new btConvexHullShape();
        // NOTE: btConvexHullShape does not have reserve(); we simply add points.
        for (const auto &v : uniqueVertices) {
            hull->addPoint(btVector3(v.x, v.y, v.z));
        }
        hull->optimizeConvexHull();
        hull->recalcLocalAabb();

        // Keep ownership to delete later
        ownedShapes.push_back(hull);

        // inertia
        btVector3 localInertia(0,0,0);
        if (massKg > 0.0f) hull->calculateLocalInertia(massKg, localInertia);

        // Motion state
        btDefaultMotionState* motion = new btDefaultMotionState(startTransform);
        ownedMotionStates.push_back(motion);

        btRigidBody::btRigidBodyConstructionInfo rbInfo(massKg, motion, hull, localInertia);
        btRigidBody* body = new btRigidBody(rbInfo);

        // set some reasonable defaults for breakable, friction, restitution
        body->setFriction(0.7f);
        body->setRestitution(0.1f);

        // store payload
        RigidBodyPayload* payload = new RigidBodyPayload();
        payload->formationID = formationID;
        payload->userData = userData;
        payloadsOwned.push_back(payload); // remember to free
        body->setUserPointer(payload);

        // contact response threshold used by contact handling logic
        bodyContactThresholds[body] = contactResponseThreshold;

        dynamicsWorld->addRigidBody(body);

        return body;
    }

    // Convenience: build unique vertex list from triangles (OutVertex contains vec4 pos in many repos)
    // Input: triangles as triplets of glm::vec3 (flat list where size%3 == 0)
    static std::vector<glm::vec3> buildUniqueVertexList(const std::vector<glm::vec3>& triangleVerts) {
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

    // Remove a rigid body later (schedules removal next safe point).
    void removeRigidBody(btRigidBody* body) {
        if (!body) return;
        bodiesToRemove.push_back(body);
    }

    // Change gravity
    void setGravity(const glm::vec3& g) {
        if (dynamicsWorld) dynamicsWorld->setGravity(btVector3(g.x, g.y, g.z));
    }

private:
    // Internal: immediate removal (assumes dynamicsWorld exists)
    void removeRigidBodyImmediate(btRigidBody* body) {
        if (!body || !dynamicsWorld) return;

        // remove from world, delete shape & motion & payload mapping
        dynamicsWorld->removeRigidBody(body);

        // delete payload
        if (body->getUserPointer()) {
            RigidBodyPayload* p = reinterpret_cast<RigidBodyPayload*>(body->getUserPointer());
            auto it = std::find(payloadsOwned.begin(), payloadsOwned.end(), p);
            if (it != payloadsOwned.end()) {
                delete p;
                payloadsOwned.erase(it);
            }
            body->setUserPointer(nullptr);
        }

        delete body;
    }

    // Internal tick: inspect manifolds and call impactCallback when impulse exceeds threshold
    static void internalTickCallback(btDynamicsWorld* world, btScalar /*timeStep*/) {
        // Use world->getWorldUserInfo() to find manager instance
        void* info = world->getWorldUserInfo();
        if (!info) return;
        SpellPhysicsManager* mgr = reinterpret_cast<SpellPhysicsManager*>(info);
        if (!mgr) return;

        // iterate manifolds
        int numManifolds = world->getDispatcher()->getNumManifolds();
        for (int i = 0; i < numManifolds; ++i) {
            btPersistentManifold* manifold = world->getDispatcher()->getManifoldByIndexInternal(i);
            const btCollisionObject* obA = manifold->getBody0();
            const btCollisionObject* obB = manifold->getBody1();

            // skip if no contacts
            int numContacts = manifold->getNumContacts();
            if (numContacts == 0) continue;

            // find max applied impulse for this manifold
            float maxImpulse = 0.0f;
            btVector3 contactPointWorld(0,0,0);
            for (int p = 0; p < numContacts; ++p) {
                const btManifoldPoint& pt = manifold->getContactPoint(p);
                float impulse = pt.getAppliedImpulse();
                if (impulse > maxImpulse) {
                    maxImpulse = impulse;
                    contactPointWorld = pt.getPositionWorldOnB();
                }
            }

            if (maxImpulse <= 0.0f) continue;

            btRigidBody* aBody = (btRigidBody*)btRigidBody::upcast(const_cast<btCollisionObject*>(obA));
            btRigidBody* bBody = (btRigidBody*)btRigidBody::upcast(const_cast<btCollisionObject*>(obB));

            auto handleBodyImpact = [&](btRigidBody* body) {
                if (!body) return;
                void* up = body->getUserPointer();
                if (!up) return;
                RigidBodyPayload* payload = reinterpret_cast<RigidBodyPayload*>(up);
                // resolve threshold for this body
                float threshold = 1.0f;
                auto it = mgr->bodyContactThresholds.find(body);
                if (it != mgr->bodyContactThresholds.end()) threshold = it->second;

                // only act if impulse exceeds threshold
                if (maxImpulse > threshold) {
                    glm::vec3 pt(contactPointWorld.x(), contactPointWorld.y(), contactPointWorld.z());
                    float damageRadius = mgr->computeDamageRadiusFromImpulse(maxImpulse, body);
                    if (mgr->impactCallback) {
                        mgr->impactCallback(pt, damageRadius, maxImpulse, payload);
                    }
                }
            };

            handleBodyImpact(aBody);
            handleBodyImpact(bBody);
        }
    }

    // compute damage radius heuristics: scale mass and impulse into a radius
    float computeDamageRadiusFromImpulse(float impulse, btRigidBody* body) {
        float invMass = body->getInvMass();
        float mass = invMass > 0.0f ? 1.0f / invMass : 1000.0f;
        float base = 0.5f * std::cbrt((mass * impulse));
        return std::clamp(base, 0.25f, 64.0f);
    }

private:
    std::unique_ptr<btDefaultCollisionConfiguration> collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> dispatcher;
    std::unique_ptr<btDbvtBroadphase> broadphase;
    std::unique_ptr<btSequentialImpulseConstraintSolver> solver;
    std::unique_ptr<btDiscreteDynamicsWorld> dynamicsWorld;

    // Owned shapes, motionstates and payloads lifetime managed here
    std::vector<btCollisionShape*> ownedShapes;
    std::vector<btMotionState*> ownedMotionStates;
    std::vector<RigidBodyPayload*> payloadsOwned;

    // schedule-or-delete list
    std::vector<btRigidBody*> bodiesToRemove;

    // impact callback invoked when collisions exceed threshold
    ImpactCallback impactCallback;

    // contact thresholds for bodies
    std::unordered_map<btRigidBody*, float> bodyContactThresholds;
};