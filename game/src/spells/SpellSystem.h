#pragma once
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <queue>
#include <future>

#include "SpellWorldContext.h"
#include "SpellEffect.h"
#include "SpellCastAsync.h"


namespace gl3 {

    class SpellSystem : public std::enable_shared_from_this<SpellSystem> {
    public:
        explicit SpellSystem(SpellWorldContext ctx);
        ~SpellSystem();

        void update(float dt);

        void pumpAsyncResults();

        void castSphere(const glm::vec3& center, float radius, uint64_t material, float strength);
        void castWall(const glm::vec3& center, const glm::vec3& normal,
                      float width, float height, float thickness,
                      uint64_t material, float strength);

        const std::deque<SpellEffect>& spells() const { return activeSpells; }
        std::deque<SpellEffect>& spellsMutable() { return activeSpells; }

        const std::vector<AnimatedVoxel>& animated() const { return animatedVoxels; }

        void clear();

        SpellEffect* findSpellById(uint64_t id);
        const SpellEffect* findSpellById(uint64_t id) const;

    private:
        SpellWorldContext ctx;

        std::deque<SpellEffect> activeSpells;
        std::vector<AnimatedVoxel> animatedVoxels;

        std::unordered_map<uint64_t, size_t> animatedVoxelIndexMap;
        uint64_t nextAnimatedVoxelID = 1;

        std::unique_ptr<SpellCastAsync> spellCastAsync;
        std::mutex spellApplyMutex;

        SpellCastRequest buildSpellCastRequestSnapshot(
                const glm::vec3& center,
                float searchRadius,
                uint64_t targetMaterial,
                float strength,
                const FormationParams& baseFormationParams
        );

        struct GpuTrianglesReadback {
            std::vector<glm::vec3> vertsLocal;   // local-space (world - spell.center)
            std::vector<glm::vec3> normals;
            std::vector<glm::vec3> colors;
        };
        void updateSpells(float dt);
        void cleanupExpiredSpells();
        void destroyPhysicsBodyForSpell(SpellEffect& spell);
        void forceCleanupSpellAnimatedVoxels(SpellEffect& s);
        void createPartialFormation(const SpellEffect& spell, float completionRatio);
        void createSpellFormation(const glm::vec3& center,
                                  const FormationParams& formationParams,
                                  float strength, uint64_t material,
                                  const glm::vec3& color, size_t collectedVoxels,
                                  uint8_t dominantType);
        void startSpellBurn(SpellEffect& s, float radius, float duration);
        bool isSpellTooSmall(const SpellEffect& s);
        bool isSpellTooSlowNow(const SpellEffect& s, float speedThreshold);
        float burn01(float t, float duration);
        void carveFormationWithSDF(const WorldPlanet& formation, uint64_t material,
                                   const FormationParams& params);
        void removeFormationVoxels(SpellEffect& spell);
        void rebuildDestructibleMeshIfNeeded(DestructibleObject& destruct);
        void initSpellDestructibleVolume(SpellEffect& spell);

        struct SphereMesh {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec3> normals;
            std::vector<uint32_t> indices;
            float radius;
        };
        std::unordered_map<int, SphereMesh> sphereMeshCache;

        void createPhysicsMeshData(PhysicsMeshData& mesh,
                                                const std::vector<glm::vec3>& vertices,
                                                const std::vector<glm::vec3>& normals,
                                                const std::vector<glm::vec3>& colors);

        void initSphereMeshCache();

        SphereMesh generateIcosphere(float radius, int subdivisions);

        static glm::vec3 calculateSphereDistribution(size_t index, size_t total, const FormationParams& params);
        static glm::vec3 calculatePlatformDistribution(size_t index, size_t total, const FormationParams& params);
        static glm::vec3 calculateWallDistribution(size_t index, size_t total, const FormationParams& params);
        static glm::vec3 calculateCubeDistribution(size_t index, size_t total, const FormationParams& params);
        static glm::vec3 calculateCylinderDistribution(size_t index, size_t total, const FormationParams& params);
        static glm::vec3 calculatePyramidDistribution(size_t index, size_t total, const FormationParams& params);

        static float haltonSequence(int index, int base);

        struct AsyncFormationRequest {
            glm::vec3 center;
            FormationParams params;
            float strength;
            uint64_t material;
            glm::vec3 color;
            size_t voxelCount;
            uint8_t dominantType;
            std::future<void> result;

            // Add move constructor
            AsyncFormationRequest(AsyncFormationRequest&& other) noexcept
                    : center(other.center)
                    , params(std::move(other.params))
                    , strength(other.strength)
                    , material(other.material)
                    , color(other.color)
                    , voxelCount(other.voxelCount)
                    , dominantType(other.dominantType)
                    , result(std::move(other.result))
            {}

            // Delete copy constructor
            AsyncFormationRequest(const AsyncFormationRequest&) = delete;
            AsyncFormationRequest& operator=(const AsyncFormationRequest&) = delete;

            // Default constructor
            AsyncFormationRequest() = default;
        };

        struct AsyncPhysicsRequest {
            size_t spellIndex;
            SpellEffect spellData;
            std::future<void> result;

            AsyncPhysicsRequest(AsyncPhysicsRequest&& other) noexcept
                    : spellIndex(other.spellIndex)
                    , spellData(std::move(other.spellData))
                    , result(std::move(other.result))
            {}

            AsyncPhysicsRequest(const AsyncPhysicsRequest&) = delete;
            AsyncPhysicsRequest& operator=(const AsyncPhysicsRequest&) = delete;

            AsyncPhysicsRequest() = default;
        };

        std::queue<AsyncFormationRequest> formationQueue;
        std::queue<AsyncPhysicsRequest> physicsQueue;
        std::mutex queueMutex;
        std::vector<std::thread> workerThreads;

        void processAsyncFormations();
        void processAsyncPhysics();
        void generateMeshAsync(Chunk* chunk);

        void processAnimatedVoxelsForSpell(SpellEffect &s, float dt);

        void processSingleSpell(SpellEffect &s, float dt, float speedThreshold, float timeToBurn, float burnDuration);

        void cleanupNonAnimatingVoxels();

        void queueAsyncPhysicsCreation(SpellEffect &s);

        void queueAsyncFormationCreation(const SpellEffect &s, float arrivalRatio);

        std::vector<SpellEffect> pendingPhysicsResults;
        std::mutex physicsResultMutex;

        void mergePhysicsBodyResult(const SpellEffect &result);

        void scheduleSpellRemoval(SpellEffect &effect);

        void createPhysicsBodyForSpell(SpellEffect &spell);

        GpuTrianglesReadback readbackTrianglesMainThread(const SpellEffect &spell);

        uint64_t nextSpellID = 1;

        std::atomic<bool> shuttingDown{false};
    };

}