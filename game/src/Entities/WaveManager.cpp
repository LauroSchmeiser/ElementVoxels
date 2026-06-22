#include "WaveManager.h"
#include "EnemyManager.h"
#include <random>
#include <cmath>

namespace gl3 {

    static std::mt19937 rng(std::random_device{}());

    void WaveManager::init(EnemyManager* enemyMgr) {
        enemyManager = enemyMgr;
        currentWave = 0;
        waveActive = false;
        bossWaveActive = false;
        bossId = 0;
    }

    void WaveManager::update(float dt, const glm::vec3& playerPos) {
        if (!waveActive) return;

        this->playerPos=playerPos;

        // Check for dead enemies and update remaining count
        auto& allEnemies = enemyManager->all();
        uint32_t aliveCount = 0;

        for (auto& e : allEnemies) {
            if (e.inst.hp > 0.0f && !e.inst.pendingRemoval) {
                aliveCount++;

                // Track boss
                if (bossWaveActive && e.inst.id == bossId) {
                    // Boss is still alive
                }
            }
        }

        // If boss wave, check if boss is dead
        if (bossWaveActive && bossId != 0) {
            bool bossFound = false;
            for (auto& e : allEnemies) {
                if (e.inst.id == bossId && e.inst.hp > 0.0f) {
                    bossFound = true;
                    break;
                }
            }

            if (!bossFound) {
                // Boss defeated
                bossId = 0;
                enemiesRemaining = 0;
                checkWaveCompletion();
                return;
            }
        }

        // Update spawn timer if we need to spawn more enemies
        if (enemiesSpawned < enemiesToSpawn) {
            // Only spawn if we haven't reached the concurrent limit
            if (aliveCount < config.maxConcurrentEnemies) {
                spawnTimer += dt;

                if (spawnTimer >= spawnInterval) {
                    spawnEnemy();
                    spawnTimer = 0.0f;
                }
            }
        }

        // Update remaining count
        enemiesRemaining = (enemiesToSpawn - enemiesSpawned) + aliveCount;

        // Check if wave is complete
        if (enemiesSpawned >= enemiesToSpawn && aliveCount == 0) {
            checkWaveCompletion();
        }
    }

    void WaveManager::startNextWave() {
        currentWave++;

        waveActive = true;
        spawnTimer = 0.0f;
        enemiesSpawned = 0;
        bossId = 0;

        bossWaveActive = (currentWave % BOSS_WAVE_INTERVAL == 0);

        if (bossWaveActive) {
            config.waveNumber = currentWave;
            config.totalEnemies = 1;
            config.isBossWave = true;
            enemiesToSpawn = 1;

            spawnBoss();
        } else {
            config.waveNumber = currentWave;
            config.totalEnemies = 2 + (currentWave - 1)*2;
            config.isBossWave = false;
            enemiesToSpawn = config.totalEnemies;
        }

        enemiesRemaining = enemiesToSpawn;
    }

    void WaveManager::spawnEnemy() {
        if (!enemyManager) return;

        // Get player position (passed in update, but we'll use a dummy for now)
        glm::vec3 spawnPos = getRandomSpawnPosition(playerPos, MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE);

        static EnemyArchetype basic;
        basic.name = "Basic";
        basic.maxHP = 1000.0f;
        basic.moveSpeed = 30.0f;
        basic.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        basic.mass = 50.0f;
        basic.radius = 2.5f * VOXEL_SIZE;
        basic.cooldownsSec = { 4.0f, 0.0f, 0.0f };

        static EnemyArchetype dasher;
        dasher.name = "Dasher";
        dasher.maxHP = 500.0f;
        dasher.moveSpeed = 50.0f;
        dasher.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        dasher.mass = 10.0f;
        dasher.radius = 2.0f * VOXEL_SIZE;
        dasher.cooldownsSec = { 0.0f, 3.0f, 0.0f };


        enemyManager->spawn(basic, spawnPos);
        enemiesSpawned++;
    }

    void WaveManager::spawnBoss() {
        if (!enemyManager) return;

        glm::vec3 spawnPos = getRandomSpawnPosition(playerPos, MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE);

        EnemyArchetype bossArchetype;
        bossArchetype.name = "Boss";
        bossArchetype.maxHP = config.bossHealth;
        bossArchetype.moveSpeed = 40.0f;
        bossArchetype.radius = config.bossRadius * VOXEL_SIZE;
        bossArchetype.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        bossArchetype.mass = 50.0f;
        bossArchetype.cooldownsSec = { 8.0f, 0.0f, 0.0f };


        EnemyRuntime& boss = enemyManager->spawn(bossArchetype, spawnPos);
        bossId = boss.inst.id;
        enemiesSpawned++;
    }

    void WaveManager::checkWaveCompletion() {
        waveActive = false;
        bossWaveActive = false;
        bossId = 0;

        // Auto-start next wave for now (you can add a delay or manual start later)
        // For now, just set waveActive to false and let the game logic start the next wave
    }

    glm::vec3 WaveManager::getRandomSpawnPosition(const glm::vec3& playerPos, float minDist, float maxDist) {
        std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * glm::pi<float>());
        std::uniform_real_distribution<float> distDist(minDist, maxDist);

        float angle = angleDist(rng);
        float distance = distDist(rng);

        glm::vec3 offset(
                std::cos(angle) * distance,
                0.0f, // Spawn at player's Y level
                std::sin(angle) * distance
        );

        return playerPos + offset;
    }

    float WaveManager::getRandomEnemyRadius() const {
        std::uniform_real_distribution<float> radiusDist(config.minEnemyRadius, config.maxEnemyRadius);
        return radiusDist(rng);
    }

    float WaveManager::getBossHealthPercent() const {
        if (!bossWaveActive || bossId == 0 || !enemyManager) return 0.0f;

        auto& allEnemies = enemyManager->all();
        for (auto& e : allEnemies) {
            if (e.inst.id == bossId) {
                return glm::clamp(e.inst.hp / e.inst.type.maxHP, 0.0f, 1.0f);
            }
        }

        return 0.0f;
    }

    float WaveManager::getWaveIntensity() const {
        if (!waveActive) return 0.0f;

        // Calculate how close we are to the next boss wave
        uint32_t wavesSinceBoss = currentWave % BOSS_WAVE_INTERVAL;
        if (wavesSinceBoss == 0) wavesSinceBoss = BOSS_WAVE_INTERVAL;

        // Intensity increases as we approach boss wave
        // Wave 1: 0.0, Wave 2: 0.25, Wave 3: 0.5, Wave 4: 0.75, Wave 5: 1.0
        return static_cast<float>(wavesSinceBoss - 1) / static_cast<float>(BOSS_WAVE_INTERVAL - 1);
    }



} // namespace gl3
