#pragma once
#include <vector>
#include <cstdint>
#include "Enemy.h"

namespace gl3 {
    class EnemyManager;

    struct WaveConfig {
        uint32_t waveNumber = 0;
        uint32_t totalEnemies = 0;
        uint32_t maxConcurrentEnemies = 3;
        bool isBossWave = false;

        float minEnemyRadius = 2.0f;
        float maxEnemyRadius = 4.0f;

        // Boss settings
        float bossRadius = 8.0f;
        float bossHealth = 7000.0f;
    };

    class WaveManager {
    public:
        void init(EnemyManager* enemyMgr);
        void update(float dt, const glm::vec3& playerPos);
        void startNextWave();

        // UI data accessors
        uint32_t getCurrentWave() const { return currentWave; }
        uint32_t getEnemiesRemaining() const { return enemiesRemaining; }
        uint32_t getSpawnedEnemies() const { return enemiesSpawned;}

        bool isBossActive() const { return bossWaveActive && bossId != 0; }
        float getBossHealthPercent() const;

        // Get color lerp factor for UI (0.0 = normal, 1.0 = red/boss wave)
        float getWaveIntensity() const;

        bool isWaveActive() const { return waveActive; }
        bool isBossWave() const { return bossWaveActive; }

    private:
        void spawnEnemy();
        void spawnBoss();
        void checkWaveCompletion();
        glm::vec3 getRandomSpawnPosition(const glm::vec3& playerPos, float minDist, float maxDist);
        float getRandomEnemyRadius() const;

        EnemyManager* enemyManager = nullptr;

        // Wave state
        uint32_t currentWave = 0;
        uint32_t enemiesRemaining = 0;
        uint32_t enemiesSpawned = 0;
        uint32_t enemiesToSpawn = 0;
        bool waveActive = false;
        bool bossWaveActive = false;
        uint64_t bossId = 0;

        // Spawn timing
        float spawnTimer = 0.0f;
        float spawnInterval = 2.0f; // Time between spawns

        // Config
        WaveConfig config;

        // Constants
        static constexpr uint32_t BOSS_WAVE_INTERVAL = 3;
        static constexpr float MIN_SPAWN_DISTANCE = 50.0f*VOXEL_SIZE;
        static constexpr float MAX_SPAWN_DISTANCE = 150.0f*VOXEL_SIZE;

        glm::vec3 playerPos;
    };

}
