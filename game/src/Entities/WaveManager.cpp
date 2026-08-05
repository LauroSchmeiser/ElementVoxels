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
        if((timeBetween>0&&timeElapsed>timeBetween)||objectiveCompleted)
        {
            startNextWaveMode();
        } else if(timeBetween>0)
        {
            timeElapsed += dt;
        }

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
        if(currentWaveMode==WaveMode::Hunt)
        {
            if(aliveCount==0&&enemiesSpawned>=enemiesToSpawn)
            {
                objectiveCompleted=true;
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

        if (enemiesToSpawn>enemiesSpawned&&currentWaveMode!=WaveMode::Preperation) {
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
        if (getRemainingBudget()<0 && aliveCount == 0) {
            checkWaveCompletion();
        }
    }

    void WaveManager::startNextWaveMode()
    {
        config.waveNumber++;
        currentWaveMode = nextWaveMode;
        timeElapsed = 0.0f;
        objectiveCompleted = false;
        nextWaveMode = WaveMode::UpgradeSelection;
        config.maxConcurrentEnemies = 5 + (config.waveNumber*2);
        config.enemyBudget= 3 +((config.waveNumber*2));

        config.currentBudget=0;
        spawnTimer = 0.0f;
        enemiesSpawned = 0;
        std::vector<uint64_t> enemyIds;

        for (auto& enemy : enemyManager->all())
        {
            enemyIds.push_back(enemy.inst.id);
        }

        for (uint64_t id : enemyIds)
        {
            enemyManager->destroyEnemy(id);
        }

        currentEnemies.clear();
        fillEnemyList(currentEnemies);
        enemiesToSpawn = static_cast<uint32_t>(currentEnemies.size());


        switch (currentWaveMode)
        {
            case WaveMode::Preperation:
            {
                setTimer(10.0f);

                static const std::vector<WaveMode> availableModes = {
                        WaveMode::Hunt,
                        WaveMode::Survival,
                        WaveMode::Mining,
                        WaveMode::Defense,
                        WaveMode::Destruction,
                        WaveMode::Infection
                };

                std::uniform_int_distribution<size_t> dist(
                        0,
                        availableModes.size() - 1
                );

                nextWaveMode = availableModes[dist(rng)];

                break;
            }

            case WaveMode::Hunt:
                setTimer(-1.0f);
                break;

            case WaveMode::Survival:
                setTimer(60.0f);
                config.maxConcurrentEnemies *=2;
                config.enemyBudget=500;
                break;

            case WaveMode::Mining:
                setTimer(1.0f); //TODO:: Change back to -1 after testing other Modes
                break;

            case WaveMode::Defense:
                setTimer(60.0f);
                break;

            case WaveMode::Destruction:
                setTimer(60.0f);
                break;

            case WaveMode::Infection:
                setTimer(60.0f);
                break;

            case WaveMode::UpgradeSelection:
                nextWaveMode = WaveMode::Preperation;
                setTimer(-1.0f);
                break;
        }
    }

    void WaveManager::startNextWave() {
        currentWave++;
        config.maxConcurrentEnemies+=glm::ceil(currentWave/2);

        config.currentBudget=0;
        waveActive = true;
        spawnTimer = 0.0f;
        enemiesSpawned = 0;
        bossId = 0;

        bossWaveActive = (currentWave % BOSS_WAVE_INTERVAL == 0);

        if (bossWaveActive) {
            config.waveNumber = currentWave;
            config.enemyBudget = 0;
            config.isBossWave = true;
            enemiesToSpawn = 0;

            spawnBoss();
            g_SoundManager.playMusic(SoundID::BossTheme, true, 1.0f);

        } else {
            config.waveNumber = currentWave;
            config.enemyBudget = 3 + (currentWave - 1)*3;
            config.enemyBaseHealth+=(currentWave)*50;
            config.isBossWave = false;
            currentEnemies.clear();

            fillEnemyList(currentEnemies);

            enemiesToSpawn = static_cast<uint32_t>(currentEnemies.size());

            std::cout << "Will be spawned: "
                      << enemiesToSpawn
                      << "enemies \n";

            g_SoundManager.playMusic(SoundID::BackgroundMusic, true, 1.0f);
        }

        enemiesRemaining = enemiesToSpawn;
    }

    void WaveManager::fillEnemyList(std::vector<EnemyArchetype>& enemies)
    {
        if (!enemyManager)
            return;

        static EnemyArchetype basic;
        basic.name = "Basic";
        basic.maxHP = config.enemyBaseHealth * 2;
        basic.moveSpeed = 10.0f;
        basic.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        basic.mass = 50.0f;
        basic.radius = 2.5f * VOXEL_SIZE;
        basic.cooldownsSec = {4.0f, 0.0f, 0.0f};
        basic.weight = 1;

        static EnemyArchetype dasher;
        dasher.name = "Dasher";
        dasher.maxHP = config.enemyBaseHealth;
        dasher.moveSpeed = 50.0f;
        dasher.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        dasher.mass = 10.0f;
        dasher.radius = 2.0f * VOXEL_SIZE;
        dasher.cooldownsSec = {0.0f, 3.0f, 0.0f};
        dasher.weight = 2;

        static EnemyArchetype consumer;
        consumer.name = "Consumer";
        consumer.maxHP = config.enemyBaseHealth * 3;
        consumer.moveSpeed = 30.0f;
        consumer.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        consumer.mass = 10.0f;
        consumer.radius = 4.0f * VOXEL_SIZE;
        consumer.cooldownsSec = {6.0f, 10.0f, 0.0f};
        consumer.weight = 3;

        static EnemyArchetype burrower;
        burrower.name = "Burrower";
        burrower.maxHP = config.enemyBaseHealth * 3;
        burrower.moveSpeed = 20.0f;
        burrower.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        burrower.mass = 10.0f;
        burrower.radius = 3.0f * VOXEL_SIZE;
        burrower.cooldownsSec = {0.0f, 6.0f, 0.0f};
        burrower.weight = 2;

        static EnemyArchetype water;
        water.name = "Water";
        water.maxHP = config.enemyBaseHealth * 3;
        water.moveSpeed = 20.0f;
        water.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        water.mass = 10.0f;
        water.radius = 3.0f * VOXEL_SIZE;
        water.cooldownsSec = {4.0f, 0.0f, 0.0f};
        water.weight = 2;

        std::vector<EnemyArchetype> enemyTypes;

        enemyTypes.push_back(basic);
        enemyTypes.push_back(dasher);

        if (currentWave > BOSS_WAVE_INTERVAL)
        {
            enemyTypes.push_back(consumer);
        }

        while (config.currentBudget < config.enemyBudget)
        {
            std::uniform_int_distribution<size_t> distEnemies(
                    0,
                    enemyTypes.size() - 1
            );

            size_t enemyPos = distEnemies(rng);

            const EnemyArchetype& selectedEnemy = enemyTypes[enemyPos];

            if (config.currentBudget + selectedEnemy.weight > config.enemyBudget)
            {
                continue;
            }

            config.currentBudget += selectedEnemy.weight;
            enemies.push_back(selectedEnemy);
        }
    }

    void WaveManager::spawnEnemy()
    {
        if (!enemyManager || currentEnemies.empty())
            return;

        glm::vec3 spawnPos = getRandomSpawnPosition(
                playerPos,
                MIN_SPAWN_DISTANCE,
                MAX_SPAWN_DISTANCE
        );

        EnemyArchetype enemy = currentEnemies.front();

        currentEnemies.erase(currentEnemies.begin());
        enemyManager->spawn(enemy, spawnPos);

        enemiesSpawned++;
    }

    void WaveManager::spawnBoss() {
        if (!enemyManager) return;

        glm::vec3 spawnPos = getRandomSpawnPosition(playerPos, MIN_SPAWN_DISTANCE, MAX_SPAWN_DISTANCE);

        EnemyArchetype bossArchetype;
        if(currentWave<2*BOSS_WAVE_INTERVAL)
        {
            bossArchetype.name = "Boss1";
        } else
        {
            bossArchetype.name = "Boss2";
        }
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

    const char* modeToString(WaveMode type) {
        switch (type) {
            case WaveMode::Hunt:
                return "Hunt";

            case WaveMode::Survival:
                return "Survival";

            case WaveMode::Mining:
                return "Mining";

            case WaveMode::Defense:
                return "Defense";

            case WaveMode::Destruction:
                return "Destruction";

            case WaveMode::Infection:
                return "Infection";

            case WaveMode::Preperation:
            default:
                return "Preparation";
        }
    }

        const char* modeToDescription(WaveMode type) {
            switch (type) {
                case WaveMode::Hunt:
                    return "Defeat all enemies to move on to the next wave!";

                case WaveMode::Survival:
                    return "Stay alive to move on to the next wave!";

                case WaveMode::Mining:
                    return "Fill the bar by mining enough of the specified material to move on to the next wave!";

                case WaveMode::Defense:
                    return "Defend the core to move on to the next wave, if it is destroyed, a catastrophe will emerge!";

                case WaveMode::Destruction:
                    return "Keep the enemies from finishing their construction, if they finish it, a catastrophe will emerge!";

                case WaveMode::Infection:
                    return "Keep the enemies from taking over the world by converting it to meat, if they convert the whole world, a catastrophe will emerge!";

                case WaveMode::Preperation:
                default:
                    return "Prepare for the next wave and level the playing field!";
            }
        }


} // namespace gl3
