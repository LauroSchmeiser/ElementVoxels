#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/Shader.h"
#include "rendering/marchingTables.h"
#include "rendering/SunBillboard.h"
#include "physics/SpellPhysicsManager.h"


namespace gl3 {

////-----Basics---------------------------------------------------------------------------------------------------------------------------------

    // CPU-side voxel format that matches the compute shader's Voxel { float density; vec4 color; }
    struct CpuVoxel {
        float density;
        // explicit padding so next member is at 16-byte offset
        float _pad0;
        float _pad1;
        float _pad2;
        glm::vec4 color;
    };
    static_assert(sizeof(CpuVoxel) == 32, "CpuVoxel size must be 32 bytes to match std430 layout");

    struct CpuTimer {
        const char* name;
        std::chrono::high_resolution_clock::time_point start;

        CpuTimer(const char* n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
        ~CpuTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << name << ": " << ms << " ms\n";
        }
    };


    void Game::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        if (height == 0) height = 1; // prevent divide-by-zero
        glViewport(0, 0, width, height);

        Game *game = static_cast<Game *>(glfwGetWindowUserPointer(window));
        if (game) {
            game->windowWidth = width;
            game->windowHeight = height;
        }
    }


    Game::Game(int width, int height, const std::string &title)
            : windowWidth(width),
              windowHeight(height),
              chunkManager(std::make_unique<MultiGridChunkManager>()),
              cameraPos(0.0f, 0.0f, 35.0f),
              cameraRotation(-90.0f, 0.0f),
              characterController(chunkManager.get(), 1.8f, 1.0f)  // Use .get() on unique_ptr
            {
        windowWidth = width;
        windowHeight = height;

        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize glfw");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("Failed to create window");
        }

        glfwMakeContextCurrent(window);

        gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
        std::cout << "GL_VERSION: " << (const char *) glGetString(GL_VERSION) << std::endl;
        std::cout << "GLSL_VERSION: " << (const char *) glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;


        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        skyboxShader = std::make_unique<Shader>("shaders/skybox.vert", "shaders/skybox.frag");
        voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");


        try {
            voxelSplatShader = std::make_unique<Shader>("shaders/metaball_splat.comp");
        } catch (std::exception &e) {
            std::cerr << "Failed to create metaballSplatShader: " << e.what() << std::endl;
            // optionally abort here
        }

        audio.init();
        audio.setGlobalVolume(0.1f);
        voxelPhysics = std::make_unique<VoxelPhysicsManager>(chunkManager.get());
                // Set collision callback for spell bodies
                voxelPhysics->setCollisionCallback(
                        [this](gl3::VoxelPhysicsBody* body,
                               const glm::vec3& hitPos,
                               const glm::vec3& hitNormal,
                               float impactSpeed) {
                            // Handle spell collision
                            SpellEffect* spell = static_cast<SpellEffect*>(body->userData);
                            if (spell) {
                                onSpellCollision(spell, hitPos, hitNormal, impactSpeed);
                            }
                        }
                );

            }



    Game::~Game() {
        glfwTerminate();
    }


////-----Run-Method-----------------------------------------------------------------------------------------------------------------------------

    void Game::run() {
        ////Initialization-Steps
        setupSkybox();
        CpuTimer t0("ssbos");
        setupSSBOsAndTables();
        setupInput();
        CpuTimer t1("generateChunks");
        generateChunks();
        setupCamera();
        setupVEffects();

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear both at once

            ////Simulation Steps
            updateDeltaTime();
            if(DebugMode1) {
                CpuTimer t7("updateSpells");
                updateSpells(deltaTime);
            } else
            {
                updateSpells(deltaTime);
            }

            ////Input-Steps
            //handleCameraInput();
            glfwPollEvents();
            updatePhysics();
            update();


            ////Post-Prod Steps?

            ////Rendering Steps
            if(!DebugMode1)
            {
                renderSkybox();
            }

            if(DebugMode1)
            {
                CpuTimer t2("renderChunks");
                renderChunks();
            } else
            {
                renderChunks();
            }
            if(DebugMode1) {
                CpuTimer t7("renderAnimatedVoxels");
                renderAnimatedVoxels();
            } else
            {
                renderAnimatedVoxels();
            }

            if(DebugMode1) {
                CpuTimer t10("renderPhysicsBodies");
                renderPhysicsFormations();
            } else
            {
                renderPhysicsFormations();
            }

            ////UI
            DisplayFPSCount();

            ////SwapBuffer
            glfwSwapBuffers(window);
        }
    }


////----Chunk Management Code-------------------------------------------------------------------------------------------------------------------

    bool Game::hasSolidVoxels(const gl3::Chunk &chunk) {
        for (int x = 0; x <= CHUNK_SIZE; ++x)
            for (int y = 0; y <= CHUNK_SIZE; ++y)
                for (int z = 0; z <= CHUNK_SIZE; ++z)
                    if (chunk.voxels[x][y][z].isSolid())
                        return true;
        return false;
    }

    int Game::worldToChunk(float worldPos){
        // Each chunk covers CHUNK_SIZE voxels, each voxel is VOXEL_SIZE world units.
        // So chunk world width = CHUNK_SIZE * VOXEL_SIZE.
        float chunkWorldSize = CHUNK_SIZE * gl3::VOXEL_SIZE;
        return (int) std::floor(worldPos / chunkWorldSize);
    }

    glm::vec3 Game::getChunkMin(const ChunkCoord& coord) const {
        // chunk origin in world units
        return glm::vec3(coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE);
    }

    glm::vec3 Game::getChunkMax(const ChunkCoord& coord) const {
        return glm::vec3((coord.x + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         (coord.y + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE,
                         (coord.z + 1) * CHUNK_SIZE * gl3::VOXEL_SIZE);
    }



    void Game::markChunkModified(const ChunkCoord &coord) {
        Chunk *chunk = chunkManager->getChunk(coord);
        if (chunk) {
            chunk->meshDirty = true;

            // Also mark neighboring chunks because Marching Cubes needs padding
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        ChunkCoord neighbor{coord.x + dx, coord.y + dy, coord.z + dz};
                        Chunk *neighborChunk = chunkManager->getChunk(neighbor);
                        if (neighborChunk) {
                            neighborChunk->meshDirty = true;
                            neighborChunk->lightingDirty=true;
                        }
                    }
                }
            }
        }
    }

    void Game::unloadChunk(const ChunkCoord &coord) {
        Chunk *chunk = chunkManager->getChunk(coord);
        if (chunk) {
            // GPU resources get cleaned up in chunk destructor or clear() method
            chunk->clear(); // This deletes VAO/VBO
        }
    }

    // Check if chunk is visible (combination of distance and frustum culling)
    bool Game::isChunkVisible(const ChunkCoord& coord) const {
        // Quick distance check first (cheaper)
        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);

        int dx = coord.x - camCX;
        int dy = coord.y - camCY;
        int dz = coord.z - camCZ;

        // Distance culling
        if (dx * dx + dy * dy + dz * dz > RenderingRange * RenderingRange) {
            return false;
        }

        // Frustum culling
        glm::vec3 chunkMin = getChunkMin(coord);
        glm::vec3 chunkMax = getChunkMax(coord);

        return currentFrustum.isAABBVisible(chunkMin, chunkMax);
    }


////----Spell-System-Code-----------------------------------------------------------------------------------------------------------------------


    void Game::castSpellWithFormation(const glm::vec3& center, float radius,
                                      uint64_t targetMaterial, float strength,
                                      const FormationParams& baseFormationParams) {
        // Create spell effect
        SpellEffect spell;
        spell.type = SpellEffect::Type::CONSTRUCT;
        spell.center = center;
        spell.radius = radius;
        spell.strength = strength;
        spell.targetMaterial = targetMaterial;
        spell.dominantType = 0;  // Will be set by findNearbyVoxelsForVisual
        spell.formationParams = baseFormationParams; // Store the initial params

        // Step 1: Find voxels
        std::vector<AnimatedVoxel> visualVoxels;
        findNearbyVoxelsForVisual(center, radius, targetMaterial,
                                  visualVoxels, strength, spell.dominantType);

        if (visualVoxels.empty()) {
            std::cout << "No voxels found for spell\n";
            return;
        }

        // Step 2: Calculate formation properties
        glm::vec3 avgColor(0.0f);

        // --- NEW: Compute optimal formation size based on collected voxels ---
        const size_t collected = visualVoxels.size();
        // Convert voxel count -> world-volume (each voxel is a cube of VOXEL_SIZE^3)
        const float voxelVolumeWorld = gl3::VOXEL_SIZE * gl3::VOXEL_SIZE * gl3::VOXEL_SIZE;
        float desiredVolumeWorld = static_cast<float>(collected) * voxelVolumeWorld;

        // Adjust formation parameters based on collected volume (in world^3)
        FormationParams adjustedFormation = baseFormationParams;
        adjustFormationForVolume(adjustedFormation, desiredVolumeWorld);
        adjustedFormation.center = center;
        spell.formationParams = adjustedFormation; // Store adjusted params
        // ----------------------------------------------------------------

        // Assign targets and set up AnimatedVoxel entries
        for (size_t i = 0; i < visualVoxels.size(); ++i) {
            AnimatedVoxel &v = visualVoxels[i];
            avgColor += v.color;

            // Calculate target position on formation surface
            v.targetPos = calculateFormationTarget(i, visualVoxels.size(),
                                                   adjustedFormation);

            v.animationSpeed = strength * 3.5f;
            v.isAnimating = true;
            v.hasArrived = false;
        }
        avgColor /= (float)visualVoxels.size();

        spell.formationColor = avgColor;
        spell.formationRadius = adjustedFormation.getBoundingRadius();

        // Step 3: Add visual voxels to global list and record their stable IDs in the spell
        for (auto &v : visualVoxels) {
            v.id = nextAnimatedVoxelID++;
            animatedVoxels.push_back(v);
            animatedVoxelIndexMap[v.id] = animatedVoxels.size() - 1;
            spell.animatedVoxelIDs.push_back(v.id);
        }

        // Step 4: Add spell to active list
        activeSpells.push_back(spell);

        std::cout << "Spell cast! " << visualVoxels.size()
                  << " voxels moving to formation (type="
                  << static_cast<int>(adjustedFormation.type) << ")\n";
    }

    // Helper function to adjust formation size based on collected volume
    void Game::adjustFormationForVolume(FormationParams& params, float volume /*world^3*/) {
        const float packingEfficiency = 0.7f;
        constexpr float PI = 3.14159265358979323846f;

        const float minWorldDim = gl3::VOXEL_SIZE * 0.15f; // don't shrink below half a voxel
        const float maxScaleFactor = 20.0f; // safety cap if you want

        switch(params.type) {
            case FormationType::SPHERE: {
                float computedRadius = std::cbrt((3.0f / (4.0f * PI)) * (volume / packingEfficiency));
                float maxRadius = std::max(minWorldDim, params.radius * 0.75f);
                float minRadius = minWorldDim;
                params.radius = glm::clamp(computedRadius, minRadius, maxRadius);
                break;
            }
            case FormationType::PLATFORM: {
                // Keep thickness (sizeY) constant (world units), adjust width/depth (world units)
                float area = volume / (params.sizeY * packingEfficiency);
                float side = std::sqrt(glm::max(0.0f, area));
                params.sizeX = glm::max(side, minWorldDim);
                params.sizeZ = glm::max(side, minWorldDim);
                break;
            }
            case FormationType::WALL: {
                // Keep thickness (sizeZ) constant (world units), adjust width/height (world units)
                float area = volume / (params.sizeZ * packingEfficiency);
                float side = std::sqrt(glm::max(0.0f, area));
                params.sizeX = glm::max(side, minWorldDim);
                params.sizeY = glm::max(params.sizeX * 0.75f, minWorldDim);
                break;
            }
            case FormationType::CUBE: {
                float side = std::cbrt(glm::max(0.0f, volume / packingEfficiency));
                params.sizeX = params.sizeY = params.sizeZ = glm::max(side, minWorldDim);
                break;
            }
            case FormationType::CYLINDER: {
                float computedRadius = std::cbrt((2.0f / (3.0f * PI)) * (volume / packingEfficiency));
                params.radius = glm::max(computedRadius, minWorldDim);
                params.sizeY = glm::max(params.radius * 2.0f, minWorldDim);
                break;
            }
            case FormationType::PYRAMID: {
                float side = std::cbrt((3.0f * volume) / (packingEfficiency * params.sizeY + 1e-6f));
                params.sizeX = glm::max(side, minWorldDim);
                params.sizeZ = glm::max(side, minWorldDim);
                break;
            }
            default:
                break;
        }
    }

// Calculate target position on formation surface
    glm::vec3 Game::calculateFormationTarget(size_t index, size_t total,
                                             const FormationParams& params) {
        // Use different distribution strategies based on formation type
        switch(params.type) {
            case FormationType::SPHERE:
                return calculateSphereDistribution(index, total, params);
            case FormationType::PLATFORM:
                return calculatePlatformDistribution(index, total, params);
            case FormationType::WALL:
                return calculateWallDistribution(index, total, params);
            case FormationType::CUBE:
                return calculateCubeDistribution(index, total, params);
            case FormationType::CYLINDER:
                return calculateCylinderDistribution(index, total, params);
            case FormationType::PYRAMID:
                return calculatePyramidDistribution(index, total, params);
            default:
                // For custom, default to sphere distribution
                return calculateSphereDistribution(index, total, params);
        }
    }

    glm::vec3 Game::calculateSphereDistribution(size_t index, size_t total,
                                                const FormationParams& params) {
        // Fibonacci sphere distribution (even distribution on sphere)
        float goldenAngle = glm::pi<float>() * (3.0f - glm::sqrt(5.0f));
        float y = 1.0f - (static_cast<float>(index) / (static_cast<float>(total) - 1.0f)) * 2.0f;
        float radiusAtY = std::sqrt(1.0f - y * y);
        float theta = goldenAngle * static_cast<float>(index);

        float x = std::cos(theta) * radiusAtY;
        float z = std::sin(theta) * radiusAtY;

        glm::vec3 localPos(x, y, z);
        glm::vec3 worldPos = params.center + localPos * params.radius;
        return worldPos;
    }

    glm::vec3 Game::calculatePlatformDistribution(size_t index, size_t total,
                                                  const FormationParams& params) {
        // Distribute evenly across platform surface (top face)
        // Use Halton sequence for better distribution
        float u = haltonSequence(index, 2) - 0.5f; // Center around 0
        float v = haltonSequence(index, 3) - 0.5f;

        // Generate points on platform surface (top face)
        glm::vec3 localPos(
                u * params.sizeX,
                params.sizeY * 0.5f, // Top surface
                v * params.sizeZ
        );

        // Apply orientation
        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateWallDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        // Distribute evenly across wall surface (front face)
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        // Generate points on wall surface (front face)
        glm::vec3 localPos(
                u * params.sizeX,
                v * params.sizeY,
                params.sizeZ * 0.5f // Front surface
        );

        // Apply orientation
        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateCubeDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        // Distribute on cube surface using face assignment
        int faceIndex = index % 6;
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos;
        switch(faceIndex) {
            case 0: // +X
                localPos = glm::vec3(params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 1: // -X
                localPos = glm::vec3(-params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 2: // +Y
                localPos = glm::vec3(u * params.sizeX, params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 3: // -Y
                localPos = glm::vec3(u * params.sizeX, -params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 4: // +Z
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, params.sizeZ * 0.5f);
                break;
            case 5: // -Z
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, -params.sizeZ * 0.5f);
                break;
        }

        glm::vec3 worldPos = params.center + localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateCylinderDistribution(size_t index, size_t total,
                                                  const FormationParams& params) {
        // Distribute on cylinder surface (side)
        float angle = static_cast<float>(index) / static_cast<float>(total) * glm::two_pi<float>();
        float height = haltonSequence(index, 2) - 0.5f;

        glm::vec3 localPos(
                std::cos(angle) * params.radius,
                height * params.sizeY,
                std::sin(angle) * params.radius
        );

        glm::vec3 worldPos = params.center + localPos;
        return worldPos;
    }

    glm::vec3 Game::calculatePyramidDistribution(size_t index, size_t total,
                                                 const FormationParams& params) {
        // Distribute on pyramid surfaces
        int surface = index % 5; // 4 sides + base

        if (surface < 4) { // Side faces
            float u = haltonSequence(index, 2);
            float v = haltonSequence(index, 3);

            // Calculate point on triangular side face
            float baseX = (u - 0.5f) * params.sizeX;
            float baseZ = (v - 0.5f) * params.sizeZ;

            // Project onto appropriate side based on surface index
            glm::vec3 localPos;
            switch(surface) {
                case 0: // +X side
                    localPos = glm::vec3(baseX, 0.0f, params.sizeZ * 0.5f);
                    break;
                case 1: // -X side
                    localPos = glm::vec3(baseX, 0.0f, -params.sizeZ * 0.5f);
                    break;
                case 2: // +Z side
                    localPos = glm::vec3(params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
                case 3: // -Z side
                    localPos = glm::vec3(-params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
            }

            // Move up to pyramid surface
            float heightRatio = haltonSequence(index, 5);
            localPos.y = heightRatio * params.sizeY;
            localPos.x *= (1.0f - heightRatio);
            localPos.z *= (1.0f - heightRatio);

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        } else { // Base
            float u = haltonSequence(index, 2) - 0.5f;
            float v = haltonSequence(index, 3) - 0.5f;

            glm::vec3 localPos(
                    u * params.sizeX,
                    0.0f, // Bottom
                    v * params.sizeZ
            );

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        }
    }

// Helper function for better distribution (Halton sequence)
    float Game::haltonSequence(int index, int base) {
        float result = 0.0f;
        float f = 1.0f;
        int i = index;

        while (i > 0) {
            f /= static_cast<float>(base);
            result += f * static_cast<float>(i % base);
            i = i / base;
        }

        return result;
    }

    void Game::findNearbyVoxelsForVisual(const glm::vec3& center, float radius,
                                         uint64_t targetMaterial,
                                         std::vector<AnimatedVoxel>& results,
                                         float strength,
                                         uint8_t& outDominantType) {  // ADD OUTPUT PARAMETER
        const float radiusSq = radius * radius;
        int maxVoxels = static_cast<int>(strength * 75.0f);

        // Use array for type counting
        int typeCounts[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        auto chunks = chunkManager->getChunksInRadius(center, radius);

        // Use a more efficient candidate collection
        struct VoxelCandidate {
            glm::vec3 worldPos;
            glm::vec3 color;
            ChunkCoord chunkCoord;
            glm::ivec3 localPos;
            float distanceSq;
            Chunk* chunk;
            uint8_t type;  // Store type here
        };

        std::vector<VoxelCandidate> candidates;

        // Pre-allocate to avoid reallocations
        candidates.reserve(maxVoxels * 2);

        // Collect candidates with early pruning
        for (const auto& [coord, chunk] : chunks) {
            if (!chunk || !hasSolidVoxels(*chunk)) continue;

            glm::vec3 chunkMin = getChunkMin(coord);
            glm::vec3 chunkCenter = chunkMin + glm::vec3(CHUNK_SIZE/2) * VOXEL_SIZE;

            // Quick sphere-AABB test
            float distToChunkCenter = glm::distance(chunkCenter, center);
            float maxChunkDist = std::sqrt(3.0f) * (CHUNK_SIZE * VOXEL_SIZE / 2.0f);

            if (distToChunkCenter > radius + maxChunkDist) {
                continue;  // Chunk is definitely outside spell radius
            }

            // Get bounds for this chunk relative to spell center
            glm::vec3 chunkRelMin = chunkMin - center;
            glm::vec3 chunkRelMax = chunkRelMin + glm::vec3(CHUNK_SIZE) * VOXEL_SIZE;

            // Early test: check if chunk bounding sphere intersects spell sphere
            float chunkRadius = glm::length(chunkRelMax - chunkRelMin) / 2.0f;
            glm::vec3 chunkCenterRel = (chunkRelMin + chunkRelMax) / 2.0f;
            float centerDist = glm::length(chunkCenterRel);

            if (centerDist > radius + chunkRadius) {
                continue;  // No intersection
            }

            // Only iterate through potentially intersecting region
            int startX = std::max(0, static_cast<int>((center.x - radius - chunkMin.x) / VOXEL_SIZE));
            int endX = std::min(CHUNK_SIZE, static_cast<int>((center.x + radius - chunkMin.x) / VOXEL_SIZE) + 1);
            int startY = std::max(0, static_cast<int>((center.y - radius - chunkMin.y) / VOXEL_SIZE));
            int endY = std::min(CHUNK_SIZE, static_cast<int>((center.y + radius - chunkMin.y) / VOXEL_SIZE) + 1);
            int startZ = std::max(0, static_cast<int>((center.z - radius - chunkMin.z) / VOXEL_SIZE));
            int endZ = std::min(CHUNK_SIZE, static_cast<int>((center.z + radius - chunkMin.z) / VOXEL_SIZE) + 1);

            for (int x = startX; x <= endX; ++x) {
                for (int y = startY; y <= endY; ++y) {
                    for (int z = startZ; z <= endZ; ++z) {
                        const Voxel& voxel = chunk->voxels[x][y][z];

                        if (voxel.isSolid() && voxel.material == targetMaterial) {
                            glm::vec3 worldPos = chunkMin + glm::vec3((float)x, (float)y, (float)z) * VOXEL_SIZE;
                            glm::vec3 diff = worldPos - center;
                            float distSq = glm::dot(diff, diff);

                            if (distSq <= radiusSq) {
                                candidates.push_back({
                                                             worldPos,
                                                             voxel.color,
                                                             coord,
                                                             {x, y, z},
                                                             distSq,
                                                             chunk,
                                                             voxel.type
                                                     });

                                // Count type immediately
                                if (voxel.type < 8) {
                                    typeCounts[voxel.type]++;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by distance
        if (candidates.size() > static_cast<size_t>(maxVoxels)) {
            std::nth_element(candidates.begin(),
                             candidates.begin() + maxVoxels,
                             candidates.end(),
                             [](const VoxelCandidate& a, const VoxelCandidate& b) {
                                 return a.distanceSq < b.distanceSq;
                             });

            // Only keep closest maxVoxels
            candidates.resize(maxVoxels);
        }

        // Determine dominant type from collected candidates
        int maxCount = 0;
        uint8_t dominantType = 1;  // Default to type 1
        for (int i = 0; i < 8; ++i) {
            if (typeCounts[i] > maxCount) {
                maxCount = typeCounts[i];
                dominantType = static_cast<uint8_t>(i);
            }
        }
        outDominantType = dominantType;  // Set output parameter

        // Process candidates (same as before, but now we have types counted)
        for (const auto& candidate : candidates) {
            AnimatedVoxel animVoxel;
            animVoxel.currentPos = candidate.worldPos;
            animVoxel.isAnimating = true;
            animVoxel.animationSpeed = strength * 1.5f;
            animVoxel.hasArrived = false;

            // --- NEW: approximate lit color at the voxel's position (cheap CPU shading) ---
            // Start with base/albedo color from the voxel
            glm::vec3 baseColor = candidate.color;
            glm::vec3 litColor = glm::vec3(0.0f);

            // 1) Ambient term (global low ambient)
            const float ambientStrength = 0.8f;
            litColor += baseColor * ambientStrength;

            // 2) Directional "sun" (cheap approximation so things look shaded)
            // You can tweak this direction and intensity to match your world lighting.
            glm::vec3 sunDir = glm::normalize(glm::vec3(-0.6f, -0.7f, -0.3f)); // direction TOWARDS light source
            glm::vec3 normal = calculateNormalAt(candidate.chunk, candidate.localPos);
            float ndotl = glm::max(glm::dot(normal, sunDir), 0.0f);
            const glm::vec3 sunColor = glm::vec3(1.0f, 0.98f, 0.92f);
            const float sunIntensity = 0.9f;
            litColor += baseColor * (sunColor * sunIntensity * ndotl);

            // 3) Add cheap contribution from emissive lights (use merged pool if available).
            // mergedEmissiveLightPool is updated periodically (every ~30 frames), stale data is OK.
            // Limit how many lights we accumulate for perf; usually this pool is small.
            const int MAX_POINT_LIGHTS = 4;
            int usedLights = 0;
            for (const auto &L : mergedEmissiveLightPool) {
                if (usedLights >= MAX_POINT_LIGHTS) break;
                glm::vec3 toLight = L.pos - candidate.worldPos;
                float distSq = glm::dot(toLight, toLight);
                if (distSq < 1e-6f) distSq = 1e-6f;
                float att = L.intensity / (distSq + 1.0f); // simple attenuation
                glm::vec3 Ldir = glm::normalize(toLight);
                float nDotL = glm::max(glm::dot(normal, Ldir), 0.0f);
                if (nDotL <= 0.0f) continue;
                // scale factor tuned so point lights don't blow out color
                const float POINT_LIGHT_SCALE = 0.025f;
                litColor += baseColor * (L.color * att * nDotL * POINT_LIGHT_SCALE);
                ++usedLights;
            }

            // 4) Fallback: if no merged lights and chunk-local emissive lights exist, use them
            if (usedLights == 0 && !candidate.chunk->emissiveLights.empty()) {
                for (const auto &L : candidate.chunk->emissiveLights) {
                    glm::vec3 toLight = L.pos - candidate.worldPos;
                    float distSq = glm::dot(toLight, toLight);
                    if (distSq < 1e-6f) distSq = 1e-6f;
                    float att = L.intensity / (distSq + 1.0f);
                    glm::vec3 Ldir = glm::normalize(toLight);
                    float nDotL = glm::max(glm::dot(normal, Ldir), 0.0f);
                    const float POINT_LIGHT_SCALE = 0.03f;
                    litColor += baseColor * (L.color * att * nDotL * POINT_LIGHT_SCALE);
                    ++usedLights;
                    if (usedLights >= MAX_POINT_LIGHTS) break;
                }
            }

            // 5) Clamp to reasonable range and ensure color remains in [0,1]
            litColor = glm::clamp(litColor, glm::vec3(0.0f), glm::vec3(1.0f));

            // Save the lit color into the animated voxel so the renderer can just use it directly.
            animVoxel.color = litColor;
            // --- END lit color calculation ---

            animVoxel.normal = calculateNormalAt(candidate.chunk, candidate.localPos);

            // Create crater
            createExteriorSmoothCrater(candidate.chunk, candidate.localPos, candidate.worldPos);

            // Mark chunk as modified
            candidate.chunk->meshDirty = true;
            candidate.chunk->lightingDirty = true;
            markChunkModified(candidate.chunkCoord);

            results.push_back(animVoxel);
        }

        // Debug output
        if (!candidates.empty()) {
            processEmissiveChunks();  // Consider calling this less frequently
            float minDist = std::sqrt(candidates.front().distanceSq);
            float maxDist = std::sqrt(candidates.back().distanceSq);
            std::cout << "Taking " << candidates.size() << " closest voxels "
                      << "(dominant type=" << (int)dominantType << ", "
                      << "distances: " << minDist << " to " << maxDist << " units)\n";
        }
    }


        void Game::createExteriorSmoothCrater(Chunk* chunk, const glm::ivec3& voxelPos,
                                              const glm::vec3& worldPos) {
            float craterRadius = 2.0f * gl3::VOXEL_SIZE; // scale crater size by voxel size
            float maxCraterDepth = 2.5f; // this is a density amount, leave unless you need to tune

            int range = static_cast<int>(std::ceil(craterRadius / gl3::VOXEL_SIZE));

            for (int dx = -range; dx <= range; ++dx) {
                for (int dy = -range; dy <= range; ++dy) {
                    for (int dz = -range; dz <= range; ++dz) {
                        int nx = voxelPos.x + dx;
                        int ny = voxelPos.y + dy;
                        int nz = voxelPos.z + dz;

                        if (nx < 0 || nx > CHUNK_SIZE || ny < 0 || ny > CHUNK_SIZE || nz < 0 || nz > CHUNK_SIZE) {
                            continue;
                        }

                        glm::vec3 offset = glm::vec3((float)dx, (float)dy, (float)dz) * gl3::VOXEL_SIZE;
                        float dist = glm::length(offset); // world distance
                        if (dist <= craterRadius) {
                            float t = dist / craterRadius;
                            float originalDensity = chunk->voxels[nx][ny][nz].density;

                            // KEY INSIGHT: For Marching Cubes to show a solid exterior,
                            // the density should be POSITIVE at the surface (0 isosurface)

                            if (originalDensity >= 0.0f) {
                                // This voxel was inside or at the surface of the planet

                                // Create crater shape that:
                                // 1. Reduces density near center
                                // 2. Keeps exterior surface positive
                                // 3. Creates smooth transition

                                float craterShape = (1.0f - t * t); // Bowl shape
                                float densityReduction = maxCraterDepth * craterShape;

                                // Apply reduction, but ensure we don't go too negative
                                float newDensity = originalDensity - densityReduction;

                                // If this was deep inside (high positive), keep it positive
                                if (originalDensity > 2.0f) {
                                    newDensity = std::max(newDensity, 0.1f);
                                }

                                chunk->voxels[nx][ny][nz].density = newDensity;

                                // Update type - critical for Marching Cubes
                                if (newDensity < -0.5f) {
                                    // Too negative = air (creates hole)
                                    chunk->voxels[nx][ny][nz].type = 0;
                                } else if (newDensity < 0.0f) {
                                    // Negative but close to 0 = surface
                                    chunk->voxels[nx][ny][nz].type = 1;
                                } else {
                                    // Positive = solid interior
                                    chunk->voxels[nx][ny][nz].type = 1;
                                }
                            }
                        }
                    }
                }
            }
        }

    void Game::createSpellFormation(const glm::vec3& center,
                                    const FormationParams& formationParams,
                                    float strength, uint64_t material,
                                    const glm::vec3& color, size_t collectedVoxels,
                                    uint8_t dominantType) {

        WorldPlanet newFormation;
        newFormation.worldPos = center;
        newFormation.color = color;
        newFormation.type = dominantType;

        float effectiveRadius = formationParams.getBoundingRadius();
        newFormation.radius = effectiveRadius;

        float preloadRadius = effectiveRadius;

        float chunkWorldSize = gl3::CHUNK_SIZE * gl3::VOXEL_SIZE;
        int minCX = worldToChunk(center.x - preloadRadius);
        int maxCX = worldToChunk(center.x + preloadRadius);
        int minCY = worldToChunk(center.y - preloadRadius);
        int maxCY = worldToChunk(center.y + preloadRadius);
        int minCZ = worldToChunk(center.z - preloadRadius);
        int maxCZ = worldToChunk(center.z + preloadRadius);

        std::cout << "createSpellFormation: effectiveRadius=" << effectiveRadius
                  << " (chunkWorldSize=" << chunkWorldSize << ")"
                  << " chunksX=[" << minCX << "," << maxCX << "]"
                  << " chunksY=[" << minCY << "," << maxCY << "]"
                  << " chunksZ=[" << minCZ << "," << maxCZ << "]"
                  << " center=(" << center.x << "," << center.y << "," << center.z << ")\n";

        // Load all chunks first (force create if missing)
        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};

                    if (!chunkManager->getChunk(coord)) {
                        chunkManager->addChunk(coord, VoxelCategory::DYNAMIC);
                        Chunk* chunk = chunkManager->getChunk(coord);
                        if (chunk) {
                            chunk->coord = coord;
                            chunk->clear();
                        }
                    }
                }
            }
        }

        FormationParams paramsCopy = formationParams;
        paramsCopy.center = center;

        carveFormationWithSDF(newFormation, material, paramsCopy);

        // Force immediate mesh regeneration
        int regenMinCX = worldToChunk(center.x - effectiveRadius);
        int regenMaxCX = worldToChunk(center.x + effectiveRadius);
        int regenMinCY = worldToChunk(center.y - effectiveRadius);
        int regenMaxCY = worldToChunk(center.y + effectiveRadius);
        int regenMinCZ = worldToChunk(center.z - effectiveRadius);
        int regenMaxCZ = worldToChunk(center.z + effectiveRadius);

        for (int cx = regenMinCX; cx <= regenMaxCX; ++cx) {
            for (int cy = regenMinCY; cy <= regenMaxCY; ++cy) {
                for (int cz = regenMinCZ; cz <= regenMaxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (chunk && chunk->meshDirty) {
                        generateChunkMesh(chunk);
                    }
                }
            }
        }

        std::cout << "createSpellFormation: collected=" << collectedVoxels
                  << " => formation type=" << static_cast<int>(formationParams.type)
                  << ", radius=" << effectiveRadius << "\n";
    }

    void Game::carveFormationWithSDF(const WorldPlanet& formation, uint64_t material,
                                     const FormationParams& params) {
        float boundingRadius = params.getBoundingRadius();
        std::cout << "carveFormationWithSDF: boundingRadius=" << boundingRadius
                  << " center=(" << formation.worldPos.x << "," << formation.worldPos.y << "," << formation.worldPos.z << ")\n";
// sample a couple of points
        float centerVal = params.evaluate(formation.worldPos);
        float atRadiusVal = params.evaluate(formation.worldPos + glm::vec3(boundingRadius,0,0));
        std::cout << " SDF(center)=" << centerVal << " SDF(center+radius)=" << atRadiusVal << "\n";


        // Determine which chunks this formation affects
        int minCX = worldToChunk(formation.worldPos.x - boundingRadius);
        int maxCX = worldToChunk(formation.worldPos.x + boundingRadius);
        int minCY = worldToChunk(formation.worldPos.y - boundingRadius);
        int maxCY = worldToChunk(formation.worldPos.y + boundingRadius);
        int minCZ = worldToChunk(formation.worldPos.z - boundingRadius);
        int maxCZ = worldToChunk(formation.worldPos.z + boundingRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};

                    // Get chunk - should exist after pre-loading
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) {
                        // Emergency fallback
                        chunkManager->addChunk(coord, VoxelCategory::DYNAMIC);
                        chunk = chunkManager->getChunk(coord);
                        if (!chunk) continue;

                        if (chunk->coord != coord) { // New chunk
                            chunk->coord = coord;
                            chunk->clear();
                        }
                    }

                    glm::vec3 chunkOrigin = getChunkMin(coord);
                    bool chunkTouched = false;

                    // Carve formation into chunk using its SDF
                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 worldPos = chunkOrigin +
                                                     glm::vec3((float)lx, (float)ly, (float)lz) * gl3::VOXEL_SIZE;

                                // Use formation SDF instead of sphere distance
                                float formationDensity = params.evaluate(worldPos);

                                float existingDensity = chunk->voxels[lx][ly][lz].density;

                                // SDF UNION: Take the MAXIMUM density (preserves existing terrain)
                                if (formationDensity > existingDensity) {
                                    chunk->voxels[lx][ly][lz].density = formationDensity;

                                    if (formationDensity >= -1.0f) {
                                        chunk->voxels[lx][ly][lz].type = formation.type;
                                        chunk->voxels[lx][ly][lz].color = formation.color;
                                        chunk->voxels[lx][ly][lz].material = material;

                                        if (formationDensity >= 0) {
                                            chunkTouched = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (chunkTouched) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        // Force mark as modified
                        markChunkModified(coord);
                    }
                }
            }
        }
        processEmissiveChunks();
    }

    void Game::updateSpells(float dt) {

        // Update each active spell
        for (auto spellIt = activeSpells.begin(); spellIt != activeSpells.end(); ) {
            if (spellIt->lifetime > 0) {
                spellIt->creationTime += dt;  // ← Increment age
            }
            // Skip if already marked for removal
            if (spellIt->markForRemoval) {
                destroyPhysicsBodyForSpell(*spellIt);
                spellIt = activeSpells.erase(spellIt);
                continue;
            }

            // Update spell center from physics body if active
            if (spellIt->physicsBody && spellIt->isPhysicsEnabled) {
                spellIt->center = spellIt->physicsBody->position;
            }

            // Track which voxels have arrived this frame (ids)
            std::vector<uint64_t> newlyArrivedIDs;

            // Check each voxel ID in this spell
            for (uint64_t id : spellIt->animatedVoxelIDs) {
                auto itIndex = animatedVoxelIndexMap.find(id);

                if (itIndex == animatedVoxelIndexMap.end()) {
                    newlyArrivedIDs.push_back(id);
                    continue;
                }

                size_t idx = itIndex->second;
                if (idx >= animatedVoxels.size()) {
                    newlyArrivedIDs.push_back(id);
                    continue;
                }

                AnimatedVoxel &voxel = animatedVoxels[idx];

                if (voxel.isAnimating) {
                    glm::vec3 toTarget = voxel.targetPos - voxel.currentPos;
                    float distance = glm::length(toTarget);

                    if (distance < 1.0f * VOXEL_SIZE) {
                        voxel.isAnimating = false;
                        voxel.hasArrived = true;
                        newlyArrivedIDs.push_back(id);
                    } else {
                        float speed = voxel.animationSpeed;
                        float slowdown = glm::clamp(distance / 2.0f, 0.75f, 1.0f);
                        voxel.velocity = (toTarget / glm::vec3(VOXEL_SIZE * CHUNK_SIZE)) * speed * slowdown * 4.0f;
                        voxel.currentPos += voxel.velocity * deltaTime;
                    }
                } else {
                    newlyArrivedIDs.push_back(id);
                }
            }

            // Create partial geometry for newly arrived voxels
            if (!newlyArrivedIDs.empty() && !spellIt->geometryCreated) {
                int arrivedCount = 0;
                int total = (int)spellIt->animatedVoxelIDs.size();
                for (uint64_t id : spellIt->animatedVoxelIDs) {
                    auto jt = animatedVoxelIndexMap.find(id);
                    if (jt == animatedVoxelIndexMap.end()) {
                        ++arrivedCount;
                    } else {
                        AnimatedVoxel &v = animatedVoxels[jt->second];
                        if (!v.isAnimating) ++arrivedCount;
                    }
                }

                float arrivalRatio = total ? (float)arrivedCount / (float)total : 1.0f;

                if (arrivalRatio >= 0.8f) {
                    std::cout << arrivedCount << "/" << total
                              << " voxels arrived. Creating final geometry...\n";

                    createSpellFormation(spellIt->center,
                                         spellIt->formationParams,
                                         spellIt->strength,
                                         spellIt->targetMaterial,
                                         spellIt->formationColor,
                                         spellIt->animatedVoxelIDs.size(),
                                         spellIt->dominantType);
                    spellIt->geometryCreated = true;
                } else if (arrivalRatio > 0.0005f) {
                    createPartialFormation(*spellIt, arrivalRatio);
                }
            }

            // CREATE PHYSICS BODY AFTER GEOMETRY IS READY
            if (spellIt->geometryCreated && spellIt->isPhysicsEnabled &&
                spellIt->physicsBody == nullptr) {
                createPhysicsBodyForSpell(*spellIt);
            }

            // FIXED: Check if all voxels have arrived and we have physics
            if (spellIt->geometryCreated && spellIt->isPhysicsEnabled && !spellIt->voxelsCleaned) {
                int stillAnimating = 0;
                for (uint64_t id : spellIt->animatedVoxelIDs) {
                    auto jt = animatedVoxelIndexMap.find(id);
                    if (jt != animatedVoxelIndexMap.end()) {
                        AnimatedVoxel &v = animatedVoxels[jt->second];
                        if (v.isAnimating) ++stillAnimating;
                    }
                }

                if (stillAnimating == 0) {
                    // All voxels have arrived - clean them up but KEEP THE SPELL for physics
                    std::cout << "All voxels arrived for spell. Cleaning voxel data, keeping physics body.\n";

                    spellIt->voxelsCleaned = true;

                    // Remove all animated voxel IDs from the global list
                    for (uint64_t id : spellIt->animatedVoxelIDs) {
                        auto jt = animatedVoxelIndexMap.find(id);
                        if (jt != animatedVoxelIndexMap.end()) {
                            animatedVoxels[jt->second].isAnimating = false;
                        }
                    }

                    // Clear the ID list to free memory
                    spellIt->animatedVoxelIDs.clear();
                }
            }

            ++spellIt;
        }

        // Clean up non-animating voxels (same as before)
        for (size_t i = 0; i < animatedVoxels.size(); ) {
            if (!animatedVoxels[i].isAnimating) {
                uint64_t removedID = animatedVoxels[i].id;
                size_t last = animatedVoxels.size() - 1;
                if (i != last) {
                    animatedVoxels[i] = animatedVoxels[last];
                    animatedVoxelIndexMap[animatedVoxels[i].id] = i;
                }
                animatedVoxels.pop_back();
                animatedVoxelIndexMap.erase(removedID);
            } else {
                ++i;
            }
        }

        cleanupExpiredSpells();
    }

    void Game::createPartialFormation(const SpellEffect& spell, float completionRatio) {
        WorldPlanet partialFormation;
        partialFormation.worldPos = spell.center;
        partialFormation.color = spell.formationColor;
        partialFormation.type = spell.dominantType;

        // Scale the formation parameters based on completion
        FormationParams partialParams = spell.formationParams;

        switch(partialParams.type) {
            case FormationType::SPHERE:
                partialParams.radius *= (completionRatio * 0.7f + 0.3f);
                break;
            case FormationType::PLATFORM:
            case FormationType::WALL:
            case FormationType::CUBE:
                partialParams.sizeX *= (completionRatio * 0.7f + 0.3f);
                partialParams.sizeY *= (completionRatio * 0.7f + 0.3f);
                partialParams.sizeZ *= (completionRatio * 0.7f + 0.3f);
                break;
            case FormationType::CYLINDER:
                partialParams.radius *= (completionRatio * 0.7f + 0.3f);
                partialParams.sizeY *= (completionRatio * 0.7f + 0.3f);
                break;
        }

        // Temporarily carve partial formation
        carveFormationWithSDF(partialFormation, spell.targetMaterial, partialParams);
    }

    void Game::cleanupExpiredSpells() {
        for (auto spellIt = activeSpells.begin(); spellIt != activeSpells.end(); ) {
            // Check lifetime
            if (spellIt->lifetime > 0) {
                float age = spellIt->creationTime;
                if (age > spellIt->lifetime) {
                    spellIt->markForRemoval = true;
                }
            }

            // CHANGED: Only remove physics spells if they're VERY far away
            // (was 350.0f, increase to 1000.0f or remove this check entirely)
            if (spellIt->isPhysicsEnabled && spellIt->voxelsCleaned) {
                float distanceToPlayer = glm::distance(spellIt->center, cameraPos);
                if (distanceToPlayer > 10000.0f * VOXEL_SIZE) {  // Increased distance
                    spellIt->markForRemoval = true;
                }
            }

            // REMOVED: Don't mark for removal when physics body settles
            // This was causing premature deletion
            /*
            if (spellIt->physicsBody && !spellIt->physicsBody->active) {
                spellIt->markForRemoval = true;
            }
            */
            ++spellIt;
        }
    }

    void Game::onSpellCollision(SpellEffect* spell,
                                const glm::vec3& hitPos,
                                const glm::vec3& hitNormal,
                                float impactSpeed) {
        if (!spell) return;

        // Verify the spell is still valid
        bool spellValid = false;
        for (const auto& s : activeSpells) {
            if (&s == spell) {
                spellValid = true;
                break;
            }
        }

        if (!spellValid) {
            std::cout << "Collision callback for invalid spell, ignoring\n";
            return;
        }

        // Now safe to access spell
        float mass = spell->physicsBody ? spell->physicsBody->mass : 1.0f;


/*
        // Determine which chunks this formation affects
        int minCX = worldToChunk(spell->center.x - spell->radius);
        int maxCX = worldToChunk(spell->center.x + spell->radius);
        int minCY = worldToChunk(spell->center.y - spell->radius);
        int maxCY = worldToChunk(spell->center.y + spell->radius);
        int minCZ = worldToChunk(spell->center.z - spell->radius);
        int maxCZ = worldToChunk(spell->center.z + spell->radius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};

                    // Get chunk - should exist after pre-loading
                    Chunk *chunk = chunkManager->getChunk(coord);
                    if (!chunk) {
                        // Emergency fallback
                        chunkManager->addChunk(coord, VoxelCategory::DYNAMIC);
                        chunk = chunkManager->getChunk(coord);
                        if (!chunk) continue;

                        if (chunk->coord != coord) { // New chunk
                            chunk->coord = coord;
                            chunk->clear();
                        }
                    }
                    createExteriorSmoothCrater(chunkManager->getChunk(coord), glm::vec3(cy, cy, cy),
                                               glm::vec3(cy, cy, cy));
                }
            }
        }
        */
       // spell->markForRemoval=true;
        //applyImpactAtPosition(hitPos, spell->radius, impactSpeed, mass);

        /*std::cout << "Spell collided at ("
                  << hitPos.x << "," << hitPos.y << "," << hitPos.z
                  << ") with impact speed " << impactSpeed << "\n";*/

        // Handle different spell types on collision
        /*
        switch (spell->type) {
            case SpellEffect::Type::CONSTRUCT:
                // Construct spells might just bounce
                if (impactSpeed < 1.0f && spell->voxelsCleaned) {
                    // Low impact - mark for removal after a short time
                    spell->lifetime = 2.0f;
                }
                break;

            case SpellEffect::Type::GRAVITY_WELL:
                // Gravity wells might stick to surfaces
                if (spell->physicsBody) {
                    spell->physicsBody->active = false; // Freeze
                    spell->physicsBody->velocity = glm::vec3(0.0f);
                }
                break;

            default:
                break;
        }
*/
    }

    void Game::createCraterAtPosition(const glm::vec3& worldPos, float impactFactor, float spellRadius) {
        // Find which chunk this position is in
        int cx = worldToChunk(worldPos.x);
        int cy = worldToChunk(worldPos.y);
        int cz = worldToChunk(worldPos.z);

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) {
            std::cout << "No chunk found at impact position\n";
            return;
        }

        // Convert world position to local chunk coordinates
        glm::vec3 chunkMin = getChunkMin(coord);
        glm::vec3 localPos = (worldPos - chunkMin) / VOXEL_SIZE;

        // Get voxel coordinates (clamp to chunk bounds)
        int vx = glm::clamp(static_cast<int>(std::round(localPos.x)), 0, CHUNK_SIZE);
        int vy = glm::clamp(static_cast<int>(std::round(localPos.y)), 0, CHUNK_SIZE);
        int vz = glm::clamp(static_cast<int>(std::round(localPos.z)), 0, CHUNK_SIZE);

        glm::ivec3 voxelPos(vx, vy, vz);

        // Scale crater based on impact factor and spell radius
        float originalCraterRadius = 2.0f * VOXEL_SIZE;
        float craterRadius = originalCraterRadius * impactFactor * (spellRadius / (5.0f * VOXEL_SIZE));
        float maxCraterDepth = 2.5f * impactFactor;

        // Clamp to reasonable values
        craterRadius = glm::clamp(craterRadius, VOXEL_SIZE, 10.0f * VOXEL_SIZE);
        maxCraterDepth = glm::clamp(maxCraterDepth, 1.0f, 5.0f);

        int range = static_cast<int>(std::ceil(craterRadius / VOXEL_SIZE));

        bool chunkModified = false;

        for (int dx = -range; dx <= range; ++dx) {
            for (int dy = -range; dy <= range; ++dy) {
                for (int dz = -range; dz <= range; ++dz) {
                    int nx = voxelPos.x + dx;
                    int ny = voxelPos.y + dy;
                    int nz = voxelPos.z + dz;

                    // Check if we're still in this chunk
                    if (nx < 0 || nx > CHUNK_SIZE || ny < 0 || ny > CHUNK_SIZE || nz < 0 || nz > CHUNK_SIZE) {
                        // Handle neighboring chunks if needed
                        handleCraterInNeighboringChunk(worldPos, dx, dy, dz, craterRadius, maxCraterDepth, impactFactor);
                        continue;
                    }

                    glm::vec3 offset = glm::vec3((float)dx, (float)dy, (float)dz) * VOXEL_SIZE;
                    float dist = glm::length(offset);

                    if (dist <= craterRadius) {
                        float t = dist / craterRadius;
                        float originalDensity = chunk->voxels[nx][ny][nz].density;

                        // Only modify solid or semi-solid voxels
                        if (originalDensity > -1.0f) {
                            // Create crater shape
                            float craterShape = (1.0f - t * t); // Smooth falloff
                            float densityReduction = maxCraterDepth * craterShape;

                            float newDensity = originalDensity - densityReduction;

                            // Ensure we don't go too negative
                            if (originalDensity > 2.0f) {
                                newDensity = std::max(newDensity, 0.1f);
                            }

                            chunk->voxels[nx][ny][nz].density = newDensity;

                            // Update type based on new density
                            if (newDensity < -0.5f) {
                                chunk->voxels[nx][ny][nz].type = 0; // Air
                            } else {
                                chunk->voxels[nx][ny][nz].type = 1; // Solid
                            }

                            chunkModified = true;
                        }
                    }
                }
            }
        }

        if (chunkModified) {
            chunk->meshDirty = true;
            chunk->lightingDirty = true;
            markChunkModified(coord);

            // Also mark neighboring chunks that might have been affected
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        ChunkCoord neighbor{coord.x + dx, coord.y + dy, coord.z + dz};
                        Chunk* neighborChunk = chunkManager->getChunk(neighbor);
                        if (neighborChunk) {
                            neighborChunk->meshDirty = true;
                        }
                    }
                }
            }
        }
    }

    void Game::handleCraterInNeighboringChunk(const glm::vec3& worldPos,
                                              int dx, int dy, int dz,
                                              float craterRadius,
                                              float maxCraterDepth,
                                              float impactFactor) {
        // Calculate the world position of this voxel
        glm::vec3 voxelWorldPos = worldPos + glm::vec3(dx, dy, dz) * VOXEL_SIZE;

        // Find which chunk this belongs to
        int cx = worldToChunk(voxelWorldPos.x);
        int cy = worldToChunk(voxelWorldPos.y);
        int cz = worldToChunk(voxelWorldPos.z);

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) return;

        // Convert to local chunk coordinates
        glm::vec3 chunkMin = getChunkMin(coord);
        glm::vec3 localPos = (voxelWorldPos - chunkMin) / VOXEL_SIZE;

        int nx = glm::clamp(static_cast<int>(std::round(localPos.x)), 0, CHUNK_SIZE);
        int ny = glm::clamp(static_cast<int>(std::round(localPos.y)), 0, CHUNK_SIZE);
        int nz = glm::clamp(static_cast<int>(std::round(localPos.z)), 0, CHUNK_SIZE);

        // Calculate distance from impact center
        glm::vec3 offset = voxelWorldPos - worldPos;
        float dist = glm::length(offset);

        if (dist <= craterRadius) {
            float t = dist / craterRadius;
            float originalDensity = chunk->voxels[nx][ny][nz].density;

            if (originalDensity > -1.0f) {
                float craterShape = (1.0f - t * t);
                float densityReduction = maxCraterDepth * craterShape;

                float newDensity = originalDensity - densityReduction;

                if (originalDensity > 2.0f) {
                    newDensity = std::max(newDensity, 0.1f);
                }

                chunk->voxels[nx][ny][nz].density = newDensity;

                if (newDensity < -0.5f) {
                    chunk->voxels[nx][ny][nz].type = 0;
                } else {
                    chunk->voxels[nx][ny][nz].type = 1;
                }

                chunk->meshDirty = true;
                chunk->lightingDirty = true;
                markChunkModified(coord);
            }
        }
    }

        float Game::randomFloat(float min, float max) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

            return min + dis(gen) * (max - min);
        }

    void Game::castSpellSphere(const glm::vec3& center, float radius,
                               uint64_t material, float strength) {
        FormationParams params = FormationParams::Sphere(center, radius);
        castSpellWithFormation(center, radius * 10.5f, material, strength, params);

        if (!activeSpells.empty()) {
            SpellEffect& lastSpell = activeSpells.back();

            // Get camera direction
            glm::vec3 launchDir = glm::normalize(getCameraFront());

            // Scale launch speed properly
            float launchSpeed = strength * 5.0f * VOXEL_SIZE;  // Adjusted for visibility

            lastSpell.isPhysicsEnabled = true;
            lastSpell.creationTime = 0.0f;
            lastSpell.lifetime = 20.0f;
            lastSpell.center=center;
            lastSpell.initialVelocity = launchDir * launchSpeed;

            std::cout << "Sphere spell: speed=" << launchSpeed
                      << " direction=(" << launchDir.x << "," << launchDir.y << "," << launchDir.z << ")\n";
        }
    }

    void Game::castSpellPlatform(const glm::vec3& center, const glm::vec3& normal,
                                 float width, float depth, float thickness,
                                 uint64_t material, float strength) {
        FormationParams params = FormationParams::Platform(center, normal,
                                                           width, depth, thickness);
        float searchRadius = 10*4.5f*VOXEL_SIZE;
        castSpellWithFormation(center, searchRadius, material, strength, params);
    }

    void Game::castSpellWall(const glm::vec3& center, const glm::vec3& normal,
                             float width, float height, float thickness,
                             uint64_t material, float strength) {
        FormationParams params = FormationParams::Wall(center, normal,
                                                       width, height, thickness);
        float searchRadius =  strength*4.5f*VOXEL_SIZE;

        castSpellWithFormation(center, searchRadius, material, strength, params);
    }

    void Game::castSpellCube(const glm::vec3& center, const glm::vec3& size,
                             uint64_t material, float strength) {
        FormationParams params = FormationParams::Cube(center, size);
        float searchRadius = glm::length(size) * 0.75f;
        castSpellWithFormation(center, searchRadius, material, strength, params);
    }

    void Game::castSpellCylinder(const glm::vec3& center, float radius, float height,
                                 uint64_t material, float strength) {
        FormationParams params = FormationParams::Cylinder(center, radius, height);
        float searchRadius = glm::max(radius, height * 0.5f) * 1.5f;
        castSpellWithFormation(center, searchRadius, material, strength, params);
    }

    // Custom spell with user-defined SDF
    void Game::castSpellCustom(const glm::vec3& center, float radius,
                               uint64_t material, float strength,
                               SDFFunction customSDF, void* userData) {
        FormationParams params;
        params.type = FormationType::CUSTOM_SDF;
        params.center = center;
        params.radius = radius;
        params.customSDF = customSDF;
        params.customUserData = userData;

        castSpellWithFormation(center, radius, material, strength, params);
    }

    void Game::createPhysicsBodyForSpell(SpellEffect& spell) {
        if (!voxelPhysics || spell.physicsBody != nullptr) return;

        float effectiveRadius = spell.formationParams.getBoundingRadius();

        // Create a spatial hash of target positions for quick lookup
        struct TargetKey {
            int64_t x, y, z;
            bool operator==(const TargetKey& other) const {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct TargetKeyHash {
            std::size_t operator()(const TargetKey& k) const {
                return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791)) % 1000000;
            }
        };

        // Map from voxel grid position to whether we expect geometry there
        std::unordered_map<TargetKey, bool, TargetKeyHash> expectedVoxels;
        std::unordered_map<TargetKey, bool, TargetKeyHash> boundaryVoxels;

        // Populate with target positions from animated voxels
        for (uint64_t id : spell.animatedVoxelIDs) {
            auto it = animatedVoxelIndexMap.find(id);
            if (it != animatedVoxelIndexMap.end()) {
                const AnimatedVoxel& voxel = animatedVoxels[it->second];

                // Round target position to nearest voxel center
                TargetKey key{
                        static_cast<int64_t>(std::round(voxel.targetPos.x / VOXEL_SIZE)),
                        static_cast<int64_t>(std::round(voxel.targetPos.y / VOXEL_SIZE)),
                        static_cast<int64_t>(std::round(voxel.targetPos.z / VOXEL_SIZE))
                };
                expectedVoxels[key] = true;
            }
        }

        // Also mark boundary voxels (neighbors of expected voxels)
        // This ensures we keep triangles on the surface
        for (const auto& [key, _] : expectedVoxels) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        TargetKey neighbor{
                                key.x + dx,
                                key.y + dy,
                                key.z + dz
                        };
                        boundaryVoxels[neighbor] = true;
                    }
                }
            }
        }

        std::cout << "Expected formation has " << expectedVoxels.size()
                  << " voxels, with " << boundaryVoxels.size() << " boundary voxels\n";

        // Collect triangle vertices from the formation chunks
        std::vector<glm::vec3> triangleVerts;
        std::vector<glm::vec3> triangleNormals;
        std::vector<glm::vec3> triangleColors;
        triangleVerts.reserve(10000);
        triangleNormals.reserve(10000);
        triangleColors.reserve(10000);

        int minCX = worldToChunk(spell.center.x - effectiveRadius);
        int maxCX = worldToChunk(spell.center.x + effectiveRadius);
        int minCY = worldToChunk(spell.center.y - effectiveRadius);
        int maxCY = worldToChunk(spell.center.y + effectiveRadius);
        int minCZ = worldToChunk(spell.center.z - effectiveRadius);
        int maxCZ = worldToChunk(spell.center.z + effectiveRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk || !chunk->gpuCache.isValid) continue;
                    if (chunk->gpuCache.vertexCount == 0) continue;

                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);
                    size_t vcount = chunk->gpuCache.vertexCount;
                    size_t byteSize = vcount * sizeof(OutVertex);

                    if (byteSize > 0) {
                        void* mapPtr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                        byteSize, GL_MAP_READ_BIT);
                        if (mapPtr) {
                            OutVertex* ov = reinterpret_cast<OutVertex*>(mapPtr);

                            // Process triangles
                            for (size_t vi = 0; vi < vcount; vi += 3) {
                                // Get all three vertices
                                glm::vec3 v0(ov[vi].pos.x, ov[vi].pos.y, ov[vi].pos.z);
                                glm::vec3 v1(ov[vi + 1].pos.x, ov[vi + 1].pos.y, ov[vi + 1].pos.z);
                                glm::vec3 v2(ov[vi + 2].pos.x, ov[vi + 2].pos.y, ov[vi + 2].pos.z);

                                // Calculate triangle center and bounding box
                                glm::vec3 center = (v0 + v1 + v2) / 3.0f;

                                // Calculate triangle bounding box to check if any part is near expected voxels
                                glm::vec3 triMin = glm::min(glm::min(v0, v1), v2);
                                glm::vec3 triMax = glm::max(glm::max(v0, v1), v2);

                                // Expand by a small margin to catch edge cases
                                triMin -= glm::vec3(VOXEL_SIZE * 0.5f);
                                triMax += glm::vec3(VOXEL_SIZE * 0.5f);

                                // Convert to voxel grid coordinates
                                TargetKey minKey{
                                        static_cast<int64_t>(std::floor(triMin.x / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::floor(triMin.y / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::floor(triMin.z / VOXEL_SIZE))
                                };

                                TargetKey maxKey{
                                        static_cast<int64_t>(std::ceil(triMax.x / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::ceil(triMax.y / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::ceil(triMax.z / VOXEL_SIZE))
                                };

                                // Check if this triangle touches any expected or boundary voxel
                                bool shouldKeep = false;

                                // First check: if triangle center is near expected voxels (strict)
                                TargetKey centerKey{
                                        static_cast<int64_t>(std::round(center.x / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::round(center.y / VOXEL_SIZE)),
                                        static_cast<int64_t>(std::round(center.z / VOXEL_SIZE))
                                };

                                // Check center and neighbors (quick test)
                                for (int dx = -1; dx <= 1 && !shouldKeep; ++dx) {
                                    for (int dy = -1; dy <= 1 && !shouldKeep; ++dy) {
                                        for (int dz = -1; dz <= 1 && !shouldKeep; ++dz) {
                                            TargetKey neighbor{
                                                    centerKey.x + dx,
                                                    centerKey.y + dy,
                                                    centerKey.z + dz
                                            };
                                            if (expectedVoxels.find(neighbor) != expectedVoxels.end()) {
                                                shouldKeep = true;
                                            }
                                        }
                                    }
                                }

                                // Second check: if any part of the triangle touches boundary voxels (more permissive)
                                if (!shouldKeep) {
                                    for (int64_t x = minKey.x; x <= maxKey.x && !shouldKeep; ++x) {
                                        for (int64_t y = minKey.y; y <= maxKey.y && !shouldKeep; ++y) {
                                            for (int64_t z = minKey.z; z <= maxKey.z && !shouldKeep; ++z) {
                                                TargetKey voxelKey{x, y, z};
                                                if (boundaryVoxels.find(voxelKey) != boundaryVoxels.end()) {
                                                    shouldKeep = true;
                                                }
                                            }
                                        }
                                    }
                                }

                                // Third check: SDF fallback with larger threshold
                                if (!shouldKeep) {
                                    float sdf = spell.formationParams.evaluate(center);
                                    // Use a larger threshold to catch everything near the surface
                                    if (std::abs(sdf) < VOXEL_SIZE * 2.0f) {
                                        shouldKeep = true;
                                    }
                                }

                                if (shouldKeep) {
                                    for (int j = 0; j < 3; ++j) {
                                        glm::vec4 p = ov[vi + j].pos;
                                        glm::vec4 n = ov[vi + j].normal;
                                        glm::vec4 c = ov[vi + j].color;

                                        triangleVerts.emplace_back(p.x, p.y, p.z);
                                        triangleNormals.emplace_back(n.x, n.y, n.z);
                                        triangleColors.emplace_back(c.x, c.y, c.z);
                                    }
                                }
                            }

                            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                        }
                    }
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
                }
            }
        }

        if (triangleVerts.empty()) {
            std::cout << "createPhysicsBody: No triangles found for spell after filtering!\n";
            return;
        }

        std::cout << "Found " << triangleVerts.size() << " triangles for physics body (after filtering)\n";

        // Optional: Ensure manifold mesh by checking for holes
        // This is a simple check - you might want more sophisticated hole detection
        if (triangleVerts.size() < 12) { // Less than 4 triangles
            std::cout << "Warning: Very small mesh, might be incomplete\n";
        }

        // Calculate the actual bounds of the collected geometry
        glm::vec3 minBound = triangleVerts[0];
        glm::vec3 maxBound = triangleVerts[0];
        for (const auto& v : triangleVerts) {
            minBound = glm::min(minBound, v);
            maxBound = glm::max(maxBound, v);
        }
        glm::vec3 geomCenter = (minBound + maxBound) * 0.5f;
        glm::vec3 geomExtents = (maxBound - minBound) * 0.5f;

        // Expand extents slightly to ensure collision hull contains mesh
        geomExtents += glm::vec3(VOXEL_SIZE * 0.1f);

        // Determine shape type based on formation
        gl3::VoxelPhysicsBody::ShapeType shapeType;
        glm::vec3 extents;

        switch(spell.formationParams.type) {
            case FormationType::SPHERE:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::SPHERE;
                extents = glm::vec3(glm::length(geomExtents)); // Use actual geometry bounds
                break;
            case FormationType::PLATFORM:
            case FormationType::WALL:
            case FormationType::CUBE:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::BOX;
                extents = geomExtents; // Use actual geometry bounds
                break;
            case FormationType::CYLINDER:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::BOX;
                extents = geomExtents;
                break;
            default:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::SPHERE;
                extents = glm::vec3(glm::length(geomExtents));
                break;
        }

        // Calculate mass based on collected voxels (use animatedVoxelIDs size)
        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
        mass = glm::clamp(mass, 1.0f, 1000.0f);

        // Create physics body - use spell.center for initial position
        spell.physicsBody = voxelPhysics->createBody(
                spell.center,
                mass,
                shapeType,
                extents
        );

        if (spell.physicsBody) {
            spell.physicsBody->userData = &spell;
            spell.physicsBody->velocity = spell.initialVelocity;

            // Set orientation to face the direction of travel (camera front)
            if (glm::length(spell.initialVelocity) > 0.001f) {
                glm::vec3 direction = glm::normalize(spell.initialVelocity); // This is camera front

                // For glm::quatLookAt, the first parameter is the direction TO look at
                // If your model's forward is +Z, then you want to rotate so that +Z points in direction
                // quatLookAt gives a rotation that makes the object's forward point TOWARDS the given direction
                spell.physicsBody->orientation = glm::quatLookAt(direction, glm::vec3(0.0f, 1.0f, 0.0f));

                // Note: If your model uses a different forward axis, adjust accordingly:
                // For +X forward: glm::quatLookAt(direction, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f))
                // For +Y forward: glm::quatLookAt(direction, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f))
            } else {
                spell.physicsBody->orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity
            }

            // Store the mesh data for rendering
            createPhysicsMeshData(spell, triangleVerts, triangleNormals, triangleColors);

            // Link the mesh to the physics body for rendering
            spell.physicsBody->renderMesh = &spell.physicsMesh;

            spell.isPhysicsEnabled = true;

            std::cout << "Physics body created: mass=" << mass
                      << ", renderVerts=" << triangleVerts.size()
                      << ", velocity=" << glm::length(spell.initialVelocity) << "\n";
            std::cout << "Spell center: (" << spell.center.x << "," << spell.center.y << "," << spell.center.z << ")\n";
            std::cout << "Geometry bounds: center(" << geomCenter.x << "," << geomCenter.y << "," << geomCenter.z
                      << ") extents(" << extents.x << "," << extents.y << "," << extents.z << ")\n";
        }
        // Remove voxels from chunks now that we have the mesh
        removeFormationVoxels(spell);
        }

    void Game::createPhysicsMeshData(SpellEffect& spell,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<glm::vec3>& normals,
                                     const std::vector<glm::vec3>& colors) {
        if (vertices.empty()) return;

        // Store for potential recreation
        spell.physicsMesh.vertices = vertices;
        spell.physicsMesh.normals = normals;
        spell.physicsMesh.colors = colors;
        spell.physicsMesh.vertexCount = vertices.size();

        // Create VAO/VBO
        glGenVertexArrays(1, &spell.physicsMesh.vao);
        glGenBuffers(1, &spell.physicsMesh.vbo);

        glBindVertexArray(spell.physicsMesh.vao);
        glBindBuffer(GL_ARRAY_BUFFER, spell.physicsMesh.vbo);

        // Interleave vertex data: pos(3) + normal(3) + color(3) = 9 floats per vertex
        std::vector<float> interleavedData;
        interleavedData.reserve(vertices.size() * 9);

        for (size_t i = 0; i < vertices.size(); ++i) {
            // Position
            interleavedData.push_back(vertices[i].x);
            interleavedData.push_back(vertices[i].y);
            interleavedData.push_back(vertices[i].z);

            // Normal
            if (i < normals.size()) {
                interleavedData.push_back(normals[i].x);
                interleavedData.push_back(normals[i].y);
                interleavedData.push_back(normals[i].z);
            } else {
                interleavedData.push_back(0.0f);
                interleavedData.push_back(1.0f);
                interleavedData.push_back(0.0f);
            }

            // Color
            if (i < colors.size()) {
                interleavedData.push_back(colors[i].x);
                interleavedData.push_back(colors[i].y);
                interleavedData.push_back(colors[i].z);
            } else {
                interleavedData.push_back(spell.formationColor.r);
                interleavedData.push_back(spell.formationColor.g);
                interleavedData.push_back(spell.formationColor.b);
            }
        }

        glBufferData(GL_ARRAY_BUFFER, interleavedData.size() * sizeof(float),
                     interleavedData.data(), GL_STATIC_DRAW);

        // Position attribute
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);

        // Normal attribute
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                              (void*)(3 * sizeof(float)));

        // Color attribute
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                              (void*)(6 * sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        spell.physicsMesh.isValid = true;
    }


    void Game::removeFormationVoxels(const SpellEffect& spell) {
        float effectiveRadius = spell.formationParams.getBoundingRadius();

        int minCX = worldToChunk(spell.center.x - effectiveRadius);
        int maxCX = worldToChunk(spell.center.x + effectiveRadius);
        int minCY = worldToChunk(spell.center.y - effectiveRadius);
        int maxCY = worldToChunk(spell.center.y + effectiveRadius);
        int minCZ = worldToChunk(spell.center.z - effectiveRadius);
        int maxCZ = worldToChunk(spell.center.z + effectiveRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkMin = getChunkMin(coord);
                    bool touched = false;

                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 vWorld = chunkMin +
                                                   glm::vec3((float)lx, (float)ly, (float)lz) * VOXEL_SIZE;

                                // Use formation SDF to determine what to remove
                                float formationDensity = spell.formationParams.evaluate(vWorld);

                                // Remove voxels that are part of the formation
                                if (formationDensity >= 0.0f) {
                                    chunk->voxels[lx][ly][lz].density = -1000.0f;
                                    chunk->voxels[lx][ly][lz].type = 0; // Air
                                    touched = true;
                                }
                            }
                        }
                    }

                    if (touched) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        markChunkModified(coord);
                    }
                }
            }
        }

        std::cout << "Removed formation voxels from chunks\n";
    }

    void Game::destroyPhysicsBodyForSpell(SpellEffect& spell) {
        if (spell.physicsBody && voxelPhysics) {
            voxelPhysics->removeBody(spell.physicsBody);
            spell.physicsBody = nullptr;
        }

        // Clean up mesh data
        if (spell.physicsMesh.vao) {
            glDeleteVertexArrays(1, &spell.physicsMesh.vao);
            glDeleteBuffers(1, &spell.physicsMesh.vbo);
            spell.physicsMesh.vao = 0;
            spell.physicsMesh.vbo = 0;
            spell.physicsMesh.isValid = false;
        }
    }


////----Debugging Code--------------------------------------------------------------------------------------------------------------------------

    void Game::debugComputeShaderState() {
        std::cout << "\n=== COMPUTE SHADER DIAGNOSTICS ===\n";

        // 1. Check if shader exists
        if (!marchingCubesShader) {
            std::cerr << "ERROR: marchingCubesShader is nullptr!\n";
            return;
        }

        // 2. Get program ID
        GLuint program = marchingCubesShader->getProgramID();
        std::cout << "Shader program ID: " << program << std::endl;

        if (program == 0) {
            std::cerr << "ERROR: Shader program ID is 0!\n";
            return;
        }

        // 3. Check if program is linked
        GLint linkStatus;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        std::cout << "Program link status: " << (linkStatus == GL_TRUE ? "SUCCESS" : "FAILED") << std::endl;

        if (linkStatus != GL_TRUE) {
            GLchar infoLog[1024];
            glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Link error: " << infoLog << std::endl;
        }

        // 4. Check if program is valid for current GL state
        glValidateProgram(program);
        GLint validateStatus;
        glGetProgramiv(program, GL_VALIDATE_STATUS, &validateStatus);
        std::cout << "Program validate status: " << (validateStatus == GL_TRUE ? "VALID" : "INVALID") << std::endl;

        if (validateStatus != GL_TRUE) {
            GLchar infoLog[1024];
            glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Validation error: " << infoLog << std::endl;
        }

        // 5. Check what shader is currently bound
        GLint currentProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);
        std::cout << "Currently bound program: " << currentProgram << std::endl;

        // 6. Check compute shader work group limits
        GLint maxWorkGroupSize[3];
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSize[1]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxWorkGroupSize[2]);

        std::cout << "Max work group size: ["
                  << maxWorkGroupSize[0] << ", "
                  << maxWorkGroupSize[1] << ", "
                  << maxWorkGroupSize[2] << "]\n";

        GLint maxWorkGroupInvocations;
        glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxWorkGroupInvocations);
        std::cout << "Max work group invocations: " << maxWorkGroupInvocations << std::endl;

        // 7. Check if it's actually a compute shader program
        GLint numShaders;
        glGetProgramiv(program, GL_ATTACHED_SHADERS, &numShaders);
        std::cout << "Number of attached shaders: " << numShaders << std::endl;

        GLuint shaders[10];
        GLsizei count;
        glGetAttachedShaders(program, 10, &count, shaders);

        for (int i = 0; i < count; i++) {
            GLint shaderType;
            glGetShaderiv(shaders[i], GL_SHADER_TYPE, &shaderType);
            const char *typeStr = "Unknown";
            if (shaderType == GL_COMPUTE_SHADER) typeStr = "COMPUTE";
            else if (shaderType == GL_VERTEX_SHADER) typeStr = "VERTEX";
            else if (shaderType == GL_FRAGMENT_SHADER) typeStr = "FRAGMENT";

            std::cout << "  Shader " << i << ": ID=" << shaders[i] << ", Type=" << typeStr << std::endl;
        }

        std::cout << "=== END DIAGNOSTICS ===\n\n";
    }

    /*
    void debugDensity()
    {
        //Debugging for Density

        std::cout << "meteor SDF min=" << minD << " max=" << maxD
                  << " center=" << meteorChunk.voxels[cx][cy][cz].density << std::endl;

        // Print density along x-axis through center (y=cy,z=cz)
        std::cout << "center-line densities: ";
        for (int x=0;x<CHUNK_SIZE;++x) {
            std::cout << meteorChunk.voxels[x][cy][cz].density << (x+1<CHUNK_SIZE? ",":"\n");
        }
    */

    void Game::DisplayFPSCount()
    {
        static double lastTime = 0.0;
        static int frames = 0;

        double currentTime = glfwGetTime();
        frames++;

        if (currentTime - lastTime >= 1.0) {
            double fps = frames / (currentTime - lastTime);
            frames = 0;
            lastTime = currentTime;

            std::string title = "Voxel Engine | FPS: " + std::to_string((int)fps);
            glfwSetWindowTitle(getWindow(), title.c_str());
        }
    }

////----Initialization Code---------------------------------------------------------------------------------------------------------------------

    void Game::setupSSBOsAndTables() {
        // Prepare SSBOs and static tables

        // 0: voxels SSBO
        glGenBuffers(1, &ssboVoxels);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxelCount * sizeof(CpuVoxel), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels); // bind to 0

        // 1: edge table SSBO
        glGenBuffers(1, &ssboEdgeTable);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboEdgeTable);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(edgeTableCPU), edgeTableCPU, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);

        // 2: tri table SSBO
        glGenBuffers(1, &ssboTriTable);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriTable);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(triTableCPU), triTableCPU, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);

        // 3: atomic counter (SSBO containing uint vertexCounter)
        glGenBuffers(1, &ssboCounter);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        unsigned int zero = 0;
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int), &zero, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);

        // 4: triangles SSBO (output)
        glGenBuffers(1, &ssboTriangles);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriangles);
        glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(OutVertex), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);

        //5 particle ssbo
        glGenBuffers(1, &particleSSBO);

        //6 fieldbits ssbo
        glGenBuffers(1, &fieldBitsSSBO);

    }

    void Game::setupInput() {
        // Track all keys we'll use
        input.trackKeys({
                                GLFW_KEY_W, GLFW_KEY_UP, GLFW_KEY_S, GLFW_KEY_DOWN,
                                GLFW_KEY_A, GLFW_KEY_LEFT, GLFW_KEY_D, GLFW_KEY_RIGHT,
                                GLFW_KEY_SPACE, GLFW_KEY_LEFT_SHIFT, GLFW_KEY_LEFT_CONTROL,
                                GLFW_KEY_TAB, GLFW_KEY_ESCAPE,GLFW_KEY_E,GLFW_KEY_R,GLFW_KEY_F,
                                GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4, GLFW_KEY_5, GLFW_KEY_6,
                                GLFW_KEY_T
                        });

        // Character movement
        actions.addAction("MoveForward", {GLFW_KEY_W, GLFW_KEY_UP});
        actions.addAction("MoveBack", {GLFW_KEY_S, GLFW_KEY_DOWN});
        actions.addAction("MoveLeft", {GLFW_KEY_A, GLFW_KEY_LEFT});
        actions.addAction("MoveRight", {GLFW_KEY_D, GLFW_KEY_RIGHT});
        actions.addAction("Jump", {GLFW_KEY_SPACE});
        actions.addAction("Sprint", {GLFW_KEY_LEFT_SHIFT});
        actions.addAction("Crouch", {GLFW_KEY_LEFT_CONTROL});

        // Debug/UI actions
        actions.addAction("ToggleDebug", {GLFW_KEY_TAB});
        actions.addAction("DebugMode1", {GLFW_KEY_1});
        actions.addAction("DebugMode2", {GLFW_KEY_2});
        actions.addAction("DebugMode3", {GLFW_KEY_3});
        actions.addAction("DebugMode4", {GLFW_KEY_4});
        actions.addAction("DebugMode5", {GLFW_KEY_5});
        actions.addAction("DebugMode6", {GLFW_KEY_6});
        actions.addAction("Wireframe", {GLFW_KEY_T});


        actions.addAction("Escape", {GLFW_KEY_ESCAPE});
        actions.addAction("CastSphere", {GLFW_KEY_E});
        actions.addAction("CastWall", {GLFW_KEY_R});
        actions.addAction("AirReset", {GLFW_KEY_F});

    }

    void Game::setupCamera() {
        // --- Camera setup ---
        cameraRotation = glm::vec2(0.0f, -90.0f);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }


    void Game::generateChunks() {
        std::mt19937 rng(std::random_device{}());

        std::uniform_real_distribution<float> distPos(-(ChunkCount * 4), (ChunkCount * 4)); // Reduced range
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);

        std::vector<WorldPlanet> worldPlanets;

        // Create solid planets (type 1)
        int planetCount = 20;

        WorldPlanet p;
        p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
        p.radius = distScale(rng) * CHUNK_SIZE;
        p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
        p.type = 1; // solid
        worldPlanets.push_back(p);
        cameraPos=p.worldPos+glm::vec3(0,VOXEL_SIZE,0);
        characterController.setPosition(cameraPos);
        for (int i = 0; i < planetCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
            p.type = 1; // solid
            worldPlanets.push_back(p);
        }

        // Create suns (type 2 - fire)
        std::uniform_real_distribution<float> lavaDistColorR(0.8f, 1.0f);
        std::uniform_real_distribution<float> lavaDistColorG(0.2f, 0.5f);
        std::uniform_real_distribution<float> lavaDistColorB(0.0f, 0.1f);

        int lavaCount = 4 + (rng() % 5);
        for (int i = 0; i < lavaCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(lavaDistColorR(rng), lavaDistColorG(rng), lavaDistColorB(rng));
            p.type = 2; // fire
            worldPlanets.push_back(p);
        }

        // Create water planets (type 3)
        std::uniform_real_distribution<float> waterDistColorR(0.0f, 0.2f);
        std::uniform_real_distribution<float> waterDistColorG(0.2f, 0.8f);
        std::uniform_real_distribution<float> waterDistColorB(0.8f, 1.0f);

        int waterCount = 0 + (rng() % 1);
        for (int i = 0; i < waterCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(waterDistColorR(rng), waterDistColorG(rng), waterDistColorB(rng));
            p.type = 3; // water
            worldPlanets.push_back(p);
        }

        // Carve planets into chunks
        size_t solidVoxels = 0;

        // DEBUG: Track how many chunks we create
        std::unordered_set<ChunkCoord, ChunkCoordHash> createdChunks;

        for (const auto &planet: worldPlanets) {
            // Determine which chunks this planet affects
            int minCX = worldToChunk(planet.worldPos.x - planet.radius);
            int maxCX = worldToChunk(planet.worldPos.x + planet.radius);
            int minCY = worldToChunk(planet.worldPos.y - planet.radius);
            int maxCY = worldToChunk(planet.worldPos.y + planet.radius);
            int minCZ = worldToChunk(planet.worldPos.z - planet.radius);
            int maxCZ = worldToChunk(planet.worldPos.z + planet.radius);

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord coord{cx, cy, cz};

                        // Get or create chunk using MultiGridChunkManager
                        Chunk *chunk = chunkManager->getChunk(coord);
                        if (!chunk) {
                            // Determine category based on planet type
                            VoxelCategory category;
                            switch (planet.type) {
                                case 2:
                                    category = VoxelCategory::EMISSIVE;
                                    break; // Fire
                                case 3:
                                    category = VoxelCategory::FLUID;
                                    break;    // Water
                                default:
                                    category = VoxelCategory::STATIC;
                                    break;  // Solid
                            }
                            chunkManager->addChunk(coord, category);
                            chunk = chunkManager->getChunk(coord);

                            if (chunk) {
                                // Initialize the chunk if newly created
                                chunk->coord = coord;
                                chunk->clear(); // IMPORTANT: Initialize all densities to -1000
                                createdChunks.insert(coord);
                            }
                        }

                        if (!chunk) continue;

                        glm::vec3 chunkOrigin(cx * CHUNK_SIZE*VOXEL_SIZE, cy * CHUNK_SIZE*VOXEL_SIZE, cz * CHUNK_SIZE*VOXEL_SIZE);
                        bool chunkTouched = false;

                        // Carve sphere into chunk - FIXED: Use SDF union operation
                        for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                            for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                                for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                    glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz)*VOXEL_SIZE;
                                    float dist = glm::distance(worldPos, planet.worldPos);

                                    // Calculate this planet's SDF value
                                    float planetDensity = planet.radius - dist; // Positive inside, negative outside

                                    // Get existing density (initialized to -1000 for air)
                                    float existingDensity = chunk->voxels[lx][ly][lz].density;

                                    // SDF UNION: Take the MAXIMUM density (most solid)
                                    // For Marching Cubes, positive = inside, negative = outside
                                    if (planetDensity > existingDensity) {
                                        // This planet is "more solid" at this point
                                        chunk->voxels[lx][ly][lz].density = planetDensity;

                                        // Set type and color if we're inside or near the surface
                                        if (planetDensity >= -1.0f) { // Near the isosurface (density ~0)
                                            chunk->voxels[lx][ly][lz].type = planet.type;
                                            chunk->voxels[lx][ly][lz].color = planet.color;

                                            if (planetDensity >= 0) { // Actually inside the solid
                                                solidVoxels++;
                                                chunkTouched = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (chunkTouched) {
                            chunk->meshDirty = true;
                            chunk->lightingDirty = true;
                            FilledChunks++;

                            // Rebuild lights for this chunk if it contains fire
                            if (planet.type == 2) {
                                rebuildChunkLights(coord);
                            }
                        }
                    }
                }
            }
        }

        // Statistics
        std::cout << "Generated " << worldPlanets.size() << " planets\n";
        std::cout << "Touched " << FilledChunks << " chunks\n";
        std::cout << "Created " << solidVoxels << " solid voxels\n";
    }

    void Game::setupVEffects() {
        sunBillboards.init(12); // maxInstances, adjust based on max expected suns

    }


////----Simulation Code-------------------------------------------------------------------------------------------------------------------------

//------Physics-Code----------------------------------------------------------------------------------------------------------------------------

    void Game::updatePhysics() {
        const float fixedTimeStep = 1.0f / 60.0f;
        const int subStepCount = 4; // recommended sub-step count
        accumulator += deltaTime;
        if (accumulator >= fixedTimeStep) {
            // Update the entities based on what happened in the physics step
            // Get input for character
            glm::vec3 moveInput(0.0f);
            if (actions["MoveForward"].isPressed) moveInput.z += 10.0f;
            if (actions["MoveBack"].isPressed) moveInput.z -= 10.0f;
            if (actions["MoveLeft"].isPressed) moveInput.x -= 10.0f;
            if (actions["MoveRight"].isPressed) moveInput.x += 10.0f;

            bool jump = actions["Jump"].wasJustPressed;
            bool sprint = actions["Sprint"].isPressed;
            bool crouch = actions["Crouch"].isPressed;
            bool airSlam = actions["AirReset"].isPressed;


            // Get mouse delta (you'll need to implement this)
            glm::vec3 cameraFront = getCameraFront();
            glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, glm::vec3(0.0f, 1.0f, 0.0f)));

            // Get mouse delta
            glm::vec2 mouseDelta = getMouseDelta();

            std::cout<<"move Input: "<<moveInput.x<<" X,"<<moveInput.y<<" Y,"<<moveInput.z<<" Z,"<<"\n";
            // Update character with camera-relative movement
            characterController.update(deltaTime, moveInput, jump, sprint, crouch, mouseDelta, cameraFront, cameraRight, airSlam );

            accumulator -= fixedTimeStep;

            // Update physics and get bodies that were removed
            std::vector<uint64_t> removedBodyIds;
            if (voxelPhysics) {
                voxelPhysics->update(deltaTime, removedBodyIds);  // Pass in the vector
            }

            // Mark spells for removal whose physics bodies were removed
            for (uint64_t id : removedBodyIds) {
                for (auto& spell : activeSpells) {
                    if (spell.physicsBody && spell.physicsBody->id == id) {
                        spell.markForRemoval = true;
                        std::cout << "Marking spell for removal due to physics body removal\n";
                        break;
                    }
                }
            }
        }
    }

    void Game::applyImpactAtPosition(const glm::vec3 &worldPos, float radius, float impulse, float mass) {
        // tune these
        const float damageScale = glm::clamp(impulse * 0.005f*deltaTime, 0.5f, 30.0f); // larger impulse -> stronger carving
        const float maxRadius = glm::max(radius, damageScale);
        const float radiusSq = maxRadius * maxRadius;

        // Determine affected chunks in world-space (chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE)
        float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int minCX = worldToChunk(worldPos.x - maxRadius);
        int maxCX = worldToChunk(worldPos.x + maxRadius);
        int minCY = worldToChunk(worldPos.y - maxRadius);
        int maxCY = worldToChunk(worldPos.y + maxRadius);
        int minCZ = worldToChunk(worldPos.z - maxRadius);
        int maxCZ = worldToChunk(worldPos.z + maxRadius);

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkMin = getChunkMin(coord);

                    bool touched = false;
                    // iterate chunk-local voxels
                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 vWorld = chunkMin + glm::vec3((float)lx, (float)ly, (float)lz) * VOXEL_SIZE;
                                glm::vec3 d = vWorld - worldPos;
                                float distSq = glm::dot(d,d);
                                if (distSq > radiusSq) continue;
                                float dist = std::sqrt(distSq);
                                float fall = 1.0f - (dist / maxRadius); // 1 at center -> 0 at edge
                                float carveAmount = damageScale * fall;
                                // reduce density
                                float &density = chunk->voxels[lx][ly][lz].density;
                                density -= carveAmount;
                                if (density < 0.0f) {
                                    chunk->voxels[lx][ly][lz].type = 0; // air
                                }
                                touched = true;
                            }
                        }
                    }

                    if (touched) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        markChunkModified(coord);
                    }
                }
            }
        }

        // If you want visual effects or fragmentation hooks you can use payload->formationID/userData here
        std::cout << "applyImpactAtPosition: hit at " << worldPos.x << "," << worldPos.y << "," << worldPos.z
                  << " radius=" << maxRadius << " impulse=" << impulse << "\n";
    }

    void Game::updateDeltaTime() {
        float frameTime = glfwGetTime();
        deltaTime = frameTime - lastFrameTime;
        lastFrameTime = frameTime;
    }

//------Lighting-Code---------------------------------------------------------------------------------------------------------------------------

    // Replace updateGlobalLightGrid with a spatial hash approach and a flat list and performs
    // a simple greedy merging of lights that are close to each other.
    void Game::updateLightSpatialHash() {
        lightSpatialHash.clear();
        flatEmissiveLightList.clear();
        mergedEmissiveLightPool.clear();

        // 1) Gather raw pointers (lights stored inside chunks) and fill spatial-hash as before
        std::vector<const VoxelLight *> rawLights;
        chunkManager->forEachEmissiveChunk([this, &rawLights](Chunk *chunk) {
            for (auto &light: chunk->emissiveLights) {
                // coarse bucket size (same as before)
                ChunkCoord gridCell{
                        (int) std::floor(light.pos.x / (DIM * 2)),
                        (int) std::floor(light.pos.y / (DIM * 2)),
                        (int) std::floor(light.pos.z / (DIM * 2))
                };
                lightSpatialHash[gridCell].push_back(&light);
                rawLights.push_back(&light);
            }
        });

        // 2) If no lights, done
        if (rawLights.empty()) {
            std::cout << "Light spatial hash updated: 0 grid cells; 0 emissive blobs\n";
            return;
        }

        // 3) Simple greedy clustering (merge lights that are spatially close)
        // Tune this merge radius. Using CHUNK_SIZE * 1.5 means lights that spill over
        // into adjacent chunks are folded into a single logical emitter.
        const float MERGE_RADIUS = DIM * 12.0f;
        const float MERGE_RADIUS_SQ = MERGE_RADIUS * MERGE_RADIUS;

        std::vector<char> used(rawLights.size(), 0);
        for (size_t i = 0; i < rawLights.size(); ++i) {
            if (used[i]) continue;
            used[i] = 1;

            // accumulate weighted by intensity (so stronger lights dominate)
            float totalIntensity = 0.0f;
            glm::vec3 accumPos(0.0f);
            glm::vec3 accumColor(0.0f);
            uint32_t mergedId = rawLights[i]->id; // base id
            int amountMerged = 0;

            // merge any other lights that lie within MERGE_RADIUS of rawLights[i]
            for (size_t j = i; j < rawLights.size(); ++j) {
                if (used[j]) continue;
                float d2 = glm::dot(rawLights[i]->pos - rawLights[j]->pos, rawLights[i]->pos - rawLights[j]->pos);
                if (d2 <= MERGE_RADIUS_SQ) {
                    used[j] = 1;
                    const VoxelLight *L = rawLights[j];
                    float w = glm::max(1.0f, L->intensity); // weight (avoid zero)
                    accumPos += L->pos * w;
                    accumColor += L->color * w;
                    totalIntensity += L->intensity;
                    amountMerged++;
                    // You can combine ids in a deterministic way if needed; keep first for now
                }
            }

            // construct merged light (fallbacks)
            VoxelLight merged;
            if (totalIntensity > 0.0f) {
                merged.intensity = (totalIntensity / amountMerged);
                merged.pos = accumPos / (totalIntensity > 0.0f ? totalIntensity : 1.0f);
                merged.color = accumColor / (totalIntensity > 0.0f ? totalIntensity : 1.0f);
            } else {
                // fallback: single entry (should not typically occur)
                merged = *rawLights[i];
            }
            merged.id = mergedId;

            // store into pool and push pointer into flat list
            mergedEmissiveLightPool.push_back(merged);
        }

        // 4) Build final flat list of pointers into mergedEmissiveLightPool
        flatEmissiveLightList.reserve(mergedEmissiveLightPool.size());
        for (auto &m: mergedEmissiveLightPool) {
            flatEmissiveLightList.push_back(&m);
        }

        std::cout << "Light spatial hash updated: " << lightSpatialHash.size()
                  << " grid cells; raw=" << rawLights.size()
                  << " merged=" << mergedEmissiveLightPool.size() << " emissive blobs\n";
    }

    void Game::rebuildChunkLights(const ChunkCoord &coord) {
        Chunk *chunk = chunkManager->getChunk(coord);
        if (!chunk) return;

        chunk->emissiveLights.clear();

        // chunk origin in world units
        glm::vec3 chunkOrigin(
                coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
        );

        // Cluster emissive voxels within this chunk
        glm::vec3 sumPos(0.0f);
        glm::vec3 sumColor(0.0f);
        int count = 0;

        for (int x = 0; x <= CHUNK_SIZE; ++x) {
            for (int y = 0; y <= CHUNK_SIZE; ++y) {
                for (int z = 0; z <= CHUNK_SIZE; ++z) {
                    const auto &vox = chunk->voxels[x][y][z];
                    if (vox.type == 2) { // Fire / emissive voxel
                        glm::vec3 voxelWorldPos = chunkOrigin + glm::vec3((float)x, (float)y, (float)z) * gl3::VOXEL_SIZE;
                        sumPos += voxelWorldPos;
                        sumColor += vox.color;
                        ++count;
                    }
                }
            }
        }

        if (count > 0) {
            VoxelLight light;
            light.pos = sumPos / float(count);
            light.color = sumColor / float(count);
            // Tune intensity scaling if VOXEL_SIZE changes (it affects perceived scale)
            light.intensity = float(count) * 35.0f;
            light.id = makeLightID(coord.x, coord.y, coord.z);

            chunk->emissiveLights.push_back(light);
        }

        chunk->lightingDirty = false;
    }

    // New updateChunkLights: select up to MAX_LIGHTS by score = intensity / (distSq + 1)
    // This prefers strong lights even at larger distance, while still penalizing distance.
    void Game::updateChunkLights(Chunk *chunk) {
        chunk->gpuCache.nearbyLights.clear();
        chunk->gpuCache.nearbyLights.reserve(MAX_LIGHTS);

        glm::vec3 chunkOrigin(
                chunk->coord.x * DIM,
                chunk->coord.y * DIM,
                chunk->coord.z * DIM
        );
        glm::vec3 chunkCenter = chunkOrigin + glm::vec3(DIM * 0.5f);

        // Fast path
        if (flatEmissiveLightList.empty()) {
            chunk->gpuCache.lastLightUpdateFrame = frameCounter;
            return;
        }

        const float radiusSq = LIGHT_RADIUS_SQ;
        const int K = MAX_LIGHTS;

        // small stack arrays (K is tiny)
        std::array<const VoxelLight *, 8> bestPtrs{};   // pointer candidates
        std::array<float, 8> bestScore{};              // score = intensity / (distSq + 1)
        int bestCount = 0;

        // keep track of the current worst score index (min score)
        float worstScore = std::numeric_limits<float>::infinity();
        int worstIndex = -1;

        for (const VoxelLight *light: flatEmissiveLightList) {
            glm::vec3 d = light->pos - chunkCenter;
            float distSq = glm::dot(d, d);

            if (distSq > radiusSq) continue; // skip out-of-range lights

            // Score uses shader-like falloff (intensity divided by squared distance + 1)
            // +1 avoids division by zero and keeps near-zero distance finite
            float score = light->intensity / (distSq + 1.0f);

            if (bestCount < K) {
                // append
                bestPtrs[bestCount] = light;
                bestScore[bestCount] = score;
                ++bestCount;

                // find new worst
                worstScore = bestScore[0];
                worstIndex = 0;
                for (int i = 1; i < bestCount; ++i) {
                    if (bestScore[i] < worstScore) {
                        worstScore = bestScore[i];
                        worstIndex = i;
                    }
                }
            } else {
                // replace worst if this one has a higher score
                if (score > worstScore) {
                    bestPtrs[worstIndex] = light;
                    bestScore[worstIndex] = score;

                    // recompute worst (small K)
                    worstScore = bestScore[0];
                    worstIndex = 0;
                    for (int i = 1; i < K; ++i) {
                        if (bestScore[i] < worstScore) {
                            worstScore = bestScore[i];
                            worstIndex = i;
                        }
                    }
                }
            }
        }

        // Move found lights into chunk->gpuCache.nearbyLights sorted by descending score
        if (bestCount > 0) {
            std::vector<int> idx(bestCount);
            for (int i = 0; i < bestCount; ++i) idx[i] = i;
            // sort so highest score first
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return bestScore[a] > bestScore[b];
            });

            for (int i = 0; i < bestCount; ++i) {
                chunk->gpuCache.nearbyLights.push_back(const_cast<VoxelLight *>(bestPtrs[idx[i]]));
            }
        }

        chunk->gpuCache.lastLightUpdateFrame = frameCounter;
    }


    void Game::processEmissiveChunks() {
        chunkManager->forEachEmissiveChunk([this](Chunk *chunk) {
            // Process only emissive chunks
            if (chunk->lightingDirty) {
                rebuildChunkLights(chunk->coord);
            }

            // Collect emissive lights for billboards
            for (const auto &light: chunk->emissiveLights) {
                if (usedLightIDs.insert(light.id).second) {
                    SunInstance inst;
                    inst.position = light.pos;
                    inst.scale = std::sqrt(light.intensity)/VOXEL_SIZE * 0.5f;
                    inst.color = light.color * 2.5f;
                    emissiveBillboards.push_back(inst);
                }
            }
        });
    }

    uint32_t Game::makeLightID(int cx, int cy, int cz) {
        // Simple hash function for light ID
        return ((cx & 0xFFF) << 20) | ((cy & 0xFFF) << 8) | (cz & 0xFF);
    }


////----Input Code------------------------------------------------------------------------------------------------------------------------------

    void Game::update() {
            input.update(window);
            actions.update(input);

            // Now use clean, readable input checks
            if (actions["Escape"].wasJustPressed) {
                glfwSetWindowShouldClose(window, true);
            }

            if (actions["ToggleDebug"].wasJustPressed) {
                DebugMode1 = !DebugMode1;
                std::cout << "Debug mode: " << (DebugMode1 ? "ON" : "OFF") << "\n";

                // Optional: Update shaders like before
                if (DebugMode1) {
                    voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel_debug.frag");
                    activeDebugMode = 0;
                } else {
                    DebugMode2=false;
                    voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
                }
            }

        if (actions["DebugMode1"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 1;
        }
        if (actions["DebugMode2"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 2;
        }
        if (actions["DebugMode3"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 3;
        }
        if (actions["DebugMode4"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 4;
        }
        if (actions["DebugMode5"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 5;
        }
        if (actions["DebugMode6"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 6;
        }
        if (actions["Wireframe"].wasJustPressed&&DebugMode1) {
            DebugMode2=!DebugMode2;
        }

        if (actions["CastSphere"].wasJustPressed) {
            std::cout << "Sphere Spell Triggered\n";
            RayCastResult hit = rayCastFromCamera(5.0f);
            glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                                    (cameraPos + getCameraFront() * 35.0f);

            // Cast spell with physics enabled
            float spellRadius = 4.0f * VOXEL_SIZE;  // Adjust size
            float spellStrength = 4.0f;              // Affects velocity

            castSpellSphere(spellCenter, spellRadius, 0, spellStrength);
        }

        if (actions["CastWall"].wasJustPressed) {
            std::cout << "Wall Spell Triggered" << "\n";
            RayCastResult hit = rayCastFromCamera(250.0f);
            glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                                    (cameraPos + getCameraFront() * 10.0f);

            // Get camera direction for wall orientation
            glm::vec3 cameraFront = getCameraFront();

            // Wall dimensions (tune these values)
            float wallWidth = 1.0f*VOXEL_SIZE;    // Horizontal width
            float wallHeight = 0.5f*VOXEL_SIZE;   // Vertical height
            float wallThickness = 2.0f*VOXEL_SIZE; // How thick the wall is

            // Cast the wall spell
            castSpellWall(spellCenter, glm::vec3(0,0,1),
                          wallWidth, wallHeight, wallThickness,
                          0, 2.0f*VOXEL_SIZE);
        }
        if (actions["AirReset"].wasJustPressed) {
            std::cout << "Platform Spell Triggered" << "\n";
            glm::vec3 spellCenter =(cameraPos + glm::vec3(0,-1,0) * 30.0f*VOXEL_SIZE);

            // Wall dimensions (tune these values)
            float wallWidth = 0.05f*VOXEL_SIZE;    // Horizontal width
            float wallHeight = 0.05f*VOXEL_SIZE;   // Vertical height
            float wallThickness = 3.0f*VOXEL_SIZE; // How thick the wall is

            // Cast the wall spell
            castSpellWall(spellCenter, glm::vec3(0,-1,0),
                          wallWidth, wallHeight, wallThickness,
                          0, 7.0f*VOXEL_SIZE);
        }


            // Character movement - perfect for your controller

            // Update camera to follow character
            updateCamera();

        // Update dynamic chunks
        // chunkManager->forEachDynamicChunk([this](Chunk* chunk) {
        // Physics, animation, etc.
        //});

        // Update fluid chunks
        //chunkManager->forEachFluidChunk([this](Chunk* chunk) {
        // Fluid simulation
        //});
    }


    void Game::handleCameraInput() {
        float speed = 20.0f*VOXEL_SIZE * deltaTime;

        /*
        // WASD movement (relative to view)
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += speed * getCameraFront();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= speed * getCameraFront();
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(getCameraFront(), glm::vec3(0, 1, 0))) * speed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(getCameraFront(), glm::vec3(0, 1, 0))) * speed;

        // Vertical flight controls
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            cameraPos.y += speed;   // Fly up
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            cameraPos.y -= speed;   // Fly down (or use GLFW_KEY_LEFT_SHIFT if you prefer)
*/

        // Mouse rotation
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        static double lastX = xpos, lastY = ypos;

        float sensitivity = 0.1f;
        float dx = (xpos - lastX) * sensitivity;
        float dy = (lastY - ypos) * sensitivity; // invert Y movement

        cameraRotation.y += dx; // yaw
        cameraRotation.x += dy; // pitch

        // Clamp pitch to avoid gimbal lock
        if (cameraRotation.x > 89.0f) cameraRotation.x = 89.0f;
        if (cameraRotation.x < -89.0f) cameraRotation.x = -89.0f;

        lastX = xpos;
        lastY = ypos;
    }


    glm::vec3 Game::getCameraFront() const {
        glm::vec3 front = glm::vec3(
                cos(glm::radians(cameraRotation.y)) * cos(glm::radians(cameraRotation.x)),
                sin(glm::radians(cameraRotation.x)),
                sin(glm::radians(cameraRotation.y)) * cos(glm::radians(cameraRotation.x))
        );
        if (glm::length(front) < 0.001f) front = glm::vec3(0, 0, -1);
        return glm::normalize(front);

    }


////----Rendering Code--------------------------------------------------------------------------------------------------------------------------

//------General Rendering-Code------------------------------------------------------------------------------------------------------------------
    void Game::renderSkybox() {
        // Save state
        GLint oldDepthFunc;
        glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
        GLboolean depthMask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);

        // Set skybox-specific state
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE); // Don't write to depth buffer (skybox is always at far plane)

        skyboxShader->use();

        // Set uniforms
        skyboxShader->setFloat("time", glfwGetTime());

        float aspect = (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));

        skyboxShader->setMatrix("projection", projection);
        skyboxShader->setMatrix("view", view);

        // Bind noise texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cubemapTexture);
        skyboxShader->setInt("noiseTexture", 0);

        // Draw skybox
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        // Restore state
        glDepthFunc(oldDepthFunc);
        glDepthMask(depthMask);
    }

    void Game::renderChunks() {
        int built = 0;
        int lighted = 0;
        int culledChunks=0;

        std::cout<<"\nFrame:"<<(frameCounter-29)<<"\n";
        if (DebugMode2) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        frameCounter++;

        // Clear per-frame billboards
        emissiveBillboards.clear();
        usedLightIDs.clear();

        // Get camera chunk coordinates
        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);
        const int totalChunks = (2 * RenderingRange + 1) * (2 * RenderingRange + 1) * (2 * RenderingRange + 1);

        int renderedChunks = 0;
        int meshRegens = 0;
        int totalChecked = 0;


        // STEP 1: Update global light spatial hash (replaces grid)
        if (frameCounter % 30 == 0) { // Update every 30 frames
            CpuTimer t7("\n Update Light Spacial Hash");
            updateLightSpatialHash();
        }

        // STEP 2: First, generate meshes for dirty chunks
        const int renderRadius = RenderingRange;
        //if(DebugMode1) {CpuTimer t8("Generate Meshes");}
        for (int cx = camCX - renderRadius; cx <= camCX + renderRadius; ++cx) {
            for (int cy = camCY - renderRadius; cy <= camCY + renderRadius; ++cy) {
                for (int cz = camCZ - renderRadius; cz <= camCZ + renderRadius; ++cz) {
                    totalChecked++;
                    ChunkCoord coord{cx, cy, cz};
                    if(!isChunkVisible(coord))
                    {
                        culledChunks++;
                        continue;
                    }
                    Chunk *chunk = chunkManager->getChunk(coord);

                    if (!chunk) continue;
                    if (!hasSolidVoxels(*chunk)) continue;

                    // Regenerate mesh if dirty
                    if (built < MAX_CHUNKS_PER_FRAME&&chunk->meshDirty|| built < MAX_CHUNKS_PER_FRAME&&!chunk->gpuCache.isValid) {
                        built++;
                        if(built<=1&&DebugMode1)
                        {
                            CpuTimer t5("generateChunkMesh");
                            generateChunkMesh(chunk);
                        }
                        else{
                        generateChunkMesh(chunk);
                        }
                        if(built<=1) {
                            std::cout << "approximate Mesh Progress: "
                                      << ((float) (frameCounter - 29) * MAX_CHUNKS_PER_FRAME / FilledChunks) * 100.0f
                                      << "% \n";
                        }
                        meshRegens++;
                    }

                    // Rebuild lighting if dirty
                    if (lighted<MAX_CHUNKS_PER_FRAME&&chunk->lightingDirty) {
                        lighted++;
                        if(lighted==1&&DebugMode1)
                        {
                            CpuTimer t4("rebuildChunkLights");
                            rebuildChunkLights(coord);
                        }
                        else {
                            rebuildChunkLights(coord);
                        }
                        if(lighted<=1) {
                            std::cout << "approximate Lighting Progress: "
                                    <<((float) (frameCounter - 29) * MAX_CHUNKS_PER_FRAME / FilledChunks) * 100.0f
                                    << "% \n";
                        }
                        chunk->lightingDirty = false;
                    }

                }
            }
        }
        if(DebugMode1) {std::cout << "Mesh generation: culled " << culledChunks << " of " << totalChecked
                  << " chunks (" << (100.0f * culledChunks / totalChecked) << "% culled)\n";}
        culledChunks = 0;
        totalChecked = 0;
        // STEP 3: Now RENDER all visible chunks
        voxelShader->use();  // Activate RENDER shader

        // Set common uniforms that don't change per chunk

        voxelShader->setMatrix("model", glm::mat4(1.0f));
        voxelShader->setMatrix("mvp", pv);
        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.02f));

// or show signed N·L for the strongest light (index 0)
        if (DebugMode1) {
            voxelShader->setInt("debugMode", activeDebugMode % 5);
        } else {
            voxelShader->setFloat("emission", 0.0f);
        }

        std::vector<Chunk> emissiveChunks;
        chunkManager->forEachEmissiveChunk([this, &emissiveChunks](Chunk *chunk) {
                emissiveChunks.push_back(*chunk);
        });

        //if(DebugMode1) {CpuTimer t6("ChunkLighting");}
        for (int cx = camCX - renderRadius; cx <= camCX + renderRadius; ++cx) {
            for (int cy = camCY - renderRadius; cy <= camCY + renderRadius; ++cy) {
                for (int cz = camCZ - renderRadius; cz <= camCZ + renderRadius; ++cz) {
                    totalChecked++;
                    ChunkCoord coord{cx, cy, cz};
                    if(!isChunkVisible(coord))
                    {
                        culledChunks++;
                        continue;
                    }
                    Chunk *chunk = chunkManager->getChunk(coord);
                    if (!chunk || !hasSolidVoxels(*chunk)) continue;

                    // Skip if no geometry
                    if (chunk->gpuCache.vertexCount == 0 || !chunk->gpuCache.isValid) {
                        continue;
                    }

                    // Update lights for this chunk (check if outdated)
                    if (frameCounter - chunk->gpuCache.lastLightUpdateFrame > LIGHT_UPDATE_INTERVAL ||
                        chunk->gpuCache.nearbyLights.empty()) {
                        updateChunkLights(chunk);
                    }

                    // Collect billboards from emissive chunks
                    for (const auto &light: chunk->emissiveLights) {
                        if (usedLightIDs.insert(light.id).second) {
                            SunInstance inst;
                            inst.position = light.pos;
                            inst.scale = std::sqrt(light.intensity) * 0.5f;
                            inst.color = light.color * 1.0f;
                            emissiveBillboards.push_back(inst);
                        }
                    }

                    //std::cout<<(chunk->gpuCache.nearbyLights.size())<<"\n";

                    // Set per-chunk uniforms (lights)
                    /*if((chunk->gpuCache.nearbyLights.size()>0)) {
                        std::cout << "lights exist for this chunk: "<< chunk->gpuCache.nearbyLights.size() << "\n";
                    }*/
                    bool skip=false;
                    for(int k=0; k<emissiveChunks.size();k++)
                    {
                        if(emissiveChunks.at(k).coord==chunk->coord)
                        {
                            skip=true;
                            continue;
                        }
                    }
                    if(skip)
                    {
                        continue;
                    }
                    int numLights = std::min((int) chunk->gpuCache.nearbyLights.size(), MAX_LIGHTS);
                    voxelShader->setInt("numLights", numLights);
                    for (int i = 0; i < numLights; ++i) {
                        const VoxelLight *light = chunk->gpuCache.nearbyLights[i];
                        voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", light->pos);
                        voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", light->color);
                        voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", light->intensity);
                    }

                    // For remaining light slots, set them to zero
                    for (int i = numLights; i < MAX_LIGHTS; ++i) {
                        voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", glm::vec3(0.0f));
                        voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", glm::vec3(0.0f));
                        voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", 0.0f);
                    }

                    // Draw from chunk's VAO
                    glBindVertexArray(chunk->gpuCache.vao);
                    glDrawArrays(GL_TRIANGLES, 0, chunk->gpuCache.vertexCount);
                    glBindVertexArray(0);

                    renderedChunks++;
                }
            }
        }

        if(DebugMode1) {
            std::cout << "Lighting/Rendering: culled " << culledChunks << " of " << totalChecked
                      << " chunks (" << (100.0f * culledChunks / totalChecked) << "% culled)\n";

            // Also add frustum culling statistics
            std::cout << "Frustum culling effectiveness: "
                      << (100.0f * (totalChunks - totalChecked + culledChunks) / totalChunks)
                      << "% of chunks were culled\n";
        }
        // Render billboards
        if (!emissiveBillboards.empty() && !DebugMode1) {
            sunBillboards.render(emissiveBillboards, view, projection, (float) glfwGetTime());
        }

        //std::cout << "Rendered " << renderedChunks << " chunks (Regenerated: " << meshRegens << ")\n";
    }


    void Game::renderAnimatedVoxels() {
        if (animatedVoxels.empty()) return;

        // Collect instances (include animating ones; optionally include arrived ones)
        std::vector<const AnimatedVoxel*> instances;
        instances.reserve(animatedVoxels.size());
        for (const auto &v : animatedVoxels) {
            if (v.isAnimating || v.hasArrived) instances.push_back(&v);
        }
        int instanceCount = (int)instances.size();
        if (instanceCount == 0) return;

        // Use the same fragment shader (lighting) but an instanced vertex shader.
        static Shader instancedShader(resolveAssetPath("shaders/voxel_anim.vert"),
                                      resolveAssetPath("shaders/voxel.frag"));
        instancedShader.use();

        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        instancedShader.setMatrix("pv", pv);
        instancedShader.setVec3("viewPos", cameraPos);
        instancedShader.setVec3("ambientColor", glm::vec3(0.02f)); // match your chunk ambient

        // Choose lights for animated voxels:
        // Use mergedEmissiveLightPool (already kept up to date) and pick up to MAX_LIGHTS
        int numLights = std::min((int)mergedEmissiveLightPool.size(), MAX_LIGHTS);
        instancedShader.setInt("numLights", numLights);
        for (int i = 0; i < numLights; ++i) {
            const VoxelLight &L = mergedEmissiveLightPool[i];
            instancedShader.setVec3("lightPos[" + std::to_string(i) + "]", L.pos);
            instancedShader.setVec3("lightColor[" + std::to_string(i) + "]", L.color);
            instancedShader.setFloat("lightIntensity[" + std::to_string(i) + "]", L.intensity);
        }
        // Zero out remaining lights
        for (int i = numLights; i < MAX_LIGHTS; ++i) {
            instancedShader.setVec3("lightPos[" + std::to_string(i) + "]", glm::vec3(0.0f));
            instancedShader.setVec3("lightColor[" + std::to_string(i) + "]", glm::vec3(0.0f));
            instancedShader.setFloat("lightIntensity[" + std::to_string(i) + "]", 0.0f);
        }

        // Build instance arrays
        std::vector<float> posScaleData; posScaleData.reserve(instanceCount * 4);
        std::vector<float> colorData;    colorData.reserve(instanceCount * 3);
        std::vector<float> normalData;   normalData.reserve(instanceCount * 3);

        for (int i = 0; i < instanceCount; ++i) {
            const AnimatedVoxel* v = instances[i];

            // optionally apply a small pulse scale
            float pulse = 1.0f;
            if (v->hasArrived) {
                pulse = 1.0f + 0.08f * std::sin(glfwGetTime() * 3.0f);
            } else {
                pulse = 1.0f + 0.16f * std::sin(glfwGetTime() * 8.0f + v->currentPos.x);
            }

            posScaleData.push_back(v->currentPos.x);
            posScaleData.push_back(v->currentPos.y);
            posScaleData.push_back(v->currentPos.z);
            posScaleData.push_back(pulse * 0.5f); // cube half-size; tweak to taste

            colorData.push_back(v->color.r);
            colorData.push_back(v->color.g);
            colorData.push_back(v->color.b);

            // Use stored per-voxel normal (computed at collection time)
            glm::vec3 n = v->normal; // fallback if normal missing
            // Prefer an explicit stored normal if available:
            // if AnimatedVoxel has 'normal' member, use that instead; below expects v->color exists.
            normalData.push_back(n.x);
            normalData.push_back(n.y);
            normalData.push_back(n.z);
        }

        // Create cube VAO + buffers (static so only created once)
        static GLuint cubeVAO = 0, cubeVBO = 0, cubeEBO = 0;
        static GLuint instPosScaleVBO = 0, instColorVBO = 0, instNormalVBO = 0;
        if (cubeVAO == 0) {
            // cube vertices (centered unit-cube)
            float vertices[] = {
                    -0.5f, -0.5f,  0.5f,
                    0.5f, -0.5f,  0.5f,
                    0.5f,  0.5f,  0.5f,
                    -0.5f,  0.5f,  0.5f,
                    -0.5f, -0.5f, -0.5f,
                    0.5f, -0.5f, -0.5f,
                    0.5f,  0.5f, -0.5f,
                    -0.5f,  0.5f, -0.5f
            };
            unsigned int indices[] = {
                    0,1,2, 2,3,0,
                    4,5,6, 6,7,4,
                    4,0,3, 3,7,4,
                    1,5,6, 6,2,1,
                    3,2,6, 6,7,3,
                    4,5,1, 1,0,4
            };

            glGenVertexArrays(1, &cubeVAO);
            glBindVertexArray(cubeVAO);

            glGenBuffers(1, &cubeVBO);
            glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glGenBuffers(1, &cubeEBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            // per-vertex aPos (location 0)
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

            // instance buffers
            glGenBuffers(1, &instNormalVBO);
            glBindBuffer(GL_ARRAY_BUFFER, instNormalVBO);
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glVertexAttribDivisor(1, 1);

            glGenBuffers(1, &instColorVBO);
            glBindBuffer(GL_ARRAY_BUFFER, instColorVBO);
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glVertexAttribDivisor(2, 1);

            glGenBuffers(1, &instPosScaleVBO);
            glBindBuffer(GL_ARRAY_BUFFER, instPosScaleVBO);
            glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
            glVertexAttribDivisor(3, 1);

            glBindVertexArray(0);
        }

        // Upload instance buffers
        glBindBuffer(GL_ARRAY_BUFFER, instPosScaleVBO);
        glBufferData(GL_ARRAY_BUFFER, posScaleData.size() * sizeof(float), posScaleData.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, instColorVBO);
        glBufferData(GL_ARRAY_BUFFER, colorData.size() * sizeof(float), colorData.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, instNormalVBO);
        glBufferData(GL_ARRAY_BUFFER, normalData.size() * sizeof(float), normalData.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Render instanced cubes with voxel lighting
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        instancedShader.use();
        glBindVertexArray(cubeVAO);
        glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, instanceCount);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
    }

    void Game::renderPhysicsFormations() {
        if (activeSpells.empty()) return;


        voxelShader->use();
        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.02f));

        // Use merged emissive light pool for lighting
        int numLights = std::min((int)mergedEmissiveLightPool.size(), MAX_LIGHTS);
        voxelShader->setInt("numLights", numLights);

        for (int i = 0; i < numLights; ++i) {
            const VoxelLight& L = mergedEmissiveLightPool[i];
            voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", L.pos);
            voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", L.color);
            voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", L.intensity);
        }

        for (int i = numLights; i < MAX_LIGHTS; ++i) {
            voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", glm::vec3(0.0f));
            voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", glm::vec3(0.0f));
            voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", 0.0f);
        }

        static float time = 0;
        time += deltaTime;


        // Render each physics-enabled formation
        for (const auto& spell : activeSpells) {
            if (!spell.isPhysicsEnabled || !spell.physicsBody) continue;
            if (!spell.physicsMesh.isValid) {
                std::cout << "Physics mesh invalid for spell!\n";
                continue;
            }
            // Use stored position and orientation directly
            int currentChunkX = worldToChunk(spell.physicsBody->position.x);
            int currentChunkY = worldToChunk(spell.physicsBody->position.y);
            int currentChunkZ = worldToChunk(spell.physicsBody->position.z);

            glm::vec3 pos = glm::vec3(spell.physicsBody->position.x,spell.physicsBody->position.y,spell.physicsBody->position.z) ;
            glm::quat rot = spell.physicsBody->orientation;

            // Build model matrix
            glm::mat4 model = glm::translate(glm::mat4(1.0f), pos);
            //model *= glm::mat4_cast(rot);
            //model = glm::scale(model, glm::vec3(VOXEL_SIZE));  // Apply VOXEL_SIZE scaling

            voxelShader->setMatrix("model", model);
            voxelShader->setMatrix("mvp", pv * model);

            // Draw mesh
            glBindVertexArray(spell.physicsMesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, spell.physicsMesh.vertexCount);
            glBindVertexArray(0);
        }
    }

    void Game::renderFluidPlanets() {
        const int GRID_X = 128;
        const int GRID_Y = 128;
        const int GRID_Z = 128;
        const size_t FIELD_COUNT = size_t(GRID_X) * GRID_Y * GRID_Z;

        // PV matrix
        float aspect = windowHeight == 0 ? float(windowWidth) : float(windowWidth) / float(windowHeight);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), {0, 1, 0});
        glm::mat4 pv = projection * view;

        // Billboards
        std::vector<SunInstance> billboardInstances;
        //billboardInstances.reserve(fluidPlanets.size());

        // Clear field storage
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, fieldBitsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, FIELD_COUNT * sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, fieldBitsSSBO);

        // Color palette for fluid planets (blue/green)
        const glm::vec3 baseColors[5] = {
                {0.10f, 0.60f, 1.00f},
                {0.00f, 0.75f, 0.80f},
                {0.20f, 0.85f, 0.95f},
                {0.05f, 0.55f, 0.70f},
                {0.15f, 0.70f, 0.90f}
        };

        for (size_t i = 0; i < 1; i++) {
            //WorldPlanet &planet = fluidPlanets[i];

            // Pick blue/green color
            glm::vec3 planetColor = baseColors[i % 5];

            // Upload the 32³ voxel template (binding 0)
            //uploadVoxelChunk(fluidPlanetChunk, &planetColor);

            // -------------------------
            // Voxel splat into a field
            // -------------------------
            voxelSplatShader->use();
           // voxelSplatShader->setVec3("gridOrigin", planet.worldPos - 0.5f * glm::vec3(CHUNK_SIZE));
            voxelSplatShader->setFloat("voxelSize", 0.25f);
            voxelSplatShader->setFloat("influenceRadius", 10.0f);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);     // input
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, fieldBitsSSBO);  // output

            const int LOCAL = 8;
            int groups = (CHUNK_SIZE + LOCAL - 1) / LOCAL;
            glDispatchCompute(groups, groups, groups);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // -------------------------
            // Marching cubes
            // -------------------------
            marchingCubesShader->use();
            //marchingCubesShader->setVec3("gridOrigin", planet.worldPos - 0.5f * glm::vec3(GRID_X, GRID_Y, GRID_Z));
            marchingCubesShader->setFloat("voxelSize", 0.25f);

            resetAtomicCounter();
            //dispatchCompute();

            // -------------------------
            // Lighting (no emission)
            // -------------------------
            voxelShader->use();
            voxelShader->setInt("numLights", 0); // not emissive — no lights needed
            voxelShader->setVec3("ambientColor", glm::vec3(0.02f));
            voxelShader->setVec3("viewPos", cameraPos);
            voxelShader->setMatrix("model", glm::mat4(1.0f));
            voxelShader->setMatrix("mvp", pv);

            voxelShader->setFloat("emission", 100.0f);          // not emissive
            voxelShader->setVec3("emissionColor", planetColor);   // no glow in mesh
            voxelShader->setVec3("uniformColor", planetColor);

            //drawTriangles(*voxelShader);

            // -------------------------
            // Billboard glow (fluid look)
            // -------------------------
            SunInstance inst;
            //inst.position =
            //        planet.worldPos + (cameraPos - planet.worldPos) * 0.25f;

           // float r = (CHUNK_SIZE * 0.25f) * glm::length(planet.radius);
            //inst.scale = r * 2.5f;       // glow radius
            inst.color = planetColor;    // same as planet

            billboardInstances.push_back(inst);
        }

        // Draw all billboards
        if (!DebugMode1) {
            sunBillboards.render(billboardInstances, view, projection, float(glfwGetTime()));
        }
    }

//------Marching Cubes-Code---------------------------------------------------------------------------------------------------------------------

    void Game::generateChunkMesh(Chunk *chunk) {
        // Clean up old GPU resources if they exist
        if (chunk->gpuCache.vao != 0) {
            glDeleteVertexArrays(1, &chunk->gpuCache.vao);
            glDeleteBuffers(1, &chunk->gpuCache.vbo);
            chunk->gpuCache.vao = 0;
            chunk->gpuCache.vbo = 0;
        }

        if (chunk->gpuCache.triangleSSBO != 0) {
            glDeleteBuffers(1, &chunk->gpuCache.triangleSSBO);
            chunk->gpuCache.triangleSSBO = 0;
        }

        glm::vec3 chunkOrigin(
                chunk->coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
        );

        // PHASE 1: Use COMPUTE SHADER to generate mesh
        marchingCubesShader->use();  // Activate compute shader

        // Upload voxel data to SSBOs (uses member DIM)
        uploadVoxelChunk(*chunk, nullptr);

        // Reset counter and set compute shader uniforms (gridOrigin aligns with chunkOrigin)
        resetAtomicCounter();
        setComputeUniforms(chunkOrigin, glm::vec3(1.0f), *marchingCubesShader);

        // Determine cells per axis and buffer sizes using DIM
        const int cellsPerAxis = DIM - 1; // marching cubes operates on (voxelGridDim - 1) cells
        const size_t maxVertices =
                size_t(cellsPerAxis) * cellsPerAxis * cellsPerAxis * 5 * 3; // conservative upper bound
        const size_t triangleBufferSize = maxVertices * sizeof(OutVertex);

        // Create per-chunk triangle SSBO
        glGenBuffers(1, &chunk->gpuCache.triangleSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, triangleBufferSize, nullptr, GL_DYNAMIC_DRAW);

        // Bind SSBOs for compute shader (voxel SSBO already filled)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, chunk->gpuCache.triangleSSBO);

        // Dispatch compute work: groups based on number of cells (DIM-1)
        int groups = (cellsPerAxis + 7) / 8;
        glDispatchCompute(groups, groups, groups);

        // Ensure compute writes are visible to subsequent operations
        glMemoryBarrier(
                GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        // Read vertex count from the atomic counter SSBO (counter stores vertex count)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        unsigned int producedVertexCount = 0;
        void *counterPtr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (counterPtr) {
            producedVertexCount = *static_cast<unsigned int *>(counterPtr);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Store vertex count into the chunk cache
        chunk->gpuCache.vertexCount = producedVertexCount;

        // Optional debug read: map the triangle SSBO for the first N vertices (safe-read)
        if (producedVertexCount > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);
            size_t readCount = std::min<unsigned int>(producedVertexCount, 6u);
            void *vertMap = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, readCount * sizeof(OutVertex),
                                             GL_MAP_READ_BIT);
            if (vertMap) {
                // could inspect some values while debugging
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }

        // PHASE 2: Create VAO that reads directly from the triangle SSBO.
        // We set attribute sizes to 3 to match the vertex shader's vec3 inputs.
        glGenVertexArrays(1, &chunk->gpuCache.vao);
        glGenBuffers(1, &chunk->gpuCache.vbo); // vbo remains unused but created for cleanup parity

        glBindVertexArray(chunk->gpuCache.vao);

        // Use the triangle SSBO as the ARRAY_BUFFER for vertex fetch
        glBindBuffer(GL_ARRAY_BUFFER, chunk->gpuCache.triangleSSBO);

        constexpr GLsizei stride = sizeof(OutVertex);
        const void *posOffset = (void *) offsetof(OutVertex, pos);
        const void *normalOffset = (void *) offsetof(OutVertex, normal);
        const void *colorOffset = (void *) offsetof(OutVertex, color);

        // aPos (vec3 in shader) <- OutVertex.pos (vec4 in buffer, we read first 3 components)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, posOffset);

        // aNormal (vec3) <- OutVertex.normal
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, normalOffset);

        // aColor (vec3) <- OutVertex.color
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, colorOffset);

        // Unbind VAO/ARRAY_BUFFER
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        chunk->gpuCache.isValid = true;
        chunk->meshDirty = false;
    }


// debug version: if doColorByDensity==true, set per-voxel color from density (visualize SDF)
    void Game::uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor) {
        const int localDIM = DIM; // Should be CHUNK_SIZE + 2 for padding
        const size_t total = size_t(localDIM) * localDIM * localDIM;
        std::vector<CpuVoxel> voxels;
        voxels.resize(total);

        // We need to include a 1-voxel border from neighbors
        // Let's assume DIM = CHUNK_SIZE + 2 (one extra voxel on each side)
        for (int x = -1; x <= CHUNK_SIZE; ++x) {  // -1 to CHUNK_SIZE inclusive
            for (int y = -1; y <= CHUNK_SIZE; ++y) {
                for (int z = -1; z <= CHUNK_SIZE; ++z) {
                    // Map to SSBO index (0..localDIM-1)
                    int idxX = x + 1;
                    int idxY = y + 1;
                    int idxZ = z + 1;
                    int idx = idxX + idxY * localDIM + idxZ * localDIM * localDIM;

                    const Voxel* srcVoxel = nullptr;

                    // Check if we need to sample from neighbor
                    if (x == -1 || x == CHUNK_SIZE ||
                        y == -1 || y == CHUNK_SIZE ||
                        z == -1 || z == CHUNK_SIZE) {

                        // Get neighbor chunk
                        ChunkCoord neighborCoord = chunk.coord;
                        int localX = x;
                        int localY = y;
                        int localZ = z;

                        // Adjust for each axis
                        if (x == -1) {
                            neighborCoord.x -= 1;
                            localX = CHUNK_SIZE - 1;
                        } else if (x == CHUNK_SIZE) {
                            neighborCoord.x += 1;
                            localX = 0;
                        }

                        if (y == -1) {
                            neighborCoord.y -= 1;
                            localY = CHUNK_SIZE - 1;
                        } else if (y == CHUNK_SIZE) {
                            neighborCoord.y += 1;
                            localY = 0;
                        }

                        if (z == -1) {
                            neighborCoord.z -= 1;
                            localZ = CHUNK_SIZE - 1;
                        } else if (z == CHUNK_SIZE) {
                            neighborCoord.z += 1;
                            localZ = 0;
                        }

                        Chunk* neighbor = chunkManager->getChunk(neighborCoord);
                        if (neighbor && localX >= 0 && localX <= CHUNK_SIZE &&
                            localY >= 0 && localY <= CHUNK_SIZE &&
                            localZ >= 0 && localZ <= CHUNK_SIZE) {
                            srcVoxel = &neighbor->voxels[localX][localY][localZ];
                        }
                    }

                    // If no neighbor data, use current chunk or default
                    if (!srcVoxel) {
                        // Clamp to valid range for current chunk
                        int clampX = glm::clamp(x, 0, CHUNK_SIZE);
                        int clampY = glm::clamp(y, 0, CHUNK_SIZE);
                        int clampZ = glm::clamp(z, 0, CHUNK_SIZE);
                        srcVoxel = &chunk.voxels[clampX][clampY][clampZ];
                    }

                    // Copy data
                    voxels[idx].density = srcVoxel->density;
                    if (overrideColor) {
                        voxels[idx].color = glm::vec4(*overrideColor, 1.0f);
                    } else {
                        voxels[idx].color = glm::vec4(srcVoxel->color, 1.0f);
                    }
                }
            }
        }

        // Upload to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxels.size() * sizeof(CpuVoxel), voxels.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void Game::resetAtomicCounter() {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);
    }

    // Set compute shader uniforms consistently (use member DIM and chunkOrigin -> gridOrigin)
    void Game::setComputeUniforms(const glm::vec3 &chunkOrigin,
                                  const glm::vec3 & /*objectScale*/,
                                  Shader &computeShader) {
        computeShader.use();

        // tell compute shader the voxel size in world units
        const float voxelSize = gl3::VOXEL_SIZE;

        // voxelGridDim equals DIM (number of sample points along each axis)
        computeShader.setVec3("gridOrigin", chunkOrigin); // index (0,0,0) maps to chunkOrigin
        computeShader.setFloat("voxelSize", voxelSize);
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));
    }

//------Frag/Vert-Code--------------------------------------------------------------------------------------------------------------------------

    void Game::drawTriangles(Shader &voxelShader, Chunk *chunk) {
        if (!chunk || !chunk->gpuCache.isValid) return;

        // Ensure compute writes are visible to vertex fetch and buffer readbacks
        glMemoryBarrier(
                GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        // Read the triangle/vertex count from the atomic counter SSBO (binding 3)
        unsigned int triCount = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &triCount);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Each triangle has 3 vertices
        unsigned int vertexCount = triCount * 3u;
        // Store to chunk cache so renderChunks() can also use it
        chunk->gpuCache.vertexCount = vertexCount;

        if (vertexCount == 0) return;

        // Bind the chunk's triangle buffer as ARRAY_BUFFER (it was allocated as SSBO)
        glBindBuffer(GL_ARRAY_BUFFER, chunk->gpuCache.triangleSSBO);

        // Ensure VAO is set up to read from this buffer. If you created the VAO
        // at mesh generation time using this buffer, you can skip re-specifying attributes.
        // For robustness, we set them here (no harm if they're already set).
        constexpr GLsizei stride = sizeof(gl3::OutVertex);

        const void *posOffset = (void *) offsetof(gl3::OutVertex, pos);
        const void *normalOffset = (void *) offsetof(gl3::OutVertex, normal);
        const void *colorOffset = (void *) offsetof(gl3::OutVertex, color);

        glBindVertexArray(chunk->gpuCache.vao);

        // Match the vertex shader (aPos/aNormal/aColor are vec3) -> use size=3
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, posOffset);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, normalOffset);

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, colorOffset);

        // Use the voxel shader program
        voxelShader.use();

        // Draw the generated vertices
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei) vertexCount);

        // Unbind VAO/ARRAY_BUFFER to avoid accidental state leaks
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }


////----Helper Functions------------------------------------------------------------------------------------------------------------------------
    Game::RayCastResult Game::rayCastFromCamera(float maxDistance) {
        RayCastResult result;
        result.hit = false;

        glm::vec3 rayDir = getCameraFront();
        glm::vec3 rayOrigin = cameraPos;

        // Step through the ray
        float stepSize = 1.0f;
        float currentDist = 0.0f;

        while (currentDist < maxDistance) {
            glm::vec3 samplePos = rayOrigin + rayDir * currentDist;

            // Convert to chunk coordinates
            ChunkCoord coord;
            coord.x = worldToChunk(samplePos.x);
            coord.y = worldToChunk(samplePos.y);
            coord.z = worldToChunk(samplePos.z);

            Chunk* chunk = chunkManager->getChunk(coord);
            if (chunk) {
                // Convert world position to local chunk coordinates
                glm::vec3 chunkMin = getChunkMin(coord);
                glm::ivec3 localPos = glm::ivec3(
                        samplePos.x - chunkMin.x,
                        samplePos.y - chunkMin.y,
                        samplePos.z - chunkMin.z
                );

                // Check bounds
                if (localPos.x >= 0 && localPos.x <= CHUNK_SIZE &&
                    localPos.y >= 0 && localPos.y <= CHUNK_SIZE &&
                    localPos.z >= 0 && localPos.z <= CHUNK_SIZE) {

                    // Check if this voxel is solid
                    if (chunk->voxels[localPos.x][localPos.y][localPos.z].isSolid()) {
                        result.hitPosition = samplePos;
                        result.hitNormal = calculateNormalAt(chunk, localPos); // We'll implement this
                        result.distance = currentDist;
                        result.hit = true;
                        return result;
                    }
                }
            }

            currentDist += stepSize;
        }

        // If no hit, return a point at max distance along ray
        result.hitPosition = rayOrigin + rayDir * maxDistance;
        result.hitNormal = -rayDir; // Point back toward camera
        result.distance = maxDistance;
        result.hit = false;
        return result;
    }

    glm::vec3 Game::calculateNormalAt(Chunk* chunk, const glm::ivec3& pos) {
        // Simple central differences normal calculation
        if (pos.x <= 0 || pos.x >= CHUNK_SIZE ||
            pos.y <= 0 || pos.y >= CHUNK_SIZE ||
            pos.z <= 0 || pos.z >= CHUNK_SIZE) {
            return glm::vec3(0, 1, 0); // Fallback
        }

        float dx = chunk->voxels[pos.x+1][pos.y][pos.z].density -
                   chunk->voxels[pos.x-1][pos.y][pos.z].density;
        float dy = chunk->voxels[pos.x][pos.y+1][pos.z].density -
                   chunk->voxels[pos.x][pos.y-1][pos.z].density;
        float dz = chunk->voxels[pos.x][pos.y][pos.z+1].density -
                   chunk->voxels[pos.x][pos.y][pos.z-1].density;

        glm::vec3 normal(dx, dy, dz);
        if (glm::length(normal) > 0.0001f) {
            return glm::normalize(normal);
        }
        return glm::vec3(0, 1, 0);
    }

    // New: get mouse delta (single place to read cursor movement)
    glm::vec2 Game::getMouseDelta() {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        glm::dvec2 curPos(xpos, ypos);
        glm::vec2 delta(0.0f, 0.0f);

        if (!hasPreviousMousePos) {
            previousMousePos = curPos;
            hasPreviousMousePos = true;
            return delta;
        }

        glm::dvec2 d = curPos - previousMousePos;
        previousMousePos = curPos;

        // return as floats (x,y)
        delta.x = static_cast<float>(d.x);
        delta.y = static_cast<float>(d.y);
        return delta;
    }


    // -----------------------
    // SDF sampling helpers
    // -----------------------
    float Game::sampleDensityAtWorld(const glm::vec3 &worldPos) const {
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

    glm::vec3 Game::sampleNormalAtWorld(const glm::vec3 &worldPos) const {
        const float e = VOXEL_SIZE * 0.5f;
        float dx = sampleDensityAtWorld(worldPos + glm::vec3(e,0,0)) - sampleDensityAtWorld(worldPos - glm::vec3(e,0,0));
        float dy = sampleDensityAtWorld(worldPos + glm::vec3(0,e,0)) - sampleDensityAtWorld(worldPos - glm::vec3(0,e,0));
        float dz = sampleDensityAtWorld(worldPos + glm::vec3(0,0,e)) - sampleDensityAtWorld(worldPos - glm::vec3(0,0,e));
        glm::vec3 g(dx,dy,dz);
        float len = glm::length(g);
        if (len < 1e-6f) return glm::vec3(0.0f, 1.0f, 0.0f);
        // density increases INTO solid => gradient points inward; we want outward normal
        return -glm::normalize(g);
        }

    // -----------------------
    // Camera update with collision-safe offset behind player
    // -----------------------
    void Game::updateCamera() {
        // Get player head/eye position
        glm::vec3 headPos = characterController.getCameraPosition();
        // Desired camera offset behind the player (tweak this)
        const float cameraFollowDistance = 1.75f * VOXEL_SIZE; // how far behind the head
        const float cameraHeightOffset = 2.5f*VOXEL_SIZE;                // extra vertical offset if needed
        glm::vec3 viewDir = getCameraFront();
        glm::vec3 desiredCam = headPos - viewDir * cameraFollowDistance + glm::vec3(0.0f, cameraHeightOffset, 0.0f);
        // Camera collision params
        const float cameraRadius = 0.35f * VOXEL_SIZE; // radius of camera sphere
        const float skinWidth = 0.02f * VOXEL_SIZE;   // small gap to keep off surface
        const int steps = glm::max(4, (int)std::ceil(glm::length(desiredCam - headPos) / (VOXEL_SIZE * 0.25f)));
        // Walk from headPos outward toward desiredCam and find first penetration
        glm::vec3 safePos = headPos; // starts at head (should be non-penetrating)
        bool collided = false;
        glm::vec3 collidedNormal(0.0f);
        for (int i = 1; i <= steps; ++i) {
            float t = (float)i / (float)steps;
                    glm::vec3 samplePos = glm::mix(headPos, desiredCam, t);
                    float sdf = sampleDensityAtWorld(samplePos); // positive = inside solid
                    float sphereSigned = sdf - cameraRadius;     // positive => penetration
            if (sphereSigned > 0.0f) {
                            // penetration at this sample -- step back to previous safe sample and push out
                                    collided = true;
                            // previous sample (clamped)
                                    float prevT = (float)(i - 1) / (float)steps;
                            glm::vec3 prevPos = glm::mix(headPos, desiredCam, prevT);
                            // get normal at collision location (best attempt)
                                    glm::vec3 n = sampleNormalAtWorld(samplePos);
                            if (glm::length(n) < 1e-6f) n = glm::vec3(0,1,0);
                            collidedNormal = glm::normalize(n);
                            // place camera at prevPos plus offset OUT of surface
                                    safePos = prevPos + collidedNormal * (cameraRadius + skinWidth);
                            break;
                        } else {
                            safePos = samplePos; // this sample is safe; remember it
                        }
                }

                    // If we never collided, but desiredCam is safe then use desiredCam (we already set safePos along the loop)
                    // If we started inside geometry (rare), project outward using normal at headPos#//
                    if (!collided) {
                    float headSdf = sampleDensityAtWorld(safePos);
                    float headSphereSigned = headSdf - cameraRadius;
                    if (headSphereSigned > 0.0f) {
                            glm::vec3 n = sampleNormalAtWorld(safePos);
                            safePos = safePos + glm::normalize(n) * (headSphereSigned + skinWidth);
                        }
               }

                    // Smooth camera movement to reduce snapping (tweak lerp factor)
                            const float smoothingLerp = 0.55f; // 0 => no smoothing (immediate), <1 => smooth
            cameraPos = glm::mix(cameraPos, safePos, smoothingLerp);

                    // Finally update camera rotation based on cursor
                            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            static double lastX = xpos, lastY = ypos;

                    float sensitivity = 0.1f;
            float dx = (xpos - lastX) * sensitivity;
            float dy = (lastY - ypos) * sensitivity; // invert Y movement

                    cameraRotation.y += dx; // yaw
           cameraRotation.x += dy; // pitch

                    // Clamp pitch to avoid gimbal lock
                            if (cameraRotation.x > 89.0f) cameraRotation.x = 89.0f;
           if (cameraRotation.x < -89.0f) cameraRotation.x = -89.0f;

                    lastX = xpos;
            lastY = ypos;
        }

    void Game::setupSkybox() {
        // Create shader
        skyboxShader = std::make_unique<Shader>("shaders/skybox.vert", "shaders/skybox.frag");

        // Create the same noise texture you already have
        createNoiseTexture();

        // Setup cube vertices
        float skyboxVertices[] = {
                // positions
                -1.0f, 1.0f, -1.0f,
                -1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,

                -1.0f, -1.0f, 1.0f,
                -1.0f, -1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f, -1.0f, 1.0f,

                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,

                -1.0f, -1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, -1.0f, 1.0f,
                -1.0f, -1.0f, 1.0f,

                -1.0f, 1.0f, -1.0f,
                1.0f, 1.0f, -1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f, -1.0f,

                -1.0f, -1.0f, -1.0f,
                -1.0f, -1.0f, 1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                -1.0f, -1.0f, 1.0f,
                1.0f, -1.0f, 1.0f
        };

        glGenVertexArrays(1, &skyboxVAO);
        glGenBuffers(1, &skyboxVBO);

        // --- THIS PART WAS MISSING ---
        glBindVertexArray(skyboxVAO);
        glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindVertexArray(0);
        // --- END OF MISSING CODE ---
    }

    void Game::createNoiseTexture() {
        const int size = 128;
        std::vector<unsigned char> data(size * size * 3);

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int index = (y * size + x) * 3;

                // Simple random noise
                data[index] = rand() % 128;
                data[index + 1] = rand() % 128;
                data[index + 2] = rand() % 128;
            }
        }

        glGenTextures(1, &cubemapTexture);
        glBindTexture(GL_TEXTURE_2D, cubemapTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
}
