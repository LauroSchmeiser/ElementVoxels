#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/Shader.h"
#include "rendering/marchingTables.h"
#include "rendering/SunBillboard.h"


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
            : windowWidth(width), windowHeight(height),
              chunkManager(std::make_unique<MultiGridChunkManager>())  // Initialize manager
            , cameraPos(0.0f, 0.0f, 50.0f), cameraRotation(-90.0f, 0.0f) {
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
    }


    Game::~Game() {
        glfwTerminate();
    }


////-----Run-Method-----------------------------------------------------------------------------------------------------------------------------

    void Game::run() {
        ////Initialization-Steps
        CpuTimer t0("ssbos");
        setupSSBOsAndTables();
        setupCamera();
        CpuTimer t1("generateChunks");
        generateChunks();
        setupVEffects();

        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            ////Simulation Steps
            updateDeltaTime();
            updateSpells(deltaTime);
            //cleanupFinishedSpells();

            ////Input-Steps
            handleCameraInput();
            glfwPollEvents();
            update();


            ////Post-Prod Steps?

            ////Rendering Steps
            if(DebugMode1)
            {
                CpuTimer t2("renderChunks");
            }
            renderChunks();
            renderAnimatedVoxels();

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

    int Game::worldToChunk(float worldPos) const {
        return (int) std::floor(worldPos / CHUNK_SIZE);
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

    // Get chunk bounding box in world space
    glm::vec3 Game::getChunkMin(const ChunkCoord& coord) const {
        return glm::vec3(coord.x * CHUNK_SIZE,
                         coord.y * CHUNK_SIZE,
                         coord.z * CHUNK_SIZE);
    }

    glm::vec3 Game::getChunkMax(const ChunkCoord& coord) const {
        return glm::vec3((coord.x + 1) * CHUNK_SIZE,
                         (coord.y + 1) * CHUNK_SIZE,
                         (coord.z + 1) * CHUNK_SIZE);
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

    void Game::castGravityWellSpell(const glm::vec3& center, float radius,
                                    uint64_t targetMaterial, float strength) {
        // Create spell effect
        SpellEffect spell;
        spell.type = SpellEffect::Type::GRAVITY_WELL;
        spell.center = center;
        spell.radius = radius;
        spell.strength = strength;
        spell.targetMaterial = targetMaterial;

        // Step 1: Find voxels
        std::vector<AnimatedVoxel> visualVoxels;
        findNearbyVoxelsForVisual(center, radius, targetMaterial, visualVoxels, strength);

        if (visualVoxels.empty()) {
            std::cout << "No voxels found for spell\n";
            return;
        }

        // Step 2: Calculate formation properties
        glm::vec3 avgColor(0.0f);
        float formationRadius = std::cbrt(visualVoxels.size()) * 0.8f;

        // Calculate final positions on sphere immediately
        for (size_t i = 0; i < visualVoxels.size(); ++i) {
            avgColor += visualVoxels[i].color;

            // Calculate final position on sphere surface
            float angle = randomFloat(0.0f, glm::two_pi<float>());
            float height = randomFloat(-1.0f, 1.0f);
            float circleRadius = std::sqrt(1.0f - height * height);

            visualVoxels[i].targetPos = center + glm::vec3(
                    std::cos(angle) * circleRadius * formationRadius,
                    height * formationRadius,
                    std::sin(angle) * circleRadius * formationRadius
            );

            visualVoxels[i].animationSpeed = strength * 1.5f;
            visualVoxels[i].isAnimating = true;

            // Store index in spell
            spell.animatedVoxelIndices.push_back(animatedVoxels.size() + i);
        }
        avgColor /= visualVoxels.size();

        spell.formationColor = avgColor;
        spell.formationRadius = formationRadius;

        // Step 3: Add visual voxels to global list
        size_t startIndex = animatedVoxels.size();
        animatedVoxels.insert(animatedVoxels.end(), visualVoxels.begin(), visualVoxels.end());

        // Step 4: Add spell to active list
        activeSpells.push_back(spell);

        std::cout << "Spell cast! " << visualVoxels.size()
                  << " voxels moving directly to formation\n";
    }

    void Game::findNearbyVoxelsForVisual(const glm::vec3& center, float radius,
                                         uint64_t targetMaterial,
                                         std::vector<AnimatedVoxel>& results,
                                         float strength) {
        const float radiusSq = radius * radius;
        int maxVoxels = static_cast<int>(strength * 50);

        auto chunks = chunkManager->getChunksInRadius(center, radius);

        // First, collect ALL potential voxels with their distances
        struct VoxelCandidate {
            glm::vec3 worldPos;
            glm::vec3 color;
            ChunkCoord chunkCoord;
            glm::ivec3 localPos;
            float distanceSq;
            Chunk* chunk; // Pointer to chunk
        };

        std::vector<VoxelCandidate> candidates;

        // Collect all candidates
        for (const auto& [coord, chunk] : chunks) {
            if (!chunk || !hasSolidVoxels(*chunk)) continue;

            glm::vec3 chunkMin = getChunkMin(coord);

            for (int x = 0; x <= CHUNK_SIZE; ++x) {
                for (int y = 0; y <= CHUNK_SIZE; ++y) {
                    for (int z = 0; z <= CHUNK_SIZE; ++z) {
                        const Voxel& voxel = chunk->voxels[x][y][z];

                        if (voxel.isSolid() && voxel.material == targetMaterial) {
                            glm::vec3 worldPos = chunkMin + glm::vec3(x, y, z);
                            glm::vec3 diff = worldPos - center;
                            float distSq = glm::dot(diff, diff);

                            if (distSq <= radiusSq) {
                                VoxelCandidate candidate;
                                candidate.worldPos = worldPos;
                                candidate.color = voxel.color;
                                candidate.chunkCoord = coord;
                                candidate.localPos = glm::ivec3(x, y, z);
                                candidate.distanceSq = distSq;
                                candidate.chunk = chunk;

                                candidates.push_back(candidate);
                            }
                        }
                    }
                }
            }
        }

        // Sort candidates by distance (closest first)
        std::sort(candidates.begin(), candidates.end(),
                  [](const VoxelCandidate& a, const VoxelCandidate& b) {
                      return a.distanceSq < b.distanceSq;
                  });

        // Take only the closest N voxels
        int takeCount = std::min(maxVoxels, static_cast<int>(candidates.size()));

        for (int i = 0; i < takeCount; ++i) {
            const auto& candidate = candidates[i];

            AnimatedVoxel animVoxel;
            animVoxel.currentPos = candidate.worldPos;
            animVoxel.color = candidate.color;
            animVoxel.isAnimating = true;
            animVoxel.animationSpeed = strength * 1.5f;
            animVoxel.hasArrived = false;

            // Create crater at this voxel position
            createExteriorSmoothCrater(candidate.chunk, candidate.localPos, candidate.worldPos);

            // Mark chunk as modified
            candidate.chunk->meshDirty = true;
            candidate.chunk->lightingDirty = true;
            markChunkModified(candidate.chunkCoord);

            results.push_back(animVoxel);
        }

        // Debug output
        if (takeCount > 0) {
            float minDist = std::sqrt(candidates[0].distanceSq);
            float maxDist = std::sqrt(candidates[takeCount-1].distanceSq);
            std::cout << "Taking " << takeCount << " closest voxels (distances: "
                      << minDist << " to " << maxDist << " units)\n";
        }
    }

    void Game::createExteriorSmoothCrater(Chunk* chunk, const glm::ivec3& voxelPos,
                                          const glm::vec3& worldPos) {
        float craterRadius = 2.0f;
        float maxCraterDepth = 2.5f;

        int range = static_cast<int>(std::ceil(craterRadius));

        for (int dx = -range; dx <= range; ++dx) {
            for (int dy = -range; dy <= range; ++dy) {
                for (int dz = -range; dz <= range; ++dz) {
                    int nx = voxelPos.x + dx;
                    int ny = voxelPos.y + dy;
                    int nz = voxelPos.z + dz;

                    if (nx < 0 || nx > CHUNK_SIZE || ny < 0 || ny > CHUNK_SIZE || nz < 0 || nz > CHUNK_SIZE) {
                        continue;
                    }

                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
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

    void Game::createSpellFormation(const glm::vec3& center, float radius,
                                    float strength, uint64_t material,
                                    const glm::vec3& color) {
        // Create a new "planet" at the spell center
        WorldPlanet newFormation;
        newFormation.worldPos = center;
        newFormation.radius = radius;
        newFormation.color = color;
        newFormation.type = 1; // Solid

        // NEW: Ensure all affected chunks are loaded BEFORE carving
        int minCX = worldToChunk(center.x - radius);
        int maxCX = worldToChunk(center.x + radius);
        int minCY = worldToChunk(center.y - radius);
        int maxCY = worldToChunk(center.y + radius);
        int minCZ = worldToChunk(center.z - radius);
        int maxCZ = worldToChunk(center.z + radius);

        // Load all chunks first
        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};

                    // Force chunk creation if it doesn't exist
                    if (!chunkManager->getChunk(coord)) {
                        chunkManager->addChunk(coord, VoxelCategory::DYNAMIC);
                        Chunk* chunk = chunkManager->getChunk(coord);
                        if (chunk) {
                            // Initialize chunk properly
                            chunk->coord = coord;
                            chunk->clear();
                        }
                    }
                }
            }
        }

        // Now carve into chunks
        carveFormationIntoChunks(newFormation, material);

        // NEW: Force immediate mesh regeneration for affected chunks
        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (chunk) {
                        // Force immediate mesh generation (bypassing MAX_CHUNKS_PER_FRAME)
                        if (chunk->meshDirty) {
                            generateChunkMesh(chunk);
                        }
                    }
                }
            }
        }
    }


    void Game::carveFormationIntoChunks(const WorldPlanet& formation, uint64_t material) {
        // Determine which chunks this formation affects
        int minCX = worldToChunk(formation.worldPos.x - formation.radius);
        int maxCX = worldToChunk(formation.worldPos.x + formation.radius);
        int minCY = worldToChunk(formation.worldPos.y - formation.radius);
        int maxCY = worldToChunk(formation.worldPos.y + formation.radius);
        int minCZ = worldToChunk(formation.worldPos.z - formation.radius);
        int maxCZ = worldToChunk(formation.worldPos.z + formation.radius);

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

                    // Carve sphere into chunk
                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz);
                                float dist = glm::distance(worldPos, formation.worldPos);

                                float formationDensity = formation.radius - dist;

                                float existingDensity = chunk->voxels[lx][ly][lz].density;

                                // SDF UNION: Take the MAXIMUM density
                                if (formationDensity > existingDensity) {
                                    chunk->voxels[lx][ly][lz].density = formationDensity;

                                    // Set type and color
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
    }

    void Game::updateSpells(float deltaTime) {
        for (auto spellIt = activeSpells.begin(); spellIt != activeSpells.end(); ) {
            // NEW: Track newly arrived voxels per frame
            int arrivedThisFrame = 0;
            int totalVoxels = spellIt->animatedVoxelIndices.size();

            for (size_t idx : spellIt->animatedVoxelIndices) {
                if (idx < animatedVoxels.size()) {
                    AnimatedVoxel& voxel = animatedVoxels[idx];
                    if (voxel.isAnimating) {
                        glm::vec3 toTarget = voxel.targetPos - voxel.currentPos;
                        float distance = glm::length(toTarget);

                        if (distance < 0.35f) {
                            voxel.isAnimating = false;
                            arrivedThisFrame++;
                        } else {
                            glm::vec3 dir = glm::normalize(toTarget);
                            float speed = voxel.animationSpeed;
                            float slowdown = glm::clamp(distance / 3.0f, 0.2f, 1.0f);
                            voxel.velocity = dir * speed * slowdown;
                            voxel.currentPos += voxel.velocity * deltaTime;
                        }
                    }
                }
            }

            // NEW: Create geometry immediately when ANY voxels arrive
            if (!spellIt->geometryCreated) {
                int totalArrived = 0;
                for (size_t idx : spellIt->animatedVoxelIndices) {
                    if (idx < animatedVoxels.size() && !animatedVoxels[idx].isAnimating) {
                        totalArrived++;
                    }
                }

                // Create formation when at least 25% have arrived (or after 2 seconds)
                static std::unordered_map<size_t, float> spellTimers;
                size_t spellId = std::distance(activeSpells.begin(), spellIt);

                if (!spellTimers.count(spellId)) {
                    spellTimers[spellId] = 0.0f;
                }
                spellTimers[spellId] += deltaTime;

                float arrivalRatio = (float)totalArrived / totalVoxels;

                // Create if enough voxels arrived OR if timer exceeded
                if (arrivalRatio >= 0.25f) {
                    std::cout << "Creating final geometry for spell ("
                              << totalArrived << "/" << totalVoxels
                              << " voxels arrived after " << spellTimers[spellId] << "s)\n";

                    createSpellFormation(spellIt->center, spellIt->formationRadius,
                                         spellIt->strength, spellIt->targetMaterial,
                                         spellIt->formationColor);
                    spellIt->geometryCreated = true;

                    // Remove timer
                    spellTimers.erase(spellId);
                }
            }

            // Check if spell is complete
            if (spellIt->geometryCreated) {
                int stillAnimating = 0;
                for (size_t idx : spellIt->animatedVoxelIndices) {
                    if (idx < animatedVoxels.size() && animatedVoxels[idx].isAnimating) {
                        stillAnimating++;
                    }
                }

                if (stillAnimating == 0) {
                    spellIt = activeSpells.erase(spellIt);
                    continue;
                }
            }

            ++spellIt;
        }

        // Clean up non-animating voxels
        animatedVoxels.erase(
                std::remove_if(animatedVoxels.begin(), animatedVoxels.end(),
                               [](const AnimatedVoxel& v) { return !v.isAnimating; }),
                animatedVoxels.end()
        );
    }

    void Game::createPartialFormation(const SpellEffect& spell, float completionRatio) {
        // Create a weaker/smaller version of the formation
        // This shows the geometry building up gradually

        WorldPlanet partialFormation;
        partialFormation.worldPos = spell.center;

        // Scale radius based on completion
        partialFormation.radius = spell.formationRadius * (completionRatio * 0.7f + 0.3f);

        // Fade in color based on completion
        partialFormation.color = spell.formationColor * glm::vec3(completionRatio * 0.8f + 0.2f);
        partialFormation.type = 1;

        // Temporarily carve partial formation (will be overwritten by final)
        carveFormationIntoChunks(partialFormation, spell.targetMaterial);
    }

    void Game::cleanupFinishedSpells() {
        // This is now redundant since we clean up in updateSpells
        // But keep it for safety
        for (auto voxelIt = animatedVoxels.begin(); voxelIt != animatedVoxels.end(); ) {
            if (!voxelIt->isAnimating) {
                voxelIt = animatedVoxels.erase(voxelIt);
            } else {
                ++voxelIt;
            }
        }
    }

    float Game::randomFloat(float min, float max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        return min + dis(gen) * (max - min);
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

    void Game::setupCamera() {
        // --- Camera setup ---
        cameraPos = glm::vec3(0.0f, 0.0f, 80.0f);
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

        int lavaCount = 2 + (rng() % 5);
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

                        glm::vec3 chunkOrigin(cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE);
                        bool chunkTouched = false;

                        // Carve sphere into chunk - FIXED: Use SDF union operation
                        for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                            for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                                for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                    glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz);
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

            accumulator -= fixedTimeStep;
        }
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

        glm::vec3 chunkOrigin(coord.x * CHUNK_SIZE, coord.y * CHUNK_SIZE, coord.z * CHUNK_SIZE);

        // Cluster emissive voxels within this chunk
        glm::vec3 sumPos(0.0f);
        glm::vec3 sumColor(0.0f);
        int count = 0;

        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    const auto &vox = chunk->voxels[x][y][z];
                    if (vox.type == 2) { // Fire voxel
                        sumPos += chunkOrigin + glm::vec3(x, y, z);
                        sumColor += vox.color;
                        count++;
                    }
                }
            }
        }

        if (count > 0) {
            VoxelLight light;
            light.pos = sumPos / float(count);
            light.color = sumColor / float(count);
            light.intensity = float(count) * 35.0f; // Scale intensity
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
                    inst.scale = std::sqrt(light.intensity) * 0.5f;
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
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS) {
            if (DebugMode1) {
                voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
                DebugMode1 = false;
            } else {
                voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel_debug.frag");
                DebugMode1 = true;
                activeDebugMode = 0;
            }
        }
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            activeDebugMode = 1;
        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            activeDebugMode = 2;
        }
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            activeDebugMode = 3;
        }
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            activeDebugMode = 4;
        }
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            activeDebugMode = 5;
        }

        if (glfwGetKey(window, GLFW_KEY_CAPS_LOCK) == GLFW_PRESS) {
            DebugMode2 = (!DebugMode2);
        }

        // Cast gravity well spell on right mouse click
        static bool wasMousePressed = false;
        // In update() function:
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!wasMousePressed) {
                RayCastResult hit = rayCastFromCamera(250.0f);
                glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                                        (cameraPos + getCameraFront() * 125.0f);

                // Different spells with different delays
                float delay = 2.0f; // Default 2 second delay

                // Optional: Change delay based on modifier keys
                if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
                    delay = 0.5f; // Fast spell with Shift
                } else if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
                    delay = 3.0f; // Slow spell with Ctrl
                }

                castGravityWellSpell(spellCenter, 100.0f, 0, 10.0f);
                wasMousePressed = true;
            }
        } else {
            wasMousePressed = false;
        }

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
        float speed = 20.0f * deltaTime;

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
        if(DebugMode1) {CpuTimer t8("Generate Meshes");}
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
                    if (chunk->meshDirty&&++built < MAX_CHUNKS_PER_FRAME || !chunk->gpuCache.isValid&&++built < MAX_CHUNKS_PER_FRAME) {
                        if(built<=1&&DebugMode1)
                        {
                            CpuTimer t5("generateChunkMesh");
                        }
                        generateChunkMesh(chunk);
                        if(built<=1) {
                            std::cout << "Progress: "
                                      << ((float) (frameCounter - 29) * MAX_CHUNKS_PER_FRAME / FilledChunks) * 100.0f
                                      << "% \n";
                        }
                        meshRegens++;
                    }

                    // Rebuild lighting if dirty
                    if (chunk->lightingDirty&&++lighted<MAX_CHUNKS_PER_FRAME) {
                        if(lighted==1&&DebugMode1)
                        {
                            CpuTimer t4("rebuildChunkLights");
                        }
                        rebuildChunkLights(coord);
                        if(lighted<=1) {
                            std::cout << "Progress: "
                                      << ((float)(100.0f * (totalChunks - (frameCounter - 29) * MAX_CHUNKS_PER_FRAME ) / totalChunks) * 100.0f)
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
        voxelShader->setVec3("ambientColor", glm::vec3(0.0002f));

// or show signed N·L for the strongest light (index 0)
        if (DebugMode1) {
            voxelShader->setInt("debugMode", activeDebugMode % 5);
        } else {
            voxelShader->setFloat("emission", 0.0f);
        }

        if(DebugMode1) {CpuTimer t6("ChunkLighting");}
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

        // Count how many are actually animating
        int animatingCount = 0;
        for (const auto& voxel : animatedVoxels) {
            if (voxel.isAnimating) animatingCount++;
        }
        if (animatingCount == 0) return;

        // Use a simple shader
        static Shader voxelAnimShader("shaders/voxel_anim.vert", "shaders/voxel_anim.frag");
        voxelAnimShader.use();

        // Set up camera matrices
        float aspect = (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));

        voxelAnimShader.setMatrix("projection", projection);
        voxelAnimShader.setMatrix("view", view);
        voxelAnimShader.setVec3("viewPos", cameraPos);

        // Create a simple cube VAO if not already created
        static GLuint cubeVAO = 0, cubeVBO = 0;
        if (cubeVAO == 0) {
            // Simple unit cube vertices
            float vertices[] = {
                    // Front face
                    -0.4f, -0.4f,  0.4f,
                    0.4f, -0.4f,  0.4f,
                    0.4f,  0.4f,  0.4f,
                    -0.4f,  0.4f,  0.4f,
                    // Back face
                    -0.4f, -0.4f, -0.4f,
                    0.4f, -0.4f, -0.4f,
                    0.4f,  0.4f, -0.4f,
                    -0.4f,  0.4f, -0.4f
            };

            unsigned int indices[] = {
                    // Front
                    0, 1, 2, 2, 3, 0,
                    // Back
                    4, 5, 6, 6, 7, 4,
                    // Left
                    4, 0, 3, 3, 7, 4,
                    // Right
                    1, 5, 6, 6, 2, 1,
                    // Top
                    3, 2, 6, 6, 7, 3,
                    // Bottom
                    4, 5, 1, 1, 0, 4
            };

            glGenVertexArrays(1, &cubeVAO);
            glGenBuffers(1, &cubeVBO);
            GLuint cubeEBO;
            glGenBuffers(1, &cubeEBO);

            glBindVertexArray(cubeVAO);

            glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            glBindVertexArray(0);
        }

        // Enable transparency for magical effect
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Render each animated voxel
        glBindVertexArray(cubeVAO);

        for (const auto& voxel : animatedVoxels) {
            if (!voxel.isAnimating) continue;

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, voxel.currentPos);

            // Add pulsing effect based on animation progress
            float pulse = 1.0f;
            float alphaMultiplier = 0.5f;

            if (voxel.hasArrived) {
                // Arrived voxels pulse gently
                pulse = 1.0f + 0.1f * std::sin(glfwGetTime() * 3.0f);
                alphaMultiplier = 0.2f; // Slightly transparent
            } else {
                // Moving voxels pulse faster
                pulse = 1.0f + 0.2f * std::sin(glfwGetTime() * 8.0f + voxel.currentPos.x);
            }            model = glm::scale(model, glm::vec3(pulse));

            model = glm::scale(model, glm::vec3(pulse));

            voxelAnimShader.setMatrix("model", model);
            voxelAnimShader.setVec3("color", voxel.color);
            voxelAnimShader.setFloat("alpha", alphaMultiplier);

            // Draw cube
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);
        glDisable(GL_BLEND);
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
        billboardInstances.reserve(fluidPlanets.size());

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

        for (size_t i = 0; i < fluidPlanets.size(); i++) {
            WorldPlanet &planet = fluidPlanets[i];

            // Pick blue/green color
            glm::vec3 planetColor = baseColors[i % 5];

            // Upload the 32³ voxel template (binding 0)
            //uploadVoxelChunk(fluidPlanetChunk, &planetColor);

            // -------------------------
            // Voxel splat into a field
            // -------------------------
            voxelSplatShader->use();
            voxelSplatShader->setVec3("gridOrigin", planet.worldPos - 0.5f * glm::vec3(CHUNK_SIZE));
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
            marchingCubesShader->setVec3("gridOrigin", planet.worldPos - 0.5f * glm::vec3(GRID_X, GRID_Y, GRID_Z));
            marchingCubesShader->setFloat("voxelSize", 0.25f);

            resetAtomicCounter();
            //dispatchCompute();

            // -------------------------
            // Lighting (no emission)
            // -------------------------
            voxelShader->use();
            voxelShader->setInt("numLights", 0); // not emissive — no lights needed
            voxelShader->setVec3("ambientColor", {0.05f, 0.08f, 0.10f});
            voxelShader->setVec3("viewPos", cameraPos);
            voxelShader->setMatrix("model", glm::mat4(1.0f));
            voxelShader->setMatrix("mvp", pv);

            voxelShader->setFloat("emission", 0.0f);          // not emissive
            voxelShader->setVec3("emissionColor", {0, 0, 0});   // no glow in mesh
            voxelShader->setVec3("uniformColor", planetColor);

            //drawTriangles(*voxelShader);

            // -------------------------
            // Billboard glow (fluid look)
            // -------------------------
            SunInstance inst;
            inst.position =
                    planet.worldPos + (cameraPos - planet.worldPos) * 0.25f;

            float r = (CHUNK_SIZE * 0.25f) * glm::length(planet.radius);
            inst.scale = r * 2.5f;       // glow radius
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
                chunk->coord.x * CHUNK_SIZE,
                chunk->coord.y * CHUNK_SIZE,
                chunk->coord.z * CHUNK_SIZE
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
        const int localDIM = DIM; // use class member
        const size_t total = size_t(localDIM) * localDIM * localDIM;
        std::vector<CpuVoxel> voxels;
        voxels.resize(total);

        // Copy samples [0 .. DIM-1] from chunk.voxels (Chunk stores CHUNK_SIZE+2 alloc, so this is safe).
        for (int x = 0; x < localDIM; ++x) {
            for (int y = 0; y < localDIM; ++y) {
                for (int z = 0; z < localDIM; ++z) {
                    int idx = x + y * localDIM + z * localDIM * localDIM;
                    const auto &src = chunk.voxels[x][y][z];
                    voxels[idx].density = src.density;
                    if (overrideColor) {
                        voxels[idx].color = glm::vec4(*overrideColor, 1.0f);
                    } else {
                        voxels[idx].color = glm::vec4(src.color, 1.0f);
                    }
                }
            }
        }

        // Upload to SSBO binding 0 (ssboVoxels was created with size voxelCount * sizeof(CpuVoxel))
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

        const float voxelSize = 1.0f;
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

}
