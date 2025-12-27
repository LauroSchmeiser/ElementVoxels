#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/VoxelMesher.h"
#include "rendering/Shader.h"
#include "entities/VoxelEntity.h"
#include "rendering/marchingTables.h"
#include "rendering/SunBillboard.h"

namespace gl3 {
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

    void Game::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        if (height == 0) height = 1; // prevent divide-by-zero
        glViewport(0, 0, width, height);

        // ✅ Update the Game instance’s stored window size
        Game *game = static_cast<Game *>(glfwGetWindowUserPointer(window));
        if (game) {
            game->windowWidth = width;
            game->windowHeight = height;
        }
    }


    Game::Game(int width, int height, const std::string &title)
            : windowWidth(width)
            , windowHeight(height)
            , chunkManager(std::make_unique<MultiGridChunkManager>())  // Initialize manager
            , cameraPos(0.0f, 0.0f, 50.0f)
            , cameraRotation(-90.0f, 0.0f) {
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

    glm::mat4 Game::calculateMvpMatrix(glm::vec3 position, float zRotationInDegrees, glm::vec3 scale) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::scale(model, scale);
        model = glm::rotate(model, glm::radians(zRotationInDegrees), glm::vec3(0, 0, 1));

        glm::vec3 front = getCameraFront();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + front, glm::vec3(0, 1, 0));

        if (windowHeight == 0) windowHeight = 1;
        float aspectRatio = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 500.0f);

        return projection * view * model;
    }


    void Game::run() {
        //Initialization-Steps
        setupSSBOsAndTables();
        setupCamera();
        generateChunks();
        //fillChunks();
        //setSimulationVariables();
        //findBestParent();
        setupVEffects();
        //updateWorldLighting();


        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //Simulation Steps
            updateDeltaTime();
            //updateWorldLighting();

            //UpdateRotation(suns);
            //UpdateRotation(planets);

            //Input-Steps
            handleCameraInput();
            glfwPollEvents();
            update();


            //Post-Prod Steps?

            //Rendering Steps
            marchingCubesShader->use();
            renderChunks();
            //renderSuns();
            //renderFluidPlanets();

            glfwSwapBuffers(window);
        }


    }

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

    inline int worldToChunk(float v) {
        return int(std::floor(v / CHUNK_SIZE));
    }

    inline Chunk &getOrCreateChunk(
            std::unordered_map<ChunkCoord, Chunk, ChunkCoordHash> &world,
            int cx, int cy, int cz
    ) {
        ChunkCoord key{cx, cy, cz};
        return world[key]; // default-constructs if missing
    }


    void Game::generateChunks() {
        std::mt19937 rng(std::random_device{}());

        std::uniform_real_distribution<float> distPos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);

        std::vector<WorldPlanet> worldPlanets;

        // Create solid planets (type 1)
        int planetCount = 5;
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

        int lavaCount = 2   + (rng() % 2);
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
        size_t touchedChunks = 0;

        for (const auto& planet : worldPlanets) {
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
                        Chunk* chunk = chunkManager->getChunk(coord);
                        if (!chunk) {
                            // Determine category based on planet type
                            VoxelCategory category;
                            switch (planet.type) {
                                case 2: category = VoxelCategory::EMISSIVE; break; // Fire
                                case 3: category = VoxelCategory::FLUID; break;    // Water
                                default: category = VoxelCategory::STATIC; break;  // Solid
                            }
                            chunkManager->addChunk(coord, category);
                            chunk = chunkManager->getChunk(coord);
                        }

                        if (!chunk) continue;

                        glm::vec3 chunkOrigin(cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE);
                        bool chunkTouched = false;

                        // Carve sphere into chunk
                        for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                            for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                                for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                    glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz);
                                    float dist = glm::distance(worldPos, planet.worldPos);

                                    // Signed distance field: positive outside, negative inside
                                    // IMPORTANT: Always set density, not just inside sphere!
                                    chunk->voxels[lx][ly][lz].density = planet.radius - dist;

                                    // Only set type if inside or near surface
                                    if (dist <= planet.radius + 1.0f) { // Slightly beyond radius for surface
                                        chunk->voxels[lx][ly][lz].type = planet.type;
                                        chunk->voxels[lx][ly][lz].color = planet.color;
                                        chunkTouched = true;
                                        solidVoxels++;
                                    }
                                }
                            }
                        }

                        if (chunkTouched) {
                            chunk->meshDirty = true;
                            chunk->lightingDirty = true;
                            touchedChunks++;

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
        std::cout << "Touched " << touchedChunks << " chunks\n";
        std::cout << "Created " << solidVoxels << " solid voxels\n";
    }


    void Game::setupVEffects() {
        sunBillboards.init(12); // maxInstances, adjust based on max expected suns

    }

    void Game::renderChunks() {
        float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        emissiveBillboards.clear();
        usedLightIDs.clear();

        // Get camera chunk coordinates
        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);
        const int R2 = RenderingRange * RenderingRange;

        int renderedChunks = 0;

        // Step 1: Process emissive chunks for lighting updates and billboards
        chunkManager->forEachEmissiveChunk([this](Chunk* chunk) {
            // Only update lighting if dirty
            if (chunk->lightingDirty) {
                rebuildChunkLights(chunk->coord);
                chunk->lightingDirty = false;
                // IMPORTANT: Lighting changes might affect mesh (for emissive materials)
                chunk->meshDirty = true;
            }

            // Collect emissive lights for billboards
            for (const auto& light : chunk->emissiveLights) {
                if (usedLightIDs.insert(light.id).second) {
                    SunInstance inst;
                    inst.position = light.pos;
                    inst.scale = std::sqrt(light.intensity) * 0.5f;
                    inst.color = light.color * 2.5f;
                    emissiveBillboards.push_back(inst);
                }
            }
        });

        // Step 2: Build light spatial index for fast lookup
        struct GridCell {
            int gx, gy, gz;
            bool operator==(const GridCell& other) const {
                return gx == other.gx && gy == other.gy && gz == other.gz;
            }
        };

        struct GridCellHash {
            size_t operator()(const GridCell& cell) const {
                return ((cell.gx * 73856093) ^ (cell.gy * 19349663) ^ (cell.gz * 83492791));
            }
        };

        std::unordered_map<GridCell, std::vector<const VoxelLight*>, GridCellHash> lightGrid;
        const int LIGHT_GRID_SIZE = CHUNK_SIZE * 4; // Lights affect 4x4 chunk area

        chunkManager->forEachEmissiveChunk([&](Chunk* chunk) {
            for (const auto& light : chunk->emissiveLights) {
                GridCell cell{
                        (int)std::floor(light.pos.x / LIGHT_GRID_SIZE),
                        (int)std::floor(light.pos.y / LIGHT_GRID_SIZE),
                        (int)std::floor(light.pos.z / LIGHT_GRID_SIZE)
                };
                lightGrid[cell].push_back(&light);
            }
        });

        // Step 3: Iterate through visible chunks
        const int renderRadius = RenderingRange;

        for (int cx = camCX - renderRadius; cx <= camCX + renderRadius; ++cx) {
            for (int cy = camCY - renderRadius; cy <= camCY + renderRadius; ++cy) {
                for (int cz = camCZ - renderRadius; cz <= camCZ + renderRadius; ++cz) {
                    // Distance culling
                    int dx = cx - camCX;
                    int dy = cy - camCY;
                    int dz = cz - camCZ;
                    if (dx*dx + dy*dy + dz*dz > R2) continue;

                    ChunkCoord coord{cx, cy, cz};
                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkOrigin(cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE);

                    // Skip empty chunks early
                    if (!hasSolidVoxels(*chunk)) {
                        chunk->vertexCount = 0;
                        continue;
                    }

                    // Step 4: Collect nearby lights efficiently
                    std::vector<const VoxelLight*> nearbyLights;
                    nearbyLights.reserve(MAX_LIGHTS * 2);

                    glm::vec3 chunkCenter = chunkOrigin + glm::vec3(CHUNK_SIZE * 0.5f);

                    // Check 2x2x2 light grid cells around chunk
                    GridCell chunkCell{
                            (int)std::floor(chunkCenter.x / LIGHT_GRID_SIZE),
                            (int)std::floor(chunkCenter.y / LIGHT_GRID_SIZE),
                            (int)std::floor(chunkCenter.z / LIGHT_GRID_SIZE)
                    };

                    for (int gx = -1; gx <= 1; ++gx) {
                        for (int gy = -1; gy <= 1; ++gy) {
                            for (int gz = -1; gz <= 1; ++gz) {
                                GridCell checkCell{chunkCell.gx + gx, chunkCell.gy + gy, chunkCell.gz + gz};
                                auto it = lightGrid.find(checkCell);
                                if (it != lightGrid.end()) {
                                    for (const VoxelLight* light : it->second) {
                                        glm::vec3 d = light->pos - chunkCenter;
                                        if (glm::dot(d, d) <= LIGHT_RADIUS_SQ) {
                                            nearbyLights.push_back(light);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Step 5: Sort and limit lights
                    std::sort(nearbyLights.begin(), nearbyLights.end(),
                              [chunkCenter](const VoxelLight* a, const VoxelLight* b) {
                                  return glm::dot(a->pos - chunkCenter, a->pos - chunkCenter) <
                                         glm::dot(b->pos - chunkCenter, b->pos - chunkCenter);
                              }
                    );

                    if (nearbyLights.size() > MAX_LIGHTS) {
                        nearbyLights.resize(MAX_LIGHTS);
                    }

                    // Step 6: Generate mesh if needed
                    if (true) {
                        // Upload voxel data to GPU
                        uploadVoxelChunk(*chunk, nullptr);

                        // Reset counters and dispatch compute shader
                        resetAtomicCounter();
                        setComputeUniforms(chunkOrigin, glm::vec3(1.0f), *marchingCubesShader);
                        dispatchCompute();

                        // Get vertex count back from GPU
                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                                           sizeof(unsigned int), &chunk->vertexCount);

                        // Only clear dirty flag if we successfully generated geometry
                        if (chunk->vertexCount > 0) {
                            chunk->meshDirty = false;
                        } else {
                            // If no geometry but chunk has solid voxels, something's wrong
                            // with density values - check your Marching Cubes threshold
                            std::cout << "Warning: Chunk at " << cx << "," << cy << "," << cz
                                      << " has solid voxels but generated 0 vertices\n";
                        }
                    }

                    // Step 7: Render if we have geometry
                    if (chunk->vertexCount == 0) {
                        continue;
                    }

                    // Setup shader
                    voxelShader->use();
                    voxelShader->setMatrix("model", glm::mat4(1.0f));
                    voxelShader->setMatrix("mvp", pv);
                    voxelShader->setVec3("viewPos", cameraPos);
                    voxelShader->setVec3("ambientColor", glm::vec3(0.02f));
                    voxelShader->setFloat("emission", 0.1f);

                    // Set lights
                    voxelShader->setInt("numLights", (int)nearbyLights.size());
                    for (int i = 0; i < (int)nearbyLights.size(); ++i) {
                        const VoxelLight* light = nearbyLights[i];
                        voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", light->pos);
                        voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", light->color);
                        voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", light->intensity);
                    }

                    // Draw
                    drawTriangles(*voxelShader);
                    renderedChunks++;
                }
            }
        }

        // Render billboards
        if (!emissiveBillboards.empty()) {
            sunBillboards.render(emissiveBillboards, view, projection, (float)glfwGetTime());
        }

        std::cout << "Rendered " << renderedChunks << " chunks\n";
    }

    void Game::buildLightSpatialHash(
            std::unordered_map<ChunkCoord, std::vector<VoxelLight*>, ChunkCoordHash>& hash) {

        chunkManager->forEachEmissiveChunk([&hash](Chunk* chunk) {
            for (auto& light : chunk->emissiveLights) {
                // Determine which chunk grid cell this light belongs to
                ChunkCoord gridCell{
                        (int)std::floor(light.pos.x / (CHUNK_SIZE * 2)),  // 2x2 chunk grid
                        (int)std::floor(light.pos.y / (CHUNK_SIZE * 2)),
                        (int)std::floor(light.pos.z / (CHUNK_SIZE * 2))
                };
                hash[gridCell].push_back(&light);
            }
        });
    }

    std::vector<VoxelLight*> Game::collectNearbyLightsFast(
            const ChunkCoord& chunkCoord,
            const glm::vec3& chunkOrigin,
            const std::unordered_map<ChunkCoord, std::vector<VoxelLight*>, ChunkCoordHash>& hash) {

        std::vector<VoxelLight*> result;
        result.reserve(MAX_LIGHTS * 8); // Reserve enough space

        glm::vec3 chunkCenter = chunkOrigin + glm::vec3(CHUNK_SIZE * 0.5f);

        // Check 3x3x3 grid cells around the chunk
        for (int gx = -1; gx <= 1; ++gx) {
            for (int gy = -1; gy <= 1; ++gy) {
                for (int gz = -1; gz <= 1; ++gz) {
                    ChunkCoord gridCell{
                            chunkCoord.x / 2 + gx,
                            chunkCoord.y / 2 + gy,
                            chunkCoord.z / 2 + gz
                    };

                    auto it = hash.find(gridCell);
                    if (it != hash.end()) {
                        for (VoxelLight* light : it->second) {
                            glm::vec3 d = light->pos - chunkCenter;
                            if (glm::dot(d, d) <= LIGHT_RADIUS_SQ) {
                                result.push_back(light);
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    void Game::processEmissiveChunks() {
        chunkManager->forEachEmissiveChunk([this](Chunk* chunk) {
            // Process only emissive chunks
            if (chunk->lightingDirty) {
                rebuildChunkLights(chunk->coord);
            }

            // Collect emissive lights for billboards
            for (const auto& light : chunk->emissiveLights) {
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

        int Game::worldToChunk(float worldPos) const {
            return (int) std::floor(worldPos / CHUNK_SIZE);
        }

        ChunkCoord Game::worldToChunkCoord(const glm::vec3 &worldPos) const {
            return ChunkCoord{
                    worldToChunk(worldPos.x),
                    worldToChunk(worldPos.y),
                    worldToChunk(worldPos.z)
            };
        }

        glm::vec3 Game::getChunkWorldPosition(const ChunkCoord &coord) {
            return glm::vec3(coord.x * CHUNK_SIZE, coord.y * CHUNK_SIZE, coord.z * CHUNK_SIZE);
        }

        uint32_t Game::makeLightID(int cx, int cy, int cz) {
            // Simple hash function for light ID
            return ((cx & 0xFFF) << 20) | ((cy & 0xFFF) << 8) | (cz & 0xFF);
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
                light.intensity = float(count) * 30.0f; // Scale intensity
                light.id = makeLightID(coord.x, coord.y, coord.z);

                chunk->emissiveLights.push_back(light);
            }

            chunk->lightingDirty = false;
        }


    /*
    void Game::renderSuns()
    {

        //build instances
        std::vector<SunInstance> instances;
        instances.reserve(suns.size());

        // compute PV once per frame (do this outside the loops)
        float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;
        glm::mat4 identityModel = glm::mat4(1.0f);


        // For each sun: upload voxels, reset counter, set uniforms, dispatch compute, read vertex count, draw
        for (auto &sun : suns) {
            uploadVoxelChunk(sunChunk, &sun.color);           // upload densities/colors, uses binding 0
            resetAtomicCounter();                     // zero counter in binding 3
            setComputeUniforms(sun.position, sun.scale, *marchingCubesShader);
            dispatchCompute();

            // read debug vertex count
            unsigned int vertexCount = 0;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
            //Vertex-Count Debug
            //std::cout << "[compute] sun vertexCount = " << vertexCount << std::endl;

            // --- Set voxel rendering shader uniforms for this sun BEFORE drawing ---
            // render

            if (sun.parent != nullptr)
            {
                sun.orbitAngle += deltaTime * sun.orbitSpeed;

                glm::vec3 flat(
                        cos(sun.orbitAngle + sun.orbitOffset) * sun.orbitRadius,
                        0.0f,
                        sin(sun.orbitAngle + sun.orbitOffset) * sun.orbitRadius
                );

                glm::mat4 tilt = glm::rotate(glm::mat4(1.0f), sun.orbitInclination, glm::vec3(1,0,0));
                glm::vec3 tilted = glm::vec3(tilt * glm::vec4(flat, 1.0f));

                sun.position = sun.parent->position + tilted;
            }



            voxelShader->use();
            voxelShader->setMatrix("model", identityModel);  // IMPORTANT: identity
            voxelShader->setMatrix("mvp", pv);               // PV only (positions are world-space)
            voxelShader->setVec3("viewPos", cameraPos);
            voxelShader->setVec3("lightPos", sun.position);
            voxelShader->setFloat("lightIntensity", 40.0f);
            voxelShader->setFloat("emission", 3.0f);
            voxelShader->setVec3("emissionColor", sun.color);

            voxelShader->setVec3("uniformColor", sun.color);

            // draw the vertices produced by the compute shader
            drawTriangles(*voxelShader);

            //render Billboards
            SunInstance inst;
            inst.position = sun.position+((cameraPos-sun.position)/glm::vec3(3));
            // choose scale for billboard so it visually surrounds voxel core; tweak as you like
            inst.scale = glm::length(sun.scale) * 3.14f;
            // compute sphere radius in world units used by your marching-cubes mesh
            const float baseVoxelSize = 1.0f;
            float scaleAvg = (sun.scale.x + sun.scale.y + sun.scale.z) / 2.5f;
            float sphereRadiusWorld = ((CHUNK_SIZE - 1) * 0.5f) * (baseVoxelSize * scaleAvg) *1.0f;

            // billboard scale = diameter (world units). small padding avoids clipping.
            inst.scale = sphereRadiusWorld * 2.0f*1.05f;
            inst.color = sun.color; // use sun.color (set when creating suns)
            instances.push_back(inst);

        }

        // render billboards (time: use glfwGetTime or your time accumulator)
        sunBillboards.render(instances, view, projection, (float)glfwGetTime());

    }



     */



    void Game::renderFluidPlanets()
{
    const int GRID_X = 128;
    const int GRID_Y = 128;
    const int GRID_Z = 128;
    const size_t FIELD_COUNT = size_t(GRID_X) * GRID_Y * GRID_Z;

    // PV matrix
    float aspect = windowHeight == 0 ? float(windowWidth) : float(windowWidth) / float(windowHeight);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), {0,1,0});
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

    for (size_t i = 0; i < fluidPlanets.size(); i++)
    {
        WorldPlanet &planet = fluidPlanets[i];

        // Pick blue/green color
        glm::vec3 planetColor = baseColors[i % 5];

        // Upload the 32³ voxel template (binding 0)
        uploadVoxelChunk(fluidPlanetChunk, &planetColor);

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
        marchingCubesShader->setVec3("gridOrigin", planet.worldPos - 0.5f * glm::vec3(GRID_X,GRID_Y,GRID_Z));
        marchingCubesShader->setFloat("voxelSize", 0.25f);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, fieldBitsSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);

        resetAtomicCounter();
        dispatchCompute();
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

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
        voxelShader->setVec3("emissionColor", {0,0,0});   // no glow in mesh
        voxelShader->setVec3("uniformColor", planetColor);

        drawTriangles(*voxelShader);

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
    sunBillboards.render(billboardInstances, view, projection, float(glfwGetTime()));
}



    void Game::update() {
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

            // Update lighting only for dirty chunks
            chunkManager->forEachEmissiveChunk([this](Chunk *chunk) {
                if (chunk->lightingDirty) {
                    rebuildChunkLights(chunk->coord);
                }
            });

            // Update dynamic chunks
            // chunkManager->forEachDynamicChunk([this](Chunk* chunk) {
            // Physics, animation, etc.
            //});

            // Update fluid chunks
            //chunkManager->forEachFluidChunk([this](Chunk* chunk) {
            // Fluid simulation
            //});
    }


    void Game::updateDeltaTime() {
        float frameTime = glfwGetTime();
        deltaTime = frameTime - lastFrameTime;
        lastFrameTime = frameTime;
    }

    void Game::updatePhysics() {
        const float fixedTimeStep = 1.0f / 60.0f;
        const int subStepCount = 4; // recommended sub-step count
        accumulator += deltaTime;
        if(accumulator >= fixedTimeStep){
            // Update the entities based on what happened in the physics step

            accumulator -= fixedTimeStep;
        }
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
        if (glm::length(front) < 0.001f) front = glm::vec3(0,0,-1);
        return glm::normalize(front);

    }

// debug version: if doColorByDensity==true, set per-voxel color from density (visualize SDF)
    void Game::uploadVoxelChunk(const Chunk& chunk, const glm::vec3* overrideColor)
    {
        // Debug toggle: set to true to upload grayscale colors based on density (visualize SDF)
        // Set false to use overrideColor / chunk colors.
        bool doColorByDensity = false;

        const int DIM = CHUNK_SIZE + 1;
        size_t voxelCount = DIM * DIM * DIM;
        std::vector<CpuVoxel> voxels(voxelCount);


        // find min/max density for normalization (optional, fast)
        float minD = FLT_MAX, maxD = -FLT_MAX;
        if (doColorByDensity) {
            for(int x=0;x<CHUNK_SIZE;x++) for(int y=0;y<CHUNK_SIZE;y++) for(int z=0;z<CHUNK_SIZE;z++) {
                        float d = chunk.voxels[x][y][z].density;
                        if (d < minD) minD = d;
                        if (d > maxD) maxD = d;
                    }
            // avoid degenerate range
            if (maxD - minD < 1e-6f) { maxD = minD + 1.0f; }
        }

        for(int x=0;x<=CHUNK_SIZE;x++) {
            for(int y=0;y<=CHUNK_SIZE;y++) {
                for(int z=0;z<=CHUNK_SIZE;z++) {
                    int sx = std::min(x, CHUNK_SIZE+1 );
                    int sy = std::min(y, CHUNK_SIZE+1 );
                    int sz = std::min(z, CHUNK_SIZE+1 );

                    const auto &v = chunk.voxels[sx][sy][sz];
                    int idx = x + y * DIM + z * DIM * DIM;

                    voxels[idx].density = v.density;
                    voxels[idx].color = overrideColor ? glm::vec4(*overrideColor, 1.0f) : glm::vec4(v.color, 1.0f);
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxels.size()*sizeof(CpuVoxel), voxels.data());
    }
    void Game::resetAtomicCounter()
    {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);

    }

    void Game::setComputeUniforms(const glm::vec3& chunkOrigin,
                                  const glm::vec3& /*objectScale*/,
                                  Shader& computeShader)
    {
        computeShader.use();

        const float voxelSize = 1.0f;
        const int DIM = CHUNK_SIZE + 1;

        // Shift grid so voxel corners are centered correctly
        glm::vec3 gridOrigin = chunkOrigin; // no -0.5f * voxelSize


        computeShader.setVec3("gridOrigin", gridOrigin);
        computeShader.setFloat("voxelSize", voxelSize);
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));
    }


    void Game::dispatchCompute()
    {
        int groups = (CHUNK_SIZE - 1 + 7) / 8; // local_size = 8 in compute shader

        // Bind SSBOs to match compute shader bindings:
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);

        glDispatchCompute(groups, groups, groups);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    }
    void Game::drawTriangles(Shader& voxelShader)
    {
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Use SSBO as VBO
        glBindBuffer(GL_ARRAY_BUFFER, ssboTriangles);

        // Stride for OutVertex
        constexpr GLsizei stride = sizeof(gl3::OutVertex);

        // Attribute offsets (use offsetof for safety)
        const void* posOffset    = (void*)offsetof(gl3::OutVertex, pos);
        const void* normalOffset = (void*)offsetof(gl3::OutVertex, normal);
        const void* colorOffset  = (void*)offsetof(gl3::OutVertex, color);

        // -- IMPORTANT --
        // All are vec4 (4 floats), not vec3.

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, posOffset);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, normalOffset);

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, colorOffset);

        // Ensure compute writes are visible
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

        // Read counter (triangles)
        unsigned int triCount = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &triCount);

        //std::cout << "Triangles: " << triCount << "\n";

        // Each triangle has 3 vertices
        unsigned int vertexCount = triCount * 3;

        voxelShader.use();

        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        glDeleteVertexArrays(1, &vao);
    }


    float Game::getVoxelPlanetRadius(const glm::vec3& scale, float baseChunkRadius) {
        return baseChunkRadius * std::max(scale.x, std::max(scale.y, scale.z));
    }


    bool Game::isOverlapping(const glm::vec3 &pos, float rad, const std::vector<gl3::Game::WorldPlanet> &others) {
        for (const WorldPlanet& p : others) {
            float r = p.radius;
            float dist = glm::distance(pos, p.worldPos);

            if (dist < (rad + r)) {
                return true;  // collision
            }
        }
        return false;
    }

    void debugDensity()
    {
        //Debugging for Density
        /*
        std::cout << "meteor SDF min=" << minD << " max=" << maxD
                  << " center=" << meteorChunk.voxels[cx][cy][cz].density << std::endl;

        // Print density along x-axis through center (y=cy,z=cz)
        std::cout << "center-line densities: ";
        for (int x=0;x<CHUNK_SIZE;++x) {
            std::cout << meteorChunk.voxels[x][cy][cz].density << (x+1<CHUNK_SIZE? ",":"\n");
        }
*/
    }

    /*
    void Game::findBestParent()
    {
        std::sort(suns.begin(), suns.end(),
                  [](const Planet &a, const Planet &b) {
                      return glm::length(a.scale) > glm::length(b.scale);
                  });

        for (int i = 1; i < suns.size(); i++) {

            Planet &p = suns[i];
            float mySize = glm::length(p.scale);

            float bestDist = std::numeric_limits<float>::max();
            Planet* bestParent = nullptr;
            /*
            for (int j = 0; j < suns.size(); j++)
            {
                if (i == j) continue;

                float otherSize = glm::length(suns[j].scale);
                if (otherSize <= mySize) continue;  // must be larger

                float d = glm::distance(p.position, suns[j].position);
                if (d < bestDist)
                {
                    bestDist = d;
                    bestParent = &suns[j];
                }
            }



            p.parent = &suns[0]; // NULL if no larger sun exists
        }

        for (int i = 0; i < planets.size(); i++)
        {
            Planet &p = planets[i];
            float mySize = glm::length(p.scale);

            float bestDist = std::numeric_limits<float>::max();
            Planet* bestParent = nullptr;

            for (int j = 0; j < suns.size(); j++)
            {
                if (i == j) continue;

                float otherSize = glm::length(suns[j].scale);
                if (otherSize <= mySize) continue;  // must be larger

                float d = glm::distance(p.position, suns[j].position);
                if (d < bestDist)
                {
                    bestDist = d;
                    bestParent = &suns[j];
                }
            }

            p.parent = &suns[0]; // NULL if no larger sun exists
        }
    }
*/

    /*void Game::updateWorldLighting() {

        for (auto& [coord, chunk] : data->gameWorld) {
            if (chunk.lightingDirty) {
                rebuildChunkLights(coord.x, coord.y, coord.z);
            }
        }
    }
     */



    bool Game::hasSolidVoxels(const gl3::Chunk& chunk) {
        for (int x = 0; x <= CHUNK_SIZE; ++x)
            for (int y = 0; y <= CHUNK_SIZE; ++y)
                for (int z = 0; z <= CHUNK_SIZE; ++z)
                    if (chunk.voxels[x][y][z].isSolid())
                        return true;
        return false;
    }


}
