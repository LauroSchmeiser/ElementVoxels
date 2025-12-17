#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/VoxelMesher.h"
#include "rendering/TestChunk.h"
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

    void Game::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        if (height == 0) height = 1; // prevent divide-by-zero
        glViewport(0, 0, width, height);

        // ✅ Update the Game instance’s stored window size
        Game* game = static_cast<Game*>(glfwGetWindowUserPointer(window));
        if (game) {
            game->windowWidth = width;
            game->windowHeight = height;
        }
    }


    Game::Game(int width, int height, const std::string &title)
    {
        windowWidth = width;
        windowHeight = height;

        if(!glfwInit()) {
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

        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        std::cout << "GL_VERSION: " << (const char*)glGetString(GL_VERSION) << std::endl;
        std::cout << "GLSL_VERSION: " << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;


        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");


        try {
            voxelSplatShader= std::make_unique<Shader>("shaders/metaball_splat.comp");
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


        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            //Simulation Steps
            updateDeltaTime();
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

    void Game::setupSSBOsAndTables()
    {
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

    void Game::setupCamera()
    {
        // --- Camera setup ---
        cameraPos = glm::vec3(0.0f, 0.0f, 80.0f);
        cameraRotation = glm::vec2(0.0f, -90.0f);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    void Game::generateChunks() {

        const float centerOffset = (CHUNK_SIZE - 1) * 0.5f;
        const float radius = centerOffset; // radius in voxel units to roughly touch inside
        const float densityEpsilon = 1e-4f; // small bias to avoid exact zeros


        // Clear all chunks to "air" first
        for(auto &chunkb : data->gameWorld) {
            for (auto &chunka : chunkb) {
                for (auto &chunk : chunka) {
                    for (int x = 0; x <= CHUNK_SIZE; ++x) {
                        for (int y = 0; y <= CHUNK_SIZE; ++y) {
                            for (int z = 0; z <= CHUNK_SIZE; ++z) {
                                chunk.voxels[x][y][z].type = 0; // Air
                                chunk.voxels[x][y][z].density = -1.0f;
                            }
                        }
                    }
                }
            }
        }

        std::mt19937 rng(std::random_device{}());
         int chunkGridSize = ChunkCount * DIM;

         std::uniform_real_distribution<float> distPos(0.0f, chunkGridSize);
         std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distAxis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> distSpeed(5.0f, 5.2f);

        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);
        std::uniform_real_distribution<float> distColor2(0.66f, 1.0f);



        struct WorldPlanet {
            glm::vec3 worldPos;     // Position in world coordinates
            float radius;           // World-space radius
            glm::vec3 color;
            glm::vec3 scale;
            // ... other planet properties
        };
        std::vector<WorldPlanet> worldPlanets; // New struct
        std::vector<WorldPlanet> suns; // special planets
        std::vector<WorldPlanet> waterPlanets; // special planets


        const float VoxelRadius = (CHUNK_SIZE - 1) * 0.5f;


// Generate planets in world space
        for (int i = 0; i < 5 ; ++i) {
            glm::vec3 worldPos;
            glm::vec3 scale;
            float worldRadius;

            int attempts = 0;
            const int maxAttempts = 25;

            do {
                worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                scale = glm::vec3(distScale(rng));
                worldRadius = getVoxelPlanetRadius(scale, VoxelRadius);
                attempts++;
            } while (isOverlapping(worldPos, worldRadius, CollisionEntities) && attempts < maxAttempts);

            if (attempts >= maxAttempts) continue;

            WorldPlanet wp = {
                    worldPos,
                    worldRadius,
                    glm::vec3(distColor(rng), distColor(rng), distColor(rng)),
                    scale
            };

            worldPlanets.push_back(wp);
        }

        // --- Generate 2–3 suns
        std::uniform_real_distribution<float> lavaDistColorR(0.8f, 1.0f);
        std::uniform_real_distribution<float> lavaDistColorG(0.2f, 0.5f);
        std::uniform_real_distribution<float> lavaDistColorB(0.0f, 0.1f);

        int lavaCount = 2 + (rng() % 2); // 2 or 3 planets

        for (int i = 0; i < lavaCount; ++i) {
            glm::vec3 worldPos;
            glm::vec3 scale;
            float worldRadius;

            int attempts = 0;
            const int maxAttempts = 25;

            do {
                worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                scale = glm::vec3(distScale(rng));
                worldRadius = getVoxelPlanetRadius(scale, VoxelRadius);
                attempts++;
            } while (isOverlapping(worldPos, worldRadius, CollisionEntities) && attempts < maxAttempts);

            if (attempts >= maxAttempts) continue;

            WorldPlanet lp = {
                    worldPos,
                    worldRadius,
                    glm::vec3(
                            lavaDistColorR(rng),  // reddish
                            lavaDistColorG(rng),  // orange-ish
                            lavaDistColorB(rng)   // small amount of dark red
                    ),
                    scale
            };

            suns.push_back(lp);
        }


        // --- Generate 2–3 waterPlanets
        std::uniform_real_distribution<float> waterDistColorR(0.0f, 0.2f);
        std::uniform_real_distribution<float> waterDistColorG(0.2f, 0.8f);
        std::uniform_real_distribution<float> waterDistColorB(0.8f, 1.0f);

        int waterCount = 0 + (rng() % 1); // 2 or 3 planets

        for (int i = 0; i < waterCount; ++i) {
            glm::vec3 worldPos;
            glm::vec3 scale;
            float worldRadius;

            int attempts = 0;
            const int maxAttempts = 25;

            do {
                worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                scale = glm::vec3(distScale(rng));
                worldRadius = getVoxelPlanetRadius(scale, VoxelRadius);
                attempts++;
            } while (isOverlapping(worldPos, worldRadius, CollisionEntities) && attempts < maxAttempts);

            if (attempts >= maxAttempts) continue;

            WorldPlanet lp = {
                    worldPos,
                    worldRadius,
                    glm::vec3(
                            waterDistColorR(rng),  // reddish
                            waterDistColorG(rng),  // orange-ish
                            waterDistColorB(rng)   // small amount of dark red
                    ),
                    scale
            };

            waterPlanets.push_back(lp);
        }


        int solidCount=0;
// After generating world planets, carve them into chunks
        for (const auto& planet : worldPlanets) {
            // Determine which chunks this planet affects
            // For simplicity, mark all voxels in sphere
            for (int chunkX = 0; chunkX < ChunkCount; ++chunkX) {
                for (int chunkY = 0; chunkY < ChunkCount; ++chunkY) {
                    for (int chunkZ = 0; chunkZ < ChunkCount; ++chunkZ) {
                        auto& chunk = data->gameWorld[chunkX][chunkY][chunkZ];

                        // Convert chunk coordinates to world coordinates
                        glm::vec3 chunkWorldOffset = glm::vec3(
                                chunkX * DIM,
                                chunkY * DIM,
                                chunkZ * DIM
                        );

                        // Fill voxels in this chunk
                        for (int localX = 0; localX <= CHUNK_SIZE; ++localX) {
                            for (int localY = 0; localY <= CHUNK_SIZE; ++localY) {
                                for (int localZ = 0; localZ <= CHUNK_SIZE; ++localZ) {
                                    // Convert to world coordinates
                                    glm::vec3 worldPos = glm::vec3(
                                            chunkX * CHUNK_SIZE + localX,
                                            chunkY * CHUNK_SIZE + localY,
                                            chunkZ * CHUNK_SIZE + localZ
                                    );

                                    float dist = glm::distance(worldPos, planet.worldPos);

                                    //std::cout<<"Atomic: "<<(dist <= planet.radius)<<"\n dist: "<<dist<<"\n radius: "<<planet.radius<<"\n";

                                    // If inside planet
                                    if (dist <= planet.radius) {
                                        chunk.voxels[localX][localY][localZ].type = 1; // Solid
                                        chunk.voxels[localX][localY][localZ].density = planet.radius - dist;
                                        chunk.voxels[localX][localY][localZ].color = planet.color;
                                    }
                                }
                            }


                        }

                        int solidVoxels = 0;
                        for(int x=0;x<=CHUNK_SIZE;x++)
                            for(int y=0;y<=CHUNK_SIZE;y++)
                                for(int z=0;z<=CHUNK_SIZE;z++)
                                    if(chunk.voxels[x][y][z].isSolid()) solidVoxels++;

                        /*else {
                            std::cout << "Chunk (" << chunkX << "," << chunkY << "," << chunkZ << ") has "
                                      << solidVoxels << " solid voxels\n";
                        }
                         */
                        if(checkEmptyChunks(chunk))
                        {
                            solidCount++;
                        }
                    }

                }
            }

            //planet.worldPos
        }
        std::cout<<"ChunkCount: "<<(ChunkCount*ChunkCount*ChunkCount)<<"\n";
        std::cout<<"Solid Chunks: "<<solidCount<<"\n";
        std::cout<<"Empty Chunks: "<<((ChunkCount*ChunkCount*ChunkCount)-solidCount)<<"\n";

        std::cout<<"VoxelsPerChunk: "<<CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE<<"\n";
        std::cout<<"Number of Voxels: "<<((CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)*(ChunkCount*ChunkCount*ChunkCount))<<"\n";

        //std::cout<<"ChunkCount: "<<ChunkCount<<"\n";

        //std::cout<<"VoxelCount: "<<CHUNK_SIZE<<"\n";
        //std::cout<<"VoxelCountx3: "<<CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE<<"\n";

        // --- Carve lava planets (type = 2) ---
        for (const auto& planet : suns) {
            for (int chunkX = 0; chunkX < ChunkCount; ++chunkX) {
                for (int chunkY = 0; chunkY < ChunkCount; ++chunkY) {
                    for (int chunkZ = 0; chunkZ < ChunkCount; ++chunkZ) {

                        auto &chunk = data->gameWorld[chunkX][chunkY][chunkZ];

                        for (int localX = 0; localX <= CHUNK_SIZE; ++localX) {
                            for (int localY = 0; localY <= CHUNK_SIZE; ++localY) {
                                for (int localZ = 0; localZ <= CHUNK_SIZE; ++localZ) {

                                    glm::vec3 worldPos = glm::vec3(
                                            chunkX * CHUNK_SIZE + localX,
                                            chunkY * CHUNK_SIZE + localY,
                                            chunkZ * CHUNK_SIZE + localZ
                                    );

                                    float dist = glm::distance(worldPos, planet.worldPos);

                                    if (dist <= planet.radius) {
                                        chunk.voxels[localX][localY][localZ].type = 2; // planet made of fire
                                        chunk.voxels[localX][localY][localZ].density = planet.radius - dist;
                                        chunk.voxels[localX][localY][localZ].color = planet.color;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // --- Carve water planets (type = 3) ---
            for (const auto &planet: waterPlanets) {
                for (int chunkX = 0; chunkX < ChunkCount; ++chunkX) {
                    for (int chunkY = 0; chunkY < ChunkCount; ++chunkY) {
                        for (int chunkZ = 0; chunkZ < ChunkCount; ++chunkZ) {

                            auto &chunk = data->gameWorld[chunkX][chunkY][chunkZ];

                            for (int localX = 0; localX <= CHUNK_SIZE; ++localX) {
                                for (int localY = 0; localY <= CHUNK_SIZE; ++localY) {
                                    for (int localZ = 0; localZ <= CHUNK_SIZE; ++localZ) {

                                        glm::vec3 worldPos = glm::vec3(
                                                chunkX * CHUNK_SIZE + localX,
                                                chunkY * CHUNK_SIZE + localY,
                                                chunkZ * CHUNK_SIZE + localZ
                                        );

                                        float dist = glm::distance(worldPos, planet.worldPos);

                                        if (dist <= planet.radius) {
                                            chunk.voxels[localX][localY][localZ].type = 3; // planet made of fire
                                            chunk.voxels[localX][localY][localZ].density = planet.radius - dist;
                                            chunk.voxels[localX][localY][localZ].color = planet.color;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        /*for (size_t i = 0; i < worldPlanets.size(); ++i) {
            glm::vec3 toPlanet = worldPlanets[i].worldPos - cameraPos;   // vector from camera to planet
            glm::vec3 forward = getCameraFront();                        // normalized camera direction

            float dot = glm::dot(glm::normalize(toPlanet), forward);

            if (dot > 0.0f) {
                std::cout << "Planet " << i << " is in front of the camera. Pos: "
                          << worldPlanets[i].worldPos.x << ", "
                          << worldPlanets[i].worldPos.y << ", "
                          << worldPlanets[i].worldPos.z << "\n";
            } else {
                std::cout << "Planet " << i << " is behind the camera.\n";
            }
        }

*/
        /*
        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - centerOffset;
                    float dy = y - centerOffset;
                    float dz = z - centerOffset;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = baseChunk.voxels[x][y][z];
                    // SDF: positive inside, negative outside
                    voxel.density = radius - dist + densityEpsilon;
                    voxel.type = (voxel.density > 0.0f) ? 1 : 0;

                    // default chunk color (will be overridden per-object if desired)
                    voxel.color = glm::vec3(0.5f);
                }
            }
        }

        // sunChunk: same but optionally different color
        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - centerOffset;
                    float dy = y - centerOffset;
                    float dz = z - centerOffset;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = sunChunk.voxels[x][y][z];
                    voxel.density = radius - dist + densityEpsilon;
                    voxel.type = (voxel.density > 0.0f) ? 1 : 0;
                    voxel.color = glm::vec3(1.0f, 0.75f, 0.0f);
                }
            }
        }

        //fluid Chunk
        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - centerOffset;
                    float dy = y - centerOffset;
                    float dz = z - centerOffset;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = fluidPlanetChunk.voxels[x][y][z];
                    voxel.density = radius - dist + densityEpsilon;
                    voxel.type = (voxel.density > 0.0f) ? 1 : 0;
                    voxel.color = glm::vec3(0.0f, 0.2f, 1.0f);
                }
            }
        }


        // robust lumpy meteor generator (drop-in)
        const float rx = centerOffset * 0.75f; // ensure meteor fits inside chunk
        const float ry = centerOffset * 0.55f;
        const float rz = centerOffset * 0.55f;
        const float minRadius = std::min(std::min(rx, ry), rz);

        // base eps and noise
        const float noiseAmp = minRadius * 0.06f; // start small (6%)
        const uint32_t seed = 1337u;
        auto intNoise = [&](int xi, int yi, int zi)->float {
            uint32_t h = uint32_t(xi+17) * 73856093u ^ uint32_t(yi+31) * 19349663u ^ uint32_t(zi+97) * 83492791u ^ seed;
            h = (h ^ (h >> 13u)) * 0x5bd1e995u;
            h ^= h >> 15u;
            return (float)(h & 0xFFFFu) / float(0xFFFFu); // [0..1]
        };

        // optional extra lumps (offsets & scales) to break symmetry
        struct Lump { float ox, oy, oz, sx, sy, sz; };
        std::vector<Lump> lumps = {
                { 0.2f*rx,  -0.1f*ry, 0.05f*rz, 0.6f, 0.6f, 0.6f },
                {-0.3f*rx,   0.15f*ry,-0.12f*rz, 0.5f, 0.5f, 0.5f },
                { 0.0f,      0.25f*ry, 0.2f*rz,  0.4f, 0.4f, 0.4f },
        };

        for (int x=0;x<CHUNK_SIZE;++x) {
            for (int y=0;y<CHUNK_SIZE;++y) {
                for (int z=0;z<CHUNK_SIZE;++z) {
                    float dx = float(x) - centerOffset;
                    float dy = float(y) - centerOffset;
                    float dz = float(z) - centerOffset;

                    // normalized ellipsoid distance (1 == surface)
                    float ellMain = sqrtf((dx*dx)/(rx*rx) + (dy*dy)/(ry*ry) + (dz*dz)/(rz*rz));
                    float baseDensity = (1.0f - ellMain) * minRadius;

                    // add a few small ellipsoidal lumps and take max (union)
                    float maxDensity = baseDensity;
                    for (auto &L : lumps) {
                        float lx = (dx - L.ox) / (rx * L.sx);
                        float ly = (dy - L.oy) / (ry * L.sy);
                        float lz = (dz - L.oz) / (rz * L.sz);
                        float ell = sqrtf(lx*lx + ly*ly + lz*lz);
                        float d = (1.0f - ell) * (minRadius * 0.6f);
                        maxDensity = glm::max(maxDensity, d);
                    }

                    // gentle procedural noise, fade toward the surface (0 at ell>=1)
                    float n = intNoise(x,y,z);
                    float rawNoise = (n - 0.5f) * 2.0f * noiseAmp;
                    float interiorFactor = glm::clamp(1.0f - ellMain, 0.0f, 1.0f);
                    interiorFactor = pow(interiorFactor, 0.8f); // keep some roughness near surface
                    float noise = rawNoise * interiorFactor;

                    float density = maxDensity + noise + densityEpsilon;

                    meteorChunk.voxels[x][y][z].density = density;
                    meteorChunk.voxels[x][y][z].type = (density > 0.0f) ? 1 : 0;
                    meteorChunk.voxels[x][y][z].color = glm::vec3(0.45f,0.37f,0.28f);
                }
            }
        }
        // Debug: inspect meteorChunk SDF
        float minD = 1e9f, maxD = -1e9f;
        int cx = CHUNK_SIZE/2, cy = CHUNK_SIZE/2, cz = CHUNK_SIZE/2;
        for (int x=0;x<CHUNK_SIZE;++x){
            for (int y=0;y<CHUNK_SIZE;++y){
                for (int z=0;z<CHUNK_SIZE;++z){
                    float d = meteorChunk.voxels[x][y][z].density;
                    minD = std::min(minD, d);
                    maxD = std::max(maxD, d);
                }
            }
        }

*/
        //std::cout<<"Generated Planets: "<<worldPlanets.size();
    }

    /*
    void Game::fillChunks()
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distPos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distAxis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> distSpeed(5.0f, 5.2f);

        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);
        std::uniform_real_distribution<float> distColor2(0.66f, 1.0f);

/*
        for(auto  &chunkb : data->gameWorld) {
            for (auto &chunka: chunkb) {
                for (auto &chunk: chunka) {
                    glm::vec3 pos;
                    glm::vec3 scale;
                    float radius;

                    // Try up to N times to find a valid non-overlapping position
                    int attempts = 0;
                    const int maxAttempts = 10 * 2;

                    const float VoxelRadius = (CHUNK_SIZE - 1) * 0.5f;
                    do {
                        pos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                        scale = glm::vec3(distScale(rng));
                        radius = getVoxelPlanetRadius(scale, VoxelRadius);
                        attempts++;
                    } while (isOverlapping(pos, radius, CollisionEntities) && attempts < maxAttempts);

                    // If all attempts failed, skip this one
                    if (attempts >= maxAttempts) {
                        std::cout << "failed collision checks";
                        continue;

                    }

                    glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
                    Planet p = {pos, scale, 0.0f, axis, distSpeed(rng),
                                glm::vec3(distColor(rng), distColor(rng), distColor(rng))};

                    planets.push_back(p);
                    CollisionEntities.push_back(p);

                }
            }
        }


        // --- Generate mesh from voxel chunk ---
        for (int i = 0; i < 4; ++i) {
            glm::vec3 pos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            glm::vec3 scale = glm::vec3(1.5f);
            glm::vec3 color = glm::vec3(0.5f, 0.45f, 0.35f);
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            float speed = distSpeed(rng);

            Planet m = {pos, scale, 0.0f, axis, speed, color};

            meteors.push_back(m);
            CollisionEntities.push_back(m); // SAME object
        }


        int sunsCount=3;
        for (int j = 0; j < sunsCount; ++j) {
            glm::vec3 pos;
            glm::vec3 scale;
            float radius;

            // Try up to N times to find a valid non-overlapping position
            int attempts = 0;
            const int maxAttempts = sunsCount*25;

            const float VoxelRadius = (CHUNK_SIZE - 1) * 0.5f ;

            do {
                pos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                scale = glm::vec3(distScale(rng)*5);
                radius = getVoxelPlanetRadius(scale,VoxelRadius);
                attempts++;
            }
            while (isOverlapping(pos, radius, CollisionEntities) && attempts < maxAttempts);

            // If all attempts failed, skip this one
            if (attempts >= maxAttempts)
            {
                std::cout<<"failed collision checks";
                continue;

            }
            glm::vec3 color = glm::vec3(distColor2(rng), distColor(rng) , 0.0f);
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            float speed = distSpeed(rng)*(5/scale.x);
            Planet s = {pos, scale, 0.0f, axis, speed, color};
            suns.push_back(s);
            CollisionEntities.push_back(s);
        }

        glm::vec3 pos;
        float radius;
        pos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
        glm::vec3 color = glm::vec3(0.1f,0.6f,1.0f);
        glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
        float speed = distSpeed(rng);
        Planet s = {pos, glm::vec3(10,10,10), 0.0f, axis, speed, color};
        fluidPlanets.push_back(s);
        int fluidPlanetsCount=15;
        for (int j = 0; j < fluidPlanetsCount; ++j) {
            glm::vec3 pos;
            float radius;
            pos = fluidPlanets.at(0).position+(glm::vec3(distPos(rng), distPos(rng), distPos(rng))/25.0f);
            glm::vec3 color = glm::vec3(0.1f,0.6f,1.0f);
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            float speed = distSpeed(rng);
            Planet s = {pos, glm::vec3(0.25f,0.25f,0.25f), 0.0f, axis, speed, color};
            fluidPlanets.push_back(s);
        }

        int planetsCount=25;
        for (int i = 0; i < planetsCount; ++i) {

            glm::vec3 pos;
            glm::vec3 scale;
            float radius;

            // Try up to N times to find a valid non-overlapping position
            int attempts = 0;
            const int maxAttempts = planetsCount*2;

            const float VoxelRadius = (CHUNK_SIZE - 1) * 0.5f ;
            do {
                pos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
                scale = glm::vec3(distScale(rng));
                radius = getVoxelPlanetRadius(scale,VoxelRadius);
                attempts++;
            }
            while (isOverlapping(pos, radius, CollisionEntities) && attempts < maxAttempts);

            // If all attempts failed, skip this one
            if (attempts >= maxAttempts)
            {
                std::cout<<"failed collision checks";
                continue;

            }

            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            Planet p = { pos, scale, 0.0f, axis, distSpeed(rng), glm::vec3(distColor(rng),distColor(rng),distColor(rng)) };

            planets.push_back(p);
            CollisionEntities.push_back(p);

        }


    }
    */

    /*
    void Game::setSimulationVariables()
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distOrbit(0.0f, 1.0f);
        for(auto &planet : planets)
        {
            planet.orbitOffset      = distOrbit(rng) * glm::two_pi<float>();       // random 0–2π
            planet.orbitInclination = (distOrbit(rng) * 2.0f - 1.0f) * glm::radians(30.0f); // -30°..+30°
            planet.orbitRadius      = 100.0f + distOrbit(rng) * 200.0f;            // 50–200 units
            planet.orbitSpeed       = (0.001f + distOrbit(rng)) * (planet.orbitOffset/3.0f)*(1.0f/planet.scale.length());           // 0.001–0.01 rad/sec (slow)
            std::cout<<"orbit speed: "<<planet.orbitSpeed<<"\n";
            std::coutstd::cout<<"orbitOffset: "<<planet.orbitOffset<<"\n";
            std::cout<<"1/planet.scale.length(): "<<(1/planet.scale.length())<<"\n";
        }

        for(auto &sun : suns)
        {
            sun.orbitOffset      = distOrbit(rng) * glm::two_pi<float>();       // random 0–2π
            sun.orbitInclination = (distOrbit(rng) * 2.0f - 1.0f) * glm::radians(30.0f); // -30°..+30°
            sun.orbitRadius      = 100.0f + distOrbit(rng) * 150.0f;            // 50–200 units
            sun.orbitSpeed       = (0.001f + distOrbit(rng)) * (sun.orbitOffset/3.0f)*(1.0f/sun.scale.length());           //        // 0.001–0.01 rad/sec (slow)
        }
    }

     */

    void Game::setupVEffects() {
        sunBillboards.init(64); // maxInstances, adjust based on max expected suns

    }

    void Game::renderChunks() {
        float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        std::vector<VoxelLight> globalVoxelLights;
        emissiveBillboards.clear();
        globalVoxelLights.reserve(50); // adjust depending on expected lights

        int chunkIndex = 0;

        for(int chunkX = 0; chunkX < ChunkCount; ++chunkX) {
            for(int chunkY = 0; chunkY < ChunkCount; ++chunkY) {
                for(int chunkZ = 0; chunkZ < ChunkCount; ++chunkZ) {
                    auto& chunk = data->gameWorld[chunkX][chunkY][chunkZ];
                    glm::vec3 chunkOrigin(chunkX * CHUNK_SIZE, chunkY * CHUNK_SIZE, chunkZ * CHUNK_SIZE);

                    EmissiveBlob blob{glm::vec3(0.0f), 0, glm::vec3(1.0f)};
                    std::vector<VoxelLight> voxelLights;
                    voxelLights.reserve(3);

                    if(!checkEmptyChunks(chunk)) {
                        chunkIndex++;
                        continue; // Skip empty chunks
                    }

                    for(int x = 0; x < CHUNK_SIZE; ++x) {
                        for(int y = 0; y < CHUNK_SIZE; ++y) {
                            for(int z = 0; z < CHUNK_SIZE; ++z) {
                                const auto &vox = chunk.voxels[x][y][z];

                                if(vox.type == 2) {
                                    glm::vec3 worldPos = chunkOrigin + glm::vec3(x, y, z);
                                    float intensity = 20.0f + vox.density * vox.density * 10.0f;

                                    globalVoxelLights.push_back({worldPos, intensity, vox.color});
                                    voxelLights.push_back({worldPos, intensity, vox.color});

                                    blob.sumPos += worldPos;
                                    blob.color = vox.color;
                                    blob.count++;
                                }
                            }
                        }
                    }

                    // Create billboard for chunk if emissive voxels found
                    if(blob.count > 0) {
                        glm::vec3 center = blob.sumPos / (float)blob.count;
                        float radius = pow((float)blob.count, 1.0f/2.0f);

                        SunInstance inst;
                        inst.position = center;
                        inst.scale = radius;
                        inst.color = blob.color;

                        emissiveBillboards.push_back(inst);
                    }

                    // Upload chunk geometry
                    uploadVoxelChunk(chunk, nullptr);

                    glm::vec3 chunkWorldPos = chunkOrigin;

                    resetAtomicCounter();
                    setComputeUniforms(chunkWorldPos, glm::vec3(1.0f), *marchingCubesShader);
                    dispatchCompute();

                    unsigned int vertexCount = 0;
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
                    if(vertexCount == 0) {
                        chunkIndex++;
                        continue;
                    }

                    // Setup shader
                    glm::mat4 model = glm::translate(glm::mat4(1.0f), chunkWorldPos);
                    voxelShader->use();
                    voxelShader->setMatrix("model", model);
                    voxelShader->setMatrix("mvp", pv);
                    voxelShader->setVec3("viewPos", cameraPos);
                    voxelShader->setVec3("uniformColor", glm::vec3(1.0f));
                    voxelShader->setFloat("emission", 0.1f);

                    // Pick nearest lights using squared distance
                    int maxLights = 2;
                    int numToPick = std::min(maxLights, (int)globalVoxelLights.size());
                    std::vector<VoxelLight> lightsForChunk(numToPick); // fixed-size vector

                    glm::vec3 chunkCenter = chunkOrigin + glm::vec3(CHUNK_SIZE / 2.0f);

                    std::partial_sort_copy(
                            globalVoxelLights.begin(), globalVoxelLights.end(),
                            lightsForChunk.begin(), lightsForChunk.end(),
                            [chunkCenter](const VoxelLight &a, const VoxelLight &b){
                                // Compare squared distances
                                glm::vec3 diffA = a.pos - chunkCenter;
                                glm::vec3 diffB = b.pos - chunkCenter;
                                return (diffA.x*diffA.x + diffA.y*diffA.y + diffA.z*diffA.z) <
                                       (diffB.x*diffB.x + diffB.y*diffB.y + diffB.z*diffB.z);
                            }
                    );

                    //TODO: Optimize light

                    // Option A: Only use lights in nearby chunks
                    //Maintain a spatial hash / chunk grid of voxel lights. Then each chunk only checks lights in the 27 neighboring chunks (3×3×3). That drastically reduces comparisons.
                    //
                    //Option B: Precompute nearest lights to camera
                    //If the lights’ effect falls off quickly with distance, you can select the nearest maxLights once per frame relative to the camera instead of per chunk.



                    int numLights = (int)std::min(maxLights, (int)globalVoxelLights.size());
                    voxelShader->setInt("numLights", numLights);
                    for(int i = 0; i < numLights; ++i){
                        voxelShader->setVec3("lightPos[" + std::to_string(i) + "]", lightsForChunk[i].pos);
                        voxelShader->setVec3("lightColor[" + std::to_string(i) + "]", lightsForChunk[i].color);
                        voxelShader->setFloat("lightIntensity[" + std::to_string(i) + "]", lightsForChunk[i].intensity);
                    }

                    voxelShader->setVec3("ambientColor", glm::vec3(0.05f));

                    drawTriangles(*voxelShader);

                    chunkIndex++;
                }
            }
        }

        // Render all emissive billboards
        sunBillboards.render(emissiveBillboards, view, projection, (float)glfwGetTime());
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
        Planet &planet = fluidPlanets[i];

        // Pick blue/green color
        glm::vec3 planetColor = baseColors[i % 5];

        // Upload the 32³ voxel template (binding 0)
        uploadVoxelChunk(fluidPlanetChunk, &planetColor);

        // -------------------------
        // Voxel splat into a field
        // -------------------------
        voxelSplatShader->use();
        voxelSplatShader->setVec3("gridOrigin", planet.position - 0.5f * glm::vec3(CHUNK_SIZE));
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
        marchingCubesShader->setVec3("gridOrigin", planet.position - 0.5f * glm::vec3(GRID_X,GRID_Y,GRID_Z));
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
            planet.position + (cameraPos - planet.position) * 0.25f;

        float r = (CHUNK_SIZE * 0.25f) * glm::length(planet.scale);
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

        for(auto &entity: entities) {
            entity->update(this, deltaTime);

        }
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
                    const auto &v = chunk.voxels[x][y][z];
                    const int DIM = CHUNK_SIZE + 1;

                    int idx = x + y * DIM + z * DIM * DIM;


                    voxels[idx].density = v.density;

                    if (doColorByDensity) {
                        // normalize to [0..1
                        float nv = (v.density - minD) / (maxD - minD);
                        nv = glm::clamp(nv, 0.0f, 1.0f);
                        voxels[idx].color = glm::vec4(nv, nv, nv, 1.0f);
                    } else if (overrideColor) {
                        voxels[idx].color = glm::vec4(*overrideColor, 1.0f); // uniform color
                    } else {
                        voxels[idx].color = glm::vec4(v.color, 1.0f);
                    }
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

    void Game::setComputeUniforms(const glm::vec3& position, const glm::vec3& objectScale, Shader& computeShader)
    {
        /*
        computeShader.use();

        // base voxel size (world units per voxel) you used previously
        const float baseVoxelSize = 1.0f;

        // If the object has a uniform scale, apply it by scaling voxelSize.
        // If objectScale is vec3, you can pick x (uniform scale) or average.
        float scale = (objectScale.x + objectScale.y + objectScale.z) / 3.0f;
        float effectiveVoxelSize = baseVoxelSize * scale;

        // We want the chunk centered at 'position'. The compute shader expects gridOrigin
        // to be the world position of voxel (0,0,0). So offset by half the grid extents:
        glm::vec3 halfExtents = (glm::vec3(CHUNK_SIZE - 1) * 0.5f) * effectiveVoxelSize;
        glm::vec3 gridOrigin = position - halfExtents;

        // upload uniforms
        computeShader.setVec3("gridOrigin", gridOrigin);
        computeShader.setFloat("voxelSize", effectiveVoxelSize);
        // shader expects ivec3 voxelGridDim
        int DIM = CHUNK_SIZE + 1;
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));
        */


        computeShader.use();

        float voxelSize = 1.0f;

        // gridOrigin = world position of voxel
        computeShader.setVec3("gridOrigin", position);

        computeShader.setFloat("voxelSize", voxelSize);

        int DIM = CHUNK_SIZE+1;
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


    bool Game::isOverlapping(const glm::vec3 &pos, float rad, const std::vector<gl3::Game::Planet> &others) {
        for (const Planet& p : others) {
            float r = getVoxelPlanetRadius(p.scale,(CHUNK_SIZE - 1) * 0.5f );
            float dist = glm::distance(pos, p.position);

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

    bool Game::checkEmptyChunks(gl3::Chunk chunk) {
        // Single loop: check solids & collect emissive voxels
        for(int x = 0; x < CHUNK_SIZE ; ++x) {
            for(int y = 0; y < CHUNK_SIZE; ++y) {
                for(int z = 0; z < CHUNK_SIZE; ++z) {
                    if(chunk.voxels[x][y][z].isSolid()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    void Game::simulatePhysics(const std::vector<gl3::Game::Planet> &others)
    {

    }

}
