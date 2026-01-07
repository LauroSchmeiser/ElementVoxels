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

        std::uniform_real_distribution<float> distPos(-(ChunkCount*4), (ChunkCount*4)); // Reduced range
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
        size_t touchedChunks = 0;

        // DEBUG: Track how many chunks we create
        std::unordered_set<ChunkCoord, ChunkCoordHash> createdChunks;

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

    void Game::generateChunkMesh(Chunk* chunk) {
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
        const size_t maxVertices = size_t(cellsPerAxis) * cellsPerAxis * cellsPerAxis * 5 * 3; // conservative upper bound
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
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        // Read vertex count from the atomic counter SSBO (counter stores vertex count)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        unsigned int producedVertexCount = 0;
        void* counterPtr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (counterPtr) {
            producedVertexCount = *static_cast<unsigned int*>(counterPtr);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Store vertex count into the chunk cache
        chunk->gpuCache.vertexCount = producedVertexCount;

        // Optional debug read: map the triangle SSBO for the first N vertices (safe-read)
        if (producedVertexCount > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, chunk->gpuCache.triangleSSBO);
            size_t readCount = std::min<unsigned int>(producedVertexCount, 6u);
            void* vertMap = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, readCount * sizeof(OutVertex), GL_MAP_READ_BIT);
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
        const void* posOffset    = (void*)offsetof(OutVertex, pos);
        const void* normalOffset = (void*)offsetof(OutVertex, normal);
        const void* colorOffset  = (void*)offsetof(OutVertex, color);

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
            const char* typeStr = "Unknown";
            if (shaderType == GL_COMPUTE_SHADER) typeStr = "COMPUTE";
            else if (shaderType == GL_VERTEX_SHADER) typeStr = "VERTEX";
            else if (shaderType == GL_FRAGMENT_SHADER) typeStr = "FRAGMENT";

            std::cout << "  Shader " << i << ": ID=" << shaders[i] << ", Type=" << typeStr << std::endl;
        }

        std::cout << "=== END DIAGNOSTICS ===\n\n";
    }

    void Game::renderChunks() {

        if(DebugMode2)
        {
            glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        } else
        {
            glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
        }
        float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
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
        const int R2 = RenderingRange * RenderingRange;

        int renderedChunks = 0;
        int meshRegens = 0;

        // STEP 1: Update global light spatial hash (replaces grid)
        if (frameCounter % 30 == 0) { // Update every 30 frames
            updateLightSpatialHash();
        }

        // STEP 2: First, generate meshes for dirty chunks
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
                    if(!hasSolidVoxels(*chunk)) continue;

                    // Rebuild lighting if dirty
                    if (chunk->lightingDirty) {
                        rebuildChunkLights(coord);
                        chunk->lightingDirty = false;
                    }

                    // Regenerate mesh if dirty
                    if (chunk->meshDirty || !chunk->gpuCache.isValid) {
                        generateChunkMesh(chunk);
                        meshRegens++;
                    }

                }
            }
        }

        // STEP 3: Now RENDER all visible chunks
        voxelShader->use();  // Activate RENDER shader

        // Set common uniforms that don't change per chunk

        voxelShader->setMatrix("model", glm::mat4(1.0f));
        voxelShader->setMatrix("mvp", pv);
        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.0002f));
        voxelShader->setFloat("emission", 0.0f);
// or show signed N·L for the strongest light (index 0)
        voxelShader->setInt("debugMode", 0);

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
                    for (const auto& light : chunk->emissiveLights) {
                        if (usedLightIDs.insert(light.id).second) {
                            SunInstance inst;
                            inst.position = light.pos;
                            inst.scale = std::sqrt(light.intensity) * 0.5f;
                            inst.color = light.color * 2.5f;
                            emissiveBillboards.push_back(inst);
                        }
                    }

                    //std::cout<<(chunk->gpuCache.nearbyLights.size())<<"\n";

                    // Set per-chunk uniforms (lights)
                    /*if((chunk->gpuCache.nearbyLights.size()>0)) {
                        std::cout << "lights exist for this chunk: "<< chunk->gpuCache.nearbyLights.size() << "\n";
                    }*/

                    int numLights = std::min((int)chunk->gpuCache.nearbyLights.size(), MAX_LIGHTS);
                    voxelShader->setInt("numLights", numLights);
                    for (int i = 0; i < numLights; ++i) {
                        const VoxelLight* light = chunk->gpuCache.nearbyLights[i];
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

        // Render billboards
        if (!emissiveBillboards.empty()&&!DebugMode1) {
            sunBillboards.render(emissiveBillboards, view, projection, (float)glfwGetTime());
        }

        //std::cout << "Rendered " << renderedChunks << " chunks (Regenerated: " << meshRegens << ")\n";
    }

    // Replace updateGlobalLightGrid with a spatial hash approach and a flat list and performs
    // a simple greedy merging of lights that are close to each other.
    void Game::updateLightSpatialHash() {
        lightSpatialHash.clear();
        flatEmissiveLightList.clear();
        mergedEmissiveLightPool.clear();

        // 1) Gather raw pointers (lights stored inside chunks) and fill spatial-hash as before
        std::vector<const VoxelLight*> rawLights;
        chunkManager->forEachEmissiveChunk([this, &rawLights](Chunk* chunk) {
            for (auto &light : chunk->emissiveLights) {
                // coarse bucket size (same as before)
                ChunkCoord gridCell{
                        (int)std::floor(light.pos.x / (DIM * 2)),
                        (int)std::floor(light.pos.y / (DIM * 2)),
                        (int)std::floor(light.pos.z / (DIM * 2))
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
            int amountMerged=0;

            // merge any other lights that lie within MERGE_RADIUS of rawLights[i]
            for (size_t j = i; j < rawLights.size(); ++j) {
                if (used[j]) continue;
                float d2 = glm::dot(rawLights[i]->pos - rawLights[j]->pos, rawLights[i]->pos - rawLights[j]->pos);
                if (d2 <= MERGE_RADIUS_SQ) {
                    used[j] = 1;
                    const VoxelLight* L = rawLights[j];
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
                merged.intensity = (totalIntensity/amountMerged);
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
        for (auto &m : mergedEmissiveLightPool) {
            flatEmissiveLightList.push_back(&m);
        }

        std::cout << "Light spatial hash updated: " << lightSpatialHash.size()
                  << " grid cells; raw=" << rawLights.size()
                  << " merged=" << mergedEmissiveLightPool.size() << " emissive blobs\n";
    }


    // New updateChunkLights: select up to MAX_LIGHTS by score = intensity / (distSq + 1)
// This prefers strong lights even at larger distance, while still penalizing distance.
    void Game::updateChunkLights(Chunk* chunk) {
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
        std::array<const VoxelLight*, 8> bestPtrs{};   // pointer candidates
        std::array<float, 8> bestScore{};              // score = intensity / (distSq + 1)
        int bestCount = 0;

        // keep track of the current worst score index (min score)
        float worstScore = std::numeric_limits<float>::infinity();
        int worstIndex = -1;

        for (const VoxelLight* light : flatEmissiveLightList) {
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
                chunk->gpuCache.nearbyLights.push_back(const_cast<VoxelLight*>(bestPtrs[idx[i]]));
            }
        }

        chunk->gpuCache.lastLightUpdateFrame = frameCounter;
    }

    void Game::markChunkModified(const ChunkCoord& coord) {
        Chunk* chunk = chunkManager->getChunk(coord);
        if (chunk) {
            chunk->meshDirty = true;

            // Also mark neighboring chunks because Marching Cubes needs padding
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

    void Game::unloadChunk(const ChunkCoord& coord) {
        Chunk* chunk = chunkManager->getChunk(coord);
        if (chunk) {
            // GPU resources get cleaned up in chunk destructor or clear() method
            chunk->clear(); // This deletes VAO/VBO
        }
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
                light.intensity = float(count) * 35.0f; // Scale intensity
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
        voxelShader->setVec3("emissionColor", {0,0,0});   // no glow in mesh
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
    if(!DebugMode1)
    {
        sunBillboards.render(billboardInstances, view, projection, float(glfwGetTime()));
    }
}



    void Game::update() {
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        if(glfwGetKey(window,GLFW_KEY_TAB)==GLFW_PRESS)
        {
            if(DebugMode1)
            {
                voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
                DebugMode1= false;
            }
            else
            {
                voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel_debug.frag");
                DebugMode1= true;
            }
        }
        if(glfwGetKey(window,GLFW_KEY_CAPS_LOCK)==GLFW_PRESS)
        {
            DebugMode2=(!DebugMode2);
        }


            // Update lighting only for dirty chunks
            /*chunkManager->forEachEmissiveChunk([this](Chunk *chunk) {
                if (chunk->lightingDirty) {
                    rebuildChunkLights(chunk->coord);
                }
            });*/

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

    void Game::resetAtomicCounter()
    {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);

    }

    // Set compute shader uniforms consistently (use member DIM and chunkOrigin -> gridOrigin)
    void Game::setComputeUniforms(const glm::vec3& chunkOrigin,
                                  const glm::vec3& /*objectScale*/,
                                  Shader& computeShader)
    {
        computeShader.use();

        const float voxelSize = 1.0f;
        // voxelGridDim equals DIM (number of sample points along each axis)
        computeShader.setVec3("gridOrigin", chunkOrigin); // index (0,0,0) maps to chunkOrigin
        computeShader.setFloat("voxelSize", voxelSize);
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));
    }

    void Game::dispatchCompute(Chunk* chunk) {
        int groups = (CHUNK_SIZE - 1 + 7) / 8;

        // Bind the chunk's triangle SSBO instead of the shared one
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, chunk->gpuCache.triangleSSBO); // <-- CHUNK'S SSBO

        glDispatchCompute(groups, groups, groups);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    }

    // Replace your existing drawTriangles() with this chunk-aware version.

    void Game::drawTriangles(Shader& voxelShader, Chunk* chunk)
    {
        if (!chunk || !chunk->gpuCache.isValid) return;

        // Ensure compute writes are visible to vertex fetch and buffer readbacks
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

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

        const void* posOffset    = (void*)offsetof(gl3::OutVertex, pos);
        const void* normalOffset = (void*)offsetof(gl3::OutVertex, normal);
        const void* colorOffset  = (void*)offsetof(gl3::OutVertex, color);

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
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertexCount);

        // Unbind VAO/ARRAY_BUFFER to avoid accidental state leaks
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
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
