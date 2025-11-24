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


    Game::Game(int width, int height, const std::string &title) {
        windowWidth=width;
        windowHeight=height;
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

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
        if(glGetError() != GL_NO_ERROR) {
            throw std::runtime_error("gl error");
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
        // Prepare SSBOs and static tables
        size_t voxelCount = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        size_t maxVerts = CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE * 5 * 3;

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

        // --- Setup shaders ---
        Shader voxelShader("shaders/voxel.vert", "shaders/voxel.frag");
        Shader computeShader("shaders/marching_cubes.comp");
        // Create chunks and set densities properly (density = radius - dist)
        Chunk baseChunk;
        Chunk sunChunk;
        // inside run(), replace the chunk population loops with this:

        const float centerOffset = (CHUNK_SIZE - 1) * 0.5f;
        const float radius = centerOffset; // radius in voxel units to roughly touch inside
        const float densityEpsilon = 1e-4f; // small bias to avoid exact zeros

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

        Chunk meteorChunk;
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

        // --- Generate mesh from voxel chunk ---
        //Mesh planetMesh = generateVoxelChunkMesh(baseChunk);
        //Mesh sunMesh = generateVoxelChunkMesh(sunChunk);

        std::vector<Planet> suns;
        std::vector<Planet> planets;
        std::vector<Planet> meteors;


        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distPos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distAxis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> distSpeed(5.0f, 10.0f);

        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);


        for (int i = 0; i < 4; ++i) {
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            if (glm::length(axis) < 0.001f) axis = glm::vec3(0, 1, 0);

            meteors.push_back({
                                      glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                      glm::vec3(1.5f),
                                      0.0f,
                                      axis,
                                      distSpeed(rng),
                                      glm::vec3(0.5f, 0.45f, 0.35f)
                              });
        }


        for (int j = 0; j < 2; ++j) {
                glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
                if (glm::length(axis) < 0.001f) axis = glm::vec3(0, 1, 0);

                suns.push_back({
                                       glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                       glm::vec3(3* distScale(rng)),
                                       0.0f,
                                       axis,
                                       distSpeed(rng),
                                       glm::vec3(distColor(rng), distColor(rng), 0)  // << ONE color per planet
                               });
            }

        for (int i = 0; i < 25; ++i) {
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            if (glm::length(axis) < 0.001f) axis = glm::vec3(0, 1, 0);

            planets.push_back({
                                      glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                      glm::vec3(distScale(rng)),
                                      0.0f,
                                      axis,
                                      distSpeed(rng),
                                      glm::vec3(distColor(rng), distColor(rng), distColor(rng))  // << ONE color per planet
                              });
        }


        // --- Camera setup ---
        cameraPos = glm::vec3(0.0f, 0.0f, 80.0f);
        cameraRotation = glm::vec2(0.0f, -90.0f);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


        /*
        // --- Main loop ---
        while (!glfwWindowShouldClose(window)) {
            update();
            updateDeltaTime();
            handleCameraInput();

            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.2f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            voxelShader.use();

            for(auto & sun : suns){
                // Each planet spins around its own random axis
                sun.rotationAngle += deltaTime * sun.rotationSpeed;
                if (sun.rotationAngle > 360.0f) sun.rotationAngle -= 360.0f;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, sun.position);
                model = glm::rotate(model, glm::radians(sun.rotationAngle), sun.rotationAxis);
                model = glm::scale(model, sun.scale);

                glm::mat4 mvp = calculateMvpMatrix(sun.position, sun.rotationAngle, sun.scale);

                voxelShader.setVec3("uniformColor", sun.color);
                float combinedLight=1.0f;
                glm::vec3 combinedDir= glm::vec3(0,0,0);
                combinedLight= 1.0f;
                combinedDir=glm::vec3(0,0,0);
                voxelShader.setFloat("emission", 1.0f);
                voxelShader.setFloat("lightIntensity",combinedLight);
                voxelShader.setMatrix("model", model);
                voxelShader.setVec3("lightPos",combinedDir);
                voxelShader.setMatrix("mvp", mvp);
                sunMesh.draw();
            }

            for (auto &planet : planets) {
                // Each planet spins around its own random axis
                planet.rotationAngle += deltaTime * planet.rotationSpeed;
                if (planet.rotationAngle > 360.0f) planet.rotationAngle -= 360.0f;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, planet.position);
                model = glm::rotate(model, glm::radians(planet.rotationAngle), planet.rotationAxis);
                model = glm::scale(model, planet.scale);

                glm::mat4 mvp = calculateMvpMatrix(planet.position, planet.rotationAngle, planet.scale);

                voxelShader.setVec3("uniformColor", planet.color);
                float combinedLight = 0.0f;
                glm::vec3 lightDir = glm::vec3(0);

                for (auto &sun : suns) {
                    float dist = glm::distance(planet.position, sun.position);

                    combinedLight += 100 / (dist * dist);
                    lightDir += glm::normalize(sun.position - planet.position);
                }

                lightDir = glm::normalize(lightDir);

                voxelShader.setFloat("emission", 0.0f);
                voxelShader.setFloat("lightIntensity",combinedLight);
                voxelShader.setMatrix("model", model);
                voxelShader.setVec3("lightPos",lightDir);
                voxelShader.setMatrix("mvp", mvp);
                planetMesh.draw();
            }

            //TODO new code
            mcShader.use(); // use your marching cubes compute shader

            // --- Render Suns ---
            for (auto &sun : suns) {
                uploadVoxelChunk(sunChunk);       // step 3
                resetAtomicCounter();              // step 4
                setComputeUniforms(sun.position); // step 5a
                dispatchCompute();                 // step 5b
                drawTriangles();                   // step 5c
            }



            glfwSwapBuffers(window);
            glfwPollEvents();
        }
         */

        SunBillboard sunBillboards;
        sunBillboards.init(16);

        while (!glfwWindowShouldClose(window)) {
            update();
            updateDeltaTime();
            handleCameraInput();

            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // --- Rotate suns and planets ---
            UpdateRotation(suns);
            UpdateRotation(planets);

            computeShader.use(); // use your marching cubes compute shader

            // compute PV once per frame (do this outside the loops)
            float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
            glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 pv = projection * view;
            glm::mat4 identityModel = glm::mat4(1.0f);

            // build instances
            std::vector<SunInstance> instances;
            instances.reserve(suns.size());

            // For each sun: upload voxels, reset counter, set uniforms, dispatch compute, read vertex count, draw
            for (auto &sun : suns) {
                uploadVoxelChunk(sunChunk, &sun.color);           // upload densities/colors, uses binding 0
                resetAtomicCounter();                     // zero counter in binding 3
                setComputeUniforms(sun.position, sun.scale, computeShader);
                dispatchCompute();

                // read debug vertex count
                unsigned int vertexCount = 0;
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
                //Vertex-Count Debug
                //std::cout << "[compute] sun vertexCount = " << vertexCount << std::endl;

                // --- Set voxel rendering shader uniforms for this sun BEFORE drawing ---
                // render
                voxelShader.use();
                voxelShader.setMatrix("model", identityModel);  // IMPORTANT: identity
                voxelShader.setMatrix("mvp", pv);               // PV only (positions are world-space)
                voxelShader.setVec3("viewPos", cameraPos);
                voxelShader.setVec3("lightPos", sun.position);
                voxelShader.setFloat("lightIntensity", 40.0f);
                voxelShader.setFloat("emission", 3.0f);
                voxelShader.setVec3("emissionColor", sun.color);

                voxelShader.setVec3("uniformColor", sun.color);

                // draw the vertices produced by the compute shader
                drawTriangles(voxelShader);

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


            // Render planets
            for (auto &planet : planets) {
                uploadVoxelChunk(baseChunk, &planet.color);
                resetAtomicCounter();
                setComputeUniforms(planet.position, planet.scale, computeShader);
                dispatchCompute();

                unsigned int vertexCount = 0;
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
                //Vertex-Count Debug
                //std::cout << "[compute] planet vertexCount = " << vertexCount << std::endl;

                // choose top-N lights to send to shader
                constexpr int MAX_LIGHTS = 4;
                struct LightEntry {
                    glm::vec3 pos;
                    glm::vec3 color;
                    float intensity; // scaled brightness including inverse-square
                };

                // build list of candidate lights (with computed intensity)
                std::vector<LightEntry> candidates;
                candidates.reserve(suns.size());

                const float EPS = 1e-5f;
                for (auto &s : suns) {
                    float dist = glm::distance(planet.position, s.position);
                    float invSq = 2.0f / (dist * dist + EPS);

                    // sunBrightness: tune to your scene. using scale length as proxy
                    float sunBase = glm::length(s.scale)* glm::length(s.scale) * (125.0f); // tweak multiplier
                    float intensity = sunBase * invSq;

                    candidates.push_back({ s.position, s.color, intensity });
                }

                // sort by intensity descending
                std::sort(candidates.begin(), candidates.end(),
                          [](const LightEntry &a, const LightEntry &b) {
                              return a.intensity > b.intensity;
                          });

                // take top MAX_LIGHTS
                int numLights = std::min<int>((int)candidates.size(), MAX_LIGHTS);

                // optional: if all intensities tiny, you can set numLights = 0

                // send uniforms
                voxelShader.use();
                voxelShader.setInt("numLights", numLights);

                for (int i = 0; i < numLights; ++i) {
                    std::string idx = std::to_string(i);
                    voxelShader.setVec3(("lightPos[" + idx + "]").c_str(), candidates[i].pos);
                    voxelShader.setVec3(("lightColor[" + idx + "]").c_str(), candidates[i].color);
                    voxelShader.setFloat(("lightIntensity[" + idx + "]").c_str(), candidates[i].intensity);
                }

                // fallback ambient if desired
                voxelShader.setVec3("ambientColor", glm::vec3(0.001f)); // very small ambient term

                // other per-object uniforms
                voxelShader.setMatrix("model", identityModel);
                voxelShader.setMatrix("mvp", pv);
                voxelShader.setVec3("viewPos", cameraPos);
                voxelShader.setFloat("emission", 0.0f);
                voxelShader.setVec3("emissionColor", glm::vec3(0.0f));
                voxelShader.setVec3("uniformColor", planet.color); // albedo


                drawTriangles(voxelShader);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

    }

    void Game::UpdateRotation(std::vector<Planet>& planets)
    {
        for(auto & planet : planets){
            planet.rotationAngle += deltaTime * planet.rotationSpeed;
            if (planet.rotationAngle > 360.0f) planet.rotationAngle -= 360.0f;
        }
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

        size_t voxelCount = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
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

        for(int x=0;x<CHUNK_SIZE;x++) {
            for(int y=0;y<CHUNK_SIZE;y++) {
                for(int z=0;z<CHUNK_SIZE;z++) {
                    const auto &v = chunk.voxels[x][y][z];
                    int idx = x + y*CHUNK_SIZE + z*CHUNK_SIZE*CHUNK_SIZE;

                    voxels[idx].density = v.density;

                    if (doColorByDensity) {
                        // normalize to [0..1]
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
        computeShader.setIVec3("voxelGridDim", glm::ivec3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE));
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

        // Use output SSBO as VBO
        glBindBuffer(GL_ARRAY_BUFFER, ssboTriangles);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)0);                   // position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)(sizeof(glm::vec4))); // normal
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)(2 * sizeof(glm::vec4))); // color

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);

        unsigned int vertexCount = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);

        voxelShader.use();

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        glDeleteVertexArrays(1, &vao);
    }

}
