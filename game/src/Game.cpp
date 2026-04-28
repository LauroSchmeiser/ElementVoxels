#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/Shader.h"
#include "rendering/marchingTables.h"
#include "rendering/SunBillboard.h"
#include <tracy/Tracy.hpp>
#include <cstdint>
#include "rendering/GpuStructsStd430.h"
#include "tracy/TracyOpenGL.hpp"
#include "scenes/MainMenuScene.h"
#include "scenes/GameplayScene.h"
#include "scenes/LoadingScene.h"
#include "scenes/GameOverScene.h"
#include <imgui.h>
#include "entities/EnemyManager.h"
#include "physics//DestructibleObject.h"
#include "Entities/LocalMarchingCubes.h"

#ifndef TRACY_GPU_ENABLED
#define TRACY_CPU_ZONE(nameStr) ZoneScopedN(nameStr)
#define TRACY_GPU_ZONE(nameStr) TracyGpuZone(nameStr)
#endif
namespace gl3 {

////-----Basics---------------------------------------------------------------------------------------------------------------------------------
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

    // Put near top of Game.cpp (static in translation unit)
    static GLuint gPreviewCubeVAO = 0;
    static GLuint gPreviewCubeVBO = 0;

    void Game::ensurePreviewCube() {
        if (previewCubeVAO != 0) return;

        // positions+dummy normals, 36 verts
        const float cubeVerts[] = {
                // pos                normal
                -0.5f,-0.5f,-0.5f,   0,0,1,  0.5f,-0.5f,-0.5f,   0,0,1,  0.5f, 0.5f,-0.5f,   0,0,1,
                0.5f, 0.5f,-0.5f,   0,0,1, -0.5f, 0.5f,-0.5f,   0,0,1, -0.5f,-0.5f,-0.5f,   0,0,1,

                -0.5f,-0.5f, 0.5f,   0,0,1,  0.5f,-0.5f, 0.5f,   0,0,1,  0.5f, 0.5f, 0.5f,   0,0,1,
                0.5f, 0.5f, 0.5f,   0,0,1, -0.5f, 0.5f, 0.5f,   0,0,1, -0.5f,-0.5f, 0.5f,   0,0,1,

                -0.5f, 0.5f, 0.5f,   0,0,1, -0.5f, 0.5f,-0.5f,   0,0,1, -0.5f,-0.5f,-0.5f,   0,0,1,
                -0.5f,-0.5f,-0.5f,   0,0,1, -0.5f,-0.5f, 0.5f,   0,0,1, -0.5f, 0.5f, 0.5f,   0,0,1,

                0.5f, 0.5f, 0.5f,   0,0,1,  0.5f, 0.5f,-0.5f,   0,0,1,  0.5f,-0.5f,-0.5f,   0,0,1,
                0.5f,-0.5f,-0.5f,   0,0,1,  0.5f,-0.5f, 0.5f,   0,0,1,  0.5f, 0.5f, 0.5f,   0,0,1,

                -0.5f,-0.5f,-0.5f,   0,0,1,  0.5f,-0.5f,-0.5f,   0,0,1,  0.5f,-0.5f, 0.5f,   0,0,1,
                0.5f,-0.5f, 0.5f,   0,0,1, -0.5f,-0.5f, 0.5f,   0,0,1, -0.5f,-0.5f,-0.5f,   0,0,1,

                -0.5f, 0.5f,-0.5f,   0,0,1,  0.5f, 0.5f,-0.5f,   0,0,1,  0.5f, 0.5f, 0.5f,   0,0,1,
                0.5f, 0.5f, 0.5f,   0,0,1, -0.5f, 0.5f, 0.5f,   0,0,1, -0.5f, 0.5f,-0.5f,   0,0,1
        };

        glGenVertexArrays(1, &previewCubeVAO);
        glGenBuffers(1, &previewCubeVBO);

        glBindVertexArray(previewCubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, previewCubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    void Game::ensurePreviewSphereMesh() {
        if (previewSphereVAO != 0) return;

        // Pick any sphere mesh from cache (we just need topology; we scale in model)
        if (sphereMeshCache.empty()) {
            std::cerr << "ensurePreviewSphereMesh: sphereMeshCache is empty\n";
            return;
        }

        const SphereMesh& m = sphereMeshCache.begin()->second;

        // Interleave pos+normal for your shader layout (loc0 pos, loc1 normal)
        struct PVN { glm::vec3 p; glm::vec3 n; };
        std::vector<PVN> vtx;
        vtx.reserve(m.vertices.size());
        for (size_t i = 0; i < m.vertices.size(); ++i) {
            vtx.push_back({ m.vertices[i] / m.radius, m.normals[i] });
            // NOTE: divide by m.radius to make it a unit sphere in object space
            // so model scaling by radius works cleanly.
        }

        glGenVertexArrays(1, &previewSphereVAO);
        glGenBuffers(1, &previewSphereVBO);
        glGenBuffers(1, &previewSphereEBO);

        glBindVertexArray(previewSphereVAO);

        glBindBuffer(GL_ARRAY_BUFFER, previewSphereVBO);
        glBufferData(GL_ARRAY_BUFFER, vtx.size() * sizeof(PVN), vtx.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, previewSphereEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.indices.size() * sizeof(uint32_t), m.indices.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PVN), (void*)offsetof(PVN, p));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PVN), (void*)offsetof(PVN, n));

        glBindVertexArray(0);

        previewSphereIndexCount = (GLsizei)m.indices.size();
    }

    void Game::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        if (height == 0) height = 1; // prevent divide-by-zero
        glViewport(0, 0, width, height);

        Game *game = static_cast<Game *>(glfwGetWindowUserPointer(window));
        if (game) {
            game->windowWidth = width;
            game->windowHeight = height;
            game->initPostFBO();
        }
    }


    Game::Game(int width, int height, const std::string &title)
            : windowWidth(width),
              windowHeight(height),
              chunkManager(std::make_unique<FixedGridChunkManager>(WORLD_RADIUS_CHUNKS)),
              cameraPos(0.0f, 0.0f, 35.0f),
              cameraRotation(-90.0f, 0.0f)
            {
        //Window Setup
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

        //UI Setup
        imguiLayer.init(window, "#version 460");

        //Shader Setup
        skyboxRuntimeShader = std::make_unique<Shader>("shaders/skybox.vert", "shaders/skybox_runtime.frag");
        voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");
        spellPreviewShader = std::make_unique<Shader>("shaders/spell_prev.vert", "shaders/spell_prev.frag");
        postShader = std::make_unique<Shader>("shaders/post_fullscreen.vert", "shaders/post_fog_glow.frag");

        try {
            voxelSplatShader = std::make_unique<Shader>("shaders/metaball_splat.comp");
        } catch (std::exception &e) {
            std::cerr << "Failed to create metaballSplatShader: " << e.what() << std::endl;
        }


        //Sound Setup
        audio.init();
        audio.setGlobalVolume(0.1f);

        //SceneManager Setup
        sceneManager.registerScene(SceneId::MainMenu, std::make_unique<MainMenuScene>());
        sceneManager.registerScene(SceneId::Gameplay, std::make_unique<GameplayScene>());
        sceneManager.registerScene(SceneId::Loading, std::make_unique<LoadingScene>());
        sceneManager.registerScene(SceneId::GameOver, std::make_unique<GameOverScene>());

        sceneManager.requestChange(SceneId::MainMenu);
        sceneManager.applyPendingChange();
    }



    Game::~Game() {
        imguiLayer.shutdown();
        shutdownSpellCastAsync();
        glfwTerminate();
    }


    void Game::initGameplayIfNeeded() {
        if (gameplayInitialized) return;
        initGameplaySystems();
        gameplayInitialized = true;
    }

    void Game::initGameplaySystems() {
        ////Initialization-Steps
        setupSkybox();
        bakeNebulaCubemap(512);
        setupSSBOsAndTables();
        setupPhysics();
        setupControls();
        generateAssets();
        setupInput();
        generateChunks();
        setupCamera();
        setupVEffects();
        fillMaterialTable();
    }

    void Game::fillMaterialTable()
    {
        std::vector<std::string> mats;
        mats.push_back(gl3::resolveAssetPath("textures/cobble.jpg").string());
        mats.push_back(gl3::resolveAssetPath("textures/wood.jpg").string());
        //mats.push_back(gl3::resolveAssetPath("textures/cobble.jpg").string());

        mats.push_back(gl3::resolveAssetPath("textures/water.png").string());
        mats.push_back(gl3::resolveAssetPath("textures/dirt.jpg").string());
        mats.push_back(gl3::resolveAssetPath("textures/water2.jpg").string());
        mats.push_back(gl3::resolveAssetPath("textures/metal.jpg").string());
        mats.push_back(gl3::resolveAssetPath("textures/cobble.jpg").string());

        //mats.push_back(gl3::resolveAssetPath("textures/gem.jpg").string());


        materials.initTextureArrayFromFiles(mats);
        materialAlbedoArrayTexId = materials.albedoArrayTex;

        materials.params[0].roughness = 1.0f;
        materials.params[0].specular  = 0.05f;
        materials.params[0].uvScale   = 0.05f;

        materials.params[1].roughness = 0.5f;
        materials.params[1].specular  = 0.15f;
        materials.params[1].uvScale   = 0.05f;

        materials.params[2].roughness = 0.05f;
        materials.params[2].specular  = 0.5f;
        materials.params[2].uvScale   = 0.02f;

        materials.params[3].roughness = 1.0f;
        materials.params[3].specular  = 0.05f;
        materials.params[3].uvScale   = 0.05f;

        materials.params[4].roughness = 1.0f;
        materials.params[4].specular  = 0.05f;
        materials.params[4].uvScale   = 0.005f;

        materials.params[5].roughness = 1.0f;
        materials.params[5].specular  = 0.05f;
        materials.params[5].uvScale   = 0.005f;

        materials.params[6].roughness = 1.0f;
        materials.params[6].specular  = 0.05f;
        materials.params[6].uvScale   = 0.05f;

        materials.params[7].roughness = 1.0f;
        materials.params[7].specular  = 0.05f;
        materials.params[7].uvScale   = 0.05f;

        for (int i = 0; i < 64; ++i) {
            rough[i] = materials.params[i].roughness;
            spec[i]  = materials.params[i].specular;
            uvScale[i] = materials.params[i].uvScale;
        }
    }

    void Game::generateAssets()
    {
        initSphereMeshCache();
        initSpellCastAsync();
    }

    void Game::setupControls() {
        characterController = std::make_unique<CharacterController>(
                chunkManager.get(),
                voxelPhysics.get(),
                1.8f,
                1.0f
        );

        //player-body collision callback
        characterController->setPlayerBodyCollisionCallback(
                [this](gl3::VoxelPhysicsBody* body,
                       const glm::vec3& hitPos,
                       const glm::vec3& hitNormal,
                       float playerSpeed) {
                    onPlayerBodyCollision(body, hitPos, hitNormal, playerSpeed);
                }
        );

    }

    void Game::setupPhysics() {
        voxelPhysics = std::make_unique<VoxelPhysicsManager>(chunkManager.get());
        // body-body collision callback
        voxelPhysics->setBodyBodyCollisionCallback(
                [this](gl3::VoxelPhysicsBody* bodyA,
                       gl3::VoxelPhysicsBody* bodyB,
                       const glm::vec3& hitPos,
                       const glm::vec3& hitNormal,
                       float impactSpeed) {
                    onBodyBodyCollision(bodyA, bodyB, hitPos, hitNormal, impactSpeed*impactSpeed);
                }
        );
        // body-world collision callback
        voxelPhysics->setVoxelCollisionCallback(
                [this](gl3::VoxelPhysicsBody* body,
                       const glm::vec3& hitPos,
                       const glm::vec3& hitNormal,
                       float impactSpeed) {
                    onSpellCollision(static_cast<SpellEffect*>(body->userData),
                                     hitPos, hitNormal, impactSpeed);
                }
        );

        enemyManager = std::make_unique<EnemyManager>();
        enemyManager->init(voxelPhysics.get(),chunkManager.get(), this);

        static EnemyArchetype basic;
        basic.name = "Basic";
        basic.maxHP = 1.0f;
        basic.moveSpeed = 6.0f;
        basic.shapeType = VoxelPhysicsBody::ShapeType::SPHERE;
        basic.mass = 50.0f;
        basic.radius = 2.5f * VOXEL_SIZE;
        basic.cooldownsSec = { 2.0f, 0.0f, 0.0f };

        enemyManager->spawn(basic, cameraPos + getCameraFront() * (35.0f * VOXEL_SIZE));
    }

    void Game::updateGameplayFrame() {
        updateDeltaTime();
        pumpAsyncSpellResults();
        updateSpells(deltaTime);

        glfwPollEvents();
        update();
        if (enemyManager) {
            enemyManager->update(deltaTime, cameraPos /* player pos */);
        }
        updatePhysics();
        updateChunkBurns(deltaTime);
    }

    void Game::renderGameplayFrame() {
        // 1) Render scene into postFBO
        glBindFramebuffer(GL_FRAMEBUFFER, postFBO);
        glViewport(0, 0, windowWidth, windowHeight);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(!DebugMode1) renderSkybox();
        renderChunks();
        renderAnimatedVoxels();
        renderPhysicsFormations();
        renderEnemies();
        renderSpellPreview();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        // 2) Post pass to screen (fog + glow)
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        postShader->use();

        postShader->setInt("uSceneColor", 0);
        postShader->setInt("uSceneDepth", 1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, postColorTex);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, postDepthTex);

        postShader->setFloat("uNear", 0.1f);
        postShader->setFloat("uFar", 500.0f);

        /*postShader->setFloat("uExposure", 1.0f);
        postShader->setFloat("uGamma", 2.2f);

        postShader->setFloat("uFogStrength", 3.0f);
        postShader->setFloat("uFogStart", 50.0f);
        postShader->setFloat("uFogEnd", 1050.0f);

        postShader->setFloat("uGlowThreshold", 2.0f);  // if HDR-ish
        postShader->setFloat("uGlowStrength", 0.85f);

        postShader->setVec2("uInvResolution", glm::vec2(1.0f / windowWidth, 1.0f / windowHeight));
*/
        glBindVertexArray(postVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);


        // UI and other stuff AFTER post
        DisplayFPSCount();
        renderGameplayUI();
    }

    void gl3::Game::beginGameplayPreload(bool newRun)
    {
        doNewRun = newRun;

        gameplayInitialized = false;
        preloadStage = PreloadStage::NotStarted;
        preloadStageName = "Starting...";
    }

    float gl3::Game::tickGameplayPreload()
    {
        if (gameplayInitialized) {
            preloadStageName = "Ready";
            return 1.0f;
        }

        if (preloadStage == PreloadStage::NotStarted) {
            if (!bootLoaded) {
                preloadStage = PreloadStage::Boot_Skybox;
            } else if (doNewRun) {
                preloadStage = PreloadStage::Run_Physics;
                setPlayerHealth(getPlayerMaxHealth());
            } else {
                preloadStage = PreloadStage::Done;
            }
        }

        switch (preloadStage)
        {
            // ---------------- immutable boot ----------------
            case PreloadStage::Boot_Skybox:
                preloadStageName = "Preparing skybox...";
                setupSkybox();
                preloadStage = PreloadStage::Boot_Nebula;
                return 0.10f;

            case PreloadStage::Boot_Nebula:
                preloadStageName = "Baking nebula cubemap...";
                bakeNebulaCubemap(512);
                preloadStage = PreloadStage::Boot_PostProcessor;
                return 0.20f;

            case PreloadStage::Boot_PostProcessor:
                preloadStageName = "Preparing Post-processing...";
                initPostFBO();
                preloadStage = PreloadStage::Boot_SSBOs;
                return 0.25f;

            case PreloadStage::Boot_SSBOs:
                preloadStageName = "Setting up GPU buffers...";
                setupSSBOsAndTables();
                MAX_CHUNKS_GPU = (int)chunkManager->maxChunksGpu();
                setupChunkBatchBuffers(MAX_CHUNKS_GPU);
                setupLightSSBOs();
                preloadStage = PreloadStage::Boot_Materials;
                return 0.30f;

            case PreloadStage::Boot_Materials:
                preloadStageName = "Loading materials...";
                fillMaterialTable();
                preloadStage = PreloadStage::Boot_Assets;
                return 0.40f;

            case PreloadStage::Boot_Assets:
                preloadStageName = "Loading assets...";
                generateAssets();
                preloadStage = PreloadStage::Boot_VEffects;
                return 0.50f;

            case PreloadStage::Boot_VEffects:
                preloadStageName = "Finalizing effects...";
                setupVEffects();
                bootLoaded = true;

                preloadStage = doNewRun ? PreloadStage::Run_Physics : PreloadStage::Done;
                return 0.60f;

                // ---------------- mutable run ----------------
            case PreloadStage::Run_Physics:
                preloadStageName = "Setting up physics...";
                clearWorldAndGameplayObjects();
                setupPhysics();
                preloadStage = PreloadStage::Run_Controls;
                return 0.70f;

            case PreloadStage::Run_Controls:
                preloadStageName = "Setting up controls...";
                setupControls();
                preloadStage = PreloadStage::Run_Input;
                return 0.75f;

            case PreloadStage::Run_Input:
                preloadStageName = "Setting up input...";
                setupInput();
                preloadStage = PreloadStage::Run_World;
                return 0.80f;

            case PreloadStage::Run_World:
                preloadStageName = "Generating world...";
                generateChunks();
                preloadStage = PreloadStage::Run_Camera;
                return 0.88f;

            case PreloadStage::Run_Camera:
                preloadStageName = "Setting up camera...";
                setupCamera();
                static_assert(gl3::CHUNK_SIZE == 16);
                assert(DIM == gl3::CHUNK_SIZE + 2 && "DIM must be CHUNK_SIZE+2 for padded uploadVoxelChunk");
                preloadStage = PreloadStage::Run_Lighting;
                return 0.92f;

            case PreloadStage::Run_Lighting:
                preloadStageName = "Setting up lighting...";
                updateLightSpatialHash();
                uploadMergedLightsToGPU();
                chunkManager->forEachChunk([&](gl3::Chunk* chunk)
                {rebuildChunkLights(chunk->coord);
                    chunk->lightingDirty = false;
                    for (const auto& light : chunk->emissiveLights) {
                        if (usedLightIDs.insert(light.id).second) {
                            SunInstance inst;
                            inst.position = light.pos;
                            inst.scale = std::sqrt(light.intensity) * 0.5f;
                            inst.color = light.color * 1.0f;
                            emissiveBillboards.push_back(inst);
                        }
                    }
                });
                buildAndUploadChunkLightIndexBuffer(worldToChunk(cameraPos.x),worldToChunk(cameraPos.y),worldToChunk(cameraPos.z),RenderingRange);
                preloadStage = PreloadStage::Done;
                return 0.92f;

            case PreloadStage::Done:
            default:
                gameplayInitialized = true;
                needsNewRun = false;
                preloadStageName = "Ready";
                return 1.0f;
        }
    }

    void Game::clearWorldAndGameplayObjects()
    {
        activeSpells.clear();
        animatedVoxels.clear();
        animatedVoxelIndexMap.clear();
        nextAnimatedVoxelID = 1;

        chunkManager->clearAll();
    }

    void Game::setPaused(bool p)
    {
        if (paused == p) return;
        paused = p;

        if (paused) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            hasPreviousMousePos = false;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            hasPreviousMousePos = false;
        }
    }

    void Game::togglePaused()
    {
        setPaused(!paused);
    }

    void Game::renderGameplayUI()
    {
        // HUD + pause menu overlay
        imguiLayer.beginFrame();

        // ----------------
        // (A) Health bar
        // ----------------
        {
            ImGuiWindowFlags flags =
                    ImGuiWindowFlags_NoDecoration |
                    ImGuiWindowFlags_AlwaysAutoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoNav;

            const ImVec2 pad(12.0f, 12.0f);
            ImGui::SetNextWindowPos(pad, ImGuiCond_Always);

            ImGui::SetNextWindowBgAlpha(0.35f);

            if (ImGui::Begin("HUD_Health", nullptr, flags))
            {
                float maxH = glm::max(1.0f, playerMaxHealth);
                float frac = glm::clamp(playerHealth / maxH, 0.0f, 1.0f);

                ImGui::TextUnformatted("Health");
                ImGui::SetWindowFontScale(2);
                ImGui::PushItemWidth(320.0f);

                ImVec4 col = ImVec4(1.0f - frac, frac, 0.15f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
                ImGui::ProgressBar(frac, ImVec2(320.0f, 30.0f));
                ImGui::PopStyleColor();

                ImGui::Text("%d / %d", (int)playerHealth, (int)playerMaxHealth);
                ImGui::PopItemWidth();
            }
            ImGui::End();
        }

        // ----------------
        // (B) Pause menu
        // ----------------
        if (paused)
        {
            ImGuiIO& io = ImGui::GetIO();
            const ImVec2 center(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);

            ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(420, 220), ImGuiCond_Always);

            ImGuiWindowFlags flags =
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoTitleBar;

            ImGui::SetNextWindowBgAlpha(0.65f);

            if (ImGui::Begin("PauseMenu", nullptr, flags))
            {
                ImGui::SetWindowFontScale(2.0f);
                ImGui::SetCursorPosY(18.0f);
                ImGui::SetCursorPosX((420.0f - ImGui::CalcTextSize("Paused").x) * 0.5f);
                ImGui::TextUnformatted("Paused");
                ImGui::SetWindowFontScale(1.0f);

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                const ImVec2 btnSize(420.0f * 0.75f, 44.0f);
                ImGui::SetCursorPosX((420.0f - btnSize.x) * 0.5f);

                if (ImGui::Button("Resume", btnSize)) {
                    setPaused(false);
                }

                ImGui::Spacing();
                ImGui::SetCursorPosX((420.0f - btnSize.x) * 0.5f);

                if (ImGui::Button("Back to Main Menu", btnSize)) {
                    setPaused(false);
                    requestSceneChange(SceneId::MainMenu);
                }

                ImGui::Spacing();
                ImGui::SetCursorPosX((420.0f - btnSize.x) * 0.5f);

                if (ImGui::Button("Exit to Desktop", btnSize)) {
                    glfwSetWindowShouldClose(getWindow(), true);
                }
            }
            ImGui::End();
        }

        imguiLayer.endFrame();
    }

    void Game::initPostFBO()
    {
        // Cleanup if re-init (e.g. resize)
        if (postFBO) {
            glDeleteFramebuffers(1, &postFBO);
            glDeleteTextures(1, &postColorTex);
            glDeleteTextures(1, &postDepthTex);
            postFBO = postColorTex = postDepthTex = 0;
        }

        glGenFramebuffers(1, &postFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, postFBO);

        // Color: HDR-ish so glow has headroom
        glGenTextures(1, &postColorTex);
        glBindTexture(GL_TEXTURE_2D, postColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postColorTex, 0);

        // Depth: MUST be a texture to sample in post
        glGenTextures(1, &postDepthTex);
        glBindTexture(GL_TEXTURE_2D, postDepthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, postDepthTex, 0);

        GLenum drawBufs[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, drawBufs);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "postFBO incomplete!\n";
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Fullscreen tri + shader
        if (!postVAO) CreateFullscreenTriangle(postVAO, postVBO);

        // Load your post shader (paths should match resolveAssetPath usage)
        postShader = std::make_unique<Shader>("shaders/post_fullscreen.vert", "shaders/post_fog_glow.frag");
    }

////-----Run-Method-----------------------------------------------------------------------------------------------------------------------------

    void Game::run() {
        TracyGpuContext
        TRACY_CPU_ZONE("Game::run");

        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glfwPollEvents();

            if (sceneManager.current()) {
                sceneManager.current()->update(*this, deltaTime);
            }

            sceneManager.applyPendingChange();

            if (sceneManager.current()) {
                sceneManager.current()->render(*this);
            }

            glfwSwapBuffers(window);
            TracyGpuCollect;
            FrameMark;
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

    int Game::worldToChunk(float worldPos) {
        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        return (int)std::floor(worldPos / chunkWorldSize);
    }

    glm::vec3 Game::getChunkMin(const ChunkCoord& coord) const {
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
            if (!chunk->isCleared) chunk->meshDirty = true;

            for (int dx=-1; dx<=1; ++dx)
                for (int dy=-1; dy<=1; ++dy)
                    for (int dz=-1; dz<=1; ++dz) {
                        ChunkCoord neighbor{coord.x + dx, coord.y + dy, coord.z + dz};
                        Chunk *neighborChunk = chunkManager->getChunk(neighbor);
                        if (neighborChunk) {
                            if (!neighborChunk->isCleared) neighborChunk->meshDirty = true;
                            neighborChunk->lightingDirty = true;
                        }
                    }
        }
    }

    void Game::unloadChunk(const ChunkCoord &coord) {
        Chunk *chunk = chunkManager->getChunk(coord);
        if (chunk) {
            chunk->clear();
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
        SpellEffect spell;
        spell.type = SpellEffect::Type::CONSTRUCT;
        spell.center = center;
        spell.radius = radius;
        spell.strength = strength;
        spell.targetMaterial = targetMaterial;
        spell.dominantType = 0;
        spell.formationParams = baseFormationParams;

        std::vector<AnimatedVoxel> visualVoxels;
        findNearbyVoxelsForVisualNew(center, radius, targetMaterial,
                                  visualVoxels, strength, spell.dominantType);

        if (visualVoxels.empty()) {
            std::cout << "No voxels found for spell\n";
            return;
        }

        glm::vec3 avgColor(0.0f);
        const size_t collected = visualVoxels.size();
        const float voxelVolumeWorld = gl3::VOXEL_SIZE * gl3::VOXEL_SIZE * gl3::VOXEL_SIZE;
        float desiredVolumeWorld = static_cast<float>(collected) * voxelVolumeWorld;

        FormationParams adjustedFormation = baseFormationParams;
        adjustFormationForVolume(adjustedFormation, desiredVolumeWorld);

        adjustedFormation.center = center;
        spell.formationParams = adjustedFormation;
        spell.center = adjustedFormation.center;

        for (size_t i = 0; i < visualVoxels.size(); ++i) {
            AnimatedVoxel &v = visualVoxels[i];
            avgColor += v.color;

            v.targetPos = calculateFormationTarget(i, visualVoxels.size(), adjustedFormation);

            v.animationSpeed = (glm::pow(strength,2)/glm::sqrt(radius))*2;
            v.animationSpeed=glm::min(v.animationSpeed,20.5f);
            v.animationSpeed=glm::max(v.animationSpeed,1.5f);

            v.isAnimating = true;
            v.hasArrived = false;
        }
        avgColor /= (float)visualVoxels.size();

        spell.formationColor = avgColor;
        spell.formationRadius = adjustedFormation.getBoundingRadius();

        for (auto &v : visualVoxels) {
            v.id = nextAnimatedVoxelID++;
            animatedVoxels.push_back(v);
            animatedVoxelIndexMap[v.id] = animatedVoxels.size() - 1;
            spell.animatedVoxelIDs.push_back(v.id);
        }

        activeSpells.push_back(spell);

        std::cout << "Spell cast! " << visualVoxels.size()
                  << " voxels moving to formation (adjusted radius="
                  << adjustedFormation.radius << ")\n";
    }

    void Game::adjustFormationForVolume(FormationParams& params, float volume /*world^3*/) {
        const float packingEfficiency = 0.3f;
        constexpr float PI = 3.14159265358979323846f;

        const float minWorldDim = gl3::VOXEL_SIZE * 0.15f;
        const float maxScaleFactor = 20.0f;

        switch(params.type) {
            case FormationType::SPHERE: {
                float computedRadius = std::cbrt((3.0f / (4.0f * PI)) * (volume/2 / packingEfficiency));
                float maxRadius = std::max(minWorldDim, params.radius * 0.75f);
                float minRadius = minWorldDim;
                params.radius = glm::clamp(computedRadius, minRadius, maxRadius);
                break;
            }
            case FormationType::PLATFORM: {
                float area = volume / (params.sizeY * packingEfficiency);
                float side = std::sqrt(glm::max(0.0f, area));
                params.sizeX = glm::max(side, minWorldDim);
                params.sizeZ = glm::max(side, minWorldDim);
                break;
            }
            case FormationType::WALL: {
                float area = volume / (params.sizeZ * packingEfficiency)*20;
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

    glm::vec3 Game::calculateFormationTarget(size_t index, size_t total,
                                             const FormationParams& params) {
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
                return calculateSphereDistribution(index, total, params);
        }
    }

    glm::vec3 Game::calculateSphereDistribution(size_t index, size_t total,
                                                const FormationParams& params) {
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
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos(
                u * params.sizeX,
                params.sizeY * 0.5f,
                v * params.sizeZ
        );

        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateWallDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos(
                u * params.sizeX,
                v * params.sizeY,
                params.sizeZ * 0.5f
        );

        glm::vec3 right = glm::normalize(glm::cross(params.normal, params.up));
        glm::mat3 rotation(right, params.up, params.normal);

        glm::vec3 worldPos = params.center + rotation * localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateCubeDistribution(size_t index, size_t total,
                                              const FormationParams& params) {
        int faceIndex = index % 6;
        float u = haltonSequence(index, 2) - 0.5f;
        float v = haltonSequence(index, 3) - 0.5f;

        glm::vec3 localPos;
        switch(faceIndex) {
            case 0:
                localPos = glm::vec3(params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 1:
                localPos = glm::vec3(-params.sizeX * 0.5f, u * params.sizeY, v * params.sizeZ);
                break;
            case 2:
                localPos = glm::vec3(u * params.sizeX, params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 3:
                localPos = glm::vec3(u * params.sizeX, -params.sizeY * 0.5f, v * params.sizeZ);
                break;
            case 4:
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, params.sizeZ * 0.5f);
                break;
            case 5:
                localPos = glm::vec3(u * params.sizeX, v * params.sizeY, -params.sizeZ * 0.5f);
                break;
        }

        glm::vec3 worldPos = params.center + localPos;
        return worldPos;
    }

    glm::vec3 Game::calculateCylinderDistribution(size_t index, size_t total,
                                                  const FormationParams& params) {
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
        int surface = index % 5;

        if (surface < 4) {
            float u = haltonSequence(index, 2);
            float v = haltonSequence(index, 3);

            float baseX = (u - 0.5f) * params.sizeX;
            float baseZ = (v - 0.5f) * params.sizeZ;

            glm::vec3 localPos;
            switch(surface) {
                case 0:
                    localPos = glm::vec3(baseX, 0.0f, params.sizeZ * 0.5f);
                    break;
                case 1:
                    localPos = glm::vec3(baseX, 0.0f, -params.sizeZ * 0.5f);
                    break;
                case 2:
                    localPos = glm::vec3(params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
                case 3:
                    localPos = glm::vec3(-params.sizeX * 0.5f, 0.0f, baseZ);
                    break;
            }

            float heightRatio = haltonSequence(index, 5);
            localPos.y = heightRatio * params.sizeY;
            localPos.x *= (1.0f - heightRatio);
            localPos.z *= (1.0f - heightRatio);

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        } else {
            float u = haltonSequence(index, 2) - 0.5f;
            float v = haltonSequence(index, 3) - 0.5f;

            glm::vec3 localPos(
                    u * params.sizeX,
                    0.0f,
                    v * params.sizeZ
            );

            glm::vec3 worldPos = params.center + localPos;
            return worldPos;
        }
    }

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

    int Game::estimateAvailableVoxels(const glm::vec3& center, float radius, uint64_t targetMaterial, int maxNeeded)
    {
        const float radiusSq = radius * radius;

        const int step = 2; //accuracy

        int count = 0;
        auto chunks = chunkManager->getChunksInRadius(center, radius);

        for (const auto& [coord, chunk] : chunks)
        {
            if (!chunk) continue;

            glm::vec3 chunkMin = getChunkMin(coord);

            int startX = std::max(0, static_cast<int>((center.x - radius - chunkMin.x) / VOXEL_SIZE));
            int endX   = std::min(CHUNK_SIZE, static_cast<int>((center.x + radius - chunkMin.x) / VOXEL_SIZE) + 1);
            int startY = std::max(0, static_cast<int>((center.y - radius - chunkMin.y) / VOXEL_SIZE));
            int endY   = std::min(CHUNK_SIZE, static_cast<int>((center.y + radius - chunkMin.y) / VOXEL_SIZE) + 1);
            int startZ = std::max(0, static_cast<int>((center.z - radius - chunkMin.z) / VOXEL_SIZE));
            int endZ   = std::min(CHUNK_SIZE, static_cast<int>((center.z + radius - chunkMin.z) / VOXEL_SIZE) + 1);

            for (int x = startX; x <= endX; x += step)
                for (int y = startY; y <= endY; y += step)
                    for (int z = startZ; z <= endZ; z += step)
                    {
                        const Voxel& v = chunk->voxels[x][y][z];
                        if (!v.isSolid()) continue;
                        if (v.material != targetMaterial) continue;

                        glm::vec3 worldPos = chunkMin + glm::vec3((float)x, (float)y, (float)z) * VOXEL_SIZE;
                        glm::vec3 diff = worldPos - center;
                        if (glm::dot(diff, diff) > radiusSq) continue;

                        ++count;
                        if (count >= maxNeeded) return count;
                    }
        }

        //sampling density adjustments
        if (step == 2) count *= 8;
        else if (step == 3) count *= 27;

        return count;
    }

    void Game::findNearbyVoxelsForVisual(const glm::vec3& center, float radius,
                                         uint64_t targetMaterial,
                                         std::vector<AnimatedVoxel>& results,
                                         float strength,
                                         uint8_t& outDominantType) {
        TRACY_CPU_ZONE("Game::findNearbyVoxelsForVisual");
        const float radiusSq = radius * radius;

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        int maxVoxels = static_cast<int>((strength * 70.0f) / voxelVolume);
        maxVoxels = glm::clamp(maxVoxels, 50, 200);

        const size_t hardCandidateCap = (size_t)maxVoxels * 4;
        bool stop = false;

        std::cout << "[Spell] Targeting " << maxVoxels  << " voxels for formation\n";

        int typeCounts[8] = {0};
        auto chunks = chunkManager->getChunksInRadius(center, radius);

        struct VoxelCandidate {
            glm::vec3 worldPos;
            glm::vec3 color;
            ChunkCoord chunkCoord;
            glm::ivec3 localPos;
            float distanceSq;
            Chunk* chunk;
            uint8_t type;
        };

        std::vector<VoxelCandidate> candidates;
        candidates.reserve(maxVoxels * 2);

        for (const auto& [coord, chunk] : chunks) {
            if (stop||!chunk || !hasSolidVoxels(*chunk)) continue;

            glm::vec3 chunkMin = getChunkMin(coord);
            glm::vec3 chunkCenter = chunkMin + glm::vec3(CHUNK_SIZE * 0.5f) * VOXEL_SIZE;

            float distToChunkCenter = glm::distance(chunkCenter, center);
            float maxChunkDist = std::sqrt(3.0f) * (CHUNK_SIZE * VOXEL_SIZE * 0.5f);
            if (distToChunkCenter > radius + maxChunkDist) continue;

            int startX = std::max(0, static_cast<int>((center.x - radius - chunkMin.x) / VOXEL_SIZE));
            int endX = std::min(CHUNK_SIZE, static_cast<int>((center.x + radius - chunkMin.x) / VOXEL_SIZE) + 1);
            int startY = std::max(0, static_cast<int>((center.y - radius - chunkMin.y) / VOXEL_SIZE));
            int endY = std::min(CHUNK_SIZE, static_cast<int>((center.y + radius - chunkMin.y) / VOXEL_SIZE) + 1);
            int startZ = std::max(0, static_cast<int>((center.z - radius - chunkMin.z) / VOXEL_SIZE));
            int endZ = std::min(CHUNK_SIZE, static_cast<int>((center.z + radius - chunkMin.z) / VOXEL_SIZE) + 1);

            for (int x = startX; x <= endX&&!stop; ++x) {
                for (int y = startY; y <= endY&&!stop; ++y) {
                    for (int z = startZ; z <= endZ&&!stop; ++z) {
                        const Voxel& voxel = chunk->voxels[x][y][z];

                        if (voxel.isSolid() && voxel.material == targetMaterial) {
                            glm::vec3 worldPos = chunkMin + glm::vec3((float)x, (float)y, (float)z) * VOXEL_SIZE;
                            glm::vec3 diff = worldPos - center;
                            float distSq = glm::dot(diff, diff);

                            if (distSq <= radiusSq) {
                                glm::vec3 normal = calculateNormalAt(chunk, {x, y, z});

                                candidates.push_back({
                                                             worldPos,
                                                             voxel.color,
                                                             coord,
                                                             {x, y, z},
                                                             distSq,
                                                             chunk,
                                                             voxel.type
                                                     });

                                if (voxel.type < 8) typeCounts[voxel.type]++;
                            }
                        }
                    }
                }
            }
        }


        std::sort(candidates.begin(), candidates.end(),
                  [](const VoxelCandidate& a, const VoxelCandidate& b) {
                      return a.distanceSq < b.distanceSq;
                  });

        if (candidates.size() > static_cast<size_t>(maxVoxels)) {
            candidates.resize(maxVoxels);
        }

        memset(typeCounts, 0, sizeof(typeCounts));
        for (const auto& candidate : candidates) {
            if (candidate.type < 8) typeCounts[candidate.type]++;
        }

        int maxCount = 0;
        uint8_t dominantType = 1;
        for (int i = 0; i < 8; ++i) {
            if (typeCounts[i] > maxCount) {
                maxCount = typeCounts[i];
                dominantType = static_cast<uint8_t>(i);
            }
        }
        outDominantType = dominantType;

        results.reserve(candidates.size());

        std::vector<CraterStampBatch::Stamp> stamps;
        stamps.reserve(candidates.size());

        robin_hood::unordered_set<ChunkCoord, ChunkCoordHash> touchedChunks;
        touchedChunks.reserve(candidates.size() / 4 + 8);

        for (const auto& candidate : candidates) {
            AnimatedVoxel animVoxel;

            animVoxel.currentPos = candidate.worldPos;
            animVoxel.originalVoxelPos = candidate.worldPos;
            animVoxel.isAnimating = true;
            animVoxel.animationSpeed = 1000/strength;
            animVoxel.hasArrived = false;

            animVoxel.color = candidate.color;

            animVoxel.normal = calculateNormalAt(candidate.chunk, candidate.localPos);

            results.push_back(animVoxel);

            CraterStampBatch::Stamp s;
            s.center = candidate.worldPos;
            s.radius = 2.0f * gl3::VOXEL_SIZE;
            s.depth  = 5.5f;
            stamps.push_back(s);

            touchedChunks.insert(candidate.chunkCoord);
        }
        CraterStampBatch::apply(chunkManager.get(), stamps, /*densityThreshold=*/-0.5f);

        for (const ChunkCoord& c : touchedChunks) {
            markChunkModified(c);
        }

        std::cout << "[Spell] Collected " << results.size() << "/" << maxVoxels
                  << " closest voxels (type=" << (int)dominantType << ")\n";
    }

    //optimized version?
    void Game::findNearbyVoxelsForVisualNew(const glm::vec3& center, float radius,
                                         uint64_t targetMaterial,
                                         std::vector<AnimatedVoxel>& results,
                                         float strength,
                                         uint8_t& outDominantType) {
        TRACY_CPU_ZONE("Game::findNearbyVoxelsForVisualNew");
        const float radiusSq = radius * radius;

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        int maxVoxels = static_cast<int>((glm::pow(strength,4)) / voxelVolume);
        maxVoxels = glm::clamp(maxVoxels, 3, 100);

       // const size_t hardCandidateCap = (size_t)maxVoxels * 4;
        bool stop = false;
        const float stepSize = radius/(strength * strength);

        std::cout << "[Spell] Targeting " << maxVoxels  << " voxels for formation\n";

        int typeCounts[8] = {0};

        struct VoxelCandidate {
            glm::vec3 worldPos;
            glm::vec3 color;
            ChunkCoord chunkCoord;
            glm::ivec3 localPos;
            float distanceSq;
            Chunk* chunk;
            uint8_t type;
        };

        std::vector<VoxelCandidate> candidates;
        candidates.reserve(maxVoxels);

        for (int i = 0; i <= glm::ceil(radius/stepSize)&&!stop; ++i) {
            auto chunks = chunkManager->getChunksInRadius(center, i*stepSize);
            for (const auto& [coord, chunk] : chunks) {
                if (stop||!chunk || !hasSolidVoxels(*chunk)) continue;

                glm::vec3 chunkMin = getChunkMin(coord);
                glm::vec3 chunkCenter = chunkMin + glm::vec3(CHUNK_SIZE * 0.5f) * VOXEL_SIZE;

                float distToChunkCenter = glm::distance(chunkCenter, center);
                float maxChunkDist = std::sqrt(3.0f) * (CHUNK_SIZE * VOXEL_SIZE * 0.5f);
                if (distToChunkCenter > radius + maxChunkDist) continue;

                int startX = std::max(0, static_cast<int>((center.x - radius - chunkMin.x) / VOXEL_SIZE));
                int endX = std::min(CHUNK_SIZE, static_cast<int>((center.x + radius - chunkMin.x) / VOXEL_SIZE) + 1);
                int startY = std::max(0, static_cast<int>((center.y - radius - chunkMin.y) / VOXEL_SIZE));
                int endY = std::min(CHUNK_SIZE, static_cast<int>((center.y + radius - chunkMin.y) / VOXEL_SIZE) + 1);
                int startZ = std::max(0, static_cast<int>((center.z - radius - chunkMin.z) / VOXEL_SIZE));
                int endZ = std::min(CHUNK_SIZE, static_cast<int>((center.z + radius - chunkMin.z) / VOXEL_SIZE) + 1);

                for (int x = startX; x <= endX&&!stop; ++x) {
                    for (int y = startY; y <= endY&&!stop; ++y) {
                        for (int z = startZ; z <= endZ&&!stop; ++z) {
                            if(candidates.size()>=maxVoxels)
                            {
                                stop=true;
                                continue;
                            }
                            const Voxel& voxel = chunk->voxels[x][y][z];

                            if (voxel.isSolid() && voxel.material == targetMaterial) {
                                glm::vec3 worldPos = chunkMin + glm::vec3((float)x, (float)y, (float)z) * VOXEL_SIZE;
                                glm::vec3 diff = worldPos - center;
                                float distSq = glm::dot(diff, diff);

                                if (distSq <= radiusSq) {
                                    glm::vec3 normal = calculateNormalAt(chunk, {x, y, z});

                                    candidates.push_back({
                                                                 worldPos,
                                                                 voxel.color,
                                                                 coord,
                                                                 {x, y, z},
                                                                 distSq,
                                                                 chunk,
                                                                 voxel.type
                                                         });

                                    if (voxel.type < 8) typeCounts[voxel.type]++;
                                }
                            }
                        }
                    }
                }
            }

        }
        /*memset(typeCounts, 0, sizeof(typeCounts));
        for (const auto& candidate : candidates) {
            if (candidate.type < 8) typeCounts[candidate.type]++;
        }*/

        int maxCount = 0;
        uint8_t dominantType = 1;
        for (int i = 0; i < 8; ++i) {
            if (typeCounts[i] > maxCount) {
                maxCount = typeCounts[i];
                dominantType = static_cast<uint8_t>(i);
            }
        }
        outDominantType = dominantType;

        results.reserve(candidates.size());

        std::vector<CraterStampBatch::Stamp> stamps;
        stamps.reserve(candidates.size());

        robin_hood::unordered_set<ChunkCoord, ChunkCoordHash> touchedChunks;
        touchedChunks.reserve(candidates.size() / 4 + 8);

        for (const auto& candidate : candidates) {
            AnimatedVoxel animVoxel;

            animVoxel.currentPos = candidate.worldPos;
            animVoxel.originalVoxelPos = candidate.worldPos;
            animVoxel.isAnimating = true;
            animVoxel.animationSpeed = 1000/strength;
            animVoxel.hasArrived = false;

            animVoxel.color = candidate.color;

            animVoxel.normal = calculateNormalAt(candidate.chunk, candidate.localPos);

            results.push_back(animVoxel);

            CraterStampBatch::Stamp s;
            s.center = candidate.worldPos;
            s.radius = 2.0f * gl3::VOXEL_SIZE;
            s.depth  = 5.5f;
            stamps.push_back(s);

            touchedChunks.insert(candidate.chunkCoord);
        }

        CraterStampBatch::apply(chunkManager.get(), stamps, /*densityThreshold=*/-0.5f);

        for (const ChunkCoord& c : touchedChunks) {
            markChunkModified(c);
        }

        std::cout << "[Spell] Collected " << results.size() << "/" << maxVoxels
                  << " closest voxels (type=" << (int)dominantType << ")\n";
    }

    void Game::createSpellFormation(const glm::vec3& center,
                                    const FormationParams& formationParams,
                                    float strength, uint64_t material,
                                    const glm::vec3& color, size_t collectedVoxels,
                                    uint8_t dominantType) {
        TRACY_CPU_ZONE("Game::createSpellFormation");
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

        for (int cx = minCX; cx <= maxCX; ++cx) {
            for (int cy = minCY; cy <= maxCY; ++cy) {
                for (int cz = minCZ; cz <= maxCZ; ++cz) {
                    ChunkCoord coord{cx, cy, cz};

                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) continue; // outside fixed world
                }
            }
        }

        FormationParams paramsCopy = formationParams;
        paramsCopy.center = center;

        carveFormationWithSDF(newFormation, material, paramsCopy);

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
        TRACY_CPU_ZONE("Game::carveFormationWithSDF");
        float boundingRadius = params.getBoundingRadius();
        std::cout << "carveFormationWithSDF: boundingRadius=" << boundingRadius
                  << " center=(" << formation.worldPos.x << "," << formation.worldPos.y << "," << formation.worldPos.z << ")\n";

        float centerVal = params.evaluate(formation.worldPos);
        float atRadiusVal = params.evaluate(formation.worldPos + glm::vec3(boundingRadius,0,0));
        std::cout << " SDF(center)=" << centerVal << " SDF(center+radius)=" << atRadiusVal << "\n";


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

                    Chunk* chunk = chunkManager->getChunk(coord);
                    if (!chunk) continue;

                    glm::vec3 chunkOrigin = getChunkMin(coord);
                    bool chunkTouched = false;

                    for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                        for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                            for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                glm::vec3 worldPos = chunkOrigin +
                                                     glm::vec3((float)lx, (float)ly, (float)lz) * gl3::VOXEL_SIZE;

                                float formationDensity = params.evaluate(worldPos);

                                float existingDensity = chunk->voxels[lx][ly][lz].density;

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
    }

    void Game::updateSpells(float dt) {
        TRACY_CPU_ZONE("Game::updateSpells");

        const float kSlowSpeedThreshold = 0.0001f * VOXEL_SIZE;
        const float kSlowTimeToBurn     = 0.75f;
        const float kBurnDuration       = 3.0f;

        for (size_t i = 0; i < activeSpells.size(); /*manual*/) {
            gl3::SpellEffect& s = activeSpells[i];

            if (s.lifetime > 0.0f) s.creationTime += dt;

            if (s.physicsBodyId != 0 && voxelPhysics) {
                s.physicsBody = voxelPhysics->getBodyById(s.physicsBodyId);
                if (s.physicsBody && s.isPhysicsEnabled) {
                    s.center = s.physicsBody->position;
                    s.formationParams.center = s.physicsBody->position;
                }
            } else {
                s.physicsBody = nullptr;
            }

            if (s.physicsBodyId != 0 && !s.physicsBody) {
                s.markForRemoval = true;
            }

            if (s.markForRemoval) {
                forceCleanupSpellAnimatedVoxels(s);
                destroyPhysicsBodyForSpell(s);
                activeSpells.erase(activeSpells.begin() + (ptrdiff_t)i);
                continue;
            }

            if (s.burn.active) {
                s.burn.center = s.center;
                s.burn.t += dt;

                if (burn01(s.burn.t, s.burn.duration) >= 1.0f) {
                    forceCleanupSpellAnimatedVoxels(s);
                    destroyPhysicsBodyForSpell(s);
                    activeSpells.erase(activeSpells.begin() + (ptrdiff_t)i);
                    continue;
                }

                ++i;
                continue;
            }

            const bool tooSmall = isSpellTooSmall(s);

            const bool tooSlowNow = isSpellTooSlowNow(s, kSlowSpeedThreshold);
            s.burn.slowAccum = tooSlowNow ? (s.burn.slowAccum + dt) : 0.0f;

            const bool tooSlowLong = (s.burn.slowAccum >= kSlowTimeToBurn);

            if (tooSmall || tooSlowLong) {
                const float r = glm::max(s.formationParams.getBoundingRadius(), 1.0f * VOXEL_SIZE);
                startSpellBurn(s, r, kBurnDuration);
                ++i;
                continue;
            }

            std::vector<uint64_t> newlyArrivedIDs;

            for (uint64_t id : s.animatedVoxelIDs) {
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
                        float slowdown = glm::clamp(distance*distance / 2.0f, 0.75f, 3.0f);
                        voxel.velocity = (toTarget / glm::vec3(VOXEL_SIZE * CHUNK_SIZE)) * speed * slowdown * 4.0f;
                        voxel.currentPos += voxel.velocity * deltaTime;
                    }
                } else {
                    newlyArrivedIDs.push_back(id);
                }
            }

            if (!newlyArrivedIDs.empty() && !s.geometryCreated) {
                int arrivedCount = 0;
                int total = (int)s.animatedVoxelIDs.size();
                for (uint64_t id : s.animatedVoxelIDs) {
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

                    createSpellFormation(s.center,
                                         s.formationParams,
                                         s.strength,
                                         s.targetMaterial,
                                         s.formationColor,
                                         s.animatedVoxelIDs.size(),
                                         s.dominantType);
                    s.geometryCreated = true;
                } else if (arrivalRatio > 0.0005f) {
                    createPartialFormation(s, arrivalRatio);
                }
            }

            if (s.geometryCreated && s.isPhysicsEnabled &&
                    s.physicsBody == nullptr) {
                createPhysicsBodyForSpell(s);
            }

            if (s.geometryCreated && s.isPhysicsEnabled && !s.voxelsCleaned) {
                int stillAnimating = 0;
                for (uint64_t id : s.animatedVoxelIDs) {
                    auto jt = animatedVoxelIndexMap.find(id);
                    if (jt != animatedVoxelIndexMap.end()) {
                        AnimatedVoxel &v = animatedVoxels[jt->second];
                        if (v.isAnimating) ++stillAnimating;
                    }
                }

                if (stillAnimating == 0) {
                    std::cout << "All voxels arrived for spell. Cleaning voxel data, keeping physics body.\n";

                    s.voxelsCleaned = true;

                    for (uint64_t id : s.animatedVoxelIDs) {
                        auto jt = animatedVoxelIndexMap.find(id);
                        if (jt != animatedVoxelIndexMap.end()) {
                            animatedVoxels[jt->second].isAnimating = false;
                        }
                    }

                    s.animatedVoxelIDs.clear();
                }
            }

            ++i;
        }

        for (size_t vi = 0; vi < animatedVoxels.size(); ) {
            if (!animatedVoxels[vi].isAnimating) {
                uint64_t removedID = animatedVoxels[vi].id;
                size_t last = animatedVoxels.size() - 1;
                if (vi != last) {
                    animatedVoxels[vi] = animatedVoxels[last];
                    animatedVoxelIndexMap[animatedVoxels[vi].id] = vi;
                }
                animatedVoxels.pop_back();
                animatedVoxelIndexMap.erase(removedID);
            } else {
                ++vi;
            }
        }

        cleanupExpiredSpells();
    }

    bool Game::isSpellTooSlowNow(const gl3::SpellEffect& s, float speedThresholdWorld) const {
        if (!s.physicsBody) return false;
        return glm::length(s.physicsBody->velocity) < speedThresholdWorld;
    }

    void Game::forceCleanupSpellAnimatedVoxels(gl3::SpellEffect& s) {
        for (uint64_t id : s.animatedVoxelIDs) {
            auto it = animatedVoxelIndexMap.find(id);
            if (it == animatedVoxelIndexMap.end()) continue;
            size_t idx = it->second;
            if (idx >= animatedVoxels.size()) continue;
            animatedVoxels[idx].isAnimating = false;
            animatedVoxels[idx].hasArrived  = true;
        }
        s.animatedVoxelIDs.clear();
    }

    bool Game::isSpellTooSmall(const gl3::SpellEffect& s) {
        float r = s.formationParams.getBoundingRadius();
        return (r < (0.05f * gl3::VOXEL_SIZE));
    }

    bool Game:: isSpellTooSlow(const gl3::SpellEffect& s, float speedThreshold) {
        if (!s.physicsBody) return false;
        return glm::length(s.physicsBody->velocity) < speedThreshold;
    }

    void Game::updateChunkBurns(float dt)
    {
        TRACY_CPU_ZONE("Game::updateChunkBurns");

        chunkManager->forEachChunk([&](gl3::Chunk* chunk) {
            if (!chunk) return;
            if (!chunk->burn.active) return;

            chunk->burn.t += dt;
            if (burn01(chunk->burn.t, chunk->burn.duration) >= 1.0f)
            {
                chunk->clear();
                chunk->isCleared = true;

                chunk->gpuCache.vertexCount = 0;

                chunk->meshDirty = false;
                chunk->gpuCache.isValid = true; // "valid but empty"

                chunk->burn.active = false;
            }
        });

        static int tick = 0;
        const bool doPolicyThisFrame = (++tick % 10) == 0;
        if (!doPolicyThisFrame) return;

        chunkManager->forEachChunk([&](gl3::Chunk* chunk) {
            if (!chunk) return;
            if (chunk->isCleared) return;
            if (chunk->burn.active) return;

            if (!chunk->gpuCache.isValid || chunk->gpuCache.vertexCount == 0) return;

            const uint32_t kSmallVtx = 100;

            if (isChunkMeshTooSmall(*chunk, kSmallVtx))
            {
                glm::vec3 chunkMin = getChunkMin(chunk->coord);
                glm::vec3 center   = chunkMin + glm::vec3(CHUNK_SIZE * 0.5f) * VOXEL_SIZE;
                startChunkBurn(chunk, center, (CHUNK_SIZE * 0.6f) * VOXEL_SIZE, /*duration*/ 3.5f);
            }
        });
    }

    bool Game:: isChunkMeshTooSmall(const gl3::Chunk& c, uint32_t vtxThreshold) {
        return c.gpuCache.isValid && c.gpuCache.vertexCount > 0 && c.gpuCache.vertexCount < vtxThreshold;
    }

    void Game::createPartialFormation(const SpellEffect& spell, float completionRatio) {
        ZoneScoped;
        WorldPlanet partialFormation;
        partialFormation.worldPos = spell.center;
        partialFormation.color = spell.formationColor;
        partialFormation.type = spell.dominantType;

        FormationParams partialParams = spell.formationParams;

        partialParams.center = spell.center;

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

        carveFormationWithSDF(partialFormation, spell.targetMaterial, partialParams);
    }

    void Game::cleanupExpiredSpells() {
        for (auto spellIt = activeSpells.begin(); spellIt != activeSpells.end(); ) {
            if (spellIt->lifetime > 0) {
                float age = spellIt->creationTime;
                if (age > spellIt->lifetime) {
                    spellIt->markForRemoval = true;
                }
            }

            if (spellIt->physicsBody!= nullptr&& glm::length(spellIt->physicsBody->velocity) < 0.5) {
                spellIt->markForRemoval = true;
            }


            if (spellIt->isPhysicsEnabled && spellIt->voxelsCleaned) {
                float distanceToPlayer = glm::distance(spellIt->center, cameraPos);
                if (distanceToPlayer > 10000.0f * VOXEL_SIZE) {  // Increased distance
                    spellIt->markForRemoval = true;
                }
            }

            ++spellIt;
        }
    }

    void Game::onSpellCollision(SpellEffect* spell,
                                const glm::vec3& hitPos,
                                const glm::vec3& hitNormal,
                                float impactSpeed) {
        if (!spell) return;

        bool spellValid = false;
        for (const auto& s : activeSpells) {
            if (&s == spell) {
                spellValid = true;
                break;
            }
        }

        float mass = spell->physicsBody ? spell->physicsBody->mass : 1.0f;

        createCraterAtPosition(hitPos,glm::sqrt(impactSpeed),glm::sqrt(spell->physicsBody->radius));
    }

    void Game::createCraterAtPosition(const glm::vec3& worldPos, float impactFactor, float spellRadius) {
        ZoneScoped;
        int cx = worldToChunk(worldPos.x);
        int cy = worldToChunk(worldPos.y);
        int cz = worldToChunk(worldPos.z);

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);

        if (!chunk) {
            std::cout << "No chunk found at impact position\n";
            return;
        }

        glm::vec3 chunkMin = getChunkMin(coord);
        glm::vec3 localPos = (worldPos - chunkMin) / VOXEL_SIZE;

        int vx = glm::clamp(static_cast<int>(std::round(localPos.x)), 0, CHUNK_SIZE);
        int vy = glm::clamp(static_cast<int>(std::round(localPos.y)), 0, CHUNK_SIZE);
        int vz = glm::clamp(static_cast<int>(std::round(localPos.z)), 0, CHUNK_SIZE);

        glm::ivec3 voxelPos(vx, vy, vz);

        float originalCraterRadius = 2.0f * VOXEL_SIZE;
        float craterRadius = originalCraterRadius * impactFactor * (spellRadius / (5.0f * VOXEL_SIZE));
        float maxCraterDepth = 2.5f * impactFactor;

        craterRadius = glm::clamp(craterRadius, VOXEL_SIZE, 10.0f * VOXEL_SIZE);
        maxCraterDepth = glm::clamp(maxCraterDepth, 1.0f, 5.0f);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        FastCraterCarver::carveCrater(
                chunkManager.get(),
                worldPos,
                craterRadius,
                maxCraterDepth,
                -0.5f,
                &touched,
                /*autoCreateChunks=*/false
        );

        for (const ChunkCoord& c : touched) {
            markChunkModified(c);
        }
    }

    void Game::castSpellSphere(const glm::vec3& center, float radius,
                               uint64_t material, float strength) {
        float searchRadius = radius * 1.5f;

        FormationParams params = FormationParams::Sphere(center, radius);
        //castSpellWithFormation(center, searchRadius, material, strength, params);
        SpellCastRequest req = buildSpellCastRequestSnapshot(center, searchRadius, material, strength, params);
        req.physicsEnabled = true;
        req.launchDir = glm::normalize(getCameraFront());
        req.launchSpeed = strength * 20.0f * VOXEL_SIZE;
        req.lifetime = 20.0f;
        spellCastAsync->enqueueOrReplaceQueued(std::move(req));
        //TracyPlot("SpellJobsCompletedQueue", (int)completed.size());
        //TracyPlot("SpellHasQueued", queued.has_value() ? 1.0 : 0.0);
        //TracyPlot("SpellHasInFlight", inFlight.has_value() ? 1.0 : 0.0);

        /*if (!activeSpells.empty()) {
            SpellEffect& lastSpell = activeSpells.back();
            glm::vec3 launchDir = glm::normalize(getCameraFront());
            float launchSpeed = strength * 20.5f * VOXEL_SIZE;

            lastSpell.isPhysicsEnabled = true;
            lastSpell.creationTime = 0.0f;
            lastSpell.lifetime = 20.0f;
            lastSpell.center = center;
            lastSpell.initialVelocity = launchDir * launchSpeed;
        }*/
    }

    void Game::castSpellWall(const glm::vec3& center, const glm::vec3& normal,
                             float width, float height, float thickness,
                             uint64_t material, float strength) {
        FormationParams params = FormationParams::Wall(center, normal,
                                                       width, height, thickness);
        float searchRadius = glm::max(width, height) * VOXEL_SIZE* 10.5f;
        castSpellWithFormation(center, searchRadius, material, strength, params);
    }

    void Game::castSpellPlatform(const glm::vec3& center, const glm::vec3& normal,
                                 float width, float depth, float thickness,
                                 uint64_t material, float strength) {
        FormationParams params = FormationParams::Platform(center, normal,
                                                           width, depth, thickness);
        float searchRadius = 10*4.5f*VOXEL_SIZE;
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
        TRACY_CPU_ZONE("Game::createPhysicsBodyForSpell");
        if (!voxelPhysics || spell.physicsBody != nullptr) return;

        if (spell.formationParams.type == FormationType::SPHERE) {
            float radius = spell.formationParams.radius;
            int key = static_cast<int>(std::round(radius / VOXEL_SIZE));

            auto it = sphereMeshCache.find(key);
            if (it == sphereMeshCache.end()) {
                int nearestKey = sphereMeshCache.begin()->first;
                for (const auto& [k, _] : sphereMeshCache) {
                    if (std::abs(k - key) < std::abs(nearestKey - key)) {
                        nearestKey = k;
                    }
                }
                it = sphereMeshCache.find(nearestKey);
            }

            const SphereMesh& baseMesh = it->second;

            std::vector<glm::vec3> scaledVerts;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec3> colors;

            float scale = radius / baseMesh.radius;
            scaledVerts.reserve(baseMesh.indices.size());
            normals.reserve(baseMesh.indices.size());
            colors.reserve(baseMesh.indices.size());

            for (uint32_t idx : baseMesh.indices) {
                scaledVerts.push_back(baseMesh.vertices[idx] * scale);
                normals.push_back(baseMesh.normals[idx]);
                colors.push_back(spell.formationColor);
            }

            float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
            float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
            mass = glm::clamp(mass, 1.0f, 1000.0f);

            spell.physicsBody = voxelPhysics->createBody(
                    spell.center,
                    mass,
                    VoxelPhysicsBody::ShapeType::SPHERE,
                    glm::vec3(radius)
            );
            spell.physicsBodyId = spell.physicsBody ? spell.physicsBody->id : 0;

            if (spell.physicsBody) {
                spell.physicsBody->userData = &spell;
                spell.physicsBody->velocity = spell.initialVelocity;

                initSpellDestructibleVolume(spell);
                rebuildDestructibleMeshIfNeeded(spell.destruct);

                // NEW: render from destructible mesh (local-space mesh)
                spell.physicsBody->renderMesh = &spell.destruct.mesh;

                // If you still want physicsMesh for debugging, keep it, but it’s no longer the render source.
                spell.isPhysicsEnabled = true;
            }

            removeFormationVoxels(spell);
            return;
        }
        if (!voxelPhysics || spell.physicsBody != nullptr) return;

        float effectiveRadius = spell.formationParams.getBoundingRadius();

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
                                    // Inside createPhysicsBodyForSpell, when collecting triangles:

                                    for (int j = 0; j < 3; ++j) {
                                        glm::vec4 p = ov[vi + j].pos;
                                        glm::vec4 n = ov[vi + j].normal;
                                        glm::vec4 c = ov[vi + j].color;

                                        // CONVERT TO LOCAL SPACE relative to spell.center
                                        glm::vec3 worldVert(p.x, p.y, p.z);
                                        glm::vec3 localVert = worldVert - spell.center;  // ← SUBTRACT CENTER!

                                        triangleVerts.emplace_back(localVert);  // ← Now in local space
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

        std::cout << "Found " << triangleVerts.size() << " triangles for physics body\n";

        // Calculate the bounds of collected geometry (for extents calculation only)
        glm::vec3 minBound = triangleVerts[0];
        glm::vec3 maxBound = triangleVerts[0];
        for (const auto& v : triangleVerts) {
            minBound = glm::min(minBound, v);
            maxBound = glm::max(maxBound, v);
        }

        // Calculate extents from bounds
        glm::vec3 geomExtents = (maxBound - minBound) * 0.5f;
        geomExtents += glm::vec3(VOXEL_SIZE * 0.1f); // Small margin

        // ⚠️ KEY FIX: Use spell.center for physics body position, NOT geomCenter
        // The triangles are already in world space at spell.center location
        glm::vec3 physicsBodyPosition = spell.center;

        // Determine shape type based on formation
        gl3::VoxelPhysicsBody::ShapeType shapeType;
        glm::vec3 extents;

        switch(spell.formationParams.type) {
            case FormationType::PLATFORM:
            case FormationType::WALL:
            case FormationType::CUBE:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::BOX;
                extents = geomExtents;
                break;
            case FormationType::CYLINDER:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::BOX;
                extents = geomExtents;
                break;
            default:
                shapeType = gl3::VoxelPhysicsBody::ShapeType::BOX;
                extents = geomExtents;
                break;
        }

        // Calculate mass based on collected voxels
        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        float mass = static_cast<float>(spell.animatedVoxelIDs.size()) * voxelVolume * 0.5f;
        mass = glm::clamp(mass, 1.0f, 1000.0f);

        spell.physicsBody = voxelPhysics->createBody(
                physicsBodyPosition,  // Use spell.center, not geomCenter!
                mass,
                shapeType,
                extents
        );
        spell.physicsBodyId = spell.physicsBody ? spell.physicsBody->id : 0;

        if (spell.physicsBody) {
            spell.physicsBody->userData = &spell;
            spell.physicsBody->velocity = spell.initialVelocity;

            // Set orientation to face direction of travel
            if (glm::length(spell.initialVelocity) > 0.001f) {
                glm::vec3 direction = glm::normalize(spell.initialVelocity);
                spell.physicsBody->orientation = glm::quatLookAt(direction, glm::vec3(0.0f, 1.0f, 0.0f));
            } else {
                spell.physicsBody->orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            }

            // Store the mesh data for rendering
            // NEW: initialize destructible volume for this formation
            initSpellDestructibleVolume(spell);
            rebuildDestructibleMeshIfNeeded(spell.destruct);

            // render from destructible mesh instead of triangle dump
            spell.physicsBody->renderMesh = &spell.destruct.mesh;
            spell.isPhysicsEnabled = true;

            std::cout << "Physics body created at spell.center: ("
                      << spell.center.x << "," << spell.center.y << "," << spell.center.z << ")\n";
            std::cout << "Extents: (" << extents.x << "," << extents.y << "," << extents.z << ")\n";
            std::cout << "Initial velocity: (" << spell.initialVelocity.x << ","
                      << spell.initialVelocity.y << "," << spell.initialVelocity.z << ")\n";
        }

        // Remove voxels from chunks now that we have the mesh
        removeFormationVoxels(spell);
    }

    // Call this in Game constructor
    void Game::initSphereMeshCache() {
        // Generate sphere meshes at common radii (in voxels)
        const std::vector<float> commonRadii = {
                2.0f * VOXEL_SIZE,
                4.0f * VOXEL_SIZE,
                6.0f * VOXEL_SIZE,
                8.0f * VOXEL_SIZE,
                10.0f * VOXEL_SIZE
        };

        for (float radius : commonRadii) {
            int key = static_cast<int>(radius / VOXEL_SIZE);
            sphereMeshCache[key] = generateIcosphere(radius, 2); // 2 subdivisions
        }

        std::cout << "Initialized " << sphereMeshCache.size() << " sphere meshes\n";
    }

    void Game::initSpellDestructibleVolume(gl3::SpellEffect& spell)
    {
        auto& d = spell.destruct; // assume it exists
        d.voxelSize = VOXEL_SIZE;

        // Choose half extents based on formation type
        glm::vec3 halfExtWorld(2.0f * VOXEL_SIZE);

        if (spell.formationParams.type == FormationType::SPHERE) {
            halfExtWorld = glm::vec3(spell.formationParams.radius*2);
        } else {
            // bounding radius -> conservative cube
            float r = spell.formationParams.getBoundingRadius();
            halfExtWorld = glm::vec3(r);
        }

        glm::ivec3 cornerDims = dimsFromHalfExtents(halfExtWorld, VOXEL_SIZE);
        d.volume.init(cornerDims, VOXEL_SIZE);

        glm::vec3 localHalf = halfExtentsFromVolumeDims(cornerDims, VOXEL_SIZE);
        glm::vec3 centerLocal = localHalf;               // center of volume
        d.localCenterOffsetWorld = centerLocal;          // so body.position corresponds to centerLocal

        // Fill based on formation
        if (spell.formationParams.type == FormationType::SPHERE) {
            d.volume.fillSphere(centerLocal, spell.formationParams.radius, spell.formationColor, (uint32_t)spell.targetMaterial, spell.dominantType);
        } else {
            // Example: treat WALL/CUBE/PLATFORM as box-ish in the destructible proxy
            // You can derive extents from FormationParams for each type precisely.
            glm::vec3 halfBox = halfExtWorld;
            fillBox(d.volume, centerLocal, halfBox, spell.formationColor, (uint32_t)spell.targetMaterial, spell.dominantType);
        }

        d.meshDirty = true;
    }

    // Game.cpp
    void Game::rebuildDestructibleMeshIfNeeded(gl3::DestructibleObject& d)
    {
        if (!d.meshDirty) return;

        gl3::LocalMesh mesh = gl3::buildMeshLocalMC(d.volume);

        // Convert from volume-local [0..size] to body-local centered at origin
        // so renderer can do model = T(body.position) * R(body.orientation).
        const glm::vec3 centerLocal = d.localCenterOffsetWorld;

        for (auto& v : mesh.vertices) {
            v -= centerLocal;
        }

        createPhysicsMeshData(d.mesh, mesh.vertices, mesh.normals, mesh.colors);
        d.meshDirty = false;
    }

// Generate icosphere (smooth sphere)
    Game::SphereMesh Game::generateIcosphere(float radius, int subdivisions) {
        SphereMesh mesh;
        mesh.radius = radius;

        // Golden ratio
        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        // 12 vertices of icosahedron
        std::vector<glm::vec3> positions = {
                {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
                { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
                { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
        };

        // Normalize and scale
        for (auto& p : positions) {
            p = glm::normalize(p) * radius;
        }

        // 20 faces of icosahedron
        std::vector<uint32_t> indices = {
                0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
                1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
                3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
                4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1
        };

        // Subdivide (optional for smoother sphere)
        for (int sub = 0; sub < subdivisions; ++sub) {
            std::vector<uint32_t> newIndices;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpointCache;

            auto getMidpoint = [&](uint32_t i0, uint32_t i1) -> uint32_t {
                auto key = std::make_pair(std::min(i0, i1), std::max(i0, i1));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) {
                    return it->second;
                }

                glm::vec3 mid = glm::normalize((positions[i0] + positions[i1]) * 0.5f) * radius;
                uint32_t newIdx = positions.size();
                positions.push_back(mid);
                midpointCache[key] = newIdx;
                return newIdx;
            };

            for (size_t i = 0; i < indices.size(); i += 3) {
                uint32_t v0 = indices[i];
                uint32_t v1 = indices[i + 1];
                uint32_t v2 = indices[i + 2];

                uint32_t m01 = getMidpoint(v0, v1);
                uint32_t m12 = getMidpoint(v1, v2);
                uint32_t m20 = getMidpoint(v2, v0);

                newIndices.insert(newIndices.end(), {
                        v0, m01, m20,
                        v1, m12, m01,
                        v2, m20, m12,
                        m01, m12, m20
                });
            }

            indices = newIndices;
        }

        // Build final mesh
        mesh.vertices = positions;
        mesh.normals.reserve(positions.size());
        for (const auto& p : positions) {
            mesh.normals.push_back(glm::normalize(p));
        }
        mesh.indices = indices;

        return mesh;
    }


    void Game::createPhysicsMeshData(SpellEffect& spell,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<glm::vec3>& normals,
                                     const std::vector<glm::vec3>& colors) {
        /*if (vertices.empty()) return;

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

        spell.physicsMesh.isValid = true;*/
        createPhysicsMeshData(spell.physicsMesh, vertices, normals, colors);
    }

    void Game::createPhysicsMeshData(gl3::PhysicsMeshData& out,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<glm::vec3>& normals,
                                     const std::vector<glm::vec3>& colors)
    {
        if (vertices.empty()) return;

        // cleanup old
        if (out.vao) { glDeleteVertexArrays(1, &out.vao); out.vao = 0; }
        if (out.vbo) { glDeleteBuffers(1, &out.vbo); out.vbo = 0; }

        out.vertices = vertices;
        out.normals  = normals;
        out.colors   = colors;
        out.vertexCount = vertices.size();
        out.isValid = true;

        glGenVertexArrays(1, &out.vao);
        glGenBuffers(1, &out.vbo);

        glBindVertexArray(out.vao);
        glBindBuffer(GL_ARRAY_BUFFER, out.vbo);

        std::vector<float> interleaved;
        interleaved.reserve(vertices.size() * 9);

        for (size_t i=0;i<vertices.size();++i) {
            const glm::vec3 p = vertices[i];
            const glm::vec3 n = (i < normals.size()) ? normals[i] : glm::vec3(0,1,0);
            const glm::vec3 c = (i < colors.size())  ? colors[i]  : glm::vec3(1,0,1);

            interleaved.insert(interleaved.end(), {p.x,p.y,p.z, n.x,n.y,n.z, c.x,c.y,c.z});
        }

        glBufferData(GL_ARRAY_BUFFER, interleaved.size()*sizeof(float), interleaved.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,9*sizeof(float),(void*)0);
        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,9*sizeof(float),(void*)(3*sizeof(float)));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,9*sizeof(float),(void*)(6*sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }


    void Game::removeFormationVoxels(const SpellEffect& spell) {
        ZoneScoped;
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

    void Game::destroyPhysicsBodyForSpell(gl3::SpellEffect& spell) {
        // If body already gone, just clean render mesh
        if (spell.physicsBodyId != 0 && voxelPhysics) {

            // IMPORTANT: refresh pointer, because it may already be removed by physics manager
            gl3::VoxelPhysicsBody* body = voxelPhysics->getBodyById(spell.physicsBodyId);

            // If body exists, detach userData before removing (avoids callback using freed SpellEffect pointer)
            if (body) {
                body->userData = nullptr;
            }

            // If you want to "bake back into world" before deleting, do it WITHOUT using spell.physicsBody pointer.
            // (Using spell.physicsBody->mass here is a common crash.)
            // Use a safe mass fallback:
            const float safeCollectedProxy = (float)spell.physicsMesh.vertexCount; // or store spell.collectedVoxelCount when created

            createSpellFormation(
                    spell.center,
                    spell.formationParams,
                    spell.strength,
                    spell.targetMaterial,
                    spell.formationColor,
                    (size_t)safeCollectedProxy,
                    spell.dominantType
            );

            voxelPhysics->removeBody(spell.physicsBodyId);
            spell.physicsBodyId = 0;
        }

        spell.physicsBody = nullptr;           // <- critical (avoid stale pointer)
        spell.isPhysicsEnabled = false;

        // Clean up mesh data
        if (spell.physicsMesh.vao) {
            glDeleteVertexArrays(1, &spell.physicsMesh.vao);
            glDeleteBuffers(1, &spell.physicsMesh.vbo);
            spell.physicsMesh.vao = 0;
            spell.physicsMesh.vbo = 0;
        }
        spell.physicsMesh.isValid = false;
        spell.physicsMesh.vertexCount = 0;
    }

    void Game::initSpellCastAsync()
    {
        if (!spellCastAsync)
            spellCastAsync = std::make_unique<SpellCastAsync>();
    }

    void Game::shutdownSpellCastAsync()
    {
        if (spellCastAsync)
        {
            spellCastAsync->stop();
            spellCastAsync.reset();
        }
    }

    SpellCastRequest Game::buildSpellCastRequestSnapshot(
            const glm::vec3& center,
            float searchRadius,
            uint64_t targetMaterial,
            float strength,
            const FormationParams& baseFormationParams
    )
    {
        SpellCastRequest req;
        req.center = center;
        req.searchRadius = searchRadius;
        req.targetMaterial = targetMaterial;
        req.strength = strength;
        req.baseFormationParams = baseFormationParams;

        // IMPORTANT: main-thread snapshot; worker must not touch chunkManager/chunks directly.
        auto chunks = chunkManager->getChunksInRadius(center, searchRadius);

        req.chunks.reserve(chunks.size());

        for (const auto& [coord, chunk] : chunks)
        {
            if (!chunk) continue;

            SpellCastRequest::ChunkSnapshot snap;
            snap.coord = coord;
            snap.chunkMinWorld = getChunkMin(coord);
            snap.voxelsLinear.resize((CHUNK_SIZE + 1) * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1));

            for (int x = 0; x <= CHUNK_SIZE; ++x)
                for (int y = 0; y <= CHUNK_SIZE; ++y)
                    for (int z = 0; z <= CHUNK_SIZE; ++z)
                    {
                        snap.voxelsLinear[(size_t)x + (size_t)y * (CHUNK_SIZE + 1) + (size_t)z * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1)]
                                = chunk->voxels[x][y][z];
                    }

            req.chunks.push_back(std::move(snap));
        }

        return req;
    }

    void Game::pumpAsyncSpellResults()
    {
        if (!spellCastAsync) return;

        SpellCastResult r;
        while (spellCastAsync->tryPopCompleted(r))
        {
            if (!r.ok)
            {
                std::cout << "[SpellAsync] failed: " << r.debugMsg << "\n";
                continue;
            }

            std::lock_guard<std::mutex> lk(spellApplyMutex);

            // 1) Apply crater stamps (world mutation)
            if (!r.craterStamps.empty())
            {
                CraterStampBatch::apply(chunkManager.get(), r.craterStamps, -0.5f);
            }

            // 2) Mark touched chunks dirty (and neighbors via your existing helper)
            // De-dup the touched list
            robin_hood::unordered_set<ChunkCoord, ChunkCoordHash> touchedSet;
            touchedSet.reserve(r.touchedChunks.size());
            for (const auto& c : r.touchedChunks)
                touchedSet.insert(c);

            for (const auto& c : touchedSet)
                markChunkModified(c);

            // 3) Spawn animated voxels + spell in main thread (assign stable IDs here)
            SpellEffect spell = r.spell;

            for (auto& v : r.visualVoxels)
            {
                v.id = nextAnimatedVoxelID++;
                // now that world exists on main thread, you can compute a better normal if desired:
                // - locate chunk and local pos again OR use your sampleNormalAtWorld()
                v.normal = sampleNormalAtWorld(v.currentPos);

                animatedVoxels.push_back(v);
                animatedVoxelIndexMap[v.id] = animatedVoxels.size() - 1;
                spell.animatedVoxelIDs.push_back(v.id);
            }

            activeSpells.push_back(std::move(spell));

            std::cout << "[SpellAsync] applied: " << r.visualVoxels.size()
                      << " voxels, type=" << int(r.spell.dominantType) << "\n";
        }
    }

////----Debugging Code--------------------------------------------------------------------------------------------------------------------------

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
        ZoneScoped;
        // Prepare SSBOs and static tables

        // 0: voxels SSBO
        glGenBuffers(1, &ssboVoxels);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxelCount * sizeof(CpuVoxelStd430), nullptr, GL_DYNAMIC_DRAW);
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

      /*  // 4: triangles SSBO (output)
        glGenBuffers(1, &ssboTriangles);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriangles);
        glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(OutVertex), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);*/

        //5 particle ssbo
      //  glGenBuffers(1, &particleSSBO);

        //6 fieldbits ssbo
     //   glGenBuffers(1, &fieldBitsSSBO);

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
        TRACY_CPU_ZONE("Game::generateChunks");
        std::mt19937 rng(std::random_device{}());

        // world max coordinate inside grid (world units)
        float chunkWorld = CHUNK_SIZE * VOXEL_SIZE;
        float worldMax = chunkManager->radius() * chunkWorld;

        std::uniform_real_distribution<float> distPos(-worldMax * 0.9f, worldMax * 0.9f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);
        std::uniform_real_distribution<float> distMat(0.0f, 3.9f);

        std::vector<WorldPlanet> worldPlanets;

        //clear old data:
        chunkManager->clearAll();


        // Create solid planets (type 1)
        int planetCount = 20;

        WorldPlanet p;
        p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
        p.radius = distScale(rng) * CHUNK_SIZE;
        p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
        p.type = 1; // solid
        p.material=(int)distMat(rng);
        //p.material=0;
        worldPlanets.push_back(p);
        cameraPos=p.worldPos+glm::vec3(0,VOXEL_SIZE,0);
        characterController->setPosition(cameraPos);
        for (int i = 0; i < planetCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
            p.type = 1; // solid
           // p.material=(int)distMat(rng);
             p.material=0;
            worldPlanets.push_back(p);
        }

        // Create suns (type 2 - fire)
        std::uniform_real_distribution<float> lavaDistColorR(0.8f, 1.0f);
        std::uniform_real_distribution<float> lavaDistColorG(0.2f, 0.5f);
        std::uniform_real_distribution<float> lavaDistColorB(0.0f, 0.1f);

        int lavaCount = 3 + (rng() % 3);
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
                                        chunk->voxels[lx][ly][lz].density = planetDensity;

                                        // If this voxel participates in the field at all, give it a color.
                                        // (You can still keep type/material logic near the surface if you want.)
                                        chunk->voxels[lx][ly][lz].color = planet.color;

                                        if (planetDensity >= -1.0f) {
                                            chunk->voxels[lx][ly][lz].type = planet.type;
                                            chunk->voxels[lx][ly][lz].material = planet.material; // if you use it
                                            if (planetDensity >= 0) {
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
        const int subStepCount = 10;
        const float subDt = (fixedTimeStep / (float)subStepCount);

        accumulator += deltaTime;

        while (accumulator >= fixedTimeStep) {
            // (Input can be sampled once per fixed step; that's fine)
            glm::vec3 moveInput(0.0f);
            if (actions["MoveForward"].isPressed) moveInput.z += 10.0f;
            if (actions["MoveBack"].isPressed) moveInput.z -= 10.0f;
            if (actions["MoveLeft"].isPressed) moveInput.x -= 10.0f;
            if (actions["MoveRight"].isPressed) moveInput.x += 10.0f;

            bool jump = actions["Jump"].wasJustPressed;
            bool sprint = actions["Sprint"].isPressed;
            bool crouch = actions["Crouch"].isPressed;
            bool airSlam = actions["AirReset"].isPressed;

            glm::vec3 cameraFront = getCameraFront();
            glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, glm::vec3(0,1,0)));
            glm::vec2 mouseDelta = getMouseDelta();

            // IMPORTANT: advance simulation with fixedTimeStep/subDt, not deltaTime
            for (int i = 0; i < subStepCount; ++i) {
                characterController->update(subDt, moveInput, jump, sprint, crouch, mouseDelta, cameraFront, cameraRight, airSlam);

                std::vector<uint64_t> removedBodyIds;
                if (voxelPhysics) voxelPhysics->update(subDt, removedBodyIds);

                // handle removed bodies each substep (or accumulate ids and handle after)
                for (uint64_t id : removedBodyIds) {
                    for (auto& spell : activeSpells) {
                        if (spell.physicsBody && spell.physicsBody->id == id) {
                            spell.markForRemoval = true;
                            break;
                        }
                    }
                }

                // usually you only want "wasJustPressed" to apply once:
                jump = false;
            }

            accumulator -= fixedTimeStep;
        }
    }

    void Game::onPlayerBodyCollision(gl3::VoxelPhysicsBody* body,
                                     const glm::vec3& hitPos,
                                     const glm::vec3& hitNormal,
                                     float playerSpeed)
    {
        if (!body) return;

        const glm::vec3 playerPos = cameraPos;

        const float velLen = glm::length(body->velocity);
        if (velLen < 0.0001f) return; // too slow => ignore

        const glm::vec3 velDir = body->velocity / velLen;

        glm::vec3 toPlayer = playerPos - hitPos;
        const float toPlayerLen = glm::length(toPlayer);
        if (toPlayerLen < 0.0001f) return;

        const glm::vec3 toPlayerDir = toPlayer / toPlayerLen;

        // +1 = moving directly toward player, 0 = sideways, -1 = moving away
        const float approach = glm::dot(velDir, toPlayerDir);

        // Tune this: 0.5 means within ~60 degrees of directly toward player
        constexpr float kApproachThreshold = 0.25f;

        //if (approach < kApproachThreshold) {
          //  return;
        //}

        // Optional: scale damage by approach so grazing hits do less
        const float approach01 = glm::clamp((approach - kApproachThreshold) / (1.0f - kApproachThreshold), 0.0f, 1.0f);

        float hitSpeed = velLen / 400.0f;

        // Your original damage
        float dmg = glm::sqrt(body->mass * hitSpeed);

        // Apply approach scaling (comment out if you want full damage once threshold passes)
        //dmg *= approach01;
        dmg = glm::min(dmg,getPlayerMaxHealth()/5);
        dmg= glm::max(dmg,getPlayerMaxHealth()/10);
        setPlayerHealth(getPlayerHealth() - dmg);
        body->velocity=-body->velocity*0.75f;
    }

    void Game::onBodyBodyCollision(gl3::VoxelPhysicsBody *bodyA, gl3::VoxelPhysicsBody *bodyB, const glm::vec3 &hitPos,
                                   const glm::vec3 &hitNormal, float impactSpeed) {
            enemyManager->applyDamageSphere(bodyA->id,hitPos,bodyA->radius,impactSpeed);
            enemyManager->applyDamageSphere(bodyB->id,hitPos,bodyA->radius,impactSpeed);

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
        ZoneScoped;
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
        ZoneScoped;
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
            light.intensity = float(count) * 65.0f;
            light.id = makeLightID(coord.x, coord.y, coord.z);

            chunk->emissiveLights.push_back(light);
        }

        chunk->lightingDirty = false;
        chunkManager->updateEmissiveMembership(*chunk);
    }

    void Game::updateChunkLights(Chunk *chunk) {
        ZoneScoped;
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
        ZoneScoped;
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

    void Game::setupLightSSBOs()
    {
        glGenBuffers(1, &ssboLights);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4096 * sizeof(VoxelLightGpu), nullptr, GL_DYNAMIC_DRAW); // capacity
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glGenBuffers(1, &ssboChunkLightIdx);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboChunkLightIdx);
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_CHUNKS_GPU * sizeof(ChunkLightIndexGpu), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    inline uint32_t Game::lightIndexFromPtr(const VoxelLight* ptr) const {
        const VoxelLight* base = mergedEmissiveLightPool.data();
        return (uint32_t)(ptr - base);
    }

    void Game::uploadMergedLightsToGPU()
    {
        std::vector<VoxelLightGpu> gpu;
        gpu.resize(mergedEmissiveLightPool.size());

        for (size_t i = 0; i < mergedEmissiveLightPool.size(); ++i) {
            const auto& L = mergedEmissiveLightPool[i];
            gpu[i].posIntensity = glm::vec4(L.pos, L.intensity);
            gpu[i].color        = glm::vec4(L.color, 0.0f);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLights);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gpu.size() * sizeof(VoxelLightGpu), gpu.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void Game::buildAndUploadChunkLightIndexBuffer(int camCX, int camCY, int camCZ, int renderRadius)
    {
        static int lastUpdateFrame = -1;
        static int lastCamCX = camCX, lastCamCY = camCY, lastCamCZ = camCZ;
        static int lastRenderRadius = renderRadius;

        const int UPDATE_INTERVAL = 60;
        const int CAM_MOVE_THRESHOLD = 5;

        bool needsUpdate = false;

        if (frameCounter - lastUpdateFrame >= UPDATE_INTERVAL) {
            needsUpdate = true;
        }

        if (std::abs(camCX - lastCamCX) >= CAM_MOVE_THRESHOLD ||
            std::abs(camCY - lastCamCY) >= CAM_MOVE_THRESHOLD ||
            std::abs(camCZ - lastCamCZ) >= CAM_MOVE_THRESHOLD ||
            renderRadius != lastRenderRadius) {
            needsUpdate = true;
        }

        if (!needsUpdate) {
            return; // Skip update this frame
        }

        lastUpdateFrame = frameCounter;
        lastCamCX = camCX;
        lastCamCY = camCY;
        lastCamCZ = camCZ;
        lastRenderRadius = renderRadius;

        std::vector<ChunkLightIndexGpu> chunkIdx(MAX_CHUNKS_GPU);
        for (auto& e : chunkIdx) {
            e.count = 0;
            for (int i = 0; i < 4; ++i) e.indices[i] = 0;
        }

        for (int cx = camCX - renderRadius; cx <= camCX + renderRadius; ++cx) {
            for (int cy = camCY - renderRadius; cy <= camCY + renderRadius; ++cy) {
                for (int cz = camCZ - renderRadius; cz <= camCZ + renderRadius; ++cz) {
                    Chunk* chunk = chunkManager->getChunk({cx,cy,cz});
                    if (!chunk) continue;
                    if (!chunk->gpuCache.isValid) continue;

                    if (frameCounter - chunk->gpuCache.lastLightUpdateFrame > LIGHT_UPDATE_INTERVAL ||
                        chunk->gpuCache.nearbyLights.empty()) {
                        updateChunkLights(chunk);
                    }

                    auto& dst = chunkIdx[chunk->gpuSlot];
                    int num = std::min((int)chunk->gpuCache.nearbyLights.size(), MAX_LIGHTS);
                    dst.count = (uint32_t)num;

                    for (int i = 0; i < num; ++i) {
                        const VoxelLight* L = chunk->gpuCache.nearbyLights[i];
                        dst.indices[i] = lightIndexFromPtr(L);
                    }
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboChunkLightIdx);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, chunkIdx.size() * sizeof(ChunkLightIndexGpu), chunkIdx.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }


////----Input Code------------------------------------------------------------------------------------------------------------------------------

    void Game::update() {

        if(getPlayerHealth()<=0)
        {
            requestSceneChange(SceneId::MainMenu);
        }
        chunkManager->forEachEmissiveChunk([this](Chunk *chunk) {
            for (auto &light: chunk->emissiveLights) {
                glm::vec3 dist=(cameraPos-light.pos);
                float distsq = glm::sqrt(dist.x*dist.x+ dist.y*dist.y+ dist.z*dist.z);
                if(distsq<0.0006f*light.intensity)
                {
                    setPlayerHealth(getPlayerHealth()-0.0075f*distsq);
                }
            }
        });

        emissiveUpdateCounter++;
        if (++emissiveUpdateCounter >= 30) { // Every 30 frames
            processEmissiveChunks();
            emissiveUpdateCounter = 0;
        }

        input.update(window);
        actions.update(input);

        // Now use clean, readable input checks
        if (actions["Escape"].wasJustPressed) {
           // glfwSetWindowShouldClose(window, true);
        }

        if (actions["ToggleDebug"].wasJustPressed) {
            DebugMode1 = !DebugMode1;
            activeSpellMat=0;
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
        } else if(actions["DebugMode1"].wasJustPressed)
        {
            activeSpellMat=1;
        }
        if (actions["DebugMode2"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 2;
        } else if (actions["DebugMode2"].wasJustPressed) {
            activeSpellMat = 2;
        }
        if (actions["DebugMode3"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 3;
        } else  if (actions["DebugMode3"].wasJustPressed) {
            activeSpellMat = 3;
        }
        if (actions["DebugMode4"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 4;
        }
        else if (actions["DebugMode4"].wasJustPressed) {
            activeSpellMat = 4;
        }
        if (actions["DebugMode5"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 5;
        } else  if (actions["DebugMode5"].wasJustPressed) {
            activeSpellMat = 5;
        }
        if (actions["DebugMode6"].wasJustPressed&&DebugMode1) {
            activeDebugMode = 6;
        } else if (actions["DebugMode6"].wasJustPressed) {
            activeSpellMat = 6;
        }
        if (actions["Wireframe"].wasJustPressed&&DebugMode1) {
            DebugMode2=!DebugMode2;
        }

        if (actions["CastSphere"].wasJustReleased) {
            std::cout << "Sphere Spell Triggered\n";
            RayCastResult hit = rayCastFromCamera(5.0f);
            glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                                    (cameraPos + getCameraFront() * 35.0f);

            // Cast spell with physics enabled
            float spellRadius = 20.0f * VOXEL_SIZE;  // Adjust size
            float spellStrength = 3.0f;              // Affects velocity

            castSpellSphere(spellCenter, spellRadius, activeSpellMat, spellStrength);
        }

        if (actions["CastWall"].wasJustReleased) {
            std::cout << "Wall Spell Triggered" << "\n";
            RayCastResult hit = rayCastFromCamera(250.0f);
            glm::vec3 spellCenter = hit.hit ? hit.hitPosition+getCameraFront() * 20.5f :
                                    (cameraPos + getCameraFront() * 40.0f);

            // Get camera direction for wall orientation
            glm::vec3 cameraFront = getCameraFront();

            // Wall dimensions (tune these values)
            float wallWidth = 0.75f*VOXEL_SIZE;    // Horizontal width
            float wallHeight = 0.25f*VOXEL_SIZE;   // Vertical height
            float wallThickness = 3.5f*VOXEL_SIZE; // How thick the wall is

            // Cast the wall spell
            castSpellWall(spellCenter, cameraFront,
                          wallWidth, wallHeight, wallThickness,
                          0, 2.0f*VOXEL_SIZE);
        }
        if (actions["AirReset"].wasJustReleased) {
            std::cout << "Platform Spell Triggered" << "\n";
            glm::vec3 spellCenter =(cameraPos + glm::vec3(0,-1,0) * 25.0f*VOXEL_SIZE);

            // Wall dimensions (tune these values)
            float wallWidth = 1.0f*VOXEL_SIZE;    // Horizontal width
            float wallHeight = 1.05f*VOXEL_SIZE;   // Vertical height
            float wallThickness = 3.5f*VOXEL_SIZE; // How thick the wall is

            // Cast the wall spell
            castSpellWall(spellCenter, glm::vec3(0,-1,0),
                          wallWidth, wallHeight, wallThickness,
                          0, 7.5f*VOXEL_SIZE);
        }


        // Character movement - perfect for your controller

        // Update camera to follow character
        updateCamera();
        burn01(1.0f,5.0f);

        // Update dynamic chunks
        // chunkManager->forEachDynamicChunk([this](Chunk* chunk) {
        // Physics, animation, etc.
        //});

        // Update fluid chunks
        //chunkManager->forEachFluidChunk([this](Chunk* chunk) {
        // Fluid simulation
        //});
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
        TRACY_CPU_ZONE("Game::renderSkybox");
        TRACY_GPU_ZONE("Skybox");

        // If not baked yet for some reason, bake now (safe fallback)
        if (!skyboxBaked) {
            bakeNebulaCubemap(512);
        }
        if (!skyboxRuntimeShader) return;

        GLint oldDepthFunc;
        glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
        GLboolean depthMask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);

        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);

        skyboxRuntimeShader->use();
        skyboxRuntimeShader->setFloat("time", (float)glfwGetTime());

        float aspect = (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));

        skyboxRuntimeShader->setMatrix("projection", projection);
        skyboxRuntimeShader->setMatrix("view", view);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, nebulaCubemap);
        skyboxRuntimeShader->setInt("nebulaCube", 0);

        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glDepthFunc(oldDepthFunc);
        glDepthMask(depthMask);
    }

    void Game::renderChunks()
    {
        TRACY_CPU_ZONE("Game::renderChunks");
        TRACY_GPU_ZONE("Chunks (total)");

        int built = 0;
        int lighted = 0;

        glPolygonMode(GL_FRONT_AND_BACK, DebugMode2 ? GL_LINE : GL_FILL);

        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        frameCounter++;

        emissiveBillboards.clear();
        usedLightIDs.clear();

        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);
        const int renderRadius = RenderingRange;

        // Cleanup distant chunks periodically (every 60 frames)
        if (frameCounter % 60 == 0) {
            TRACY_CPU_ZONE("renderChunks::cleanupDistantSlots");
            chunkManager->cleanupDistantSlots(cameraPos, renderRadius);
        }

        // Update merged lights occasionally (CPU) + upload to GPU
        if (frameCounter % 240 == 0) {
            TRACY_CPU_ZONE("renderChunks::updateLightSpatialHash");
            updateLightSpatialHash();
            uploadMergedLightsToGPU();
        }

       // visibleDrawCmds.clear();
        visibleSlots.clear();
       // visibleDrawCmds.reserve((2*renderRadius+1)*(2*renderRadius+1)*(2*renderRadius+1));

        // Generate meshes / rebuild emissive lights
        {
            TRACY_CPU_ZONE("renderChunks::PrepareMeshesAndLights");
            const int R = chunkManager->radius();

            const int minCX = std::max(camCX - renderRadius, -R);
            const int maxCX = std::min(camCX + renderRadius,  R);
            const int minCY = std::max(camCY - renderRadius, -R);
            const int maxCY = std::min(camCY + renderRadius,  R);
            const int minCZ = std::max(camCZ - renderRadius, -R);
            const int maxCZ = std::min(camCZ + renderRadius,  R);


            for (int cx = minCX; cx <= maxCX; ++cx)
                for (int cy = minCY; cy <= maxCY; ++cy)
                    for (int cz = minCZ; cz <= maxCZ; ++cz)
                    {
                        ChunkCoord coord{cx, cy, cz};
                        Chunk* chunk = chunkManager->getChunk(coord);
                        if (!chunk) continue;

                        // Skip empty chunks (no geometry)
                        if (chunk->isCleared) continue;

                        // Allocate GPU slot if needed and chunk has content
                        if (chunk->gpuSlot == FixedGridChunkManager::INVALID_GPU_SLOT) {
                            uint32_t slot = chunkManager->allocateGpuSlot(coord);
                            if (slot == FixedGridChunkManager::INVALID_GPU_SLOT) {
                                // No GPU slots available, skip this chunk for now
                                continue;
                            }
                        }

                        if (built < MAX_CHUNKS_PER_FRAME && (chunk->meshDirty || !chunk->gpuCache.isValid)) {
                            built++;
                            generateChunkMesh(chunk);
                        }

                        if (lighted < MAX_CHUNKS_PER_FRAME && chunk->lightingDirty) {
                            lighted++;
                            rebuildChunkLights(chunk->coord);
                            chunk->lightingDirty = false;
                        }

                        for (const auto& light : chunk->emissiveLights) {
                            if (usedLightIDs.insert(light.id).second) {
                                SunInstance inst;
                                inst.position = light.pos;
                                inst.scale = std::sqrt(light.intensity) * 0.5f;
                                inst.color = light.color * 1.0f;
                                emissiveBillboards.push_back(inst);
                            }
                        }

                        if (!chunk->gpuCache.isValid) continue;

                        DrawArraysIndirectCommand cmd{};
                        cmd.count         = chunk->gpuCache.vertexCount;
                        cmd.instanceCount = 1;
                        cmd.first         = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
                        cmd.baseInstance  = chunk->gpuSlot;

                        if (cmd.count == 0) continue;

                       // visibleDrawCmds.push_back(cmd);
                        visibleSlots.push_back(chunk->gpuSlot);
                    }
        }

        // per-chunk light index buffer
        {
            TRACY_CPU_ZONE("renderChunks::buildAndUploadChunkLightIndexBuffer");
            buildAndUploadChunkLightIndexBuffer(camCX, camCY, camCZ, renderRadius);
        }

        // Draw
        {
            TRACY_CPU_ZONE("renderChunks::DrawBatched");
            TRACY_GPU_ZONE("Chunks::DrawBatched");

            voxelShader->use();

            if (actions["CastSphere"].isHeld) {
                float maxDist = 250.0f;
                RayCastResult hit = rayCastFromCamera(maxDist);
                glm::vec3 center = hit.hit ? hit.hitPosition : (cameraPos + getCameraFront() * 35.0f);

                float formationRadius = 2.0f * VOXEL_SIZE;
                float pullRadius = formationRadius * 15.5f;

                voxelShader->setInt("uOverlayEnabled", 1);
                voxelShader->setVec3("uOverlayCenter", center);
                voxelShader->setFloat("uOverlayRadius", pullRadius);
                voxelShader->setUInt("uOverlayMaterial", (uint32_t)0);
                voxelShader->setVec3("uOverlayColor", glm::vec3(0.25f, 1.0f, 0.35f));
                voxelShader->setFloat("uOverlayAlpha", 0.25f);
            } else {
                voxelShader->setInt("uOverlayEnabled", 0);
            }

            voxelShader->setMatrix("model", glm::mat4(1.0f));
            voxelShader->setMatrix("mvp", pv);
            voxelShader->setVec3("viewPos", cameraPos);
            voxelShader->setVec3("ambientColor", glm::vec3(0.85f));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
            voxelShader->setInt("uAlbedoArray", 0);

            voxelShader->setFloatArray("uMatRoughness", rough.data(), 64);
            voxelShader->setFloatArray("uMatSpecular",  spec.data(), 64);
            voxelShader->setFloatArray("uUVScale",      uvScale.data(), 64);

            if (DebugMode1) voxelShader->setInt("debugMode", activeDebugMode % 5);
            else voxelShader->setFloat("emission", 0.0f);


            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, ssboLights);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, ssboChunkLightIdx);

            glBindVertexArray(globalChunkVAO);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);

            for (uint32_t slot : visibleSlots) {
                const GLsizeiptr offset = (GLsizeiptr)slot * (GLsizeiptr)sizeof(DrawArraysIndirectCommand);
                glDrawArraysIndirect(GL_TRIANGLES, (const void*)offset);
            }

            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
            glBindVertexArray(0);
        }

        // Billboards
        if (!emissiveBillboards.empty() && !DebugMode1) {
            sunBillboards.render(emissiveBillboards, view, projection, (float)glfwGetTime());
        }
    }

    void Game::renderAnimatedVoxels() {
        TRACY_CPU_ZONE("Game::renderAnimatedVoxels");
        TRACY_GPU_ZONE("AnimatedVoxels");
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
        instancedShader.setVec3("ambientColor", glm::vec3(0.2f)); // match your chunk ambient

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

            // FIXED: Scale the cube to match VOXEL_SIZE
            // The cube is unit size (1x1x1), so multiply by VOXEL_SIZE to get correct world size
            float halfSize = (VOXEL_SIZE * 0.25f) * pulse;

            posScaleData.push_back(v->currentPos.x);
            posScaleData.push_back(v->currentPos.y);
            posScaleData.push_back(v->currentPos.z);
            posScaleData.push_back(halfSize); // Now scales with VOXEL_SIZE


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
        TRACY_CPU_ZONE("Game::renderPhysicsFormations");
        TRACY_GPU_ZONE("PhysicsFormations");
        if (activeSpells.empty()) return;


        voxelShader->use();
        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.85f));

        // texture unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
        voxelShader->setInt("uAlbedoArray", 0);

        voxelShader->setFloatArray("uMatRoughness", rough.data(), 64);
        voxelShader->setFloatArray("uMatSpecular",  spec.data(), 64);
        voxelShader->setFloatArray("uUVScale",      uvScale.data(), 64);


        voxelShader->setInt("uBurnEnabled", 0);
        voxelShader->setFloat("uBurn", 0.0f);
        voxelShader->setVec3("uBurnCenter", glm::vec3(0.0f));
        voxelShader->setFloat("uBurnRadius", 0.0f);
        voxelShader->setFloat("uBurnNoiseScale", 0.35f);
        voxelShader->setFloat("uBurnEdgeWidth", 0.12f);
        voxelShader->setVec3("uBurnEmberColor", glm::vec3(2.5f, 0.9f, 0.2f));
        voxelShader->setFloat("uBurnCharStrength", 0.85f);


        static float time = 0;
        time += deltaTime;


        // Render each physics-enabled formation
        for (const auto& spell : activeSpells) {
            if (!spell.isPhysicsEnabled || !spell.physicsBody) continue;
            // Ensure destructible mesh exists (this is what we render now)
            if (!spell.destruct.mesh.isValid || spell.destruct.mesh.vertexCount == 0) {
                //rebuildDestructibleMeshIfNeeded(const_cast<gl3::SpellEffect&>(spell).destruct);
                continue;
            }
            // Use stored position and orientation directly
            int currentChunkX = worldToChunk(spell.physicsBody->position.x);
            int currentChunkY = worldToChunk(spell.physicsBody->position.y);
            int currentChunkZ = worldToChunk(spell.physicsBody->position.z);

            glm::vec3 pos = glm::vec3(spell.physicsBody->position.x,spell.physicsBody->position.y,spell.physicsBody->position.z) ;
            glm::quat rot = spell.physicsBody->orientation;

            // Build model matrix
            glm::vec3 originWorld = spell.physicsBody->position;
            glm::mat4 model = glm::translate(glm::mat4(1.0f), originWorld);
            //glm::scale(model,glm::vec3(VOXEL_SIZE));
            //model *= glm::mat4_cast(rot);
            //model = glm::scale(model, glm::vec3(VOXEL_SIZE));  // Apply VOXEL_SIZE scaling

            voxelShader->setMatrix("model", model);
            voxelShader->setMatrix("mvp", pv * model);
            voxelShader->setFloat("scale",spell.physicsBody->radius);

            if (spell.burn.active) {
                float u = burn01(spell.burn.t, spell.burn.duration);
                voxelShader->setInt("uBurnEnabled", 1);
                voxelShader->setFloat("uBurn", u);
                voxelShader->setVec3("uBurnCenter", spell.burn.center);
                voxelShader->setFloat("uBurnRadius", spell.burn.radius);
                voxelShader->setFloat("uBurnNoiseScale", spell.burn.noiseScale);
                voxelShader->setFloat("uBurnEdgeWidth", spell.burn.edgeWidth);
                voxelShader->setVec3("uBurnEmberColor", glm::vec3(2.5f, 0.9f, 0.2f));
                voxelShader->setFloat("uBurnCharStrength", 0.85f);
            } else {
                voxelShader->setInt("uBurnEnabled", 0);
            }
            // Draw mesh
            if (!spell.destruct.mesh.isValid || spell.destruct.mesh.vertexCount == 0) continue;

            glBindVertexArray(spell.destruct.mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, spell.destruct.mesh.vertexCount);
            glBindVertexArray(0);
        }
    }


    void Game::renderEnemies() {
        if (!enemyManager) return;

        voxelShader->use();

        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0,1,0));
        glm::mat4 pv = projection * view;

        for (auto& e : enemyManager->all()) {
            if (!e.renderMesh.isValid || e.renderMesh.vertexCount == 0) continue;

            glm::vec3 localCenter =
                    0.5f * (glm::vec3(e.volume.dims) - glm::vec3(1.0f)) * e.volume.voxelSize;

            glm::mat4 model = glm::translate(glm::mat4(1.0f), e.inst.position - localCenter);
            // later: model *= glm::mat4_cast(e.inst.rotation);

            voxelShader->setMatrix("model", model);
            voxelShader->setMatrix("mvp", pv * model);

            glBindVertexArray(e.renderMesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)e.renderMesh.vertexCount);
            glBindVertexArray(0);
        }
    }

    void Game::renderSpellPreview() {
        TRACY_CPU_ZONE("Game::renderSpellPreview");
        TRACY_GPU_ZONE("SpellPreview");
        if (!spellPreviewShader) return;

        // If you only want preview in non-debug or only in debug, gate it here.
        // For now: always on.
        ensurePreviewCube();
        ensurePreviewSphereMesh();

        // ---- Determine what spell is "armed" (based on your current keybinds) ----
        // You have:
        //   E: sphere spell
        //   R: wall spell
        //   F: air reset (currently also wall-like in your code)
        //
        // For preview, choose one "active preview mode".
        // Minimal: show sphere if E is held, wall if R held, else none.
        // If you want "last selected", add a variable and update it in update().
        int previewMode = -1; // -1 none, 0 sphere, 1 wall

        if (actions["CastSphere"].isHeld) previewMode = 0;
        else if (actions["CastWall"].isHeld) previewMode = 1;
        else if (actions["AirReset"].isHeld) previewMode = 1;
        else return;

        // ---- Compute placement point (center) from camera raycast ----
        // Use a longer ray for wall preview
        float maxDist = (previewMode == 0) ? 80.0f : 250.0f;
        RayCastResult hit = rayCastFromCamera(maxDist);
        glm::vec3 center;
        if(previewMode==0)
        {
            center = hit.hit ? hit.hitPosition : (cameraPos + getCameraFront() * 35.0f);
        } else
        {
            center = hit.hit ? hit.hitPosition+ getCameraFront() * 10.5f :
                     (cameraPos + getCameraFront() * 10.5f);
        }

        // Optional: snap to voxel grid so placement feels stable
       /* auto snap = [&](float v) {
            return std::round(v / VOXEL_SIZE) * VOXEL_SIZE;
        };
        center = glm::vec3(snap(center.x), snap(center.y), snap(center.z));*/

        // ---- Build pv (same as other render passes) ----
        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 pv = projection * view;

        // ---- Spell params (match your cast code) ----
        float formationRadius = 2.0f * VOXEL_SIZE;     // your castSpellSphere uses 4*VOXEL_SIZE
        float pullRadius      = formationRadius * 6.5f; // your searchRadius behavior

        // Wall params (match your CastWall in update())
        glm::vec3 wallNormal=-getCameraFront();
        glm::vec3 wallUp(0,1,0);
        glm::vec3 wallSize(3.0f*VOXEL_SIZE, 1.75f*VOXEL_SIZE, 0.35f*VOXEL_SIZE);

        // AirReset spell currently also calls castSpellWall with normal (0,-1,0) and tiny dims
        if (actions["AirReset"].isPressed) {
            wallNormal = glm::vec3(0,-1,0);
            wallUp     = glm::vec3(0,0,1); // choose a stable up when normal ~Y
            wallSize   = glm::vec3(3.0f*VOXEL_SIZE, 1.0f*VOXEL_SIZE, 3.5f*VOXEL_SIZE);
            pullRadius = 10.0f * 2.0f * VOXEL_SIZE; // your castSpellPlatform searchRadius-ish, but you used wall; pick something visible
            center = (cameraPos + glm::vec3(0,-1,0) * 4.0f *VOXEL_SIZE);
        }

        // ---- Draw state for hologram overlay ----
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Usually looks better not writing depth for translucent overlays
        glDepthMask(GL_FALSE);

        spellPreviewShader->use();

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        int maxVoxels = (int)((4.0f * 70.0f) / voxelVolume);
        maxVoxels = glm::clamp(maxVoxels, 5 , 200);

        int available = estimateAvailableVoxels(center, pullRadius, 0, maxVoxels);
        float fillRatio = maxVoxels > 0 ? (float)available / (float)maxVoxels : 1.0f;

        spellPreviewShader->setFloat("uFillRatio", fillRatio);
        spellPreviewShader->setVec3("uLowColor",  glm::vec3(1.0f, 0.2f, 0.2f)); // red
        spellPreviewShader->setVec3("uHighColor", glm::vec3(0.2f, 1.0f, 0.2f)); // green

        spellPreviewShader->setMatrix("pv", pv);
        spellPreviewShader->setVec3("uCenter", center);
        spellPreviewShader->setFloat("uVoxelSize", VOXEL_SIZE);


        spellPreviewShader->setVec3("uFormationColor", glm::vec3(0.2f, 0.8f, 1.0f));
        spellPreviewShader->setFloat("uFormationAlpha", 0.35f);

        // ---- Draw formation preview ----
        if (previewMode == 0) {
            // Sphere: use sphere mesh
            spellPreviewShader->setInt("uPreviewMode", 0);
            spellPreviewShader->setFloat("uFormationRadius", formationRadius);

            // model = translate(center) * scale(radius)
            glm::mat4 model(1.0f);
            model = glm::translate(model, center);
            model = glm::scale(model, glm::vec3(formationRadius));
            spellPreviewShader->setMatrix("model", model);

            // wall uniforms still must be set (safe defaults)
            spellPreviewShader->setVec3("uWallNormal", glm::vec3(0,0,1));
            spellPreviewShader->setVec3("uWallUp", glm::vec3(0,1,0));
            spellPreviewShader->setVec3("uWallSize", glm::vec3(1));

            glBindVertexArray(previewSphereVAO);
            glDrawElements(GL_TRIANGLES, previewSphereIndexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        } else {
            // Wall: draw cube proxy scaled to wallSize
            spellPreviewShader->setInt("uPreviewMode", 1);
            spellPreviewShader->setFloat("uFormationRadius", formationRadius); // unused for wall

            spellPreviewShader->setVec3("uWallNormal", wallNormal);
            spellPreviewShader->setVec3("uWallUp", wallUp);
            spellPreviewShader->setVec3("uWallSize", wallSize);

            // For now: model is just translate(center) (orientation is handled in fragment SDF)
            // But the mesh still must cover the region. Use a cube scaled to wall size.
            glm::mat4 model(1.0f);
            model = glm::translate(model, center);
            model = glm::scale(model, wallSize);
            spellPreviewShader->setMatrix("model", model);

            glBindVertexArray(previewCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);
        }

        // restore state
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

//------Marching Cubes-Code---------------------------------------------------------------------------------------------------------------------

    void Game::generateChunkMesh(Chunk* chunk)
    {
        if (!chunk) return;

        if (chunk->gpuSlot >= (uint32_t)MAX_CHUNKS_GPU) {
            std::cout << "BAD SLOT: " << chunk->gpuSlot << " MAX=" << MAX_CHUNKS_GPU << "\n";
            return;
        }

        if (chunk->isCleared) {
            DrawArraysIndirectCommand cmd{};
            cmd.count = 0;
            chunk->gpuCache.vertexCount = 0;
            cmd.instanceCount = 1;
            cmd.first = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
            cmd.baseInstance = chunk->gpuSlot;

            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
            glBufferSubData(GL_DRAW_INDIRECT_BUFFER,
                            chunk->gpuSlot * sizeof(DrawArraysIndirectCommand),
                            sizeof(DrawArraysIndirectCommand),
                            &cmd);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

            chunk->gpuCache.isValid = true;
            chunk->meshDirty = false;
            return;
        }

        // TODO: optional CPU early-out: if no solid, also set cmd.count=0 as above and return.

        glm::vec3 chunkOrigin(
                chunk->coord.x * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.y * CHUNK_SIZE * gl3::VOXEL_SIZE,
                chunk->coord.z * CHUNK_SIZE * gl3::VOXEL_SIZE
        );

        marchingCubesShader->use();

        uploadVoxelChunk(*chunk, nullptr);

        resetAtomicCounter();
        setComputeUniforms(chunkOrigin, *marchingCubesShader);

        marchingCubesShader->setUInt("uChunkSlot", chunk->gpuSlot);
        marchingCubesShader->setUInt("uChunkMaxVerts", (uint32_t)CHUNK_MAX_VERTS);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, globalChunkVertexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, chunkIndirectBuffer);

        int cellsPerAxis = DIM - 1;
        int groups = (cellsPerAxis + 7) / 8;
        glDispatchCompute(groups, groups, groups);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
                        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
                        GL_COMMAND_BARRIER_BIT);

        // readback vertexCounter (4 bytes)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        uint32_t produced = 0;
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &produced);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        DrawArraysIndirectCommand cmd{};
        cmd.count = produced;
        chunk->gpuCache.vertexCount = produced;
        cmd.instanceCount = 1;
        cmd.first = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
        cmd.baseInstance = chunk->gpuSlot;

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
        glBufferSubData(GL_DRAW_INDIRECT_BUFFER,
                        chunk->gpuSlot * sizeof(DrawArraysIndirectCommand),
                        sizeof(DrawArraysIndirectCommand),
                        &cmd);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        chunk->gpuCache.isValid = true;
        chunk->meshDirty = false;
    }

    bool Game::tryResolveChunkVertexCount(Chunk* chunk)
    {
        if (!chunk->gpuCache.hasPendingCount || !chunk->gpuCache.counterFence)
            return false;

        // Non-blocking poll:
        GLenum res = glClientWaitSync(chunk->gpuCache.counterFence, 0, 0);
        if (res == GL_TIMEOUT_EXPIRED)
            return false;

        // Fence signaled (or already signaled). Read the 4 bytes.
        glDeleteSync(chunk->gpuCache.counterFence);
        chunk->gpuCache.counterFence = 0;
        chunk->gpuCache.hasPendingCount = false;

        glBindBuffer(GL_COPY_READ_BUFFER, chunk->gpuCache.counterReadbackBuffer);
        void* ptr = glMapBufferRange(GL_COPY_READ_BUFFER, 0, sizeof(uint32_t), GL_MAP_READ_BIT);
        if (ptr) {
            chunk->gpuCache.vertexCount = *reinterpret_cast<uint32_t*>(ptr);
            glUnmapBuffer(GL_COPY_READ_BUFFER);
        }
        glBindBuffer(GL_COPY_READ_BUFFER, 0);

        return true;
    }


// debug version: if doColorByDensity==true, set per-voxel color from density (visualize SDF)
    void Game::uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor) {
        const int localDIM = DIM; // Should be CHUNK_SIZE + 2 for padding
        const size_t total = size_t(localDIM) * localDIM * localDIM;
        std::vector<CpuVoxelStd430> voxels;
        voxels.resize(total);

        for (int x = -1; x <= CHUNK_SIZE; ++x) {
            for (int y = -1; y <= CHUNK_SIZE; ++y) {
                for (int z = -1; z <= CHUNK_SIZE; ++z) {
                    int idxX = x + 1;
                    int idxY = y + 1;
                    int idxZ = z + 1;
                    int idx = idxX + idxY * localDIM + idxZ * localDIM * localDIM;

                    const Voxel *srcVoxel = nullptr;

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

                        Chunk *neighbor = chunkManager->getChunk(neighborCoord);
                        if (neighbor && localX >= -1 && localX <= CHUNK_SIZE &&
                            localY > -1 && localY <= CHUNK_SIZE &&
                            localZ > -1 && localZ <= CHUNK_SIZE) {
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
                    voxels[idx].pad0 = voxels[idx].pad1 = voxels[idx].pad2 = 0.0f;
                    glm::vec3 col = overrideColor ? *overrideColor : srcVoxel->color;

                    voxels[idx].color[0] = col.r;
                    voxels[idx].color[1] = col.g;
                    voxels[idx].color[2] = col.b;
                    voxels[idx].color[3] = 1.0f;
                    voxels[idx].type = srcVoxel->type;
                    voxels[idx].material=srcVoxel->material;
                }
            }
        }

        // Upload to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxels.size() * sizeof(CpuVoxelStd430), voxels.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void Game::resetAtomicCounter() {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);
    }

    void Game::setComputeUniforms(const glm::vec3& chunkOrigin, Shader& computeShader) {
        computeShader.use();
        computeShader.setFloat("voxelSize", gl3::VOXEL_SIZE);
        computeShader.setIVec3("voxelGridDim", glm::ivec3(DIM, DIM, DIM));

        // IMPORTANT: padded voxel index (0,0,0) corresponds to world (chunkOrigin - voxelSize)
        computeShader.setVec3("gridOrigin", chunkOrigin - glm::vec3(gl3::VOXEL_SIZE));
    }
    void Game::setupChunkBatchBuffers(int maxChunksGpu)
    {
        MAX_CHUNKS_GPU = maxChunksGpu;
        CHUNK_MAX_VERTS = (DIM - 1) * (DIM - 1) * (DIM - 1) * 5 * 3;

        const int MAX_VERTS_PER_CHUNK = 12000;
        CHUNK_MAX_VERTS = std::min(
                (int)chunkMaxVertices(DIM),
                MAX_VERTS_PER_CHUNK
        );

        // Global vertex buffer
        glGenBuffers(1, &globalChunkVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, globalChunkVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
                     MAX_CHUNKS_GPU * CHUNK_MAX_VERTS * sizeof(OutVertexStd430),
                     nullptr,
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Indirect command buffer
        glGenBuffers(1, &chunkIndirectBuffer);

        std::vector<DrawArraysIndirectCommand> cmds(MAX_CHUNKS_GPU);
        for (uint32_t s = 0; s < (uint32_t)MAX_CHUNKS_GPU; ++s) {
            cmds[s].count = 0;
            cmds[s].instanceCount = 1;
            cmds[s].first = s * (uint32_t)CHUNK_MAX_VERTS;
            cmds[s].baseInstance = s;
        }

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkIndirectBuffer);
        glBufferData(GL_DRAW_INDIRECT_BUFFER,
                     cmds.size() * sizeof(DrawArraysIndirectCommand),
                     cmds.data(),
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

        // VAO setup
        glGenVertexArrays(1, &globalChunkVAO);
        glBindVertexArray(globalChunkVAO);

        glBindBuffer(GL_ARRAY_BUFFER, globalChunkVertexBuffer);

        constexpr GLsizei stride = sizeof(OutVertexStd430);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, pos));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, color));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(OutVertexStd430, uv));
        glEnableVertexAttribArray(4);
        glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, stride, (void*)offsetof(OutVertexStd430, flags));

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
        glm::vec3 headPos = characterController->getCameraPosition();
        float distsq=glm::sqrt(headPos.x*headPos.x+headPos.y*headPos.y+headPos.z*headPos.z);
       // std::cout<<"Distance: "<< distsq << "/"<< 750<<"\n";
        if(distsq>750)
        {
            setPlayerHealth(getPlayerHealth()-0.2f);
        }
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

    void Game::createNebulaCubemap(int size) {
        if (nebulaCubemap != 0) return;

        glGenTextures(1, &nebulaCubemap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, nebulaCubemap);

        // Use HDR-ish storage if you want (optional):
        // internal format GL_RGB16F + type GL_FLOAT gives nicer gradients.
        // But GL_RGB8 is fine too and smaller.
        for (int face = 0; face < 6; ++face) {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                         0,
                         GL_RGB16F,
                         size, size,
                         0,
                         GL_RGB,
                         GL_FLOAT,
                         nullptr);
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

        // FBO + depth renderbuffer for baking
        glGenFramebuffers(1, &skyboxBakeFBO);
        glGenRenderbuffers(1, &skyboxBakeRBO);

        glBindFramebuffer(GL_FRAMEBUFFER, skyboxBakeFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, skyboxBakeRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, skyboxBakeRBO);

        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Skybox bake FBO incomplete, status=" << status << "\n";
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Game::bakeNebulaCubemap(int size) {
        if (skyboxBaked) return;

        createNebulaCubemap(size);

        if (!skyboxNebulaBakeShader) {
            skyboxNebulaBakeShader = std::make_unique<Shader>("shaders/skybox.vert",
                                                              "shaders/skybox_nebula_bake.frag");
        }

        // Make sure your cube VAO exists (you already build skyboxVAO in setupSkybox()).
        // So ensure setupSkybox() has run before calling bakeNebulaCubemap().

        const glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
        const glm::mat4 captureViews[] = {
                glm::lookAt(glm::vec3(0.0f), glm::vec3( 1.0f, 0.0f, 0.0f), glm::vec3(0.0f,-1.0f, 0.0f)), // +X
                glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f,-1.0f, 0.0f)), // -X
                glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)), // +Y
                glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f,-1.0f, 0.0f), glm::vec3(0.0f, 0.0f,-1.0f)), // -Y
                glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f, 0.0f, 1.0f), glm::vec3(0.0f,-1.0f, 0.0f)), // +Z
                glm::lookAt(glm::vec3(0.0f), glm::vec3( 0.0f, 0.0f,-1.0f), glm::vec3(0.0f,-1.0f, 0.0f))  // -Z
        };

        // Save viewport
        GLint prevViewport[4];
        glGetIntegerv(GL_VIEWPORT, prevViewport);

        glBindFramebuffer(GL_FRAMEBUFFER, skyboxBakeFBO);
        glViewport(0, 0, size, size);

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        skyboxNebulaBakeShader->use();
        skyboxNebulaBakeShader->setMatrix("projection", captureProjection);

        // bind noise texture the bake shader expects (your existing 2D noise)
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cubemapTexture); // NOTE: this is your 2D noise texture, confusing name
        skyboxNebulaBakeShader->setInt("noiseTexture", 0);

        // Choose bakeTime. If you want a “base state”, 0 is fine.
        skyboxNebulaBakeShader->setFloat("bakeTime", 0.0f);

        glBindVertexArray(skyboxVAO);

        for (int face = 0; face < 6; ++face) {
            skyboxNebulaBakeShader->setMatrix("view", captureViews[face]);

            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                                   nebulaCubemap,
                                   0);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // draw the cube; fragment shader writes nebula
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Restore viewport
        glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);

        skyboxBaked = true;
        std::cout << "Baked nebula cubemap at " << size << "x" << size << "\n";
    }

    // 1) Helper to start burn on a chunk (call when you decide it should cleanup)
    void Game::startChunkBurn(gl3::Chunk* chunk, const glm::vec3& chunkCenterWorld,
                              float radiusWorld, float durationSec)
    {
        if (!chunk) return;
        if (chunk->burn.active) return; // important: do not restart

        chunk->burn.active = true;
        chunk->burn.t = 0.0f;
        chunk->burn.duration = durationSec;
        chunk->burn.center = chunkCenterWorld;
        chunk->burn.radius = radiusWorld;
        chunk->burn.noiseScale = 0.18f;
        chunk->burn.edgeWidth  = 0.10f;
        chunk->burn.slowAccum  = 0.0f;
    }

    void Game::startSpellBurn(gl3::SpellEffect& spell, float radiusWorld, float durationSec)
    {
        if (spell.burn.active) return; // important: do not restart

        spell.burn.active = true;
        spell.burn.t = 0.0f;
        spell.burn.duration = durationSec;
        spell.burn.center = spell.center; // will follow physics body in updateSpells
        spell.burn.radius = radiusWorld;
        spell.burn.noiseScale = 0.22f;
        spell.burn.edgeWidth  = 0.10f;
        spell.burn.slowAccum  = 0.0f;
    }


    // 0..1 burn progress (but use curve in shader uniform)
    float Game::burn01(float t, float duration) {
        if (duration <= 1e-4f) return 1.0f;
        return glm::clamp(t / duration, 0.0f, 1.0f);
    }


    void Game::CreateFullscreenTriangle(GLuint& vao, GLuint& vbo)
    {
        // Fullscreen triangle (no index buffer)
        // positions (x,y) + uvs (u,v)
        const float verts[] = {
                //  x,  y,   u,  v
                -1.f, -1.f, 0.f, 0.f,
                3.f, -1.f, 2.f, 0.f,
                -1.f,  3.f, 0.f, 2.f
        };

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindVertexArray(0);
    }

    void Game::spawnEnemyLaunchSphere(const glm::vec3& start,
                                      const glm::vec3& target,
                                      float radiusWorld,
                                      float speedWorld,
                                      glm::vec3 color)
    {
        float searchRadius = radiusWorld * 2.5f;

        FormationParams params = FormationParams::Sphere(start, radiusWorld);
        //castSpellWithFormation(center, searchRadius, material, strength, params);
        SpellCastRequest req = buildSpellCastRequestSnapshot(start, searchRadius, 0, 5, params);
        req.physicsEnabled = true;
        req.launchDir = glm::normalize(target-start);
        req.launchSpeed = speedWorld * VOXEL_SIZE;
        req.lifetime = 20.0f;
        spellCastAsync->enqueueOrReplaceQueued(std::move(req));
    }

    void Game::buildSphereMeshData(gl3::PhysicsMeshData& outMesh,
                                   float radiusWorld,
                                   const glm::vec3& color)
    {
        if (sphereMeshCache.empty()) return;

        int key = static_cast<int>(std::round(radiusWorld / VOXEL_SIZE));
        auto it = sphereMeshCache.find(key);
        if (it == sphereMeshCache.end()) {
            // nearest fallback
            it = sphereMeshCache.begin();
            int bestKey = it->first;
            for (auto& [k, _] : sphereMeshCache) {
                if (std::abs(k - key) < std::abs(bestKey - key)) bestKey = k;
            }
            it = sphereMeshCache.find(bestKey);
        }

        const SphereMesh& base = it->second;

        std::vector<glm::vec3> verts;
        std::vector<glm::vec3> norms;
        std::vector<glm::vec3> cols;

        float scale = radiusWorld / base.radius;

        verts.reserve(base.indices.size());
        norms.reserve(base.indices.size());
        cols.reserve(base.indices.size());

        for (uint32_t idx : base.indices) {
            verts.push_back(base.vertices[idx] * scale);
            norms.push_back(base.normals[idx]);
            cols.push_back(color);
        }

        // uses the generic helper you added earlier:
        createPhysicsMeshData(outMesh, verts, norms, cols);
    }

}
