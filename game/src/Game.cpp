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
#include "Entities/LocalMarchingCubes.h"
#include "physics/MaterialCollisionPolicy.h"

#ifndef TRACY_GPU_ENABLED
#define TRACY_CPU_ZONE(nameStr) ZoneScopedN(nameStr)
#define TRACY_GPU_ZONE(nameStr) TracyGpuZone(nameStr)
#endif
namespace gl3 {

////-----Basics---------------------------------------------------------------------------------------------------------------------------------

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
            game->initFluidFBO();
            //game->initGasFBO();
            game->initCompositeFBO();
        }
    }


    Game::Game(int width, int height, const std::string &title)
            : windowWidth(width),
              windowHeight(height),
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

        chunkManager = std::make_unique<FixedGridChunkManager>(WORLD_RADIUS_CHUNKS);
        chunkRenderer = std::make_unique<ChunkRenderer>(chunkManager.get());
        chunkRenderer->initialize();

        //UI Setup
        imguiLayer.init(window, "#version 460");

        //Shader Setup
        skyboxRuntimeShader = std::make_unique<Shader>("shaders/skybox.vert", "shaders/skybox_runtime.frag");
        voxelShader = std::make_unique<Shader>("shaders/voxel.vert", "shaders/voxel.frag");
        fluidShader = std::make_unique<Shader>("shaders/fluid_voxel.vert", "shaders/fluid_voxel.frag");
        gasRayMarchShader = std::make_unique<Shader>("shaders/gas_ray_march.comp");
        marchingCubesShader = std::make_unique<Shader>("shaders/marching_cubes.comp");
        spellPreviewShader = std::make_unique<Shader>("shaders/spell_prev.vert", "shaders/spell_prev.frag");
        postShader = std::make_unique<Shader>("shaders/post_fullscreen.vert", "shaders/post_fog_glow.frag");
        damageShader = std::make_unique<Shader>("shaders/speedlines.vert", "shaders/damage.frag");

        //SceneManager Setup
        sceneManager.registerScene(SceneId::MainMenu, std::make_unique<MainMenuScene>());
        sceneManager.registerScene(SceneId::Gameplay, std::make_unique<GameplayScene>());
        sceneManager.registerScene(SceneId::Loading, std::make_unique<LoadingScene>());
        sceneManager.registerScene(SceneId::GameOver, std::make_unique<GameOverScene>());

        sceneManager.requestChange(SceneId::MainMenu);
        sceneManager.applyPendingChange();
        applyDisplaySettings();
        //pickResolutionFromNativeMonitor(true);

        //init Music:
        g_SoundManager.init();
        initAudio();

        SoLoud::handle musicHandle = g_SoundManager.playMusic(SoundID::MainMenuTheme, true, 1.0f);

        skillTree.SetMainMenuCallback([&]() {
            g_SoundManager.playSound(SoundID::MenuClose, 1.0f, 1.0f);
            pauseSubmenu = PauseSubmenu::Main;
        });

            }



    Game::~Game() {
        imguiLayer.shutdown();
        if (impactNoiseTexId != 0) {
            glDeleteTextures(1, &impactNoiseTexId);
            impactNoiseTexId = 0;
        }
        if (materialAlbedoArrayTexId != 0) {
            glDeleteTextures(1, &materialAlbedoArrayTexId);
            materialAlbedoArrayTexId = 0;
        }
        if (materialHeightArrayTexId != 0) {
            glDeleteTextures(1, &materialHeightArrayTexId);
            materialHeightArrayTexId = 0;
        }
        if (materialAOArrayTexId != 0) {
            glDeleteTextures(1, &materialAOArrayTexId);
            materialAOArrayTexId = 0;
        }
        if (materialNormalArrayTexId != 0) {
            glDeleteTextures(1, &materialNormalArrayTexId);
            materialNormalArrayTexId = 0;
        }
        if (materialRoughArrayTexId != 0) {
            glDeleteTextures(1, &materialRoughArrayTexId);
            materialRoughArrayTexId = 0;
        }
        g_SoundManager.shutdown();
        glfwTerminate();
    }

    void Game::fillMaterialTable()
    {
        using MS = gl3::MaterialSystem;
        constexpr int M = MS::kMaxMaterials;

        // One path per material id (empty string = use fallback layer)
        std::array<std::string, M> albedoPaths{};
        std::array<std::string, M> normalPaths{};
        std::array<std::string, M> roughPaths{};
        std::array<std::string, M> aoPaths{};
        std::array<std::string, M> heightPaths{};

        albedoPaths[0] = gl3::resolveAssetPath("textures/marble_rock_03_diff_4k.jpg").string();
        normalPaths[0] = gl3::resolveAssetPath("textures/marble_rock_03_nor_gl_4k.png").string();
        roughPaths[0]  = gl3::resolveAssetPath("textures/marble_rock_03_rough_4k.png").string();
        heightPaths[0] = gl3::resolveAssetPath("textures/marble_rock_03_disp_4k.png").string();

        albedoPaths[1] = gl3::resolveAssetPath("textures/ground_0035_color_1k.jpg").string();
        aoPaths[1] = gl3::resolveAssetPath("textures/ground_0035_ao_1k.jpg").string();
        normalPaths[1] = gl3::resolveAssetPath("textures/ground_0035_normal_opengl_1k.png").string();
        roughPaths[1] = gl3::resolveAssetPath("textures/ground_0035_roughness_1k.jpg").string();
        heightPaths[1] = gl3::resolveAssetPath("textures/ground_0035_height_1k.png").string();

        albedoPaths[2] = gl3::resolveAssetPath("textures/sand/ground_0024_color_1k.jpg").string();
        normalPaths[2] = gl3::resolveAssetPath("textures/sand/ground_0024_normal_opengl_1k.png").string();
        roughPaths[2]  = gl3::resolveAssetPath("textures/sand/ground_0024_roughness_1k.jpg").string();
        aoPaths[2] = gl3::resolveAssetPath("textures/sand/ground_0024_ao_1k.jpg").string();;
        heightPaths[2] = gl3::resolveAssetPath("textures/sand/ground_0024_height_1k.png").string();

        albedoPaths[3] = gl3::resolveAssetPath("textures/rock_0001_color_1k.jpg").string();
        normalPaths[3] = gl3::resolveAssetPath("textures/rock_0001_normal_opengl_1k.png").string();
        roughPaths[3]  = gl3::resolveAssetPath("textures/rock_0001_roughness_1k.jpg").string();
        //aoPaths[3] = gl3::resolveAssetPath("textures/rock_0001_ao_1k.jpg").string();;
        heightPaths[3] = gl3::resolveAssetPath("textures/rock_0001_height_1k.png").string();

        //albedoPaths[2] = gl3::resolveAssetPath("textures/aerial_rocks_04_diff_4k.jpg").string();
        //normalPaths[2] = gl3::resolveAssetPath("textures/aerial_rocks_04_nor_gl_4k.png").string();
        //roughPaths[2]  = gl3::resolveAssetPath("textures/aerial_rocks_04_rough_4k.jpg").string();
        //heightPaths[2] = gl3::resolveAssetPath("textures/aerial_rocks_04_disp_4k.png").string();

        //albedoPaths[3] = gl3::resolveAssetPath("textures/aerial_rocks_02_diff_4k.jpg").string();
        //normalPaths[3] = gl3::resolveAssetPath("textures/aerial_rocks_02_nor_gl_4k.png").string();
        //roughPaths[3]  = gl3::resolveAssetPath("textures/aerial_rocks_02_rough_4k.jpg").string();
        //heightPaths[3] = gl3::resolveAssetPath("textures/aerial_rocks_02_disp_4k.png").string();

        albedoPaths[4] = gl3::resolveAssetPath("textures/metal/metal_0081_color_1k_edited.jpg").string();
        normalPaths[4] = gl3::resolveAssetPath("textures/metal/metal_0081_normal_opengl_1k.png").string();
        roughPaths[4]  = gl3::resolveAssetPath("textures/metal/metal_0081_roughness_1k.jpg").string();
        aoPaths[4] = gl3::resolveAssetPath("textures/metal/metal_0081_ao_1k.jpg").string();;
        heightPaths[4] = gl3::resolveAssetPath("textures/metal/metal_0081_height_1k.png").string();

        //albedoPaths[4] = gl3::resolveAssetPath("textures/cobble.jpg").string();

        albedoPaths[5] = gl3::resolveAssetPath("textures/ice/others_0002_color_1k.jpg").string();
        normalPaths[5] = gl3::resolveAssetPath("textures/ice/others_0002_normal_opengl_1k.png").string();
        roughPaths[5]  = gl3::resolveAssetPath("textures/ice/others_0002_roughness_1k.jpg").string();
        aoPaths[5] = gl3::resolveAssetPath("textures/ice/others_0002_ao_1k.jpg").string();;
        heightPaths[5] = gl3::resolveAssetPath("textures/ice/others_0002_height_1k.png").string();

        albedoPaths[6] = gl3::resolveAssetPath("textures/crystal/ground_0025_color_1k_edited.jpg").string();
        normalPaths[6] = gl3::resolveAssetPath("textures/crystal/ground_0025_normal_opengl_1k.png").string();
        aoPaths[6] = gl3::resolveAssetPath("textures/crystal/ground_0025_ao_1k.jpg").string();;
        roughPaths[6]  = gl3::resolveAssetPath("textures/crystal/ground_0025_roughness_1k.jpg").string();
        heightPaths[6] = gl3::resolveAssetPath("textures/crystal/ground_0025_height_1k.png").string();

        //albedoPaths[6] = gl3::resolveAssetPath("textures/water.png").string();

        albedoPaths[7] = gl3::resolveAssetPath("textures/flesh/flesh.jpg").string();
        normalPaths[7] = gl3::resolveAssetPath("textures/flesh/flesh_normal_opengl_1k.png").string();
        roughPaths[7]  = gl3::resolveAssetPath("textures/flesh/flesh_roughness_1k.jpg").string();
        aoPaths[7] = gl3::resolveAssetPath("textures/flesh/flesh_ao_1k.jpg").string();;
        heightPaths[7] = gl3::resolveAssetPath("textures/flesh/flesh_height_1k.png").string();

        albedoPaths[8] = gl3::resolveAssetPath("textures/EyeMaterial_new.png").string();

        albedoPaths[9] = gl3::resolveAssetPath("textures/lava.jpg").string();
        normalPaths[9] = gl3::resolveAssetPath("textures/lava_NormalGL.jpg").string();
        roughPaths[9] = gl3::resolveAssetPath("textures/lava_Roughness.jpg").string();
        heightPaths[9] = gl3::resolveAssetPath("textures/lava_Displacement.jpg").string();

        ///Fluid-Materials
        albedoPaths[10] = gl3::resolveAssetPath("textures/marble_rock_03_diff_4k.jpg").string();
        normalPaths[10] = gl3::resolveAssetPath("textures/marble_rock_03_nor_gl_4k.png").string();
        roughPaths[10]  = gl3::resolveAssetPath("textures/marble_rock_03_rough_4k.png").string();
        heightPaths[10] = gl3::resolveAssetPath("textures/marble_rock_03_disp_4k.png").string();

        albedoPaths[11] = gl3::resolveAssetPath("textures/ground_0035_color_1k.jpg").string();
        aoPaths[11] = gl3::resolveAssetPath("textures/ground_0035_ao_1k.jpg").string();
        normalPaths[11] = gl3::resolveAssetPath("textures/ground_0035_normal_opengl_1k.png").string();
        roughPaths[11] = gl3::resolveAssetPath("textures/ground_0035_roughness_1k.jpg").string();
        heightPaths[11] = gl3::resolveAssetPath("textures/ground_0035_height_1k.png").string();

        albedoPaths[12] = gl3::resolveAssetPath("textures/sand/ground_0024_color_1k.jpg").string();
        normalPaths[12] = gl3::resolveAssetPath("textures/sand/ground_0024_normal_opengl_1k.png").string();
        roughPaths[12]  = gl3::resolveAssetPath("textures/sand/ground_0024_roughness_1k.jpg").string();
        aoPaths[12] = gl3::resolveAssetPath("textures/sand/ground_0024_ao_1k.jpg").string();;
        heightPaths[12] = gl3::resolveAssetPath("textures/sand/ground_0024_height_1k.png").string();

        albedoPaths[13] = gl3::resolveAssetPath("textures/rock_0001_color_1k.jpg").string();
        normalPaths[13] = gl3::resolveAssetPath("textures/rock_0001_normal_opengl_1k.png").string();
        roughPaths[13]  = gl3::resolveAssetPath("textures/rock_0001_roughness_1k.jpg").string();
        //aoPaths[13] = gl3::resolveAssetPath("textures/rock_0001_ao_1k.jpg").string();;
        heightPaths[13] = gl3::resolveAssetPath("textures/rock_0001_height_1k.png").string();

        albedoPaths[14] = gl3::resolveAssetPath("textures/metal/metal_0081_color_1k_edited.jpg").string();
        normalPaths[14] = gl3::resolveAssetPath("textures/metal/metal_0081_normal_opengl_1k.png").string();
        roughPaths[14]  = gl3::resolveAssetPath("textures/metal/metal_0081_roughness_1k.jpg").string();
        aoPaths[14] = gl3::resolveAssetPath("textures/metal/metal_0081_ao_1k.jpg").string();;
        heightPaths[14] = gl3::resolveAssetPath("textures/metal/metal_0081_height_1k.png").string();

        //albedoPaths[4] = gl3::resolveAssetPath("textures/cobble.jpg").string();

        albedoPaths[15] = gl3::resolveAssetPath("textures/water/water.png").string();
        normalPaths[15] = gl3::resolveAssetPath("textures/ice/others_0002_normal_opengl_1k.png").string();
        roughPaths[15]  = gl3::resolveAssetPath("textures/ice/others_0002_roughness_1k.jpg").string();
        aoPaths[15] = gl3::resolveAssetPath("textures/ice/others_0002_ao_1k.jpg").string();;
        heightPaths[15] = gl3::resolveAssetPath("textures/ice/others_0002_height_1k.png").string();

        albedoPaths[16] = gl3::resolveAssetPath("textures/crystal/ground_0025_color_1k_edited.jpg").string();
        normalPaths[16] = gl3::resolveAssetPath("textures/crystal/ground_0025_normal_opengl_1k.png").string();
        aoPaths[16] = gl3::resolveAssetPath("textures/crystal/ground_0025_ao_1k.jpg").string();;
        roughPaths[16]  = gl3::resolveAssetPath("textures/crystal/ground_0025_roughness_1k.jpg").string();
        heightPaths[16] = gl3::resolveAssetPath("textures/crystal/ground_0025_height_1k.png").string();

        //albedoPaths[6] = gl3::resolveAssetPath("textures/water.png").string();

        albedoPaths[17] = gl3::resolveAssetPath("textures/blood/blood.png").string();
        normalPaths[17] = gl3::resolveAssetPath("textures/flesh/flesh_normal_opengl_1k.png").string();
        roughPaths[17]  = gl3::resolveAssetPath("textures/flesh/flesh_roughness_1k.jpg").string();
        aoPaths[17] = gl3::resolveAssetPath("textures/flesh/flesh_ao_1k.jpg").string();;
        heightPaths[17] = gl3::resolveAssetPath("textures/flesh/flesh_height_1k.png").string();

        albedoPaths[18] = gl3::resolveAssetPath("textures/EyeMaterial_new.png").string();

        albedoPaths[19] = gl3::resolveAssetPath("textures/lava.jpg").string();
        normalPaths[19] = gl3::resolveAssetPath("textures/lava_NormalGL.jpg").string();
        roughPaths[19] = gl3::resolveAssetPath("textures/lava_Roughness.jpg").string();
        heightPaths[19] = gl3::resolveAssetPath("textures/lava_Displacement.jpg").string();


        // Build all arrays with fallback for empty entries
        materials.initAllTextureArraysFromFiles(
                albedoPaths, normalPaths, roughPaths, aoPaths, heightPaths
        );

        materialAlbedoArrayTexId = materials.albedoArrayTex;
        materialNormalArrayTexId = materials.normalArrayTex;
        materialRoughArrayTexId  = materials.roughArrayTex;
        materialAOArrayTexId     = materials.aoArrayTex;
        materialHeightArrayTexId = materials.heightArrayTex;

        // Your existing scalar params unchanged
        materials.params[0].roughness = 1.0f; materials.params[0].specular = 0.05f; materials.params[0].uvScale = 0.05f;

        materials.params[1].roughness = 1.0f;
        materials.params[1].specular  = 0.05f;
        materials.params[1].uvScale   = 0.05f;

        materials.params[2].roughness = 1.0f;
        materials.params[2].specular  = 0.05f;
        materials.params[2].uvScale   = 0.02f;

        materials.params[3].roughness = 1.0f;
        materials.params[3].specular  = 0.05f;
        materials.params[3].uvScale   = 0.01f;

        materials.params[4].roughness = 0.3f;
        materials.params[4].specular  = 0.5f;
        materials.params[4].uvScale   = 0.015f;

        materials.params[5].roughness = 0.0f;
        materials.params[5].specular  = 1.0f;
        materials.params[5].uvScale   = 0.015f;

        materials.params[6].roughness = 0.1f;
        materials.params[6].specular  = 0.75f;
        materials.params[6].uvScale   = 0.015f;

        materials.params[7].roughness = 0.85f;
        materials.params[7].specular  = 0.2f;
        materials.params[7].uvScale   = 0.125f;

        materials.params[8].roughness = 0.05f;
        materials.params[8].specular  = 0.9f;
        materials.params[8].uvScale   = 3.9f;

        materials.params[9].roughness = 0.85f;
        materials.params[9].specular  = 0.2f;
        materials.params[9].uvScale   = 0.125f;

        materials.params[10].roughness = 1.0f;
        materials.params[10].specular = 0.05f;
        materials.params[10].uvScale = 0.05f;

        materials.params[11].roughness = 1.0f;
        materials.params[11].specular  = 0.05f;
        materials.params[11].uvScale   = 0.05f;

        materials.params[12].roughness = 1.0f;
        materials.params[12].specular  = 0.05f;
        materials.params[12].uvScale   = 0.02f;

        materials.params[13].roughness = 1.0f;
        materials.params[13].specular  = 0.05f;
        materials.params[13].uvScale   = 0.01f;

        materials.params[14].roughness = 0.3f;
        materials.params[14].specular  = 0.5f;
        materials.params[14].uvScale   = 0.015f;

        materials.params[15].roughness = 0.0f;
        materials.params[15].specular  = 1.0f;
        materials.params[15].uvScale   = 0.015f;

        materials.params[16].roughness = 0.1f;
        materials.params[16].specular  = 0.75f;
        materials.params[16].uvScale   = 0.015f;

        materials.params[17].roughness = 0.0f;
        materials.params[17].specular  = 0.2f;
        materials.params[17].uvScale   = 0.0125f;

        materials.params[18].roughness = 0.05f;
        materials.params[18].specular  = 0.9f;
        materials.params[18].uvScale   = 3.9f;

        materials.params[19].roughness = 0.85f;
        materials.params[19].specular  = 0.2f;
        materials.params[19].uvScale   = 0.125f;

        for (int i = 0; i < M; ++i) {
            rough[i]   = materials.params[i].roughness;
            spec[i]    = materials.params[i].specular;
            uvScale[i] = materials.params[i].uvScale;
        }
    }

    void Game::setupSpellContext()
    {
        SpellWorldContext ctx;
        ctx.chunks = chunkManager.get();
        ctx.physics = voxelPhysics.get();
        ctx.worldToChunk = [](float w){ return Game::worldToChunk(w); };
        ctx.getCameraFront = [this](){ return getCameraFront(); };
        ctx.getChunkMin = [this](const ChunkCoord& c){ return getChunkMin(c); };
        ctx.markChunkModified = [this](const ChunkCoord& c){ markChunkModified(c); };
        ctx.sampleNormalAtWorld = [this](const glm::vec3& p){ return sampleNormalAtWorld(p); };
        ctx.sampleDensityAtWorld = [this](const glm::vec3& p){ return sampleDensityAtWorld(p); };
        ctx.generateChunkMesh = [this](Chunk* ch){ chunkRenderer->generateChunkMesh(ch); };
        const uint64_t myEpoch = mainDispatcher.epoch();
        ctx.mainThreadDispatcher = [this, myEpoch](std::function<void()> task) {
            mainDispatcher.dispatch([this, myEpoch, task = std::move(task)]() mutable {
                if (mainDispatcher.epoch() != myEpoch) return;
                task();
            });
        };
        spellSystem = std::make_shared<SpellSystem>(ctx);
    };

    void Game::setupControls() {
        characterController = std::make_unique<CharacterController>(
                chunkManager.get(),
                voxelPhysics.get(),
                5.8f,
                2.5f
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
                    onBodyBodyCollision(bodyA, bodyB, hitPos, hitNormal, impactSpeed);
                }
        );
        // body-world collision callback
        voxelPhysics->setVoxelCollisionCallback(
                [this](gl3::VoxelPhysicsBody* body, const glm::vec3& hitPos,
                       const glm::vec3& hitNormal, float impactSpeed)
                {
                    onSpellCollision(body, hitPos, hitNormal, impactSpeed);
                }
        );

        //BodyBodyPreCollision Callback
        voxelPhysics->setBodyBodyPreSolveCallback(
                [this](gl3::VoxelPhysicsBody* bodyA,
                       gl3::VoxelPhysicsBody* bodyB,
                       const glm::vec3& hitPos,
                       const glm::vec3& hitNormal,
                       float impactSpeed) -> bool
                {
                    CollisionDecision dA = decideCollisionResponse(*bodyA, bodyB, hitPos, hitNormal, impactSpeed);
                    CollisionDecision dB = decideCollisionResponse(*bodyB, bodyA, hitPos, -hitNormal, impactSpeed);

                    // apply stick immediately (pre-solve, before bounce)
                    if (dA.stick) {
                        bodyA->stuck = true;
                        bodyA->stuckToBodyId = bodyB->id;
                        bodyA->velocity = glm::vec3(0.0f);
                        bodyA->angularVelocity = glm::vec3(0.0f);
                        bodyA->stuckOffset = bodyA->position - bodyB->position;
                    }
                    if (dB.stick) {
                        bodyB->stuck = true;
                        bodyB->stuckToBodyId = bodyA->id;
                        bodyB->velocity = glm::vec3(0.0f);
                        bodyB->angularVelocity = glm::vec3(0.0f);
                        bodyB->stuckOffset = bodyB->position - bodyA->position;
                    }

                    // If either side says no physics resolve, skip bounce
                    if (dA.ignoreCollision || dB.ignoreCollision || !dA.resolvePhysics || !dB.resolvePhysics) {
                        return false;
                    }

                    return true;
                }
        );

        voxelPhysics->setBodyStepCallback(
                [this](gl3::VoxelPhysicsBody* body, const glm::vec3& from, const glm::vec3& to)
                {
                    if (!body) return;
                    if (body->material != 9) return;

                    const float burnRadius = glm::max(body->radius * 1.5f, VOXEL_SIZE);
                    applyMaterial9BurnAlongSegment(from, to, burnRadius);
                }
        );

        enemyManager = std::make_unique<EnemyManager>();
        enemyManager->init(voxelPhysics.get(),chunkManager.get(), this);

        waveManager.init(enemyManager.get());
    }

    bool Game::initAudio() {
        if (!g_SoundManager.init()) {
            return false;
        }

        // Load all sounds using SoundIDs
        g_SoundManager.loadMusic(SoundID::MainMenuTheme,
                                 resolveAssetPath("audio/charlvera-eclipse-of-the-cosmos-241713.mp3").string());
        g_SoundManager.loadMusic(SoundID::BackgroundMusic,
                                 resolveAssetPath("audio/charlvera-dancing-among-comets-241708.mp3").string());
        g_SoundManager.loadMusic(SoundID::BossTheme,
                                 resolveAssetPath("audio/charlvera-galactic-sonata-241706.mp3").string());

        g_SoundManager.loadSound(SoundID::Collision,
                                 resolveAssetPath("audio/lordsonny-small-rock-break-194553.mp3").string());
        g_SoundManager.loadSound(SoundID::Fire,
                                 resolveAssetPath("audio/alice_soundz-fire-sound-effects-224089.mp3").string());
        g_SoundManager.loadSound(SoundID::WaterSplash,
                                 resolveAssetPath("audio/universfield-water-splash-199583.mp3").string());
        g_SoundManager.loadSound(SoundID::Suffocate,
                                 resolveAssetPath("audio/kuzu420-running-out-of-oxygen-suffocation-sfx-485955.mp3").string());

        g_SoundManager.loadSound(SoundID::Jump,
                                 resolveAssetPath("audio/dragon-studio-whoosh-cinematic-376875.mp3").string());
        g_SoundManager.loadSound(SoundID::Run,
                                 resolveAssetPath("audio/blendertimer-person-running-loop-245173_2.mp3").string());
        g_SoundManager.loadSound(SoundID::Step,
                                 resolveAssetPath("audio/blendertimer-person-walking-on-gravel-loop-528372_2.mp3").string());
        g_SoundManager.loadSound(SoundID::Velocity,
                                 resolveAssetPath("audio/dragon-studio-whoosh-cinematic-376875_VelocityVersion.mp3").string());
        g_SoundManager.loadSound(SoundID::Land,
                                 resolveAssetPath("audio/universfield-character-fall-impact-352287.mp3").string());
        g_SoundManager.loadSound(SoundID::MeatStick,
                                 resolveAssetPath("audio/universfield-slime-impact-352473.mp3").string());
        g_SoundManager.loadSound(SoundID::Crunch,
                                 resolveAssetPath("audio/saboteurcomics-superfast-crunch-405118.mp3").string());
        g_SoundManager.loadSound(SoundID::PlayerDamage,
                                 resolveAssetPath("audio/dragon-studio-punch-431475.mp3").string());

        g_SoundManager.loadSound(SoundID::ButtonClick,
                                 resolveAssetPath("audio/creatorshome-digital-click-357350_2.mp3").string());
        g_SoundManager.loadSound(SoundID::ButtonHover,
                                 resolveAssetPath("audio/lesiakower-minimalist-button-hover-sound-effect-399749.mp3").string());
        g_SoundManager.loadSound(SoundID::MenuClose,
                                 resolveAssetPath("audio/litupsubway-ui-close-sfx-513359.mp3").string());

        // Set initial volumes
        g_SoundManager.setMusicVolume(0.8f);
        g_SoundManager.setSFXVolume(1.0f);

        return true;
    }

    void Game::updateGameplayFrame() {
        updateDeltaTime();

        spellSystem->update(deltaTime);

        glfwPollEvents();
        update();
        if (enemyManager) {
           enemyManager->update(deltaTime, cameraPos /* player pos */);
        }

        //std::cout<<waveManager.isWaveActive()<<" wave active!\n";
        if (enemyManager && waveManager.isWaveActive()) {
            waveManager.update(deltaTime, cameraPos);
        }

        if (!waveManager.isWaveActive()) {
            bool hasLivingEnemies = false;
            for (auto& e : enemyManager->all()) {
                if (e.inst.hp > 0.0f && !e.inst.pendingRemoval) {
                    hasLivingEnemies = true;
                    break;
                }
            }

            if (!hasLivingEnemies&&waveManager.getRemainingBudget()<=0) {
                waveManager.startNextWave();
            }
        }

        updatePhysics();
        updateImpactEffects(deltaTime);
        updateChunkBurns(deltaTime);

        mainDispatcher.execute();
    }

    void Game::renderGameplayFrame()
    {
        // 1) main scene - render to postFBO
        glBindFramebuffer(GL_FRAMEBUFFER, postFBO);
        glViewport(0, 0, windowWidth, windowHeight);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!DebugMode1) renderSkybox();
        renderChunks();
        renderAnimatedVoxels();
        renderPhysicsFormations();
        renderImpactEffects();
        renderEnemies();
        renderSpellPreview();

        // 2) fluid pass - render to fluidFBO
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);
        glViewport(0, 0, windowWidth, windowHeight);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderFluids();

        // 3) final composite - render to compositeFBO
        glBindFramebuffer(GL_FRAMEBUFFER, compositeFBO);
        glViewport(0, 0, windowWidth, windowHeight);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        postShader->use();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, postColorTex);
        postShader->setInt("uSceneColor", 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, postDepthTex);
        postShader->setInt("uSceneDepth", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, fluidColorTex);
        postShader->setInt("uFluidColor", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, fluidDepthTex);
        postShader->setInt("uFluidDepth", 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, fluidThicknessTex);
        postShader->setInt("uFluidThickness", 4);

        postShader->setFloat("uNear", 0.1f);
        postShader->setFloat("uFar", 500.0f);

        bool inside = isPointInsideFluid(cameraPos);
        postShader->setInt("uCameraInsideFluid", inside ? 1 : 0);
        glm::vec3 currentFluidTint = sampleFluidColorAtWorld(cameraPos);
        postShader->setVec3("uFluidFogColor", currentFluidTint);
        postShader->setFloat("uFluidFogDensity", 0.1f);

        glBindVertexArray(postVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        // 4) Post-processing chain (ping-pong)

        GLuint currentTexture = compositeColorTex;
        int ping = 0;


        if(renderSpeedLines(currentTexture, postProcessFBO[ping]))
        {
            currentTexture = postProcessColor[ping];
            ping ^= 1;
        }


        if(renderDamageFeedback(currentTexture, postProcessFBO[ping]))
        {
            currentTexture = postProcessColor[ping];
            ping ^= 1;
        }

        // 5) Render final result to screen

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        renderTextureToScreen(currentTexture);

        renderGameplayUI();

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
    }

    void gl3::Game::beginGameplayPreload(bool newRun)
    {
        doNewRun = newRun;

        mainDispatcher.bumpEpochAndClear();

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
                initFluidFBO();
                //initGasFBO();
                initCompositeFBO();
                initPostProcessBuffers();
                preloadStage = PreloadStage::Boot_SSBOs;
                return 0.25f;

            case PreloadStage::Boot_SSBOs:
                preloadStageName = "Setting up GPU buffers...";
                //setupSSBOsAndTables();
                //MAX_CHUNKS_GPU = (int)chunkManager->maxChunksGpu();
                //setupChunkBatchBuffers(MAX_CHUNKS_GPU);
                preloadStage = PreloadStage::Boot_Materials;
                return 0.30f;

            case PreloadStage::Boot_Materials:
                preloadStageName = "Loading materials...";
                fillMaterialTable();
                initMaterialRules();
                preloadStage = PreloadStage::Boot_Assets;
                return 0.40f;

            case PreloadStage::Boot_Assets:
                preloadStageName = "Loading assets...";
                initSphereMeshCache();
                preloadStage = PreloadStage::Boot_VEffects;
                return 0.50f;

            case PreloadStage::Boot_VEffects:
                preloadStageName = "Finalizing effects...";
                setupVEffects();
                createImpactNoiseTexture();
                initImpactInstancing();
                bootLoaded = true;

                preloadStage = doNewRun ? PreloadStage::Run_Physics : PreloadStage::Done;
                return 0.60f;

                // ---------------- mutable run ----------------
            case PreloadStage::Run_Physics:
                preloadStageName = "Setting up physics...";
                clearWorldAndGameplayObjects();
                setupPhysics();
                preloadStage = PreloadStage::Run_SpellSystem;
                return 0.65f;

            case PreloadStage::Run_SpellSystem:
                preloadStageName = "Setting up magic System...";
                if (spellSystem) {
                    spellSystem->clear();
                    spellSystem.reset();
                }
                setupSpellContext();
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
                waveManager.resetWaveState();
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
            {
                preloadStageName = "Building meshes & lighting...";
                emissiveBillboards.clear();

                chunkManager->forEachChunk([&](gl3::Chunk* chunk) {
                    if (!chunk) return;
                    if (!chunk->lightingDirty) return;
                    rebuildChunkLights(chunk->coord);   // fills chunk->emissiveLights
                    chunk->lightingDirty = false;
                });
                // 1) Keep merged light pool fresh
                chunkRenderer->updateLightSpatialHash();
                chunkRenderer->uploadMergedLightsToGPU();

                // 2) Drain dirty chunks during loading (this is the important part)
                // Uses your existing budgeted manager path (MAX_CALC_PER_FRAME in manager)
                chunkManager->rebuildDirtyChunks([this](Chunk* chunk) {
                    chunkRenderer->generateChunkMesh(chunk);
                    chunkRenderer->generateFluidMesh(chunk);
                }, cameraPos);

                // 3) Rebuild light-index buffer for current camera neighborhood
                chunkRenderer->buildAndUploadChunkLightIndexBuffer(
                        worldToChunk(cameraPos.x),
                        worldToChunk(cameraPos.y),
                        worldToChunk(cameraPos.z),
                        RenderingRange
                );

                chunkRenderer->collectMergedEmissiveBillboards(emissiveBillboards);

                // 4) Stay in this stage until no dirty chunks remain
                if (chunkManager->hasDirtyChunks()) {
                    return 0.92f; // keep loading screen up
                }

                preloadStage = PreloadStage::Done;
                return 0.99f;
            }

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

            renderWaveModeUI();

            // In ImGui rendering:
            float intensity = waveManager.getWaveIntensity();
            ImVec4 textColor = ImVec4(1.0f, 1.0f - intensity, 1.0f - intensity, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_Text, textColor);
            ImGui::Text("Wave: %d", waveManager.getCurrentWave());
         //   ImGui::Text("Enemies: %d", waveManager.getEnemiesRemaining());
            ImGui::PopStyleColor();

            if (waveManager.isBossActive()) {
                float bossHP = waveManager.getBossHealthPercent();
                ImGui::ProgressBar(bossHP, ImVec2(-1, 0), "BOSS");
            };

            ImGui::PushStyleColor(ImGuiCol_Text, textColor);
           //ImGui::Text("Enemies spawned: %d", waveManager.getSpawnedEnemies());
            //ImGui::Text("Enemies alive : %d", enemyManager->getEnemiesAlive());
           /* for (int i = 0; i < enemyManager->getEnemiesAlive(); ++i) {
                glm::vec3 diff = (enemyManager->getEnemyPos(i)-cameraPos);
                float dist = glm::sqrt(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
              //  ImGui::Text("Enemy: %d", i+1);
                ImGui::Text("distance : %f", dist);
                //ImGui::Text("Mesh valid : %d", enemyManager->getEnemyRenderingMesh(i));
                ImGui::Text("HP : %f", enemyManager->getEnemyHP(i));
                //ImGui::Text("Material : %f", enemyManager->getEnemyMat(i));

            }*/
            ImGui::PopStyleColor();

            ImGui::End();
        }

        // ----------------
        // (B) Pause menu
        // ----------------
        if (paused&&waveManager.getCurrentWaveMode()!=WaveMode::UpgradeSelection)
        {
            ImGuiIO& io = ImGui::GetIO();
            const ImVec2 center(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);

            ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(520, 420), ImGuiCond_Always);

            ImGuiWindowFlags flags =
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoTitleBar;

            ImGui::SetNextWindowBgAlpha(0.80f);

            if (ImGui::Begin("PauseMenu", nullptr, flags))
            {
                const ImVec2 btnSize(520.0f * 0.75f, 42.0f);

                if (pauseSubmenu == PauseSubmenu::Main)
                {
                    ImGui::SetWindowFontScale(1.8f);
                    ImGui::SetCursorPosY(14.0f);
                    ImGui::SetCursorPosX((520.0f - ImGui::CalcTextSize("Paused").x) * 0.5f);
                    ImGui::TextUnformatted("Paused");
                    ImGui::SetWindowFontScale(1.0f);

                    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

                    ImGui::SetCursorPosX((520.0f - btnSize.x) * 0.5f);
                    if (ImGui::Button("Resume", btnSize))
                    {
                        g_SoundManager.playSound(SoundID::MenuClose, 1.0f, 1.0f);
                        setPaused(false);
                    }
                    const char* resumeBtnId = "menu_resume";
                    if (ImGui::IsItemHovered() && lastHoveredButton != resumeBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = resumeBtnId;
                    }

                    ImGui::Spacing();
                    ImGui::SetCursorPosX((520.0f - btnSize.x) * 0.5f);
                    if (ImGui::Button("Settings", btnSize))
                    {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        pauseSubmenu = PauseSubmenu::Settings;
                    }
                    const char* settingsBtnId = "menu_settings";
                    if (ImGui::IsItemHovered() && lastHoveredButton != settingsBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = settingsBtnId;
                    }

                    ImGui::Spacing();
                    ImGui::SetCursorPosX((520.0f - btnSize.x) * 0.5f);
                    if (ImGui::Button("Skill Tree", btnSize))
                    {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        pauseSubmenu = PauseSubmenu::SkillTree;
                    }
                    const char* skillTreeBtnId = "menu_skills";
                    if (ImGui::IsItemHovered() && lastHoveredButton != settingsBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = settingsBtnId;
                    }

                    ImGui::Spacing();
                    ImGui::SetCursorPosX((520.0f - btnSize.x) * 0.5f);
                    if (ImGui::Button("Back to Main Menu", btnSize)) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        setPaused(false);
                        requestSceneChange(SceneId::MainMenu);
                        SoLoud::handle musicHandle = g_SoundManager.playMusic(SoundID::MainMenuTheme, true, 1.0f);
                    }
                    const char* mainMenuBtnId = "menu_mainMenu";
                    if (ImGui::IsItemHovered() && lastHoveredButton != mainMenuBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = mainMenuBtnId;
                    }

                    ImGui::Spacing();
                    ImGui::SetCursorPosX((520.0f - btnSize.x) * 0.5f);
                    if (ImGui::Button("Exit to Desktop", btnSize)) {
                        g_SoundManager.playSound(SoundID::MenuClose, 1.0f, 1.0f);
                        glfwSetWindowShouldClose(getWindow(), true);
                    }
                    const char* escapeBtnId = "menu_escape";
                    if (ImGui::IsItemHovered() && lastHoveredButton != escapeBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = escapeBtnId;
                    }
                }
                else if(pauseSubmenu == PauseSubmenu::SkillTree)
                {
                    skillTree.Draw();
                }
                else // Settings submenu
                {
                    ImGui::SetWindowFontScale(1.4f);
                    ImGui::TextUnformatted("Settings");
                    ImGui::SetWindowFontScale(1.0f);
                    ImGui::Separator();

                    bool changed = false;

                    changed |= ImGui::SliderFloat("Mouse Sensitivity", &settings.sensitivity, 0.01f, 1.0f, "%.3f");
                    const char* sensitivityBtnId = "settings_sensitivity";
                    if (ImGui::IsItemHovered() && lastHoveredButton != sensitivityBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = sensitivityBtnId;
                    }

                    changed |= ImGui::SliderFloat("Master Volume", &settings.masterVolume, 0.0f, 1.0f, "%.2f");
                    const char* masterBtnId = "settings_master";
                    if (ImGui::IsItemHovered() && lastHoveredButton != masterBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = masterBtnId;
                    }

                    changed |= ImGui::SliderFloat("SFX Volume", &settings.sfxVolume, 0.0f, 1.0f, "%.2f");
                    const char* sfxBtnId = "settings_sfx";
                    if (ImGui::IsItemHovered() && lastHoveredButton != sfxBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = sfxBtnId;
                    }

                    changed |= ImGui::SliderFloat("Music Volume", &settings.musicVolume, 0.0f, 1.0f, "%.2f");
                    const char* musicBtnId = "settings_music";
                    if (ImGui::IsItemHovered() && lastHoveredButton != musicBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = musicBtnId;
                    }

                    changed |= ImGui::SliderFloat("Gamma", &settings.gamma, 0.5f, 2.5f, "%.2f");
                    const char* gammaBtnId = "settings_gamma";
                    if (ImGui::IsItemHovered() && lastHoveredButton != gammaBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = gammaBtnId;
                    }

                    changed |= ImGui::SliderFloat("Brightness", &settings.brightness, 0.1f, 2.0f, "%.2f");
                    const char* brightnessBtnId = "settings_brightness";
                    if (ImGui::IsItemHovered() && lastHoveredButton != brightnessBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = brightnessBtnId;
                    }

                    changed |= ImGui::SliderFloat("Field of View", &settings.fov, 1.0f, 2.0f, "%.2f");
                    const char* fovBtnId = "settings_fov";
                    if (ImGui::IsItemHovered() && lastHoveredButton != fovBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = fovBtnId;
                    }

                    const char* modeLabels[] = {"Fullscreen", "Windowed", "Borderless"};
                    int mode = static_cast<int>(settings.displayMode);
                    if (ImGui::Combo("Display Mode", &mode, modeLabels, IM_ARRAYSIZE(modeLabels))) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        settings.displayMode = static_cast<DisplayMode>(mode);
                        changed = true;
                    }
                    const char* displayModeBtnId = "settings_displayMode";
                    if (ImGui::IsItemHovered() && lastHoveredButton != displayModeBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = displayModeBtnId;
                    }

                    std::vector<const char*> resLabels;
                    resLabels.reserve(commonResolutions.size());
                    for (auto& r : commonResolutions) resLabels.push_back(r.label);

                    if (ImGui::Combo("Resolution", &settings.resolutionIndex, resLabels.data(), (int)resLabels.size())) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        changed = true;
                    }
                    const char* pickResolutionBtnId = "settings_pickResolution";
                    if (ImGui::IsItemHovered() && lastHoveredButton != pickResolutionBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = pickResolutionBtnId;
                    }
                    if (ImGui::Button("Set native display resolution.", ImVec2(180, 38))) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        pickResolutionFromNativeMonitor(false);
                    }
                    const char* setNativeBtnId = "settings_setNative";
                    if (ImGui::IsItemHovered() && lastHoveredButton != setNativeBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = setNativeBtnId;
                    }

                    ImGui::Spacing();
                    if (changed) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        applyAudioSettings();
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Apply Display", ImVec2(180, 38))) {
                        g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f, true, true,4);
                        applyDisplaySettings();
                    }
                    const char* applyBtnId = "settings_apply";
                    if (ImGui::IsItemHovered() && lastHoveredButton != applyBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = applyBtnId;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Back", ImVec2(140, 38))) {
                        g_SoundManager.playSound(SoundID::MenuClose, 1.0f, 1.0f);
                        pauseSubmenu = PauseSubmenu::Main;
                    }
                    const char* backBtnId = "settings_back";
                    if (ImGui::IsItemHovered() && lastHoveredButton != backBtnId) {
                        g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f, true, true, 4);
                        lastHoveredButton = backBtnId;
                    }
                }

                ImGui::End();
            }
        }

        imguiLayer.endFrame();
    }

    void Game::renderWaveModeUI()
    {
        switch (waveManager.getCurrentWaveMode())
        {
            case WaveMode::Preperation:
                renderPreparationUI();
                break;

            case WaveMode::Hunt:
                renderHuntUI();
                break;

            case WaveMode::Survival:
                renderSurvivalUI();
                break;

            case WaveMode::Mining:
                renderMiningUI();
                break;

            case WaveMode::Defense:
                renderDefenseUI();
                break;

            case WaveMode::Destruction:
                renderDestructionUI();
                break;

            case WaveMode::Infection:
                renderInfectionUI();
                break;

            case WaveMode::UpgradeSelection:
                renderUpgradeSelectionUI();
                break;
        }
    }

    void Game::renderCenteredTopText(const std::string& text)
    {
        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 20.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_CenterText", nullptr, flags))
        {
            ImGui::SetWindowFontScale(5.0f);
            ImGui::TextUnformatted(text.c_str());
            ImGui::SetWindowFontScale(1.0f);
        }

        ImGui::End();
    }

    void Game::renderPreparationUI()
    {
        renderCenteredTopText(
                "Next up: " +
                std::string(waveManager.getNextWaveModeString())
        );

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );
        if (ImGui::Begin("HUD_Description", nullptr, flags))
        {
            ImGui::SetWindowFontScale(3.0f);
            ImGui::TextUnformatted(waveManager.getNextWaveModeDescription());
            ImGui::SetWindowFontScale(1.0f);
        }
        ImGui::End();

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 180.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );
        if (ImGui::Begin("HUD_Timer", nullptr, flags))
        {
            renderProgressBar(
                    "Time remaining",
                    1-waveManager.getRemainingTimerPercent(),
                    nullptr
            );
        }
        ImGui::End();
    }

    void Game::renderHuntUI()
    {
        renderCenteredTopText("Hunt");

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Hunt", nullptr, flags))
        {
            ImGui::SetWindowFontScale(3.0f);

            ImGui::Text(
                    "Enemies remaining: %d",
                    waveManager.getEnemiesRemaining()
            );
            ImGui::SetWindowFontScale(1.0f);

        }

        ImGui::End();
    }

    void Game::renderProgressBar(
            const char* label,
            float progress,
            const char* overlay)
    {
        ImGui::SetWindowFontScale(3.0f);
        ImGui::TextUnformatted(label);
        ImGui::SetWindowFontScale(1.0f);

        ImGui::ProgressBar(
                glm::clamp(progress, 0.0f, 1.0f),
                ImVec2(400.0f, 30.0f),
                overlay
        );
    }

    void Game::renderSurvivalUI()
    {
        renderCenteredTopText("Survival");

        float progress = 1.0f-waveManager.getRemainingTimerPercent();

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Survival", nullptr, flags))
        {
            renderProgressBar(
                    "Time remaining",
                    progress,
                    nullptr
            );
        }

        ImGui::End();
    }

    void Game::renderMiningUI()
    {
        renderCenteredTopText(
                std::string("Mining material: ") +
                materialToString(waveManager.materialToMine, 1)
        );

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Mining", nullptr, flags))
        {
            float progress = ((float)waveManager.materialMined/(float)waveManager.materialNeeded);

            ImGui::ProgressBar(
                    progress,
                    ImVec2(400.0f, 30.0f)
            );

            ImGui::SameLine();

            ImGui::Text(
                    "%d%%",
                    static_cast<int>(progress * 100.0f)
            );
        }

        ImGui::End();
    }

    void Game::renderDefenseUI()
    {
        renderCenteredTopText("Defense");

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Defense", nullptr, flags))
        {
            //float hp = waveManager.getDefenseHealthPercent();
            float hp = 70.0f;
            float time = 1.0f-waveManager.getRemainingTimerPercent();

            ImGui::TextUnformatted("Base Health");

            ImGui::ProgressBar(
                    hp,
                    ImVec2(400.0f, 30.0f),
                    "HP"
            );

            ImGui::Spacing();

            ImGui::TextUnformatted("Time Remaining");

            ImGui::ProgressBar(
                    time,
                    ImVec2(400.0f, 30.0f)
            );
        }

        ImGui::End();
    }

    void Game::renderDestructionUI()
    {
        renderCenteredTopText("Destruction");

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Defense", nullptr, flags))
        {
            //float hp = waveManager.getDefenseHealthPercent();
            float hp = 70.0f;
            float time = 1.0f-waveManager.getRemainingTimerPercent();

            ImGui::TextUnformatted("Base Health");

            ImGui::ProgressBar(
                    hp,
                    ImVec2(400.0f, 30.0f),
                    "HP"
            );

            ImGui::Spacing();

            ImGui::TextUnformatted("Time Remaining");

            ImGui::ProgressBar(
                    time,
                    ImVec2(400.0f, 30.0f)
            );
        }

        ImGui::End();
    }

    void Game::renderInfectionUI()
    {
        renderCenteredTopText("Infection");

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(displaySize.x * 0.5f, 100.0f),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.0f)
        );

        if (ImGui::Begin("HUD_Infection", nullptr, flags))
        {
            //float progress = waveManager.getInfectionProgress();
            float progress = 0.7f;


            ImVec4 color(
                    1.0f,
                    1.0f - progress,
                    1.0f - progress,
                    1.0f
            );

            ImGui::PushStyleColor(
                    ImGuiCol_PlotHistogram,
                    color
            );

            ImGui::TextUnformatted("Infection Progress:");

            ImGui::ProgressBar(
                    progress,
                    ImVec2(400.0f, 30.0f),
                    nullptr
            );

            ImGui::PopStyleColor();

            ImGui::TextUnformatted("Time Remaining:");
            float time = 1.0f-waveManager.getRemainingTimerPercent();

            ImGui::ProgressBar(
                    time,
                    ImVec2(400.0f, 30.0f)
            );
        }

        ImGui::End();
    }

    void Game::renderUpgradeSelectionUI()
    {
        setPaused(true);

        ImVec2 displaySize = ImGui::GetIO().DisplaySize;

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoBackground;

        ImGui::SetNextWindowPos(
                ImVec2(
                        displaySize.x * 0.5f,
                        displaySize.y * 0.5f
                ),
                ImGuiCond_Always,
                ImVec2(0.5f, 0.5f)
        );

        if (ImGui::Begin("HUD_UpgradeSelection", nullptr, flags))
        {
            ImGui::SetWindowFontScale(2.0f);

            const ImVec2 buttonSize(250.0f, 150.0f);

            if (ImGui::Button("Heal 25% Health", buttonSize))
            {
                // Select upgrade A
                setPlayerHealth(glm::clamp(getPlayerHealth()+getPlayerMaxHealth()*0.25f, getPlayerHealth(), getPlayerMaxHealth()));
                waveManager.setCompletion(true);
                setPaused(false);
            }

            ImGui::SameLine();

            if (ImGui::Button("Gain Max Health", buttonSize))
            {
                // Select upgrade B
                setPlayerMaxHealth(getPlayerMaxHealth()*2);
                waveManager.setCompletion(true);
                togglePaused();
                setPaused(false);

            }

            ImGui::SameLine();

            const char* skillPageNames[] =
                    {
                            "Fire",
                            "Water",
                            "Lightning",
                            "Crystal",
                            "Stone",
                            "Blood"
                    };

            ImGui::BeginGroup();

            const float comboWidth = buttonSize.x;
            const float comboHeight = 35.0f;

            ImGui::SetNextItemWidth(comboWidth);

            if (ImGui::BeginCombo(
                    "##SkillPage",
                    skillPageNames[selectedSkillPage]))
            {
                for (int i = 0; i < 6; ++i)
                {
                    bool selected = selectedSkillPage == i;

                    if (ImGui::Selectable(
                            skillPageNames[i],
                            selected))
                    {
                        selectedSkillPage = i;
                    }

                    if (selected)
                        ImGui::SetItemDefaultFocus();
                }

                ImGui::EndCombo();
            }

            ImGui::SetCursorPosY(
                    ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y
            );

            if (ImGui::Button(
                    "Gain a Level",
                    ImVec2(buttonSize.x, buttonSize.y - comboHeight)))
            {
                skillTree.GiveLevel(selectedSkillPage);

                waveManager.setCompletion(true);
                togglePaused();
                setPaused(false);
            }

            ImGui::EndGroup();
        }

        ImGui::End();
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

    void Game::initPostProcessBuffers()
    {
        // cleanup
        glDeleteFramebuffers(2, postProcessFBO);
        glDeleteTextures(2, postProcessColor);

        glGenFramebuffers(2, postProcessFBO);
        glGenTextures(2, postProcessColor);

        for (int i = 0; i < 2; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO[i]);

            glBindTexture(GL_TEXTURE_2D, postProcessColor[i]);

            glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA16F,
                    windowWidth,
                    windowHeight,
                    0,
                    GL_RGBA,
                    GL_FLOAT,
                    nullptr);

            glTexParameteri(GL_TEXTURE_2D,
                            GL_TEXTURE_MIN_FILTER,
                            GL_LINEAR);

            glTexParameteri(GL_TEXTURE_2D,
                            GL_TEXTURE_MAG_FILTER,
                            GL_LINEAR);

            glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    postProcessColor[i],
                    0);

            GLenum drawBuffers[] =
                    {
                            GL_COLOR_ATTACHMENT0
                    };

            glDrawBuffers(1, drawBuffers);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER)
                != GL_FRAMEBUFFER_COMPLETE)
            {
                std::cerr
                        << "Post process framebuffer "
                        << i
                        << " incomplete\n";
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Game::beginPostProcess(GLuint destinationFBO)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, destinationFBO);

        glViewport(0, 0, windowWidth, windowHeight);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glClear(GL_COLOR_BUFFER_BIT);
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
            if (!chunk->isCleared)
            {
                chunk->meshDirty = true;
                chunkManager->markChunkDirty(coord);
            }

            for (int dx=-1; dx<=1; ++dx)
                for (int dy=-1; dy<=1; ++dy)
                    for (int dz=-1; dz<=1; ++dz) {
                        ChunkCoord neighbor{coord.x + dx, coord.y + dy, coord.z + dz};
                        Chunk *neighborChunk = chunkManager->getChunk(neighbor);
                        if (neighborChunk) {
                            if (!neighborChunk->isCleared) {
                                neighborChunk->meshDirty = true;
                                neighborChunk->lightingDirty = true;
                                chunkManager->markChunkDirty(neighborChunk->coord);
                            }
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

    int Game::estimateAvailableVoxels(const glm::vec3& center, float radius,std::vector<uint32_t> excludedMaterials, uint32_t targetType, int maxNeeded)
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
                        if (v.type != targetType) continue;
                        bool excluded = false;
                        for(auto& material : excludedMaterials)
                        {
                            if(v.material==material) {
                                excluded = true;
                                break;
                            }
                        }
                        if (excluded) continue;

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

    //optimized version if we can remove voxels from previous iterations?
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


    void Game::onSpellCollision(gl3::VoxelPhysicsBody* body,
                                const glm::vec3& hitPos,
                                const glm::vec3& hitNormal,
                                float impactSpeed)
    {
        if (!body) return;

        const CollisionDecision decision = decideCollisionResponse(*body, nullptr, hitPos, hitNormal, impactSpeed);
        const auto& rule = materialRules[body->material];

        if (decision.ignoreCollision) return;
        EnemyRuntime* enemy = nullptr;
        for (auto& e : enemyManager->all())
        {
            if (body == e.inst.body)
            {
                enemy = &e;
                break;
            }
        }

        if (enemy && std::strcmp(enemy->inst.type.name, "Consumer") == 0)
        {
            int amount = consumeWorldOfMaterial(body->position, body->radius * 1.5f, 7);
            enemy->inst.hp += amount;
            std::cout << "amount is: " << amount << "\n";
        }

        if (decision.stick&&((body&&enemy&&body!=enemy->inst.body)||!enemy)) {
            body->stuck = true;
            body->angularVelocity = glm::vec3(0.0f);
            body->position = hitPos + hitNormal * body->radius;
            g_SoundManager.playSound(SoundID::MeatStick);
            //body->stuckOffset=glm::normalize(hitPos);
            if (decision.convertWorld) {
                float r = (rule.convertRadius > 0.0f) ? rule.convertRadius*glm::sqrt(body->radius) : (body->radius * 2.5f);
               convertSolidWorldToMaterial(hitPos, r, body->material);
            }
            return;
        }
        if (decision.convertWorld) {
            float r = (rule.convertRadius > 0.0f) ? rule.convertRadius*glm::sqrt(body->radius) : (body->radius * 2.0f);
            std::cout<<"radius? "<<r<<"\n";
            convertSolidWorldToMaterial(hitPos, r, body->material);
        }

        /*if (decision.destroyOther) {
            convertSolidWorldToType(hitPos, (body->radius * 1.5f), 0);
            return;
        }*/
        SoLoud::handle h = g_SoundManager.playSound3D(
                SoundID::Collision,
                hitPos,
                glm::sqrt(impactSpeed)/100,  // volume
                1.0f   // pitch
        );

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        uint64_t spellId = (uint64_t)(uintptr_t)body->userData;
        if (auto* spell = spellSystem->findSpellById(spellId)) {
            float mass = spell->physicsBody ? spell->physicsBody->mass : 1.0f;

            float craterStrength = glm::sqrt((impactSpeed * glm::pow(mass,1)*spell->physicsBody->radius)) / 30.0f;
            float spellRadius = glm::max(spell->physicsBody->radius, 0.001f);
            createCraterAtPosition(hitPos, craterStrength, spellRadius);

            // Estimate removed voxels from crater strength
            float removedVoxelEstimate = craterStrength * 40.0f;

            /*glm::vec3 tint = spell->formationColor;
            if (glm::length(tint) < 0.001f) {
                tint = glm::vec3(0.45f, 0.45f, 0.45f);
            }*/
            glm::vec3 tint = glm::vec3(0.45f, 0.45f, 0.45f);
            glm::vec3 variance = glm::vec3(dist(rng));
            tint+=(spell->formationColor/glm::vec3(10));
            tint+=variance;

            spawnImpactEffect(hitPos, hitNormal, impactSpeed, removedVoxelEstimate, tint);
        }
    }

    void Game::createCraterAtPosition(const glm::vec3& worldPos, float impactFactor, float spellRadius) {
        TRACY_CPU_ZONE("CraterAtPos");
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

    void Game::initImpactInstancing()
    {
        glBindVertexArray(impactQuadVAO);

        glGenBuffers(1, &impactInstanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, impactInstanceVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     kMaxImpactInstances * sizeof(ImpactInstanceGPU),
                     nullptr,
                     GL_STREAM_DRAW);

        // layout(location = 2) vec4 iPosSize;
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(ImpactInstanceGPU), (void*)offsetof(ImpactInstanceGPU, pos_size));
        glVertexAttribDivisor(2, 1);

        // layout(location = 3) vec4 iColor;
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(ImpactInstanceGPU), (void*)offsetof(ImpactInstanceGPU, color));
        glVertexAttribDivisor(3, 1);

        // layout(location = 4) vec4 iRotLifeKind;
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(ImpactInstanceGPU), (void*)offsetof(ImpactInstanceGPU, rot_life_kind));
        glVertexAttribDivisor(4, 1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void Game::createImpactNoiseTexture()
    {
        if (impactNoiseTexId != 0) return;

        const int W = 128;
        const int H = 128;

        std::vector<unsigned char> pixels(W * H);

        auto hash = [](uint32_t x) -> uint32_t {
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        };

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                uint32_t h = hash((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ 0xA53C49E5u);
                unsigned char n = (unsigned char)(h & 0xFF);
                pixels[y * W + x] = n;
            }
        }

        glGenTextures(1, &impactNoiseTexId);
        glBindTexture(GL_TEXTURE_2D, impactNoiseTexId);

        // R8 single-channel noise
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0, GL_RED, GL_UNSIGNED_BYTE, pixels.data());

        // Tileable behavior for scrolling UVs
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // Linear is fine for smoke
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void Game::initMaterialRules() {
        materialRules.resize(64);

        // default
        for (auto& r : materialRules) r = {};

        // examples
        materialRules[0].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[1].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[2].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[3].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[4].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[5].mode = MaterialCollisionMode::DefaultBounce;
        materialRules[6].mode = MaterialCollisionMode::DefaultBounce;


        /*materialRules[3].mode = MaterialCollisionMode::PassThroughFirstBody;
        materialRules[3].passThroughBodiesLeft = 1;

        materialRules[4].mode = MaterialCollisionMode::CollidePlayerOnly;
        materialRules[4].collideWorld = false;*/

        materialRules[7].mode = MaterialCollisionMode::StickOnWorld;
        materialRules[7].convertOnStick = true;
        materialRules[7].convertRadius = 5.0f;


        materialRules[9].mode = MaterialCollisionMode::DestroyTargetKeepFlying;
        materialRules[9].collideWorld = true;
        materialRules[9].collideBodies = true;

        materialRules[9].destroyWorldOnImpact = true;
        materialRules[9].destroyWorldSolidOnly = true;
        materialRules[9].destroyWorldToType = 0;
        materialRules[9].destroyWorldRadius = 1.5f;
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

    void Game::setupInput() {
        // Track all keys we'll use
        input.trackKeys({
                                GLFW_KEY_W, GLFW_KEY_UP, GLFW_KEY_S, GLFW_KEY_DOWN,
                                GLFW_KEY_A, GLFW_KEY_LEFT, GLFW_KEY_D, GLFW_KEY_RIGHT,
                                GLFW_KEY_SPACE, GLFW_KEY_LEFT_SHIFT, GLFW_KEY_LEFT_CONTROL,
                                GLFW_KEY_TAB, GLFW_KEY_ESCAPE,GLFW_KEY_E,GLFW_KEY_R,GLFW_KEY_F,
                                GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4, GLFW_KEY_5, GLFW_KEY_6,
                                GLFW_KEY_T, GLFW_KEY_Q, GLFW_KEY_I,GLFW_KEY_X,GLFW_KEY_C
                        });

        // Character movement
        actions.addAction("MoveForward", {GLFW_KEY_W, GLFW_KEY_UP});
        actions.addAction("MoveBack", {GLFW_KEY_S, GLFW_KEY_DOWN});
        actions.addAction("MoveLeft", {GLFW_KEY_A, GLFW_KEY_LEFT});
        actions.addAction("MoveRight", {GLFW_KEY_D, GLFW_KEY_RIGHT});
        actions.addAction("Jump", {GLFW_KEY_SPACE});
        actions.addAction("Sprint", {GLFW_KEY_LEFT_SHIFT});
        actions.addAction("Crouch", {GLFW_KEY_LEFT_CONTROL});
        actions.addAction("RollRight", {GLFW_KEY_E});
        actions.addAction("RollLeft", {GLFW_KEY_Q});

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
        actions.addAction("CastSphere", {GLFW_KEY_F});
        actions.addAction("CastFleshSphere", {GLFW_KEY_X});
        actions.addAction("CastFireSphere", {GLFW_KEY_I});


        actions.addAction("CastWall", {GLFW_KEY_R});
        actions.addAction("AirReset", {GLFW_KEY_C});

    }

    void Game::setupCamera() {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);
        cameraUp      = glm::vec3(0.0f, 1.0f, 0.0f);
        cameraRight   = glm::normalize(glm::cross(cameraForward, cameraUp));
    }

    static inline int decideMaterial(int random)
    {
        //0=Stone, 1=Earth, 2 = Sand, 3 = Shatterstone, 4 = Ice, 5 = Metal, 6= Crystal,7 = Flesh, 8 = Eye, 9 = Magma
        if(random<15)
        {
            return 0;
        } else if(random<27.5)
        {
          return 6;
        }
        else if(random<40)
        {
            return 4;
        }
        else if(random<55)
        {
            return 1;
        }
        else if(random<70)
        {
            return 2;
        }
        else if(random<80)
        {
            return 3;
        }
        else if(random<90)
        {
            return 5;
        }
        else
        {
            return 7;
        }
    }

    static inline float decideMassFromMaterial(int mat)
    {

        switch (mat) {
            case 0:
                return 1.0;
                break;
            case 1:
                return 1.5;
                break;
            case 2:
                return 2.0;
                break;
            case 3:
                return 2.5;
                break;
            case 4:
                return 3.0;
                break;
            case 5:
                return 0.5;
                break;
            case 6:
                return 4.0;
                break;
            default:
                return 5.0;
                break;
        }
    }



    void Game:: generateChunks() {
        TRACY_CPU_ZONE("Game::generateChunks");
        int FilledChunks = 0;
        std::mt19937 rng(std::random_device{}());

        float chunkWorld = CHUNK_SIZE * VOXEL_SIZE;
        float worldMax = chunkManager->radius() * chunkWorld;

        std::uniform_real_distribution<float> distPos(-worldMax * 0.9f, worldMax * 0.9f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);
        std::uniform_real_distribution<float> distMat(0.0f, 100.9f);

        std::uniform_real_distribution<float> waterDistColorR(0.0f, 0.1f);
        std::uniform_real_distribution<float> waterDistColorG(0.4f, 0.9f);
        std::uniform_real_distribution<float> waterDistColorB(0.5f, 1.0f);

        std::vector<WorldPlanet> worldPlanets;

        //clear old data:
        chunkManager->clearAll();


        // Create solid planets (type 1)
        int planetCount = 24;

        int testMat = -1;

        WorldPlanet p;
        p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
        p.radius = distScale(rng) * CHUNK_SIZE;
        p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
        p.type = 1; // solid
        p.material= 0;
        if(testMat>=0)
        {
            p.material= testMat;
        }
        if(p.material==7)
        {
            p.color= glm::vec3(1.0, 0.0, 1.0);
        }
        else if(p.material==8)
        {
            p.color= glm::vec3(1.0, 0.6, 0.0);
        }
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
            p.material=decideMaterial((int)distMat(rng));
            if(testMat>=0)
            {
                p.material= testMat;
            }
            if(p.material==4)
            {
                p.color = glm::vec3(waterDistColorR(rng), waterDistColorG(rng), waterDistColorB(rng));
            }
            //p.material=0;
            worldPlanets.push_back(p);
        }

        // Create water planets (type 3)

        int waterCount = 12 ;
        for (int i = 0; i < waterCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(distColor(rng), distColor(rng), distColor(rng));
            //p.color = glm::vec3(waterDistColorR(rng), waterDistColorG(rng), waterDistColorB(rng));
            p.type = 3;
            p.material=decideMaterial((int)distMat(rng));
            if(testMat>=0)
            {
                p.material= testMat;
            }

            if(p.material==4)
            {
                p.color = glm::vec3(waterDistColorR(rng), waterDistColorG(rng), waterDistColorB(rng));
            }
            if(p.material==7)
            {
                p.color= glm::vec3(1.0, 0.0, 0.0);
            }
            if(p.material==6)
            {
                p.color= glm::vec3(1.0, 0.0, 1.0);
            }
            if(p.material==5)
            {
                p.color= glm::vec3(0.7, 0.7, 0.7);
            }
            worldPlanets.push_back(p);
        }

        // Create suns (type 2 - fire)
        std::uniform_real_distribution<float> lavaDistColorR(0.7f, 1.0f);
        std::uniform_real_distribution<float> lavaDistColorG(0.3f, 0.6f);
        std::uniform_real_distribution<float> lavaDistColorB(0.0f, 0.1f);

        int lavaCount = 3 + (rng() % 3);
        for (int i = 0; i < lavaCount; ++i) {
            WorldPlanet p;
            p.worldPos = glm::vec3(distPos(rng), distPos(rng), distPos(rng));
            p.radius = distScale(rng) * CHUNK_SIZE;
            p.color = glm::vec3(lavaDistColorR(rng), lavaDistColorG(rng), lavaDistColorB(rng));
            p.type = 2; // fire
            p.material = 9; // fire

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

                        for (int lx = 0; lx <= CHUNK_SIZE; ++lx) {
                            for (int ly = 0; ly <= CHUNK_SIZE; ++ly) {
                                for (int lz = 0; lz <= CHUNK_SIZE; ++lz) {
                                    glm::vec3 worldPos = chunkOrigin + glm::vec3(lx, ly, lz) * VOXEL_SIZE;
                                    float dist = glm::distance(worldPos, planet.worldPos);

                                    // Calculate this planet's SDF value
                                    float planetDensity = planet.radius - dist; // Positive inside, negative outside

                                    Voxel& vox = chunk->voxels[lx][ly][lz];

                                    if (planet.type == 3) {
                                        if (planetDensity > vox.fluidDensity) {
                                            vox.fluidDensity = planetDensity;
                                            vox.color = planet.color;

                                            if (planetDensity >= 0.0f) {
                                                chunk->hasFluid=true;
                                                vox.material = planet.material;
                                                chunkTouched = true;
                                            }
                                        }
                                    } else {
                                        float existingDensity = vox.density;
                                        if (planetDensity > existingDensity) {
                                            vox.density = planetDensity;
                                            vox.color = planet.color;

                                            if (planetDensity >= -1.0f) {
                                                vox.type = planet.type;
                                                vox.material = planet.material;
                                                vox.fluidDensity = -1.0f;

                                                if (planetDensity >= 0) {
                                                    solidVoxels++;
                                                    chunkTouched = true;
                                                }
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
                            chunkManager->markChunkDirty(coord);

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

    void Game::setupImpactEffects()
    {
        impactShader = std::make_unique<Shader>(
                "shaders/impact_billboard.vert",
                "shaders/impact_billboard.frag"
        );

        ensureImpactQuad();
        impactParticles.reserve(512);
    }

    void Game::ensureImpactQuad()
    {
        if (impactQuadVAO != 0) return;

        const float quad[] = {
                // x, y, u, v
                -0.5f, -0.5f, 0.0f, 0.0f,
                0.5f, -0.5f, 1.0f, 0.0f,
                0.5f,  0.5f, 1.0f, 1.0f,

                -0.5f, -0.5f, 0.0f, 0.0f,
                0.5f,  0.5f, 1.0f, 1.0f,
                -0.5f,  0.5f, 0.0f, 1.0f
        };

        glGenVertexArrays(1, &impactQuadVAO);
        glGenBuffers(1, &impactQuadVBO);

        glBindVertexArray(impactQuadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, impactQuadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void Game::setupVEffects() {
        sunBillboards.init(12);
        setupImpactEffects();
        initSpeedLinesShader();
    }


////----Simulation Code-------------------------------------------------------------------------------------------------------------------------

//------Physics-Code----------------------------------------------------------------------------------------------------------------------------

    void Game::updatePhysics() {
        TRACY_CPU_ZONE("Game::updatePhysics");

        const float fixedTimeStep = 1.0f / 60.0f;
        const int subStepCount = 10;
        const float subDt = (fixedTimeStep / (float)subStepCount);

        float localDt = deltaTime;
        localDt = glm::clamp(deltaTime, deltaTime, 1.0f / 100.0f);
        accumulator += localDt;

        while (accumulator >= fixedTimeStep) {
            glm::vec3 moveInput(0.0f);
            if (actions["MoveForward"].isPressed) moveInput.z += 1.0f;
            if (actions["MoveBack"].isPressed) moveInput.z -= 1.0f;
            if (actions["MoveLeft"].isPressed) moveInput.x -= 1.0f;
            if (actions["MoveRight"].isPressed) moveInput.x += 1.0f;

            float swimVertical = 0.0f;
            if (actions["Jump"].isPressed) swimVertical += 1.0f;      // Swim up (Space)
            if (actions["Crouch"].isPressed) swimVertical -= 1.0f;    // Swim down (Ctrl)

            float len = glm::length(moveInput);
            if (len > 1.0f) moveInput /= len;

            bool jump = actions["Jump"].wasJustPressed;
            bool sprint = actions["Sprint"].isPressed;
            bool crouch = actions["Crouch"].isPressed;
            bool airSlam = actions["AirReset"].isPressed;

            if (glm::length(cameraRight) > 1e-5f) {
                cameraRight = glm::normalize(cameraRight);
            } else {
                cameraRight = glm::vec3(1, 0, 0);
            }

            if (characterController->hasWorldContact()) {
                uint32_t mat = characterController->getCurrentContactMaterial();

                glm::vec3 velocity = characterController->getVelocity();
                float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);

                switch (mat) {
                    case 9: // lava
                        break;
                    case 7: // flesh
                        if(speed>3.0f&&!characterController->getState().isSprinting)
                        {
                            g_SoundManager.playSound(SoundID::MeatStick,0.75f,1.0f,true,true,2);
                        }

                        break;
                    default:
                        if(!characterController->getState().isSprinting&&speed>3.0f)
                        {
                            g_SoundManager.playSound(SoundID::Step,1.0f,1.0f,true,true,2);
                        }
                        break;
                }
            }

            glm::vec2 mouseDelta = getMouseDelta();
            for (int i = 0; i < subStepCount; ++i) {
                {
                    TRACY_CPU_ZONE("Game::updatePlayer()");
                    if (characterController->hasWorldContact()) {
                        uint32_t mat = characterController->getCurrentContactMaterial();

                        glm::vec3 velocity = characterController->getVelocity();
                        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);

                        switch (mat) {
                            case 9: // lava
                                registerPlayerDamage({
                                                             1.75f * deltaTime,
                                                             (-characterController->getUpDirection()),
                                                             (float)glfwGetTime(),
                                                             5.5f
                                                     });
                                break;
                            case 7: // flesh
                                registerPlayerDamage({
                                                             0.85f*deltaTime,
                                                             (-characterController->getUpDirection()),
                                                             (float)glfwGetTime(),
                                                             5.5f
                                                     });
                                moveInput *= 0.5f;
                                break;
                            default:
                                break;
                        }
                    }
                    characterController->update(subDt, moveInput, swimVertical, jump, sprint, crouch, mouseDelta, cameraForward,
                                                cameraRight, airSlam);
                }
                jump = false;
            }

            std::vector<uint64_t> removedBodyIds;
            {
                TRACY_CPU_ZONE("Game::updateBodies()");
                if (voxelPhysics) voxelPhysics->update(localDt, removedBodyIds);
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

        for(auto &enemy: enemyManager->all())
        {
            if(body==enemy.inst.body)
            {
                registerPlayerDamage({
                                             10.0f,
                                             cameraPos-body->position,
                                             (float)glfwGetTime(),
                                             5.5f
                                     });
                g_SoundManager.playSound(SoundID::PlayerDamage);
                body->position-=body->velocity*glm::vec3(1);
                return;
            }
        }

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
        dmg = glm::min(dmg,getPlayerMaxHealth()/10);
        dmg= glm::max(dmg,getPlayerMaxHealth()/20);
        registerPlayerDamage({
                                     dmg,
                                     cameraPos-body->position,
                                     (float)glfwGetTime(),
                                     5.5f
                             });
        body->velocity=-body->velocity*0.75f;


        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        uint64_t spellId = (uint64_t)(uintptr_t)body->userData;
        if (auto* spell = spellSystem->findSpellById(spellId)) {
            float mass = spell->physicsBody ? spell->physicsBody->mass : 1.0f;

            float craterStrength = glm::sqrt((hitSpeed * mass)) / 30.0f;
            float spellRadius = glm::max(spell->physicsBody->radius, 0.001f);
           // createCraterAtPosition(hitPos, craterStrength, spellRadius);

            // Estimate removed voxels from crater strength
            //float removedVoxelEstimate = craterStrength * 40.0f;

            glm::vec3 tint = spell->formationColor;
            if (glm::length(tint) < 0.001f) {
                tint = glm::vec3(0.45f, 0.45f, 0.45f);
            }
            glm::vec3 variance = glm::vec3(dist(rng));
            tint+=(spell->formationColor/glm::vec3(10));
            tint+=variance;
            if(body->impactVfxCooldown<=0)
            {
                body->impactVfxCooldown=2.0f;
                spawnImpactEffect(hitPos, hitNormal, hitSpeed, 100, tint);

            }
        }
    }

    void Game::onBodyBodyCollision(gl3::VoxelPhysicsBody* bodyA,
                                   gl3::VoxelPhysicsBody* bodyB,
                                   const glm::vec3& hitPos,
                                   const glm::vec3& hitNormal,
                                   float impactSpeed)
    {
        if (!bodyA || !bodyB) return;

        CollisionDecision dA = decideCollisionResponse(*bodyA, bodyB, hitPos,  hitNormal,  impactSpeed);
        CollisionDecision dB = decideCollisionResponse(*bodyB, bodyA, hitPos, -hitNormal, impactSpeed);

        if (dA.stick) {
            bodyA->stuck = true;
            bodyA->stuckToBodyId = bodyB->id;
            bodyA->velocity = glm::vec3(0.0f);
            bodyA->angularVelocity = glm::vec3(0.0f);
            bodyA->stuckOffset = bodyA->position - bodyB->position;
        }
        if (dB.stick) {
            bodyB->stuck = true;
            bodyB->stuckToBodyId = bodyA->id;
            bodyB->velocity = glm::vec3(0.0f);
            bodyB->angularVelocity = glm::vec3(0.0f);
            bodyB->stuckOffset = bodyB->position - bodyA->position;
        }

        if (dA.ignoreCollision && dB.ignoreCollision) return;

        if (dA.ignoreCollision && bodyA->bodiesCanPassThrough > 0) bodyA->bodiesCanPassThrough--;
        if (dB.ignoreCollision && bodyB->bodiesCanPassThrough > 0) bodyB->bodiesCanPassThrough--;

        if(bodyA->material==9&&bodyB->material!=9)
        {
            voxelPhysics->removeBody(bodyB->id);
            return;
        }
        if(bodyB->material==9&&bodyA->material!=9)
        {
            voxelPhysics->removeBody(bodyA->id);
            return;
        }

        auto applyDamageToOneBody = [&](gl3::VoxelPhysicsBody* self,
                                        gl3::VoxelPhysicsBody* other,
                                        bool ignoreThisSide)
        {
            if (!self || ignoreThisSide) return;

            // 1) Enemy damage path
            if (enemyManager) {
                enemyManager->applyDamageSphere(self->id, hitPos, self->radius, impactSpeed);
            }

            // 2) Spell/destructible damage path
            SpellEffect* spell = self->ownerSpell;
            if (!spell) {
                uint64_t spellId = (uint64_t)(uintptr_t)self->userData;
                spell = spellSystem ? spellSystem->findSpellById(spellId) : nullptr;
            }

            if (spell) {
                // Convert world hit to local spell volume and carve damage
                const glm::vec3 localHit = spell->destruct.worldToLocal(hitPos, spell->center);

                // Tune these two to taste:
                const float damageRadius   = glm::max(self->radius * 0.35f, 1.5f * VOXEL_SIZE);
                const float damageStrength = glm::clamp(impactSpeed * 0.05f, 0.5f, 8.0f);

                spell->destruct.volume.carveSphere(localHit, damageRadius, damageStrength);
                spell->destruct.meshDirty = true; // ensure remesh
            }
        };

        // Apply to BOTH sides
        if(bodyB&&bodyA) {
            applyDamageToOneBody(bodyA, bodyB, dA.ignoreCollision);
            applyDamageToOneBody(bodyB, bodyA, dB.ignoreCollision);
        }

        if (dA.stick || dB.stick) return;

        // keep your existing impact VFX (optional)
        spawnImpactEffect(hitPos, hitNormal, impactSpeed, 100.0f, glm::vec3(0.45f));
    }

void Game::updateDeltaTime() {
    float frameTime = glfwGetTime();
    deltaTime = deltaTime = frameTime - lastFrameTime;
    lastFrameTime = frameTime;
}

//------Lighting-Code---------------------------------------------------------------------------------------------------------------------------

void Game::refreshMergedEmissiveBillboards()
{
        TRACY_CPU_ZONE("Game::refreshMergedEmissiveBillboards");

        emissiveBillboards.clear();
        chunkRenderer->collectMergedEmissiveBillboards(emissiveBillboards);
        emissiveBillboardsDirty = false;
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

uint32_t Game::makeLightID(int cx, int cy, int cz) {
// Simple hash function for light ID
return ((cx & 0xFFF) << 20) | ((cy & 0xFFF) << 8) | (cz & 0xFF);
}


////----Input Code------------------------------------------------------------------------------------------------------------------------------

void Game::updateImpactEffects(float dt)
{
for (auto& p : impactParticles) {
    if (!p.active) continue;

    p.age += dt;
    if (p.age >= p.lifetime) {
        p.active = false;
        continue;
    }

    // Basic motion
    p.position += p.velocity * dt;

    // Drag
    p.velocity *= 0.96f;

    // Gentle upward drift for smoke
    if (p.kind == 0) {
        p.velocity += glm::vec3(0.0f, 0.35f * VOXEL_SIZE, 0.0f) * dt;
    }

    p.rotation += p.rotationSpeed * dt;
}

// compact occasionally
impactParticles.erase(
        std::remove_if(impactParticles.begin(), impactParticles.end(),
                       [](const ImpactParticle& p) { return !p.active; }),
        impactParticles.end()
);
}


void Game::update() {
    TRACY_CPU_ZONE("Game::update()");

    // Update merged lights occasionally (CPU) + upload to GPU
    if (frameCounter % 283 == 0) {
        chunkRenderer->updateLightSpatialHash();
        emissiveBillboardsDirty = true;
    }
    if(frameCounter%261==0)
    {
        chunkRenderer->uploadMergedLightsToGPU();
    }
    if (frameCounter % 247 == 0) {
        TRACY_CPU_ZONE("Game::refreshMergedLights");

        chunkManager->forEachChunk([&](gl3::Chunk* chunk) {
            if (!chunk) return;
            if (!chunk->lightingDirty) return;

            rebuildChunkLights(chunk->coord);
            chunk->lightingDirty = false;
        });

        chunkRenderer->updateLightSpatialHash();
    }

    if (emissiveBillboardsDirty) {
        refreshMergedEmissiveBillboards();
    }
    {
    TRACY_CPU_ZONE("Game::Recalc Solid Meshes and Lights");
    chunkManager->rebuildDirtyChunks([this](Chunk* chunk) {
        chunkRenderer->generateChunkMesh(chunk);
    },cameraPos);

    {
        TRACY_CPU_ZONE("Game::Recalc Fluid Meshes and Lights");
            chunkManager->rebuildDirtyChunks([this](Chunk *chunk) {
                chunkRenderer->generateFluidMesh(chunk);
            }, cameraPos);
        }
    }

// per-chunk light index buffer
if(frameCounter % 311 == 0)
{
    TRACY_CPU_ZONE("renderChunks::buildAndUploadChunkLightIndexBuffer");
    const int camCX = worldToChunk(cameraPos.x);
    const int camCY = worldToChunk(cameraPos.y);
    const int camCZ = worldToChunk(cameraPos.z);
    const int renderRadius = RenderingRange;
    chunkRenderer->buildAndUploadChunkLightIndexBuffer(camCX, camCY, camCZ, renderRadius);
}

if(getPlayerHealth()<=0)
{
    requestSceneChange(SceneId::MainMenu);
    SoLoud::handle musicHandle = g_SoundManager.playMusic(SoundID::MainMenuTheme, true, 1.0f);
}
{
    glm::vec3 dir = characterController->getPosition();
    float dist = glm::sqrt(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
    if(dist>550.0f)
    {
        g_SoundManager.playSound(SoundID::Suffocate);
        registerPlayerDamage({
                                     2.25f*deltaTime,
                                     dir,
                                     (float)glfwGetTime(),
                                     5.5f
                             });
    }

    TRACY_CPU_ZONE("SunBurns()");
    chunkManager->forEachEmissiveChunk([this](Chunk *chunk) {
        VoxelLight best{
                glm::vec3(0, 0, 0),
                1.0f,
                glm::vec3(0, 0, 0),
                0
        };
    for (auto &light: chunk->emissiveLights) {
        if(light.intensity>best.intensity)
        {
            best=light;
        }
    }
    float bestGravity=0;
    for (auto &light: chunk->emissiveLights) {
        glm::vec3 dist=(cameraPos-light.pos);
        float distsq = glm::sqrt(dist.x*dist.x+ dist.y*dist.y+ dist.z*dist.z);
        if(distsq<std::sqrt(light.intensity) * 0.15f)
        {
            registerPlayerDamage({
                                         deltaTime*0.25f*distsq*(glm::sqrt(light.intensity*0.00001f)),
                                         dist,
                                         (float)glfwGetTime(),
                                         5.5f
                                 });
            g_SoundManager.playSound(SoundID::Fire);
        }
        float gravity = glm::pow(light.intensity,2.0f)/distsq;
        if(gravity>bestGravity&&!characterController->isSurfaceAdhered())
        {
            bestGravity=gravity;
            best=light;
            characterController->setGravityIntensity(gravity);
        }
    }
        if(best.pos!=characterController->settings.lastGravPoint&&!characterController->isSurfaceAdhered())
        {
            characterController->settings.lastGravPoint=best.pos;
            glm::vec3 gravDir = glm::normalize(best.pos - cameraPos);
            characterController->setGravityDirection(gravDir);
        }
});
    for(auto &spell: spellSystem->spells())
    {
        if(spell.physicsBody&&spell.physicsBody->material==9) {
            glm::vec3 dist = (cameraPos - spell.physicsBody->position);
            float distsq = glm::sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
            if (distsq < (spell.radius*1.5f)) {
                registerPlayerDamage({
                                             deltaTime*0.25f * distsq*(glm::sqrt(spell.physicsBody->radius*0.01f)),
                                             dist,
                                             (float)glfwGetTime(),
                                             5.5f
                                     });
            }
        }
    }
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
    activeSpellMat = 9;
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
    float spellRadius = 2.0f * VOXEL_SIZE;  // Adjust size
    float spellStrength = 3.0f;              // Affects velocity

    if (spellSystem)
        spellSystem->castSphere(spellCenter, spellRadius, activeSpellMat, spellStrength, getCameraFront(), VOXEL_SIZE*CHUNK_SIZE*3, makeVoxelTypeMask({1}));
}

if (actions["CastFleshSphere"].wasJustReleased) {
    std::cout << "Flesh Sphere Spell Triggered\n";
    RayCastResult hit = rayCastFromCamera(5.0f);
    glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
            (cameraPos + getCameraFront() * 35.0f);

    // Cast spell with physics enabled
    float spellRadius = 2.0f * VOXEL_SIZE;  // Adjust size
    float spellStrength = 3.0f;              // Affects velocity

    if (spellSystem)
        spellSystem->castSphere(spellCenter, spellRadius, 7, spellStrength, getCameraFront(), VOXEL_SIZE*CHUNK_SIZE*3, makeVoxelTypeMask({1}));
    }

    if (actions["CastFireSphere"].wasJustReleased) {
        std::cout << "Fire Sphere Spell Triggered\n";
        RayCastResult hit = rayCastFromCamera(5.0f);
        glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                                (cameraPos + getCameraFront() * 35.0f);

        // Cast spell with physics enabled
        float spellRadius = 2.0f * VOXEL_SIZE;  // Adjust size
        float spellStrength = 1.0f;              // Affects velocity

        if (spellSystem)
            spellSystem->castSphere(spellCenter, spellRadius, 9, spellStrength, getCameraFront(), VOXEL_SIZE*CHUNK_SIZE*50,  makeVoxelTypeMask({1}));
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
    spellSystem->castWall(spellCenter, cameraFront,
                  wallWidth, wallHeight, wallThickness,
                  0, 2.0f*VOXEL_SIZE,makeVoxelTypeMask({1}));
}
if (actions["AirReset"].wasJustReleased) {
    std::cout << "Platform Spell Triggered" << "\n";
    glm::vec3 spellCenter =(cameraPos + glm::vec3(0,-1,0) * 25.0f*VOXEL_SIZE);

    // Wall dimensions (tune these values)
    float wallWidth = 1.0f*VOXEL_SIZE;    // Horizontal width
    float wallHeight = 1.05f*VOXEL_SIZE;   // Vertical height
    float wallThickness = 3.5f*VOXEL_SIZE; // How thick the wall is

    // Cast the wall spell
    spellSystem->castWall(spellCenter, glm::vec3(0,-1,0),
                  wallWidth, wallHeight, wallThickness,
                  0, 7.5f*VOXEL_SIZE,makeVoxelTypeMask({1}));
}
if(actions["RollLeft"].wasJustPressed)
{
    RayCastResult hit = rayCastFromCamera(5.0f);
    glm::vec3 spellCenter = hit.hit ? hit.hitPosition :
                            (cameraPos + getCameraFront() * 35.0f);

    waveManager.materialMined+=mineSolidWorld(spellCenter, 12.0f);
    std::cout<<"current material collected: "<<waveManager.materialMined<<"\n";
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
    damageTimer+=deltaTime;
    while (damageTimer >= damageTimeframe)
    {
        damageTimer -= damageTimeframe;
        applyPlayerDamage();
    }

}

glm::vec3 Game::getCameraFront() const {
        return glm::normalize(cameraForward);
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
if (!skyboxRuntimeShader)
{
    std::cout<<"No skybox shader found";
    return;
}

GLint oldDepthFunc;
glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
GLboolean depthMask;
glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);

glDepthFunc(GL_LEQUAL);
glDepthMask(GL_FALSE);

skyboxRuntimeShader->use();
skyboxRuntimeShader->setFloat("uBrightness",settings.brightness);
skyboxRuntimeShader->setFloat("uGamma",settings.gamma);
skyboxRuntimeShader->setFloat("time", (float)glfwGetTime());

float aspect = (float)windowWidth / (float)windowHeight;
glm::mat4 projection;
if(characterController)
{
    glm::vec3 velocity = characterController->getVelocity();
    float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
    projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
} else
{
    projection = glm::perspective(glm::radians(45.0f*settings.fov), aspect, nearPlane, farPlane);
}
glm::vec3 camUp = getCameraUp();
glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
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
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        frameCounter++;
        chunkRenderer->frameCounter += 1;

        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);
        const int renderRadius = RenderingRange;

        // Cleanup distant chunks periodically (every 60 frames)
        if (frameCounter % 60 == 0) {
            TRACY_CPU_ZONE("renderChunks::cleanupDistantSlots");
            chunkManager->cleanupDistantSlots(cameraPos, renderRadius);
        }

        visibleSlots.clear();
        visibleFluidSlots.clear();

        // Generate meshes / rebuild emissive lights
        {
            TRACY_CPU_ZONE("renderChunks::PrepareMeshesAndLights");
            const int R = chunkManager->radius();

            const int minCX = std::max(camCX - renderRadius, -R);
            const int maxCX = std::min(camCX + renderRadius, R);
            const int minCY = std::max(camCY - renderRadius, -R);
            const int maxCY = std::min(camCY + renderRadius, R);
            const int minCZ = std::max(camCZ - renderRadius, -R);
            const int maxCZ = std::min(camCZ + renderRadius, R);

            for (int cx = minCX; cx <= maxCX; ++cx) {
                for (int cy = minCY; cy <= maxCY; ++cy) {
                    for (int cz = minCZ; cz <= maxCZ; ++cz) {
                        ChunkCoord coord{cx, cy, cz};
                        Chunk* chunk = chunkManager->getChunk(coord);
                        if (!chunk) continue;

                        // Skip empty chunks (no geometry)
                        if (chunk->isCleared) continue;

                        // If chunk has no solid mesh but has fluid, add to fluid list
                        if (!chunk->gpuCache.isValid) {
                            if (chunk->hasFluid) {
                                visibleFluidSlots.push_back(chunk->gpuSlot);
                            }
                            continue;
                        }

                        // Chunk has solid mesh
                        DrawArraysIndirectCommand cmd{};
                        cmd.count = chunk->gpuCache.vertexCount;
                        cmd.instanceCount = 1;
                        cmd.first = chunk->gpuSlot * (uint32_t)CHUNK_MAX_VERTS;
                        cmd.baseInstance = chunk->gpuSlot;

                        if (cmd.count > 0) {
                            visibleSlots.push_back(chunk->gpuSlot);
                        }

                        // ALSO add to fluid list if it has fluid (even if it has solid mesh too)
                        if (chunk->hasFluid) {
                            visibleFluidSlots.push_back(chunk->gpuSlot);
                        }
                    }
                }
            }
        }

        // Draw solid chunks
        {
            TRACY_CPU_ZONE("renderChunks::DrawBatched");
            TRACY_GPU_ZONE("Chunks::DrawBatched");

            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);

            voxelShader->use();
            voxelShader->setFloat("uBrightness",settings.brightness);
            voxelShader->setFloat("uGamma",settings.gamma);


            if (actions["CastSphere"].isHeld) {
                float maxDist = 250.0f;
                RayCastResult hit = rayCastFromCamera(maxDist);
                glm::vec3 center = hit.hit ? hit.hitPosition : (cameraPos + getCameraFront() * 35.0f);

                float formationRadius = 2.0f * VOXEL_SIZE;
                float pullRadius = formationRadius * 15.5f;

                voxelShader->setInt("uOverlayEnabled", 1);
                voxelShader->setVec3("uOverlayCenter", center);
                voxelShader->setFloat("uOverlayRadius", pullRadius);
                voxelShader->setVec3("uOverlayColor", glm::vec3(0.25f, 1.0f, 0.35f));
                voxelShader->setFloat("uOverlayAlpha", 0.25f);
            } else {
                voxelShader->setInt("uOverlayEnabled", 0);
            }

            voxelShader->setFloat("scale", 1.0f);

            voxelShader->setInt("uNormalYFlip", 0);
            voxelShader->setFloat("uNormalStrength", 1.0f);
            voxelShader->setFloat("uHeightScale", 1.0f);
            voxelShader->setFloat("uAOStrength", 0.5f);

            voxelShader->setMatrix("model", glm::mat4(1.0f));
            voxelShader->setMatrix("mvp", pv);

            voxelShader->setVec3("viewPos", cameraPos);
            voxelShader->setVec3("ambientColor", glm::vec3(0.85f));

            voxelShader->setBool("uHasPlayerContact", characterController->hasWorldContact());
            voxelShader->setVec3("uPlayerContactPoint", characterController->getCurrentContactPoint());

            uint32_t excludedArray[8] = {};

            if (actions["CastSphere"].isHeld) {
                excludedArray[0] = 7;
                excludedArray[1] = 9;
            }
            else if (actions["CastWall"].isHeld) {
                excludedArray[0] = 7;
                excludedArray[1] = 9;
            }
            else if (actions["AirReset"].isHeld) {
                excludedArray[0] = 7;
                excludedArray[1] = 9;
            }
            voxelShader->setUInt("uExcludedMaterialCount", 2u);
            voxelShader->setUIntArray("uExcludedMaterials", excludedArray, 2);
            //voxelShader->setUInt("uTargetType", 1u);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
            voxelShader->setInt("uAlbedoArray", 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialNormalArrayTexId);
            voxelShader->setInt("uNormalArray", 1);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialRoughArrayTexId);
            voxelShader->setInt("uRoughArray", 2);

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialAOArrayTexId);
            voxelShader->setInt("uAOArray", 3);

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D_ARRAY, materialHeightArrayTexId);
            voxelShader->setInt("uHeightArray", 4);

            voxelShader->setFloatArray("uMatRoughness", rough.data(), 64);
            voxelShader->setFloatArray("uMatSpecular", spec.data(), 64);
            voxelShader->setFloatArray("uUVScale", uvScale.data(), 64);
            static float frameTime = 0.0f;
            frameTime += deltaTime;
            voxelShader->setFloat("uTime", frameTime / 1.5f);

            if (DebugMode1) voxelShader->setInt("debugMode", activeDebugMode % 7);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, chunkRenderer->ssboLights);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, chunkRenderer->ssboChunkLightIdx);

            glBindVertexArray(chunkRenderer->globalChunkVAO);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkRenderer->chunkIndirectBuffer);

            for (uint32_t slot : visibleSlots) {
                const GLsizeiptr offset = (GLsizeiptr)slot * (GLsizeiptr)sizeof(DrawArraysIndirectCommand);
                glDrawArraysIndirect(GL_TRIANGLES, (const void*)offset);
            }

            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
            glBindVertexArray(0);
        }

        std::vector<SunInstance> billboardRenderList = emissiveBillboards;
        appendSpellBillboards(billboardRenderList);

        if (!billboardRenderList.empty() && !DebugMode1) {
            sunBillboards.render(billboardRenderList, view, projection, (float)glfwGetTime(),settings.gamma,settings.brightness);
        }
    }

    void Game::renderAnimatedVoxels() {
        TRACY_CPU_ZONE("Game::renderAnimatedVoxels");
        TRACY_GPU_ZONE("AnimatedVoxels");
        if (spellSystem->animated().empty()) return;

        std::vector<const AnimatedVoxel*> instances;
        instances.reserve( spellSystem->animated().size());
        for (const auto &v : spellSystem->animated()) {
            if (v.isAnimating || v.hasArrived) instances.push_back(&v);
        }
        int instanceCount = (int)instances.size();
        if (instanceCount == 0) return;

        static Shader instancedShader(resolveAssetPath("shaders/voxel_anim.vert"),
                                      resolveAssetPath("shaders/voxel.frag"));
        instancedShader.use();

        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        instancedShader.setMatrix("pv", pv);
        instancedShader.setVec3("viewPos", cameraPos);
        instancedShader.setVec3("ambientColor", glm::vec3(0.85f));

        std::vector<float> posScaleData; posScaleData.reserve(instanceCount * 4);
        std::vector<float> colorData;    colorData.reserve(instanceCount * 3);
        std::vector<float> normalData;   normalData.reserve(instanceCount * 3);

        for (int i = 0; i < instanceCount; ++i) {
            const AnimatedVoxel* v = instances[i];

            float pulse = 1.0f;
            if (v->hasArrived) {
                pulse = 1.0f + 0.08f * std::sin(glfwGetTime() * 3.0f);
            } else {
                pulse = 1.0f + 0.16f * std::sin(glfwGetTime() * 8.0f + v->currentPos.x);
            }

            float halfSize = (VOXEL_SIZE * 0.25f) * pulse;

            posScaleData.push_back(v->currentPos.x);
            posScaleData.push_back(v->currentPos.y);
            posScaleData.push_back(v->currentPos.z);
            posScaleData.push_back(halfSize);


            colorData.push_back(v->color.r);
            colorData.push_back(v->color.g);
            colorData.push_back(v->color.b);

            glm::vec3 n = v->normal;
            normalData.push_back(n.x);
            normalData.push_back(n.y);
            normalData.push_back(n.z);
        }

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
        if (spellSystem->spells().empty()) return;


        voxelShader->use();
        voxelShader->setFloat("uBrightness",settings.brightness);
        voxelShader->setFloat("uGamma",settings.gamma);
        float aspect = (windowHeight == 0) ? (float) windowWidth / 1.0f : (float) windowWidth / (float) windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.85f));

        voxelShader->setFloat("scale", 1.0f);

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
        voxelShader->setInt("uAlbedoArray", 0);


        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialNormalArrayTexId);
        voxelShader->setInt("uNormalArray", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialRoughArrayTexId);
        voxelShader->setInt("uRoughArray", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAOArrayTexId);
        voxelShader->setInt("uAOArray", 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialHeightArrayTexId);
        voxelShader->setInt("uHeightArray", 4);

        voxelShader->setFloatArray("uMatRoughness", rough.data(), 64);
        voxelShader->setFloatArray("uMatSpecular",  spec.data(), 64);
        voxelShader->setFloatArray("uUVScale",      uvScale.data(), 64);

        voxelShader->setFloat("uNormalYFlip", 0.0f);
        voxelShader->setFloat("uNormalStrength", 1.0f);
        voxelShader->setFloat("uHeightScale", 1.0f);
        voxelShader->setFloat("uAOStrength", 1.0f);

        voxelShader->setInt("uBurnEnabled", 0);
        voxelShader->setFloat("uBurn", 0.0f);
        voxelShader->setVec3("uBurnCenter", glm::vec3(0.0f));
        voxelShader->setFloat("uBurnRadius", 0.0f);
        voxelShader->setFloat("uBurnNoiseScale", 0.35f);
        voxelShader->setFloat("uBurnEdgeWidth", 0.12f);
        voxelShader->setVec3("uBurnEmberColor", glm::vec3(2.5f, 0.9f, 0.2f));
        voxelShader->setFloat("uBurnCharStrength", 0.85f);
        static float frameTime = 0.0f;
        frameTime += deltaTime;
        voxelShader->setFloat("uTime", frameTime);

        // Render each physics-enabled formation
        for (const auto& spell : spellSystem->spells()) {
            if (!spell.isPhysicsEnabled || !spell.physicsBody|| spell.physicsBodyId == 0) continue;
            auto* body = voxelPhysics->getBodyById(spell.physicsBodyId);
            if (!body || !body->renderMesh) continue;

            const auto& mesh = *body->renderMesh;
            if (!mesh.isValid || mesh.vertexCount == 0) continue;

            int currentChunkX = worldToChunk(spell.physicsBody->position.x);
            int currentChunkY = worldToChunk(spell.physicsBody->position.y);
            int currentChunkZ = worldToChunk(spell.physicsBody->position.z);

            glm::vec3 pos = glm::vec3(spell.physicsBody->position.x,spell.physicsBody->position.y,spell.physicsBody->position.z) ;
            glm::quat rot = spell.physicsBody->orientation;

            // Build model matrix
            glm::vec3 originWorld = spell.physicsBody->position;
            glm::mat4 model = glm::translate(glm::mat4(1.0f), originWorld);
//            model *= glm::mat4_cast(rot);
            //          model = glm::scale(model, glm::vec3(VOXEL_SIZE));  // Apply VOXEL_SIZE scaling

            voxelShader->setMatrix("model", model);
            voxelShader->setMatrix("mvp", pv*model );

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

            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
            glBindVertexArray(0);
        }
    }



    void Game::renderEnemies() {
        if (!enemyManager) return;

        voxelShader->use();
        voxelShader->setFloat("uBrightness",settings.brightness);
        voxelShader->setFloat("uGamma",settings.gamma);

        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        voxelShader->setVec3("viewPos", cameraPos);
        voxelShader->setVec3("ambientColor", glm::vec3(0.85f));

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
        voxelShader->setInt("uAlbedoArray", 0);


        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialNormalArrayTexId);
        voxelShader->setInt("uNormalArray", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialRoughArrayTexId);
        voxelShader->setInt("uRoughArray", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAOArrayTexId);
        voxelShader->setInt("uAOArray", 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialHeightArrayTexId);
        voxelShader->setInt("uHeightArray", 4);

        voxelShader->setFloatArray("uMatRoughness", rough.data(), 64);
        voxelShader->setFloatArray("uMatSpecular",  spec.data(), 64);
        voxelShader->setFloatArray("uUVScale",      uvScale.data(), 64);

        voxelShader->setFloat("uNormalYFlip", 0.0f);
        voxelShader->setFloat("uNormalStrength", 1.0f);
        voxelShader->setFloat("uHeightScale", 1.0f);
        voxelShader->setFloat("uAOStrength", 1.0f);

        for (auto& e : enemyManager->all()) {
            if (e.renderParts.empty()) continue;

            glm::vec3 volumeCenterLocal = glm::vec3(16.0f, 16.0f, 16.0f) * VOXEL_SIZE;

            glm::mat4 model = glm::translate(glm::mat4(1.0f), e.inst.position);

            if (e.inst.rotation != glm::quat(1,0,0,0)) {
                model *= glm::mat4_cast(e.inst.rotation);
            }

            model = glm::translate(model, -volumeCenterLocal);

            voxelShader->setMatrix("model", model);
            voxelShader->setMatrix("mvp", pv * model);
            voxelShader->setFloat("emission", 0.3f);
            voxelShader->setVec3("emissionColor", glm::vec3(0.4,0.0,0.4));

            voxelShader->setFloat("scale", e.inst.body ? e.inst.body->radius : e.inst.currentRadius);
            glm::vec3 eyeCenterLocal = glm::vec3(16.0f, 16.0f, 16.0f) * VOXEL_SIZE
                                       + glm::vec3(e.inst.currentRadius * 0.65f, 0.0f, 0.0f);
            voxelShader->setVec3("uEyeCenterLocal", eyeCenterLocal / e.inst.currentRadius);
            voxelShader->setVec3("uEyeForwardLocal", glm::vec3(1.0f, 0.0f, 0.0f));
            voxelShader->setFloat("uTime", (float)glfwGetTime());

            for (auto& part : e.renderParts) {
                if (!part.mesh.isValid || part.mesh.vertexCount == 0) continue;

                glBindVertexArray(part.mesh.vao);
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)part.mesh.vertexCount);
            }

            glBindVertexArray(0);
        }
    }

    void Game::renderSpellPreview() {
        TRACY_CPU_ZONE("Game::renderSpellPreview");
        TRACY_GPU_ZONE("SpellPreview");

        if (!spellPreviewShader) return;

        ensurePreviewCube();
        ensurePreviewSphereMesh();

        int previewMode = -1; // -1 none, 0 sphere, 1 wall
        if (actions["CastSphere"].isHeld) previewMode = 0;
        else if (actions["CastWall"].isHeld) previewMode = 1;
        else if (actions["AirReset"].isHeld) previewMode = 1;
        else return;

        float maxDist = (previewMode == 0) ? 80.0f : 250.0f;
        RayCastResult hit = rayCastFromCamera(maxDist);

        glm::vec3 cameraFront = glm::normalize(getCameraFront());

        glm::vec3 center;
        if (previewMode == 0) {
            center = hit.hit ? hit.hitPosition
                             : (cameraPos + cameraFront * 35.0f);
        } else {
            center = hit.hit ? (hit.hitPosition + cameraFront * 10.5f)
                             : (cameraPos + cameraFront * 10.5f);
        }

        float aspect = (windowHeight == 0)
                       ? (float)windowWidth
                       : (float)windowWidth / (float)windowHeight;

        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = characterController->getUpDirection();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        float formationRadius = 2.0f * VOXEL_SIZE;
        float pullRadius = formationRadius * 6.5f;

        // Default wall setup
        glm::vec3 wallNormal = -cameraFront;
        glm::vec3 wallSize(3.0f * VOXEL_SIZE,
                           1.75f * VOXEL_SIZE,
                           0.35f * VOXEL_SIZE);

        FormationParams wallParams = FormationParams::Wall(
                center,
                wallNormal,
                wallSize.x,
                wallSize.y,
                wallSize.z
        );

        // Air reset override
        if (actions["AirReset"].isHeld) {
            center = cameraPos + glm::vec3(0, -1, 0) * (4.0f * VOXEL_SIZE);

            wallParams = FormationParams::Wall(
                    center,
                    glm::vec3(0, -1, 0),
                    3.0f * VOXEL_SIZE,
                    1.0f * VOXEL_SIZE,
                    3.5f * VOXEL_SIZE
            );

            pullRadius = 20.0f * VOXEL_SIZE;
        }

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);

        spellPreviewShader->use();

        float voxelVolume = VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        int maxVoxels = (int)((4.0f * 70.0f) / voxelVolume);
        maxVoxels = glm::clamp(maxVoxels, 5, 200);

        std::vector<uint32_t> excludedSpellMats{7, 9};
        int available = estimateAvailableVoxels(center, pullRadius, excludedSpellMats, 1, maxVoxels);
        float fillRatio = maxVoxels > 0 ? (float)available / (float)maxVoxels : 1.0f;

        spellPreviewShader->setFloat("uFillRatio", fillRatio);
        spellPreviewShader->setVec3("uLowColor", glm::vec3(1.0f, 0.2f, 0.2f));
        spellPreviewShader->setVec3("uHighColor", glm::vec3(0.2f, 1.0f, 0.2f));

        spellPreviewShader->setMatrix("pv", pv);
        spellPreviewShader->setFloat("uFormationAlpha", 0.35f);

        if (previewMode == 0) {
            spellPreviewShader->setInt("uPreviewMode", 0);
            spellPreviewShader->setVec3("uWallSize", glm::vec3(1.0f));

            glm::mat4 model(1.0f);
            model = glm::translate(model, center);
            model = glm::scale(model, glm::vec3(formationRadius));
            spellPreviewShader->setMatrix("model", model);

            glBindVertexArray(previewSphereVAO);
            glDrawElements(GL_TRIANGLES, previewSphereIndexCount, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        } else {
            spellPreviewShader->setInt("uPreviewMode", 1);
            spellPreviewShader->setVec3("uWallSize", glm::vec3(wallParams.sizeX, wallParams.sizeY, wallParams.sizeZ));

            glm::vec3 forward = glm::normalize(wallParams.normal);
            glm::vec3 up = glm::normalize(wallParams.up);
            glm::vec3 right = glm::normalize(glm::cross(forward, up));
            up = glm::normalize(glm::cross(right, forward));

            glm::mat4 rot(1.0f);
            rot[0] = glm::vec4(right,   0.0f);
            rot[1] = glm::vec4(up,      0.0f);
            rot[2] = glm::vec4(forward, 0.0f);

            glm::mat4 model(1.0f);
            model = glm::translate(glm::mat4(1.0f), wallParams.center)
                    * rot
                    * glm::scale(glm::mat4(1.0f),
                                 glm::vec3(wallParams.sizeX, wallParams.sizeY, wallParams.sizeZ));

            spellPreviewShader->setMatrix("model", model);

            glBindVertexArray(previewCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);
        }

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    void Game::renderImpactEffects()
    {
        TRACY_CPU_ZONE("Game::renderImpactEffects");
        TRACY_GPU_ZONE("ImpactEffects");

        if (!impactShader || impactParticles.empty()) return;
        glm::vec3 cameraFront = getCameraFront();
        glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
        glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));

        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);

        impactInstancesCPU.clear();
        impactInstancesCPU.reserve(std::min(impactParticles.size(), kMaxImpactInstances));

        for (const auto& p : impactParticles) {
            if (!p.active) continue;
            if (impactInstancesCPU.size() >= kMaxImpactInstances) break;

            float t = glm::clamp(p.age / glm::max(0.0001f, p.lifetime), 0.0f, 1.0f);
            float size = glm::mix(p.startSize, p.endSize, t);

            glm::vec4 color = p.color;
            color.a *= (1.0f - t);

            ImpactInstanceGPU inst{};
            inst.pos_size = glm::vec4(p.position, size);
            inst.color = color;
            inst.rot_life_kind = glm::vec4(p.rotation, t, float((int)p.kind), 0.0f);
            impactInstancesCPU.push_back(inst);
        }

        if (impactInstancesCPU.empty()) return;

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        glEnable(GL_DEPTH_TEST);

        impactShader->use();
        impactShader->setMatrix("view", view);
        impactShader->setMatrix("projection", projection);
        impactShader->setVec3("uCameraRight", cameraRight);
        impactShader->setVec3("uCameraUp", cameraUp);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, impactNoiseTexId);
        impactShader->setInt("uNoiseTex", 0);

        glBindVertexArray(impactQuadVAO);

        glBindBuffer(GL_ARRAY_BUFFER, impactInstanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0,
                        impactInstancesCPU.size() * sizeof(ImpactInstanceGPU),
                        impactInstancesCPU.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)impactInstancesCPU.size());

        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    void Game::renderFluids()
    {
        TRACY_CPU_ZONE("Game::renderFluids");
        TRACY_GPU_ZONE("FluidChunks (total)");
        fluidShader->use();
        fluidShader->setFloat("uBrightness",2.0f*settings.brightness);
        fluidShader->setFloat("uGamma",settings.gamma);

        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        fluidShader->setMatrix("model", glm::mat4(1.0f));
        fluidShader->setMatrix("mvp", pv);
        fluidShader->setFloat("uTime", (float)glfwGetTime());
        fluidShader->setVec3("viewPos", cameraPos);

        fluidShader->setInt("uNormalYFlip", 0);
        fluidShader->setFloat("uNormalStrength", 1.0f);
        fluidShader->setFloat("uHeightScale", 1.0f);
        fluidShader->setFloat("uAOStrength", 0.5f);

        fluidShader->setVec3("ambientColor", glm::vec3(0.85f));
        fluidShader->setFloat("uTime", (float)glfwGetTime());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAlbedoArrayTexId);
        fluidShader->setInt("uAlbedoArray", 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialNormalArrayTexId);
        fluidShader->setInt("uNormalArray", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialRoughArrayTexId);
        fluidShader->setInt("uRoughArray", 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialAOArrayTexId);
        fluidShader->setInt("uAOArray", 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D_ARRAY, materialHeightArrayTexId);
        fluidShader->setInt("uHeightArray", 4);

        fluidShader->setFloatArray("uMatRoughness", rough.data(), 64);
        fluidShader->setFloatArray("uMatSpecular", spec.data(), 64);
        fluidShader->setFloatArray("uUVScale", uvScale.data(), 64);

        fluidShader->setInt("uFluidDebugMode", 0);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, chunkRenderer->ssboLights);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, chunkRenderer->ssboChunkLightIdx);

        fluidShader->setInt("uPass", 0); // 0 = front faces

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        glBindVertexArray(chunkRenderer->globalFluidVAO);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, chunkRenderer->fluidIndirectBuffer);

        for (uint32_t slot : visibleFluidSlots) {
            const GLsizeiptr offset = (GLsizeiptr)slot * (GLsizeiptr)sizeof(DrawArraysIndirectCommand);
            glDrawArraysIndirect(GL_TRIANGLES, (const void*)offset);
        }

        fluidShader->setInt("uPass", 1); // 1 = back faces
        glCullFace(GL_FRONT);
        for (uint32_t slot : visibleFluidSlots) {
            const GLsizeiptr offset = (GLsizeiptr)slot * (GLsizeiptr)sizeof(DrawArraysIndirectCommand);
            glDrawArraysIndirect(GL_TRIANGLES, (const void*)offset);
        }

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
        glBindVertexArray(0);

        glDisable(GL_BLEND);
        glCullFace(GL_BACK);
        glDisable(GL_CULL_FACE);
    }

    void Game::renderGas() {
        if (!gasRayMarchShader) return;

        // Collect gas chunks in view
        struct GasChunkData {
            glm::vec3 origin;
            uint32_t baseIndex;
            glm::ivec3 dims;
        };
        std::vector<GasChunkData> gasChunkData;

        const int R = chunkManager->radius();
        const int camCX = worldToChunk(cameraPos.x);
        const int camCY = worldToChunk(cameraPos.y);
        const int camCZ = worldToChunk(cameraPos.z);
        const int renderRadius = RenderingRange;

        // Check each chunk in view
        for (int cx = std::max(camCX - renderRadius, -R); cx <= std::min(camCX + renderRadius, R); ++cx) {
            for (int cy = std::max(camCY - renderRadius, -R); cy <= std::min(camCY + renderRadius, R); ++cy) {
                for (int cz = std::max(camCZ - renderRadius, -R); cz <= std::min(camCZ + renderRadius, R); ++cz) {
                    Chunk* chunk = chunkManager->getChunk({cx, cy, cz});
                    if (!chunk) continue;

                    if (chunk->hasGas && chunk->gpuSlot != FixedGridChunkManager::INVALID_GPU_SLOT) {
                        chunkRenderer->uploadVoxelChunkToGasSlot(*chunk);

                        GasChunkData data;
                        data.origin = getChunkMin({cx, cy, cz});
                        data.baseIndex = chunk->gpuSlot * ((CHUNK_SIZE + 2) * (CHUNK_SIZE + 2) * (CHUNK_SIZE + 2));
                        data.dims = glm::ivec3(CHUNK_SIZE + 2);
                        gasChunkData.push_back(data);
                    }
                }
            }
        }

        if (gasChunkData.empty()) {
            return;
        }

        // Calculate matrices
        float aspect = (windowHeight == 0) ? (float)windowWidth : (float)windowWidth / (float)windowHeight;
        glm::vec3 velocity = characterController->getVelocity();
        float speed = glm::sqrt(velocity.x*velocity.x+velocity.y*velocity.y+velocity.z*velocity.z);
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);
        glm::mat4 pv = projection * view;

        // Bind gas FBO
        glBindFramebuffer(GL_FRAMEBUFFER, gasFBO);
        glViewport(0, 0, windowWidth, windowHeight);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gasRayMarchShader->use();
        glm::mat4 invPV = glm::inverse(pv);
        gasRayMarchShader->setMatrix("uInvViewProjection", invPV);
        gasRayMarchShader->setMatrix("uViewProjection", pv);
        gasRayMarchShader->setMatrix("uView", view);
        gasRayMarchShader->setVec3("uCameraPos", cameraPos);
        gasRayMarchShader->setFloat("uVoxelSize", VOXEL_SIZE);
        gasRayMarchShader->setFloat("uNear", 0.1f);
        gasRayMarchShader->setFloat("uFar", 500.0f);
        gasRayMarchShader->setInt("uChunkCount", (int)gasChunkData.size());

        // Upload gas chunk data to SSBO
        static GLuint gasChunkSSBO = 0;
        if (gasChunkSSBO == 0) {
            glGenBuffers(1, &gasChunkSSBO);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gasChunkSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gasChunkData.size() * sizeof(GasChunkData), gasChunkData.data(), GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gasChunkSSBO);

        // Bind main voxel SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, chunkRenderer->ssboGasVoxels);

        // Bind images
        glBindImageTexture(0, gasColorTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glBindImageTexture(1, gasDensityTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R16F);

        // Dispatch at half resolution for performance
        int groupsX = (windowWidth + 7) / 8;
        int groupsY = (windowHeight + 7) / 8;
        glDispatchCompute(groupsX, groupsY, 1);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

        float Game::sampleFluidDensityAtWorld(const glm::vec3 &worldPos) const {
        if (!chunkManager) return -1000.0f;
        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int baseCX = worldToChunk(worldPos.x);
        int baseCY = worldToChunk(worldPos.y);
        int baseCZ = worldToChunk(worldPos.z);
        glm::vec3 chunkMin = glm::vec3(baseCX * chunkWorldSize, baseCY * chunkWorldSize, baseCZ * chunkWorldSize);
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;
        int ix = static_cast<int>(std::floor(local.x));
        int iy = static_cast<int>(std::floor(local.y));
        int iz = static_cast<int>(std::floor(local.z));
        float fx = local.x - ix;
        float fy = local.y - iy;
        float fz = local.z - iz;

        auto sampleCorner = [&](int sx, int sy, int sz)->float {
            glm::vec3 cornerWorld = chunkMin + glm::vec3((float)(ix + sx), (float)(iy + sy), (float)(iz + sz)) * VOXEL_SIZE;
            int cx = worldToChunk(cornerWorld.x);
            int cy = worldToChunk(cornerWorld.y);
            int cz = worldToChunk(cornerWorld.z);
            ChunkCoord coord{cx, cy, cz};
            Chunk* chunk = chunkManager->getChunk(coord);
            if (!chunk) return -1000.0f;
            glm::vec3 localCorner = (cornerWorld - getChunkMin(coord)) / VOXEL_SIZE;
            int lx = glm::clamp((int)std::round(localCorner.x), 0, CHUNK_SIZE);
            int ly = glm::clamp((int)std::round(localCorner.y), 0, CHUNK_SIZE);
            int lz = glm::clamp((int)std::round(localCorner.z), 0, CHUNK_SIZE);
            return chunk->voxels[lx][ly][lz].fluidDensity;
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

     float Game::getGasDensityAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        if (!chunkManager) return 0.0f;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);
        if (!chunk) return 0.0f;

        glm::vec3 chunkMin = glm::vec3(coord.x * chunkWorldSize,
                                       coord.y * chunkWorldSize,
                                       coord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;
        int ix = glm::clamp((int)std::round(local.x), 0, CHUNK_SIZE);
        int iy = glm::clamp((int)std::round(local.y), 0, CHUNK_SIZE);
        int iz = glm::clamp((int)std::round(local.z), 0, CHUNK_SIZE);

        const Voxel& v = chunk->voxels[ix][iy][iz];
        if (v.type == 4 && v.density >= 0.0f) {
            return v.density;
        }
        return 0.0f;
    }

    glm::vec3 Game::getGasColorAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        Chunk* chunk = chunkManager->getChunk({cx, cy, cz});
        if (!chunk) return glm::vec3(0.5f, 0.6f, 0.7f);

        glm::vec3 chunkMin = getChunkMin({cx, cy, cz});
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;
        int ix = glm::clamp((int)std::round(local.x), 0, CHUNK_SIZE);
        int iy = glm::clamp((int)std::round(local.y), 0, CHUNK_SIZE);
        int iz = glm::clamp((int)std::round(local.z), 0, CHUNK_SIZE);

        const Voxel& v = chunk->voxels[ix][iy][iz];
        return (v.type == 4) ? v.color : glm::vec3(0.5f, 0.6f, 0.7f);
    }

    glm::vec3 Game::sampleNormalAtWorld(const glm::vec3 &worldPos) const {
        const float e = VOXEL_SIZE * 0.5f;
        float dx = sampleDensityAtWorld(worldPos + glm::vec3(e,0,0)) - sampleDensityAtWorld(worldPos - glm::vec3(e,0,0));
        float dy = sampleDensityAtWorld(worldPos + glm::vec3(0,e,0)) - sampleDensityAtWorld(worldPos - glm::vec3(0,e,0));
        float dz = sampleDensityAtWorld(worldPos + glm::vec3(0,0,e)) - sampleDensityAtWorld(worldPos - glm::vec3(0,0,e));
        glm::vec3 g(dx,dy,dz);
        float len = glm::length(g);
        if (len < 1e-6f) return glm::vec3(0.0f, 1.0f, 0.0f);
        return -glm::normalize(g);
        }

     uint32_t Game::sampleMaterialAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        if (!chunkManager) return 0;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);
        if (!chunk) return 0;

        glm::vec3 chunkMin = glm::vec3(coord.x * chunkWorldSize,
                                       coord.y * chunkWorldSize,
                                       coord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        int ix = glm::clamp((int)std::round(local.x), 0, CHUNK_SIZE);
        int iy = glm::clamp((int)std::round(local.y), 0, CHUNK_SIZE);
        int iz = glm::clamp((int)std::round(local.z), 0, CHUNK_SIZE);

        return chunk->voxels[ix][iy][iz].material;
    }

    uint8_t Game::sampleTypeAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos) {
        if (!chunkManager) return 0;

        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = static_cast<int>(std::floor(worldPos.x / chunkWorldSize));
        int cy = static_cast<int>(std::floor(worldPos.y / chunkWorldSize));
        int cz = static_cast<int>(std::floor(worldPos.z / chunkWorldSize));

        ChunkCoord coord{cx, cy, cz};
        Chunk* chunk = chunkManager->getChunk(coord);
        if (!chunk) return 0;

        glm::vec3 chunkMin = glm::vec3(coord.x * chunkWorldSize,
                                       coord.y * chunkWorldSize,
                                       coord.z * chunkWorldSize);

        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        int ix = glm::clamp((int)std::round(local.x), 0, CHUNK_SIZE);
        int iy = glm::clamp((int)std::round(local.y), 0, CHUNK_SIZE);
        int iz = glm::clamp((int)std::round(local.z), 0, CHUNK_SIZE);

        return chunk->voxels[ix][iy][iz].type;
    }


    void Game::updateCamera() {
        glm::vec3 targetPos = characterController->getCameraPosition();
        const auto& ccState = characterController->getState();

        glm::vec2 mouseDelta = getMouseDelta();
        if (!paused) {
            float yawAmount   = -mouseDelta.x * cameraSensitivity * settings.sensitivity;
            float pitchAmount = -mouseDelta.y * cameraSensitivity * settings.sensitivity;

            // --- YAW ---
            if (std::abs(yawAmount) > 1e-6f) {
                glm::quat qYaw = glm::angleAxis(glm::radians(yawAmount), cameraUp);
                cameraForward = glm::normalize(qYaw * cameraForward);
                cameraUp = glm::normalize(qYaw * cameraUp);
            }

            cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));

            // --- PITCH ---
            if (std::abs(pitchAmount) > 1e-6f) {
                glm::quat qPitch = glm::angleAxis(glm::radians(pitchAmount), cameraRight);
                glm::vec3 candidateForward = glm::normalize(qPitch * cameraForward);

                float d = glm::dot(candidateForward, cameraUp);
                if (d > -0.95f && d < 0.95f) {
                    cameraForward = candidateForward;
                    cameraUp = glm::normalize(glm::cross(cameraRight, cameraForward));
                }
            }

            // --- ROLL ---
            float rollAmount = 0.0f;
            const float rollSpeedDeg = 120.0f;

            bool canRoll = !characterController->getState().isSurfaceAdhered ||
                           characterController->getState().isInFluid;

            if (canRoll) {
                if (actions["RollLeft"].isPressed)  rollAmount += rollSpeedDeg * deltaTime;
                if (actions["RollRight"].isPressed) rollAmount -= rollSpeedDeg * deltaTime;
            }

            if (std::abs(rollAmount) > 1e-6f) {
                glm::quat qRoll = glm::angleAxis(glm::radians(rollAmount), cameraForward);
                cameraUp = glm::normalize(qRoll * cameraUp);
                cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));
            }

            // --- ALIGNMENT ---
            if (ccState.isSurfaceAdhered && std::abs(rollAmount) < 0.01f) {
                glm::vec3 characterUp = characterController->getUpDirection();
                alignCameraRollToUp(characterUp, deltaTime);
            }
        }

        cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));
        cameraUp = glm::normalize(glm::cross(cameraRight, cameraForward));
        cameraForward = glm::normalize(glm::cross(cameraUp, cameraRight));

        characterController->setCameraForward(cameraForward);
        characterController->setCameraRight(cameraRight);

        // --- CAMERA POSITION ---
        glm::vec3 desiredCameraPos = characterController->getCameraPosition();

        const float cameraDistance = 0.0f;
        if (cameraDistance > 0.0f) {
            desiredCameraPos = targetPos - cameraForward * cameraDistance;
        }

        float eyeRadius = glm::max(VOXEL_SIZE * 0.12f,
                                   characterController->getRadius() * 0.35f);

        characterController->resolveCameraCollision(desiredCameraPos, eyeRadius);

        float smoothTime = characterController->getState().isSurfaceAdhered ? 0.15f : 0.04f;
        float smoothFactor = 1.0f - exp(-deltaTime / smoothTime);
        cameraPos = cameraPos + (desiredCameraPos - cameraPos) * smoothFactor;

        characterController->resolveCameraCollision(cameraPos, eyeRadius);

        glm::vec3 emergencyNormal;
        float emergencyPen;
        if (characterController->checkCameraCollision(cameraPos, eyeRadius, emergencyNormal, emergencyPen)) {
            cameraPos += emergencyNormal * (emergencyPen + VOXEL_SIZE * 0.05f);
        }

        updatePlayerAudio();
    }

    void Game::alignCameraRollToUp(const glm::vec3& worldUp, float deltaTime) {
        glm::vec3 projectedUp = worldUp - cameraForward * glm::dot(worldUp, cameraForward);
        float projectedLen = glm::length(projectedUp);
        if (projectedLen < 0.001f) return;
        projectedUp /= projectedLen;

        float angle = glm::acos(glm::clamp(glm::dot(cameraUp, projectedUp), -1.0f, 1.0f));

        glm::vec3 cross = glm::cross(cameraUp, projectedUp);
        if (glm::dot(cross, cameraForward) < 0.0f) {
            angle = -angle;
        }

        float speed = 8.0f;
        float maxAngle = glm::radians(30.0f);
        float rotation = glm::clamp(angle * speed * deltaTime, -maxAngle, maxAngle);

        if (std::abs(rotation) > 0.001f) {
            glm::quat q = glm::angleAxis(rotation, cameraForward);
            cameraUp = glm::normalize(q * cameraUp);
            cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));
        }
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
        spell.burn.center = spell.center;
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
                                      glm::vec3 color, int material)
    {
        float searchRadius = radiusWorld*4;
        if(material==9)
        {
            searchRadius=glm::pow(radiusWorld,3);
        }
        FormationParams params = FormationParams::Sphere(start, radiusWorld);
        spellSystem->castSphere(
                params.center,
                params.radius,
                material,
                speedWorld,
                glm::normalize(target - start),
                searchRadius,makeVoxelTypeMask({1})
        );
        //castSpellWithFormation(center, searchRadius, material, strength, params);
        //SpellCastRequest req = buildSpellCastRequestSnapshot(start, searchRadius, 0, 5, params);
        //req.physicsEnabled = true;
        //req.launchDir = glm::normalize(target-start);
        //req.launchSpeed = speedWorld * VOXEL_SIZE;
        //req.lifetime = 20.0f;
        //spellCastAsync->enqueueOrReplaceQueued(std::move(req));
    }

    ///Impact Effects::
    void Game::spawnImpactEffect(const glm::vec3& hitPos,
                                 const glm::vec3& hitNormal,
                                 float impactSpeed,
                                 float removedVoxelEstimate,
                                 const glm::vec3& tint)
    {
        std::cout<<impactParticles.size()<<" Particles\n";

        // Normalize impact into 0..1-ish range
        float strength01 = glm::clamp(glm::sqrt(impactSpeed) / (30.0f * VOXEL_SIZE), 0.0f, 1.0f);

        // Let voxel removal contribute too
        float removal01 = glm::clamp(removedVoxelEstimate / 100.0f, 0.0f, 1.0f);

        float combined = glm::clamp(strength01 * 0.7f + removal01 * 0.3f, 0.0f, 1.0f);

        if (strength01 < 0.25f) {
            // LIGHT: flash only (no smoke)
            spawnImpactPresetLarge(hitPos, hitNormal, strength01, tint);
        } else if (strength01 < 0.65f) {
            // MEDIUM: larger flash + light smoke
            spawnImpactPresetLarge(hitPos, hitNormal, strength01, tint);
        } else {
            // STRONG: largest flash + strong smoke
            spawnImpactPresetLarge(hitPos, hitNormal, strength01, tint);
        }
    }

    void Game::spawnImpactPresetSmall(const glm::vec3& hitPos,
                                      const glm::vec3& hitNormal,
                                      float strength01,
                                      const glm::vec3& tint)
    {
        (void)tint; // unused for flash-only small preset

        // LIGHT: flash only, no smoke
        ImpactParticle flash;
        flash.position = hitPos + hitNormal * (0.9f * VOXEL_SIZE);
        flash.velocity = glm::vec3(0.0f);
        flash.color = glm::vec4(1.0f, 0.30f, 0.18f, 0.85f); // red flash
        flash.age = 0.0f;
        flash.lifetime = glm::mix(0.10f, 0.2f, strength01);
        flash.startSize = 2.4f * VOXEL_SIZE;
        flash.endSize = glm::mix(3.4f, 3.4f, strength01) * VOXEL_SIZE;
        flash.rotation = ((float)rand() / (float)RAND_MAX) * glm::two_pi<float>();
        flash.rotationSpeed = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 1.5f;
        flash.kind = 1;
        impactParticles.push_back(flash);
    }

    void Game::spawnImpactPresetMedium(const glm::vec3& hitPos,
                                       const glm::vec3& hitNormal,
                                       float strength01,
                                       const glm::vec3& tint)
    {
        // MEDIUM: light smoke
        const int smokeCount = 8 + (int)(strength01 * 6.0f);

        for (int i = 0; i < smokeCount; ++i) {
            float rx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            float ry = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            float rz = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

            glm::vec3 randDir = glm::normalize(glm::vec3(rx, ry, rz) + hitNormal * 1.7f);

            ImpactParticle p;
            p.position = hitPos + hitNormal * (1.0f * VOXEL_SIZE);
            p.velocity = randDir * glm::mix(1.8f, 4.2f, strength01) * VOXEL_SIZE;
            p.color = glm::vec4(tint * 0.95f, 0.018f); // light smoke alpha
            p.age = 0.0f;
            p.lifetime = glm::mix(0.40f, 0.95f, strength01);
            p.startSize = 0.40f * VOXEL_SIZE;
            p.endSize = glm::mix(1.8f, 3.6f, strength01) * VOXEL_SIZE;
            p.rotation = ((float)rand() / (float)RAND_MAX) * glm::two_pi<float>();
            p.rotationSpeed = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 2.2f;
            p.kind = 0;
            impactParticles.push_back(p);
        }

        // MEDIUM: larger red flash
        ImpactParticle flash;
        flash.position = hitPos + hitNormal * (1.0f * VOXEL_SIZE);
        flash.velocity = glm::vec3(0.0f);
        flash.color = glm::vec4(1.0f, 0.28f, 0.16f, 0.90f); // red flash
        flash.age = 0.0f;
        flash.lifetime = 0.3f;
        flash.startSize = 4.2f * VOXEL_SIZE;
        flash.endSize = 5.2f * VOXEL_SIZE;
        flash.rotation = ((float)rand() / (float)RAND_MAX) * glm::two_pi<float>();
        flash.rotationSpeed = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 1.8f;
        flash.kind = 1;
        impactParticles.push_back(flash);
    }

    void Game::spawnImpactPresetLarge(const glm::vec3& hitPos,
                                      const glm::vec3& hitNormal,
                                      float strength01,
                                      const glm::vec3& tint)
    {
        // STRONG: strong smoke
        const int smokeCount = 5;

        for (int i = 0; i < smokeCount; ++i) {
            float rx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            float ry = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            float rz = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

            glm::vec3 randDir = glm::normalize(glm::vec3(rx, ry, rz) + hitNormal * 2.0f);

            ImpactParticle p;
            p.position = hitPos + hitNormal * (0.6f * VOXEL_SIZE);
            p.velocity = randDir * glm::mix(3.0f, 7.5f, strength01) * VOXEL_SIZE;
            p.color = glm::vec4(tint * 0.9f, 0.05f); // strong smoke alpha
            p.age = 0.0f;
            p.lifetime = glm::mix(1.5f, 4.6f, strength01);
            p.startSize = 3.6f * VOXEL_SIZE;
            p.endSize = glm::mix(20.0f, 100.0f, strength01) * VOXEL_SIZE;
            p.rotation = ((float)rand() / (float)RAND_MAX) * glm::two_pi<float>();
            p.rotationSpeed = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 3.0f;
            p.kind = 0;
            impactParticles.push_back(p);
        }

        // STRONG: biggest red flash
        ImpactParticle flash;
        flash.position = hitPos;
        flash.velocity = glm::vec3(0.0f);
        flash.color = glm::vec4(1.0f, 0.24f, 0.14f, 0.98f); // red flash
        flash.age = 0.0f;
        flash.lifetime = 0.4f;
        flash.startSize = 5.8f * VOXEL_SIZE;
        flash.endSize = 6.8f * VOXEL_SIZE;
        flash.rotation = ((float)rand() / (float)RAND_MAX) * glm::two_pi<float>();
        flash.rotationSpeed = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 4.0f;
        flash.kind = 1;
        impactParticles.push_back(flash);
    }

    glm::vec3 Game::getCameraUp() const {
        return glm::normalize(cameraUp);
    }


    void Game::initSpeedLinesShader() {
        try {
            speedLinesShader = std::make_unique<Shader>(
                    resolveAssetPath("shaders/speedlines.vert"),
                    resolveAssetPath("shaders/speedlines.frag")
            );
            std::cout << "Speed lines shader loaded successfully\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load speed lines shader: " << e.what() << "\n";
            speedLinesShader = nullptr;
        }
    }

    bool Game::renderSpeedLines(
            GLuint inputTexture,
            GLuint destinationFBO)
    {
        if (!speedLinesShader || !enableSpeedLines)
        {
            std::cout<<"no Shader found\n";
            return false;
        }


        glm::vec3 velocity = characterController->getVelocity();

        float speed =
                glm::length(velocity);


        const float maxSpeed =
                characterController->settings.terminalVelocity / 2.5f;


        float normalizedSpeed =
                glm::clamp(speed / maxSpeed,0.0f,1.5f)
                * speedLinesIntensity;


        if(normalizedSpeed < 0.05f)
        {
           return false;
        }

        beginPostProcess(destinationFBO);

        speedLinesShader->use();

        speedLinesShader->setInt("uSceneTexture", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inputTexture);

        speedLinesShader->setVec3("uVelocity3D", velocity);
        speedLinesShader->setFloat("uSpeed", normalizedSpeed);
        speedLinesShader->setFloat("uTime", (float)glfwGetTime());

        float aspect = (float)windowWidth / (float)windowHeight;
        glm::mat4 projection = glm::perspective(glm::radians((45.0f*(1+(speed/(characterController->settings.terminalVelocity/4))))*settings.fov), aspect, nearPlane, farPlane);
        glm::vec3 camUp = getCameraUp();
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), camUp);

        speedLinesShader->setMatrix("uView", view);
        speedLinesShader->setMatrix("uProjection", projection);
        speedLinesShader->setVec3("uCameraPos", cameraPos);

        speedLinesShader->setFloat("uLineCount", 12.0f);
        speedLinesShader->setFloat("uLineWidth", 0.05f);
        speedLinesShader->setFloat("uLineSharpness", 0.95f);
        speedLinesShader->setFloat("uInnerRadius", 0.30f);
        speedLinesShader->setFloat("uOuterFade", 0.50f);
        speedLinesShader->setFloat("uVignetteStrength", 0.4f);
        speedLinesShader->setFloat("uLineOpacity", 0.7f);

        glm::vec3 lineColor = glm::vec3(1.0f, 1.0f, 1.0f);

        if (characterController->getState().isSprinting) {
            lineColor = glm::vec3(0.3f, 0.7f, 1.0f);
            speedLinesShader->setFloat("uLineCount", 22.0f);
            speedLinesShader->setFloat("uLineWidth", 0.075f);
            speedLinesShader->setFloat("uInnerRadius", 0.20f);
            speedLinesShader->setFloat("uOuterFade", 0.50f);
            speedLinesShader->setFloat("uVignetteStrength", 0.4f);
            speedLinesShader->setFloat("uLineOpacity", 0.9f);
        }
        if (characterController->getState().isAirSlamming) {
            lineColor = glm::vec3(1.0f, 0.3f, 0.3f);
            speedLinesShader->setFloat("uLineCount", 18.0f);
            speedLinesShader->setFloat("uLineWidth", 0.09f);
            speedLinesShader->setFloat("uInnerRadius", 0.25f);
            speedLinesShader->setFloat("uOuterFade", 0.50f);
            speedLinesShader->setFloat("uVignetteStrength", 0.4f);
            speedLinesShader->setFloat("uLineOpacity", 0.8f);
        }

        speedLinesShader->setVec3("uLineColor", lineColor);

        // Enable blending for speed lines
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Draw fullscreen triangle
        glBindVertexArray(postVAO);
        glDrawArrays(GL_TRIANGLES,0,3);
        glBindVertexArray(0);


        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        return true;
    }

    bool Game::renderDamageFeedback(GLuint sceneTexture, GLuint destinationFBO)
    {
        if(!damageShader)
        {
            std::cout << "no Damage Shader found\n";
            return false;
        }

        float now = glfwGetTime();

        float intensity = 0.0f;
        glm::vec2 direction(0.0f);

        glm::vec2 accumulatedDirection(0.0f);
        float totalWeight = 0.0f;


        glm::vec3 right =
                characterController->getState().cameraRight;

        glm::vec3 up =
                characterController->getUpDirection();


        for(auto& dmg : playerDamageFeedback)
        {
            float age = now - dmg.time;

            if(age < dmg.duration)
            {
                float fade =
                        1.0f - (age / dmg.duration);


                float weight =
                        dmg.amount* 55 * fade;


                glm::vec3 dir =
                        dmg.worldPosition - cameraPos;


                glm::vec2 screenDir(
                        glm::dot(dir, right),
                        glm::dot(dir, up)
                );


                accumulatedDirection +=
                        screenDir * weight;


                intensity += weight;

                totalWeight += weight;
            }
        }


        playerDamageFeedback.erase(
                std::remove_if(
                        playerDamageFeedback.begin(),
                        playerDamageFeedback.end(),
                        [&](DamageInstance& d)
                        {
                            return now - d.time > d.duration;
                        }),
                playerDamageFeedback.end()
        );


        if(intensity < 0.01f)
        {
            return false;
        }


        if(totalWeight > 0.001f)
        {
            direction =
                    accumulatedDirection / totalWeight;
        }
        else
        {
            direction = glm::vec2(0.0f, 1.0f);
        }


        if (glm::length(direction) > 0.001f)
            direction = glm::normalize(direction);
        else
            direction = glm::vec2(0.0f, 1.0f);



        glBindFramebuffer(GL_FRAMEBUFFER, destinationFBO);
        glViewport(0, 0, windowWidth, windowHeight);

        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glClear(GL_COLOR_BUFFER_BIT);


        damageShader->use();


        glActiveTexture(GL_TEXTURE0);

        glBindTexture(
                GL_TEXTURE_2D,
                sceneTexture
        );

        damageShader->setInt(
                "uSceneTexture",
                0
        );


        damageShader->setFloat(
                "uTime",
                now
        );


        float visualIntensity =
                glm::clamp(
                        intensity / 50.0f,
                        0.0f,
                        1.0f
                );

        damageShader->setFloat(
                "uDamageIntensity",
                visualIntensity
        );


        damageShader->setVec2(
                "uDamageDirection",
                direction
        );


        glBindVertexArray(postVAO);

        glDrawArrays(
                GL_TRIANGLES,
                0,
                3
        );

        glBindVertexArray(0);


        glDepthMask(GL_TRUE);

        return true;
    }

    void Game::createPhysicsMeshData(gl3::PhysicsMeshData& out,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<glm::vec3>& normals,
                                     const std::vector<glm::vec3>& colors,
                                     const std::vector<glm::vec2>& uvs,
                                     const std::vector<uint32_t>& flags)
    {
        if (vertices.empty()) return;

        if (out.vao) { glDeleteVertexArrays(1, &out.vao); out.vao = 0; }
        if (out.vbo) { glDeleteBuffers(1, &out.vbo); out.vbo = 0; }

        out.vertices = vertices;
        out.normals  = normals;
        out.colors   = colors;
        out.vertexCount = vertices.size();
        out.isValid = true;

        struct EnemyVertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec3 color;
            glm::vec2 uv;
            uint32_t flags;
        };

        std::vector<EnemyVertex> packed;
        packed.reserve(vertices.size());

        for (size_t i = 0; i < vertices.size(); ++i) {
            EnemyVertex v{};
            v.pos   = vertices[i];
            v.normal = (i < normals.size()) ? normals[i] : glm::vec3(0,1,0);
            v.color  = (i < colors.size())  ? colors[i]  : glm::vec3(1,0,1);
            v.uv     = (i < uvs.size())     ? uvs[i]     : glm::vec2(0.0f);
            v.flags  = (i < flags.size())   ? flags[i]   : (7u << 1u);
            packed.push_back(v);
        }

        glGenVertexArrays(1, &out.vao);
        glGenBuffers(1, &out.vbo);

        glBindVertexArray(out.vao);
        glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
        glBufferData(GL_ARRAY_BUFFER, packed.size() * sizeof(EnemyVertex), packed.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(EnemyVertex), (void*)offsetof(EnemyVertex, pos));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(EnemyVertex), (void*)offsetof(EnemyVertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(EnemyVertex), (void*)offsetof(EnemyVertex, color));

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(EnemyVertex), (void*)offsetof(EnemyVertex, uv));

        glEnableVertexAttribArray(4);
        glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, sizeof(EnemyVertex), (void*)offsetof(EnemyVertex, flags));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void Game::appendSpellBillboards(std::vector<SunInstance>& out)
    {
        for (const auto& spell : spellSystem->spells()) {
            if (!spell.isPhysicsEnabled || !spell.physicsBody) continue;

            const auto& vol = spell.destruct.volume;
            const glm::vec3 bodyPos = spell.physicsBody->position;
            const glm::vec3 centerLocal = spell.destruct.localCenterOffsetWorld;

            glm::vec3 sumPos(0.0f);
            glm::vec3 sumColor(0.0f);
            int count = 0;

            for (int z = 0; z < vol.dims.z; ++z)
                for (int y = 0; y < vol.dims.y; ++y)
                    for (int x = 0; x < vol.dims.x; ++x) {
                        const auto& c = vol.at(x,y,z);

                        if (c.density < 0.0f) continue;
                        if (c.material != 9u && c.type != 2u) continue;

                        glm::vec3 localP = glm::vec3(x,y,z) * vol.voxelSize - centerLocal;
                        glm::vec3 worldP = bodyPos + localP; // add rotation later if needed

                        sumPos += worldP;
                        sumColor += c.color;
                        ++count;
                    }

            if (count > 0) {
                SunInstance inst;
                inst.position = sumPos / float(count);
                inst.color = sumColor / float(count);
                inst.scale = std::sqrt((float)count) * vol.voxelSize * 0.75f;
                out.push_back(inst);
            }
        }
    }

    CollisionDecision Game::decideCollisionResponse(
            const gl3::VoxelPhysicsBody& self,
            const gl3::VoxelPhysicsBody* other,
            const glm::vec3& hitPos,
            const glm::vec3& hitNormal,
            float impactSpeed) const
    {
        (void)hitPos; (void)hitNormal; (void)impactSpeed;

        CollisionDecision d{};
        const auto& rule = materialRules[self.material];

        if (!other && !rule.collideWorld) { d.ignoreCollision = true; return d; }
        if ( other && !rule.collideBodies) { d.ignoreCollision = true; return d; }

        switch (rule.mode) {
            case MaterialCollisionMode::StickOnWorld:
                if (!other) { d.resolvePhysics = false; d.stick = true; }
                break;

            case MaterialCollisionMode::StickOnBody:
                if (other) { d.resolvePhysics = false; d.stick = true; }
                break;

            case MaterialCollisionMode::StickOnAll:
                d.resolvePhysics = false; d.stick = true;
                break;

            case MaterialCollisionMode::PassThroughFirstBody:
                if (other && self.bodiesCanPassThrough > 0) {
                    d.ignoreCollision = true;
                    d.keepFlying = true;
                    d.resolvePhysics = false;
                }
                break;

            case MaterialCollisionMode::CollidePlayerOnly:
                d.ignoreCollision = true;
                break;

            case MaterialCollisionMode::DestroyTargetKeepFlying:
                if (!other) {
                    d.resolvePhysics = false;
                    d.keepFlying = true;
                    d.destroyWorld = true;
                    d.destroyWorldSolidOnly = true;
                    d.worldDestroyType = 0;
                    d.worldDestroyRadius = 2;
                } else {
                    d.destroyOther = true;
                    d.keepFlying = true;
                    d.resolvePhysics = false;
                }
                break;

            default:
                break;
        }

        // layered behavior
        if (rule.convertOnStick && d.stick) {
            if (other) d.convertOtherBody = true;
            else       d.convertWorld = true;
        }

        return d;
    }

    void Game::convertWorldToMaterial(const glm::vec3& center, float radius, uint32_t material)
    {
        if (!chunkManager) return;

        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];

                                // Make voxel solid
                                v.type = 1;
                                v.material = material;

                                v.density = glm::max(v.density, 1.0f);

                                any = true;
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
    }

    int Game::consumeWorldOfMaterial(const glm::vec3& center, float radius, uint32_t material)
    {
        if (!chunkManager) return 0;

        int amount = 0;
        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];
                                if (!v.isSolid()) continue;

                                v.type = 0;
                                v.material=0;
                                v.density = -1.0;

                                any = true;
                                amount++;
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
        return amount;
    }

    void Game::convertSolidWorldToMaterial(const glm::vec3& center, float radius, uint32_t material)
    {
        if (!chunkManager) return;

        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];

                                // only convert existing solid/active voxels
                                if (v.type > 0 && v.isSolid()) {
                                    v.material = material;
                                    any = true;
                                }
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
    }

    void Game::convertWorldToType(const glm::vec3& center, float radius, uint32_t type, float strength)
    {
        if (!chunkManager) return;

        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        // optional use of strength: if <=0, do nothing
        if (strength <= 0.0f) return;

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];

                                // convert all voxels in sphere (world-to-type)
                                v.type = static_cast<uint8_t>(type);

                                // keep density coherent if setting to empty
                                if (v.type == 0) {
                                    v.density = glm::min(v.density, -1.0f);
                                }
                                any = true;
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
    }

    void Game::convertSolidWorldToType(const glm::vec3& center, float radius, uint32_t type)
    {
        if (!chunkManager) return;

        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];

                                if (v.isSolid()) {
                                    v.type = static_cast<uint8_t>(type);
                                    if (v.type == 0) {
                                        v.density = glm::min(v.density, -1.0f);
                                    }
                                    any = true;
                                }
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
    }

    int Game::mineSolidWorld(const glm::vec3& center, float radius)
    {
        if (!chunkManager) return 0;

        const float r2 = radius * radius;

        const int minCX = worldToChunk(center.x - radius);
        const int maxCX = worldToChunk(center.x + radius);
        const int minCY = worldToChunk(center.y - radius);
        const int maxCY = worldToChunk(center.y + radius);
        const int minCZ = worldToChunk(center.z - radius);
        const int maxCZ = worldToChunk(center.z + radius);

        int collected = 0;

        std::vector<ChunkCoord> touched;
        touched.reserve(64);

        for (int cx = minCX; cx <= maxCX; ++cx)
            for (int cy = minCY; cy <= maxCY; ++cy)
                for (int cz = minCZ; cz <= maxCZ; ++cz)
                {
                    ChunkCoord cc{cx,cy,cz};
                    Chunk* chunk = chunkManager->getChunk(cc);
                    if (!chunk) continue;

                    const glm::vec3 cmin = getChunkMin(cc);
                    bool any = false;

                    for (int vx = 0; vx <= CHUNK_SIZE; ++vx) {
                        const float wx = cmin.x + vx * VOXEL_SIZE;
                        const float dx = wx - center.x;
                        const float dx2 = dx * dx;

                        for (int vy = 0; vy <= CHUNK_SIZE; ++vy) {
                            const float wy = cmin.y + vy * VOXEL_SIZE;
                            const float dy = wy - center.y;
                            const float dy2 = dy * dy;
                            if (dx2 + dy2 > r2) continue;

                            for (int vz = 0; vz <= CHUNK_SIZE; ++vz) {
                                const float wz = cmin.z + vz * VOXEL_SIZE;
                                const float dz = wz - center.z;
                                const float d2 = dx2 + dy2 + dz * dz;
                                if (d2 > r2) continue;

                                Voxel& v = chunk->voxels[vx][vy][vz];

                                if (v.isSolid()) {
                                    v.type = static_cast<uint8_t>(0.0f);
                                    if (v.type == 0) {
                                        v.density = glm::min(v.density, -1.0f);
                                    }
                                    any = true;
                                    if(v.material==waveManager.materialToMine)
                                    {
                                        collected++;
                                    }
                                }
                            }
                        }
                    }

                    if (any) {
                        chunk->meshDirty = true;
                        chunk->lightingDirty = true;
                        touched.push_back(cc);
                    }
                }

        for (const auto& c : touched) {
            markChunkModified(c);
        }
        return collected;
    }

    void Game::applyDisplaySettings()
    {
        GLFWwindow* win = getWindow();
        if (!win) return;

        const auto& res = commonResolutions[std::clamp(
                settings.resolutionIndex, 0, (int)commonResolutions.size() - 1)];

        GLFWmonitor* primary = glfwGetPrimaryMonitor();
        const GLFWvidmode* vm = glfwGetVideoMode(primary);

        switch (settings.displayMode)
        {
            case DisplayMode::Fullscreen:
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_TRUE);
                glfwSetWindowMonitor(win, primary, 0, 0, res.w, res.h, vm ? vm->refreshRate : GLFW_DONT_CARE);
                break;

            case DisplayMode::Windowed:
                glfwSetWindowMonitor(win, nullptr, 100, 100, res.w, res.h, GLFW_DONT_CARE);
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_TRUE);
                break;

            case DisplayMode::Borderless:
                // Borderless window at monitor native size
                glfwSetWindowMonitor(win, nullptr, 0, 0, vm->width, vm->height, GLFW_DONT_CARE);
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_FALSE);
                break;
        }

        initPostFBO();
        initFluidFBO();
        initCompositeFBO();
        initPostProcessBuffers();
    }

    int Game::findBestResolutionIndexForMonitor(GLFWmonitor* monitor) const
    {
        if (commonResolutions.empty()) return 0;
        if (!monitor) monitor = glfwGetPrimaryMonitor();
        if (!monitor) return 0;

        const GLFWvidmode* vm = glfwGetVideoMode(monitor);
        if (!vm) return 0;

        const int targetW = vm->width;
        const int targetH = vm->height;
        const float targetAspect = (targetH > 0) ? (float)targetW / (float)targetH : 16.0f / 9.0f;

        int bestIdx = 0;
        long long bestScore = std::numeric_limits<long long>::max();

        for (int i = 0; i < (int)commonResolutions.size(); ++i) {
            const auto& r = commonResolutions[i];

            const int dw = r.w - targetW;
            const int dh = r.h - targetH;
            const long long pixelDelta = 1LL * dw * dw + 1LL * dh * dh;

            const float aspect = (r.h > 0) ? (float)r.w / (float)r.h : targetAspect;
            const float aspectDiff = std::abs(aspect - targetAspect);

            // Weight aspect mismatch strongly so ultrawide chooses ultrawide entries
            const long long aspectPenalty = (long long)(aspectDiff * 1000000.0f);

            const long long score = pixelDelta + aspectPenalty;
            if (score < bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    void Game::pickResolutionFromNativeMonitor(bool applyNow)
    {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        settings.resolutionIndex = findBestResolutionIndexForMonitor(monitor);

        if (applyNow) {
            applyDisplaySettings();
        }
    }

    void Game::applyAudioSettings()
    {
        g_SoundManager.setSFXVolume(settings.sfxVolume);
        g_SoundManager.setMusicVolume(settings.musicVolume);
        g_SoundManager.setMasterVolume(settings.masterVolume);
    }

    void Game::updatePlayerAudio() {
        g_SoundManager.set3dListenerPosition(cameraPos);

        // Update listener parameters (position, velocity, forward, up)
        g_SoundManager.set3dListenerParameters(
                cameraPos,
                characterController->getVelocity(),
                getCameraFront(),
                getCameraUp()
        );

        // Update 3D audio
        g_SoundManager.update3dAudio();
    }

    void Game::applyMaterial9BurnAlongSegment(const glm::vec3& from, const glm::vec3& to, float radius)
    {
        if (!chunkManager) return;

        glm::vec3 d = to - from;
        float len = glm::length(d);
        if (len < 1e-4f) {
            convertSolidWorldToType(from, radius, 0);
            return;
        }

        glm::vec3 dir = d / len;

        // step <= half voxel for contiguous burn tunnel
        const float step = glm::max(VOXEL_SIZE * 0.5f, radius * 0.35f);
        int n = glm::max(1, (int)std::ceil(len / step));

        for (int i = 0; i <= n; ++i) {
            float t = (float)i / (float)n;
            glm::vec3 p = from + dir * (len * t);
            convertSolidWorldToType(p, radius, 0);
        }
    }

    void Game::recomputeSpellDerivedStats(SpellPreset& p)
    {
        float sizeFactor = (p.form == CraftForm::Sphere)
                           ? p.radius
                           : (p.width * 0.35f + p.height * 0.35f + p.thickness * 0.30f);

        float materialFactor = 1.0f;
        switch (p.material) {
            case CraftMaterial::Rock:  materialFactor = 1.0f; break;
            case CraftMaterial::Flesh: materialFactor = 0.9f; break;
            case CraftMaterial::Lava:  materialFactor = 1.4f; break;
        }

        float typeFactor = (p.spellType == CraftSpellType::Projectile) ? 1.15f : 0.95f;
        float rangeFactor = glm::clamp(p.range / 35.0f, 0.5f, 2.0f);

        p.cooldown = glm::clamp(0.35f + sizeFactor * 0.22f * materialFactor * typeFactor, 0.2f, 8.0f);
        p.materialCost = glm::clamp(4.0f + sizeFactor * 6.0f * materialFactor * rangeFactor, 1.0f, 999.0f);
    }

    bool Game::isPointInsideFluid(const glm::vec3& worldPos) const
    {
        return sampleFluidDensityAtWorld(worldPos) >= 0.0f;
    }

    void Game::initFluidFBO()
    {
        if (fluidFBO) {
            glDeleteFramebuffers(1, &fluidFBO);
            glDeleteTextures(1, &fluidColorTex);
            glDeleteTextures(1, &fluidDepthTex);
            glDeleteTextures(1, &fluidThicknessTex);
            fluidFBO = fluidColorTex = fluidDepthTex = fluidThicknessTex = 0;
        }

        glGenFramebuffers(1, &fluidFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, fluidFBO);

        glGenTextures(1, &fluidColorTex);
        glBindTexture(GL_TEXTURE_2D, fluidColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fluidColorTex, 0);

        glGenTextures(1, &fluidThicknessTex);
        glBindTexture(GL_TEXTURE_2D, fluidThicknessTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, windowWidth, windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, fluidThicknessTex, 0);

        glGenTextures(1, &fluidDepthTex);
        glBindTexture(GL_TEXTURE_2D, fluidDepthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fluidDepthTex, 0);

        GLenum bufs[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, bufs);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "fluidFBO incomplete!\n";
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Game::initGasFBO() {
        if (gasFBO) {
            glDeleteFramebuffers(1, &gasFBO);
            glDeleteTextures(1, &gasColorTex);
            glDeleteTextures(1, &gasDepthTex);
            glDeleteTextures(1, &gasDensityTex);
            gasFBO = gasColorTex = gasDepthTex = gasDensityTex = 0;
        }

        glGenFramebuffers(1, &gasFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, gasFBO);

        // Color texture (gas color)
        glGenTextures(1, &gasColorTex);
        glBindTexture(GL_TEXTURE_2D, gasColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gasColorTex, 0);

        // Density texture (for thickness)
        glGenTextures(1, &gasDensityTex);
        glBindTexture(GL_TEXTURE_2D, gasDensityTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, windowWidth, windowHeight, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gasDensityTex, 0);

        // Depth texture
        glGenTextures(1, &gasDepthTex);
        glBindTexture(GL_TEXTURE_2D, gasDepthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gasDepthTex, 0);

        GLenum bufs[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, bufs);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "gasFBO incomplete!\n";
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    glm::vec3 Game::sampleFluidColorAtWorld(const glm::vec3& worldPos) const
    {
        const float chunkWorldSize = CHUNK_SIZE * VOXEL_SIZE;
        int cx = (int)std::floor(worldPos.x / chunkWorldSize);
        int cy = (int)std::floor(worldPos.y / chunkWorldSize);
        int cz = (int)std::floor(worldPos.z / chunkWorldSize);

        Chunk* chunk = chunkManager->getChunk({cx, cy, cz});
        if (!chunk) return glm::vec3(0.1f, 0.3f, 0.8f);

        glm::vec3 chunkMin = getChunkMin({cx,cy,cz});
        glm::vec3 local = (worldPos - chunkMin) / VOXEL_SIZE;

        int ix = glm::clamp((int)std::round(local.x), 0, CHUNK_SIZE);
        int iy = glm::clamp((int)std::round(local.y), 0, CHUNK_SIZE);
        int iz = glm::clamp((int)std::round(local.z), 0, CHUNK_SIZE);

        const Voxel& v = chunk->voxels[ix][iy][iz];
        return v.hasFluid() ? v.color : glm::vec3(0.1f, 0.3f, 0.8f);
    }

    void Game::initCompositeFBO() {
        if (compositeFBO) {
            glDeleteFramebuffers(1, &compositeFBO);
            glDeleteTextures(1, &compositeColorTex);
        }

        glGenFramebuffers(1, &compositeFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, compositeFBO);

        glGenTextures(1, &compositeColorTex);
        glBindTexture(GL_TEXTURE_2D, compositeColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, compositeColorTex, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void Game::renderTextureToScreen(GLuint textureID)
    {
        static Shader* screenShader = nullptr;
        if (!screenShader) {
            screenShader = new Shader(
                    "shaders/post_fullscreen.vert",
                    "shaders/screen_quad.frag"  // Simple shader that just samples a texture
            );
        }

        screenShader->use();
        screenShader->setInt("uTexture", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glBindVertexArray(postVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
    }

    const char* Game::materialToString(uint32_t material, uint8_t type)
    {
        if(type==1) {
            switch (material) {
                case 0:
                    return "Stone";
                case 1:
                    return "Earth";
                case 2:
                    return "Sand";
                case 3:
                    return "Shatterstone";
                case 4:
                    return "Metal";
                case 5:
                    return "Ice";
                case 6:
                    return "Crystal";
                case 7:
                    return "Flesh";
                case 8:
                    return "Eye";
                case 9:
                    return "Magma";
                default:
                    return "Unknown";
            }
        } else if(type==2)
        {
            switch (material) {
                case 0:
                    return "Cement";
                case 1:
                    return "Mud";
                case 2:
                    return "Quicksand";
                case 3:
                    return "Splinter Gel";
                case 4:
                    return "Molten Metal";
                case 5:
                    return "Water";
                case 6:
                    return "Liquid Crystal";
                case 7:
                    return "Blood";
                case 8:
                    return "Eye";
                case 9:
                    return "Magma";
                default:
                    return "Unknown";
            }
        }
        return "Empty";
    }
}
