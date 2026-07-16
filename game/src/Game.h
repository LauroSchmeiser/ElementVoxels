#pragma once
#include <memory>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <soloud.h>
#include <soloud_wav.h>
#include <string>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include "rendering/VoxelStructures.h"
#include "rendering/SunBillboard.h"
#include "rendering/VoxelStructures.h"
#include "rendering/Chunk.h"
#include "../robin_hood.h"
#include "rendering/FixedGridChunkManager.h"
#include "Input/InputActionMap.h"
#include "CharacterController.h"
#include "physics/VoxelPhysicsManager.h"
#include "physics/VoxelCarver.h"
#include "physics/FastCraterCarver.h"
#include "physics/CraterStampBatch.h"
#include <mutex>
#include <memory>
#include "spells/SpellCastAsync.h"
#include "rendering/MaterialSystem.h"
#include "../../extern/stb_image.h"
#include "SceneManager.h"
#include "ui/ImGuiLayer.h"
#include "entities/EnemyManager.h"
#include "physics/DestructibleObject.h"
#include "spells/SpellEffect.h"
#include "rendering/GpuStructsStd430.h"
#include "spells/SpellSystem.h"
#include "MainThreadDispatcher.h"
#include "rendering/ChunkRenderer.h"
#include "Entities/WaveManager.h"
#include "physics/MaterialCollisionPolicy.h"

#undef NEAR
#undef FAR

namespace gl3 {
    enum class SceneId : uint8_t;
    class SceneManager;
    class EnemyManager;

    class Game {
    public:
        ////basics (public):
        Game(int width, int height, const std::string &title);

        virtual ~Game();

        void run();

        GLFWwindow *getWindow() { return window; }

        int getWindowWidth() {return windowWidth;}
        int getWindowHeight() {return windowHeight;}


        gl3::ImGuiLayer& imgui() { return imguiLayer; }

        gl3::ImGuiLayer imguiLayer;
        glm::vec3 getCameraFront() const;

        void createPhysicsMeshData(PhysicsMeshData &out,
                                   const std::vector<glm::vec3> &vertices,
                                   const std::vector<glm::vec3> &normals,
                                   const std::vector<glm::vec3> &colors,
                                   const std::vector<glm::vec2> &uvs,
                                   const std::vector<uint32_t> &flags);

    private:
        ////basics (private):
        static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

        ////Chunk Management:
        //std::unique_ptr<ChunkManager> chunkManager = std::make_unique<ChunkManager>(); // other manager (old)
        std::unique_ptr<FixedGridChunkManager> chunkManager;
        std::unique_ptr<ChunkRenderer> chunkRenderer;


        bool hasSolidVoxels(const gl3::Chunk &chunk);

    public:
        static int worldToChunk(float worldPos);

    private:
        void markChunkModified(const ChunkCoord &coord);

        void unloadChunk(const ChunkCoord &coord);

        glm::vec3 getChunkMin(const ChunkCoord &coord) const;

        glm::vec3 getChunkMax(const ChunkCoord &coord) const;

        bool isChunkVisible(const ChunkCoord &coord) const;

        MainThreadDispatcher mainDispatcher;


        ////Structs::

        struct Particle {
            glm::vec3 position;
            glm::vec3 velocity;
            float lifetime;
            float radius;     // metaball influence
            unsigned int type;       // 0=water,1=fire,2=smoke...
        };

        // --- Generate planet transforms ---
        struct WorldPlanet {
            glm::vec3 worldPos;   // world-space center
            float radius;         // world-space radius
            glm::vec3 color;
            int type=0; // 1=rock, 2=lava, 3=water
            int material=0;
        };

        struct Plane {
            glm::vec3 normal;
            float distance;

            Plane() : normal(0, 0, 0), distance(0) {}

            Plane(const glm::vec3 &n, const glm::vec3 &point) {
                normal = glm::normalize(n);
                distance = glm::dot(normal, point);
            }

            float distanceToPoint(const glm::vec3 &point) const {
                return glm::dot(normal, point) - distance;
            }
        };

        struct RayCastResult {
            glm::vec3 hitPosition;
            glm::vec3 hitNormal;
            float distance;
            bool hit;
        };

        struct ImpactInstanceGPU {
            glm::vec4 pos_size;      // xyz + size
            glm::vec4 color;         // rgba
            glm::vec4 rot_life_kind; // x=rotation, y=life01, z=kind, w=unused
        };

        struct Frustum {
            Plane planes[6]; // left, right, bottom, top, near, far

            enum PlaneSide {
                LEFT = 0,
                RIGHT = 1,
                BOTTOM = 2,
                TOP = 3,
                NEAR = 4,
                FAR = 5
            };

            // Extract frustum planes from view-projection matrix
            void extractFromMatrix(const glm::mat4 &mvp) {
                // Left plane
                planes[LEFT].normal.x = mvp[0][3] + mvp[0][0];
                planes[LEFT].normal.y = mvp[1][3] + mvp[1][0];
                planes[LEFT].normal.z = mvp[2][3] + mvp[2][0];
                planes[LEFT].distance = mvp[3][3] + mvp[3][0];

                // Right plane
                planes[RIGHT].normal.x = mvp[0][3] - mvp[0][0];
                planes[RIGHT].normal.y = mvp[1][3] - mvp[1][0];
                planes[RIGHT].normal.z = mvp[2][3] - mvp[2][0];
                planes[RIGHT].distance = mvp[3][3] - mvp[3][0];

                // Bottom plane
                planes[BOTTOM].normal.x = mvp[0][3] + mvp[0][1];
                planes[BOTTOM].normal.y = mvp[1][3] + mvp[1][1];
                planes[BOTTOM].normal.z = mvp[2][3] + mvp[2][1];
                planes[BOTTOM].distance = mvp[3][3] + mvp[3][1];

                // Top plane
                planes[TOP].normal.x = mvp[0][3] - mvp[0][1];
                planes[TOP].normal.y = mvp[1][3] - mvp[1][1];
                planes[TOP].normal.z = mvp[2][3] - mvp[2][1];
                planes[TOP].distance = mvp[3][3] - mvp[3][1];

                // Near plane
                planes[NEAR].normal.x = mvp[0][3] + mvp[0][2];
                planes[NEAR].normal.y = mvp[1][3] + mvp[1][2];
                planes[NEAR].normal.z = mvp[2][3] + mvp[2][2];
                planes[NEAR].distance = mvp[3][3] + mvp[3][2];

                // Far plane
                planes[FAR].normal.x = mvp[0][3] - mvp[0][2];
                planes[FAR].normal.y = mvp[1][3] - mvp[1][2];
                planes[FAR].normal.z = mvp[2][3] - mvp[2][2];
                planes[FAR].distance = mvp[3][3] - mvp[3][2];

                // Normalize all planes
                for (int i = 0; i < 6; i++) {
                    float length = glm::length(planes[i].normal);
                    planes[i].normal /= length;
                    planes[i].distance /= length;
                }
            }

            // Test if an AABB is inside the frustum
            bool isAABBVisible(const glm::vec3 &min, const glm::vec3 &max) const {
                for (int i = 0; i < 6; i++) {
                    const Plane &plane = planes[i];

                    // Find the positive vertex (farthest in the direction of the plane normal)
                    glm::vec3 positiveVertex = min;
                    if (plane.normal.x >= 0) positiveVertex.x = max.x;
                    if (plane.normal.y >= 0) positiveVertex.y = max.y;
                    if (plane.normal.z >= 0) positiveVertex.z = max.z;

                    // If the positive vertex is behind the plane, the AABB is not visible
                    if (plane.distanceToPoint(positiveVertex) < 0) {
                        return false;
                    }
                }
                return true;
            }

            // Test if a sphere is inside the frustum
            bool isSphereVisible(const glm::vec3 &center, float radius) const {
                for (int i = 0; i < 6; i++) {
                    const Plane &plane = planes[i];
                    float distance = plane.distanceToPoint(center);
                    if (distance < -radius) {
                        return false;
                    }
                }
                return true;
            }
        };


        //// Spell System
        std::shared_ptr<SpellSystem> spellSystem;
        std::mutex spellApplyMutex;

        // Stable id logging for animated voxels
        std::unordered_map<uint64_t, size_t> animatedVoxelIndexMap;
        uint64_t nextAnimatedVoxelID = 1;

        void createCraterAtPosition(const glm::vec3& worldPos, float impactFactor, float spellRadius);

        void Game::onSpellCollision(gl3::VoxelPhysicsBody* body,
                                    const glm::vec3& hitPos,
                                    const glm::vec3& hitNormal,
                                    float impactSpeed);

        void onBodyBodyCollision(gl3::VoxelPhysicsBody* bodyA,
                                 gl3::VoxelPhysicsBody* bodyB,
                                 const glm::vec3& hitPos,
                                 const glm::vec3& hitNormal,
                                 float impactSpeed);
        void onPlayerBodyCollision(gl3::VoxelPhysicsBody* body,
                                   const glm::vec3& hitPos,
                                   const glm::vec3& hitNormal,
                                   float playerSpeed);

        int estimateAvailableVoxels(const glm::vec3& center, float radius, uint64_t targetMaterial, int maxNeeded);

        static float burn01(float t, float duration);
        void startSpellBurn(gl3::SpellEffect& spell, float radiusWorld, float durationSec);
        void startChunkBurn(gl3::Chunk* chunk, const glm::vec3& chunkCenterWorld, float radiusWorld, float durationSec);
        void updateChunkBurns(float dt);
        bool isChunkMeshTooSmall(const gl3::Chunk& c, uint32_t vtxThreshold);
        static inline float smooth01(float x) {
            x = glm::clamp(x, 0.0f, 1.0f);
            return x * x * (3.0f - 2.0f * x); // smoothstep(0,1,x)
        }

// Make burn visually clearer: spend more time "not burned", then ramp.
        static inline float burnCurve(float u01) {
            // 0..1 -> hold, then ramp:
            // tweak 0.25 to "delay" start of dissolve, makes it clearer.
            float t = glm::clamp((u01 - 0.25f) / (1.0f - 0.25f), 0.0f, 1.0f);
            return smooth01(t);
        }

            // Precomputed sphere meshes at different LODs
        struct SphereMesh {
            std::vector<glm::vec3> vertices;
            std::vector<glm::vec3> normals;
            std::vector<uint32_t> indices;
            float radius;
        };

        std::unordered_map<int, SphereMesh> sphereMeshCache; // radius -> mesh

        void initSphereMeshCache();
        SphereMesh generateIcosphere(float radius, int subdivisions);


        ////Debugging:
        void DisplayFPSCount();

        bool DebugMode1 = false;
        bool DebugMode2 = false;
        int activeDebugMode = 0;

        ////Initialization-Steps
        void setupInput();

        void setupCamera();

        void generateChunks();

        void setupVEffects();

        void setupControls();

        void setupPhysics();

        ////Simulation-Steps
        //Physics:
        void updatePhysics();

        void updateDeltaTime();

        //Lighting:
        void rebuildChunkLights(const ChunkCoord &coord);

        uint32_t makeLightID(int cx, int cy, int cz);


        ////Input-Steps
        void update();


        ////Post-Prod Steps?


        ////Rendering-Steps::
        //General Rendering:
    public:
        void renderSkybox();

    private:
        void renderChunks();
        void renderAnimatedVoxels();
        void renderPhysicsFormations();
        void renderEnemies();
        void renderSpellPreview();

        //vertex/frag Shader:
        void ensurePreviewCube();
        void ensurePreviewSphereMesh();

        ////Basic-variables:
        GLFWwindow *window = nullptr;
        int windowWidth = 800, windowHeight = 600;

        Frustum currentFrustum;


        ////Simultation-Variables:
        //Physics-Variables:
        float lastFrameTime = 1.0f / 60;
        float deltaTime = 1.0f / 60;
        float accumulator = 0.1f;
        std::vector<WorldPlanet> CollisionEntities;
        std::vector<Particle> particles;
        int maxParticles = 100;

        //Lighting-Variables:
        const int MAX_LIGHTS = 4; // has to match marching cubes shader
        const float LIGHT_RADIUS = 300.0f * CHUNK_SIZE*VOXEL_SIZE*2;
        uint64_t frameCounter = 29; // Frame counter for light update staggering
        const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;
        std::vector<const gl3::VoxelLight *> flatEmissiveLightList;
        robin_hood::unordered_map<ChunkCoord, std::vector<VoxelLight *>, ChunkCoordHash> lightSpatialHash;

        std::vector<gl3::VoxelLight> mergedEmissiveLightPool;
        static constexpr int LIGHT_UPDATE_INTERVAL = 60; // Update lights every 15 frames

        //std::unique_ptr<SpellPhysicsManager> spellPhysics;
        std::unique_ptr<VoxelPhysicsManager> voxelPhysics;


        ////Camera-Variables:
        glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
        glm::vec2 cameraRotation = glm::vec2(-90.0f, 0.0f); // pitch, yaw


        ////Rendering-Variables:
        //Background-Variables:
        GLuint skyboxVAO=0;
        GLuint skyboxVBO=0;
        GLuint cubemapTexture = 0; // This replaces your noiseTexture for the background

        // Skybox baking
        GLuint nebulaCubemap = 0;
        GLuint skyboxBakeFBO = 0;
        GLuint skyboxBakeRBO = 0;

        bool skyboxBaked = false;

        // Preview proxy geometry (simple cube fallback)
        GLuint previewCubeVAO = 0;
        GLuint previewCubeVBO = 0;

        GLuint previewSphereVAO = 0;
        GLuint previewSphereVBO = 0;
        GLuint previewSphereEBO = 0;
        GLsizei previewSphereIndexCount = 0;

        // World-Variables:
        const int DIM = CHUNK_SIZE+2; //Chunk Size with a bit off padding for marching cubes
        size_t voxelCount = DIM * DIM * DIM; //How many voxels can be in one Chunk
        static constexpr int RenderingRange = 25; //Range around Camera that is rendered

        size_t CHUNK_MAX_VERTS = 0;           // computed once from DIM

        //vEffects
        SunBillboard sunBillboards;
        std::vector<SunInstance> emissiveBillboards;
        int emissiveUpdateCounter=0;
        // Billboard cache control
        static constexpr int BILLBOARD_REFRESH_INTERVAL = 240;
        bool emissiveBillboardsDirty = true;

        void refreshMergedEmissiveBillboards();

        // Post-processing FBO
        GLuint postFBO = 0;
        GLuint postColorTex = 0;   // RGBA16F scene color
        GLuint postDepthTex = 0;   // DEPTH_COMPONENT24/32F scene depth

        // Fullscreen quad/triangle
        GLuint postVAO = 0;
        GLuint postVBO = 0;

        std::unique_ptr<Shader> postShader;

        void initPostFBO();
        void CreateFullscreenTriangle(GLuint& vao, GLuint& vbo);

        //Materials:
        gl3::MaterialSystem materials;
        GLuint materialAlbedoArrayTexId = 0;
        GLuint materialNormalArrayTexId = 0;
        GLuint materialRoughArrayTexId = 0;
        GLuint materialAOArrayTexId = 0;
        GLuint materialHeightArrayTexId = 0;

        void fillMaterialTable();

        std::array<float, 64> rough{};
        std::array<float, 64> spec{};
        std::array<float, 64> uvScale{};

        ////Input
        std::unique_ptr<CharacterController> characterController;
        InputManager input;
        InputActionMap actions;


        ////Shader:
        std::unique_ptr<Shader> skyboxShader;
        std::unique_ptr<Shader> skyboxNebulaBakeShader;
        std::unique_ptr<Shader> skyboxRuntimeShader;
        std::unique_ptr<Shader> voxelShader;
        std::unique_ptr<Shader> marchingCubesShader;
        std::unique_ptr<Shader> voxelSplatShader;
        std::unique_ptr<Shader> spellPreviewShader;
        std::unique_ptr<Shader> impactShader;


    public:
        ///Scenes
        void requestSceneChange(SceneId id) { sceneManager.requestChange(id); }
        bool isGameplayInitialized() const { return gameplayInitialized; }
        float tickGameplayPreload();
        const std::string& getGameplayPreloadStageName() const { return preloadStageName; }

        void updateGameplayFrame();
        void renderGameplayFrame();

    private:

        SceneManager sceneManager{*this};

        bool gameplayInitialized = false;

        enum class PreloadStage : uint8_t {
            NotStarted,

            // immutable
            Boot_Skybox,
            Boot_Nebula,
            Boot_PostProcessor,
            Boot_SSBOs,
            Boot_Materials,
            Boot_Assets,
            Boot_VEffects,

            // mutable
            Run_Physics,
            Run_SpellSystem,
            Run_Controls,
            Run_Input,
            Run_World,
            Run_Camera,
            Run_Lighting,

            Done
        };


        PreloadStage preloadStage = PreloadStage::NotStarted;
        std::string preloadStageName;

        bool bootLoaded = false;
        bool needsNewRun = true;
        bool doNewRun = true;
        void markNeedsNewRun() { needsNewRun = true; }

        void clearWorldAndGameplayObjects();

        ////helper functions:
        RayCastResult rayCastFromCamera(float maxDistance = 1000.0f);
        glm::vec3 calculateNormalAt(Chunk* chunk, const glm::ivec3& pos);
        float sampleDensityAtWorld(const glm::vec3 &worldPos) const;
        float sampleFluidDensityAtWorld(const glm::vec3 &worldPos) const;
        float getGasDensityAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos);
        glm::vec3 Game::getGasColorAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos);

        glm::vec3 sampleNormalAtWorld(const glm::vec3 &worldPos) const;
        void updateCamera();
        void createNoiseTexture();
        void createNebulaCubemap(int size);

    public:
        static uint8_t sampleTypeAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos);
        static uint32_t sampleMaterialAtWorld(FixedGridChunkManager* chunkManager, const glm::vec3& worldPos);

        void beginGameplayPreload(bool newRun);
        void bakeNebulaCubemap(int size);
        void setupSkybox();

        float getPlayerHealth() const { return playerHealth; }
        float getPlayerMaxHealth() const { return playerMaxHealth; }
        void setPlayerHealth(float h) { playerHealth = glm::clamp(h, 0.0f, playerMaxHealth); }
        void setPlayerMaxHealth(float h) { playerMaxHealth = glm::max(1.0f, h); playerHealth = glm::min(playerHealth, playerMaxHealth); }
        std::vector<float> playerDamageInstances;
        float maxDamagePerTimeframe = 20.0f;
        float damageTimeframe = 0.0125f;
        float damageTimer = 0;
        void registerPlayerDamage(float d) { playerDamageInstances.push_back(d);}
        void applyPlayerDamage() {
            float sum=0;
            for(auto& damageInstance: playerDamageInstances )
            {
                sum+=damageInstance;
            }
            playerDamageInstances.clear();
            if(sum>maxDamagePerTimeframe)
            {
                setPlayerHealth(maxDamagePerTimeframe);
            } else
            {
                setPlayerHealth(getPlayerHealth()-sum);
            }
        }
        bool isPaused() const { return paused; }
        void setPaused(bool p);
        void togglePaused();
        void renderGameplayUI();

        void spawnEnemyLaunchSphere(const glm::vec3& start,
                                    const glm::vec3& target,
                                    float radiusWorld,
                                    float speedWorld,
                                    glm::vec3 color, int material);


        ///V-Effects::
        struct ImpactParticle {
            glm::vec3 position = glm::vec3(0.0f);
            glm::vec3 velocity = glm::vec3(0.0f);
            glm::vec4 color = glm::vec4(1.0f);
            float age = 0.0f;
            float lifetime = 1.0f;
            float startSize = 0.25f;
            float endSize = 2.0f;
            float rotation = 0.0f;
            float rotationSpeed = 0.0f;
            uint32_t kind = 0; // 0 smoke, 1 flash, 2 debris glow etc.
            bool active = true;
        };

        enum class ImpactSizeClass : uint8_t {
            Small,
            Medium,
            Large
        };

        std::vector<ImpactParticle> impactParticles;

        GLuint impactQuadVAO = 0;
        GLuint impactQuadVBO = 0;

        void setupImpactEffects();
        void ensureImpactQuad();
        void updateImpactEffects(float dt);
        void renderImpactEffects();

        void spawnImpactEffect(const glm::vec3& hitPos,
                               const glm::vec3& hitNormal,
                               float impactSpeed,
                               float removedVoxelEstimate,
                               const glm::vec3& tint = glm::vec3(0.45f, 0.45f, 0.45f));

        void spawnImpactPresetSmall(const glm::vec3& hitPos,
                                    const glm::vec3& hitNormal,
                                    float strength01,
                                    const glm::vec3& tint);

        void spawnImpactPresetMedium(const glm::vec3& hitPos,
                                     const glm::vec3& hitNormal,
                                     float strength01,
                                     const glm::vec3& tint);

        void spawnImpactPresetLarge(const glm::vec3& hitPos,
                                    const glm::vec3& hitNormal,
                                    float strength01,
                                    const glm::vec3& tint);



    private:
        glm::vec2 getMouseDelta();
        glm::dvec2 previousMousePos = glm::dvec2(0.0, 0.0);
        bool hasPreviousMousePos = false;
        float playerMaxHealth = 100.0f;
        float playerHealth    = 100.0f;

        std::unique_ptr<EnemyManager> enemyManager;

        bool paused = false;
        int activeSpellMat=0;


        std::vector<DrawArraysIndirectCommand> visibleDrawCmds;
        std::vector<uint32_t> visibleSlots;


        void findNearbyVoxelsForVisualNew(const glm::vec3 &center, float radius, uint64_t targetMaterial,
                                          std::vector<AnimatedVoxel> &results, float strength,
                                          uint8_t &outDominantType);

        void setupSpellContext();

        glm::vec3 getCameraUp() const;

        glm::vec3 cameraForward;
        glm::vec3 cameraUp;

        glm::vec3 cameraRight;
        const float cameraSensitivity=0.075f;

        std::unique_ptr<Shader> speedLinesShader;
        bool enableSpeedLines = true;
        float speedLinesIntensity = 1.0f;

        void initSpeedLinesShader();
        void renderSpeedLines(GLuint sceneTexture);

        WaveManager waveManager;

        void alignCameraRollToUp(const glm::vec3 &targetUp, float dt);

        void appendSpellBillboards(std::vector<SunInstance> &out);

        GLuint impactInstanceVBO = 0;
        std::vector<ImpactInstanceGPU> impactInstancesCPU;
        static constexpr size_t kMaxImpactInstances = 4096;
        GLuint impactNoiseTexId = 0;
        void createImpactNoiseTexture();
        void initImpactInstancing();

        CollisionDecision
        decideCollisionResponse(const VoxelPhysicsBody &self, const VoxelPhysicsBody *other, const glm::vec3 &hitPos,
                                const glm::vec3 &hitNormal, float impactSpeed) const;
        std::vector<MaterialCollisionRule> materialRules;
        void initMaterialRules();

        enum class PauseSubmenu {
            Main,
            Settings
        };

        void applyMaterial9BurnAlongSegment(const glm::vec3 &from, const glm::vec3 &to, float radius);

        void initFluidFBO();
        bool isPointInsideFluid(const glm::vec3& worldPos) const;
        GLuint fluidFBO = 0;
        GLuint fluidColorTex = 0;
        GLuint fluidDepthTex = 0;
        GLuint fluidThicknessTex = 0;
        std::vector<uint32_t> visibleFluidSlots;
        bool chunkHasFluid(const Chunk& chunk);
        void renderFluids();
        glm::vec3 sampleFluidColorAtWorld(const glm::vec3& worldPos) const;
        std::unique_ptr<Shader> fluidShader;
        GLuint compositeFBO = 0;
        GLuint compositeColorTex = 0;

        GLuint gasFBO = 0;
        GLuint gasColorTex = 0;      // RGBA8 for gas color
        GLuint gasDepthTex = 0;      // Depth for composite
        GLuint gasDensityTex = 0;    // R16F for accumulated density

        std::unique_ptr<Shader> gasRayMarchShader;
        void initGasFBO();
        void renderGas();

    public:
        void convertWorldToMaterial(const glm::vec3& center, float radius, uint32_t material);
        int Game::consumeWorldOfMaterial(const glm::vec3& center, float radius, uint32_t material);
        void convertSolidWorldToMaterial(const glm::vec3 &center, float radius, uint32_t material);

        void convertWorldToType(const glm::vec3 &center, float radius, uint32_t type, float strength);

        void convertSolidWorldToType(const glm::vec3 &center, float radius, uint32_t type);

        enum class DisplayMode {
            Fullscreen = 0,
            Windowed = 1,
            Borderless = 2
        };

        struct ResolutionOption {
            int w, h;
            const char* label;
        };

        struct GameSettings {
            float sensitivity = 0.12f;
            float masterVolume = 1.0f;
            float sfxVolume = 1.0f;
            float musicVolume = 0.8f;
            float gamma = 2.2f;       // typical gamma baseline
            float brightness = 1.0f;  // 1.0 neutral
            DisplayMode displayMode = DisplayMode::Borderless;
            int resolutionIndex = 3;
        };

        PauseSubmenu pauseSubmenu = PauseSubmenu::Main;
        GameSettings settings;
        std::vector<ResolutionOption> commonResolutions = {
                {1280, 720,  "1280x720 (HD)"},
                {1600, 900,  "1600x900"},
                {1920, 1080, "1920x1080 (FHD)"},
                {1920, 1200, "1920x1200 (HD)"},
                {2560, 1440, "2560x1440 (QHD)"},
                {3840, 2160, "3840x2160 (4K)"},
                {2560, 1080, "2560x1080 (UW FHD)"},
                {3440, 1440, "3440x1440 (UW QHD)"},
                {3840, 1600, "3840x1600 (UW+)"},
                {5120, 1440, "5120x1440 (Super UW)"}
        };

        void applyDisplaySettings();
        void applyAudioSettings();
        //void applyVisualSettings();

        void pickResolutionFromNativeMonitor(bool applyNow);

        int findBestResolutionIndexForMonitor(GLFWmonitor *monitor) const;

        enum class CraftSpellType : uint8_t { Construct, Projectile };
        enum class CraftForm : uint8_t { Sphere, Wall };
        enum class CraftMaterial : uint8_t { Rock=0, Flesh=7, Lava=9 };

        struct SpellPreset {
            std::string name = "New Spell";
            CraftMaterial material = CraftMaterial::Rock;
            CraftForm form = CraftForm::Sphere;
            CraftSpellType spellType = CraftSpellType::Projectile;

            float radius = 2.0f;      // for sphere
            float width = 3.0f;       // for wall
            float height = 2.0f;      // for wall
            float thickness = 1.0f;   // for wall

            float range = 35.0f;      // customizable

            // derived stats (not directly editable)
            float cooldown = 1.0f;
            float materialCost = 10.0f;
        };

        std::array<SpellPreset, 3> spellPresets;
        int activeSpellPresetIndex = 0;

        const SpellPreset& getSpellPreset(int i) const { return spellPresets[i]; }
        void recomputeSpellDerivedStats(SpellPreset& p);

        SpellPreset& getSpellPreset(int i) { return spellPresets[i]; }

        void initCompositeFBO();

        void renderTextureToScreen(GLuint textureID);

        SoLoud::Soloud audio;

        //Background music
        SoLoud::handle musicHandle;

        std::unique_ptr<SoLoud::Wav> backgroundMusic;
        std::unique_ptr<SoLoud::Wav> mainMenuTheme;
        std::unique_ptr<SoLoud::Wav> bossTheme;

        //UI sounds
        SoLoud::Wav buttonClick;
        SoLoud::Wav buttonHover;
        SoLoud::Wav menuClose;

        //sound effects
        SoLoud::Wav collisionEffect;
        SoLoud::Wav waterSplashEffect;
        SoLoud::Wav fireEffect;
        SoLoud::Wav crunchEffect;
        SoLoud::Wav stepEffect;
        SoLoud::Wav runEffect;
        SoLoud::Wav jumpEffect;
        SoLoud::Wav landEffect;

        void updatePlayerAudio();
    };

}