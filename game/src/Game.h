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
#include "rendering/MultiGridChunkManager.h"
#include "Input/InputActionMap.h"
#include "CharacterController.h"
#include "physics/SpellPhysicsManager.h"
#include "physics/VoxelPhysicsManager.h"

namespace gl3 {
    class Game {
    public:
        ////basics (public):
        Game(int width, int height, const std::string &title);

        virtual ~Game();

        void run();

        GLFWwindow *getWindow() { return window; }

    private:
        ////basics (private):
        static void framebuffer_size_callback(GLFWwindow *window, int width, int height);


        ////Chunk Management:
        //std::unique_ptr<ChunkManager> chunkManager = std::make_unique<ChunkManager>(); // other manager (old)
        std::unique_ptr<MultiGridChunkManager> chunkManager = std::make_unique<MultiGridChunkManager>();

        bool hasSolidVoxels(const gl3::Chunk &chunk);

    public:
        static int worldToChunk(float worldPos);

    private:
        void markChunkModified(const ChunkCoord &coord);

        void unloadChunk(const ChunkCoord &coord);

        glm::vec3 getChunkMin(const ChunkCoord &coord) const;

        glm::vec3 getChunkMax(const ChunkCoord &coord) const;

        bool isChunkVisible(const ChunkCoord &coord) const;

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
        std::deque<SpellEffect> activeSpells;
        std::vector<AnimatedVoxel> animatedVoxels;

        // Stable id logging for animated voxels
        std::unordered_map<uint64_t, size_t> animatedVoxelIndexMap;
        uint64_t nextAnimatedVoxelID = 1;


        // Spell system methods
        void castSpellWithFormation(const glm::vec3& center, float radius,
                                          uint64_t targetMaterial, float strength,
                                          const FormationParams& baseFormationParams);

        void adjustFormationForVolume(FormationParams& params, float volume);
        glm::vec3 calculateFormationTarget(size_t index, size_t total,
                                                 const FormationParams& params);

        glm::vec3 calculateSphereDistribution(size_t index, size_t total,
                                                    const FormationParams& params);

        glm::vec3 calculatePlatformDistribution(size_t index, size_t total,
                                                      const FormationParams& params);

        glm::vec3 calculateWallDistribution(size_t index, size_t total,
                                                  const FormationParams& params);

        glm::vec3 calculateCubeDistribution(size_t index, size_t total,
                                                  const FormationParams& params);

        glm::vec3 calculateCylinderDistribution(size_t index, size_t total,
                                                      const FormationParams& params);

        glm::vec3 calculatePyramidDistribution(size_t index, size_t total,
                                                     const FormationParams& params);

        float haltonSequence(int index, int base);

        void updateSpells(float deltaTime);

        void cleanupExpiredSpells();

        void findNearbyVoxelsForVisual(const glm::vec3& center, float radius,
                                             uint64_t targetMaterial,
                                             std::vector<AnimatedVoxel>& results,
                                             float strength,
                                             uint8_t& outDominantType);

        void carveFormationWithSDF(const WorldPlanet& formation, uint64_t material,
                                         const FormationParams& params);

        void createSpellFormation(const glm::vec3& center,
                                        const FormationParams& formationParams,
                                        float strength, uint64_t material,
                                        const glm::vec3& color, size_t collectedVoxels,
                                        uint8_t dominantType);


        void createPartialFormation(const SpellEffect& spell, float completionRatio);

        void createExteriorSmoothCrater(Chunk* chunk, const glm::ivec3& voxelPos,
                                        const glm::vec3& worldPos);

        void handleCraterInNeighboringChunk(const glm::vec3& worldPos,
                                                  int dx, int dy, int dz,
                                                  float craterRadius,
                                                  float maxCraterDepth,
                                                  float impactFactor);

        void createCraterAtPosition(const glm::vec3& worldPos, float impactFactor, float spellRadius);

        float randomFloat(float min, float max);

        void castSpellSphere(const glm::vec3& center, float radius,
                                   uint64_t material = 0, float strength = 1.0f);
        void castSpellPlatform(const glm::vec3& center, const glm::vec3& normal,
                                     float width, float depth, float thickness = 0.5f,
                                     uint64_t material = 0, float strength = 1.0f);
        void castSpellWall(const glm::vec3& center, const glm::vec3& normal,
                                 float width, float height, float thickness = 0.5f,
                                 uint64_t material = 0, float strength = 1.0f);
        void castSpellCube(const glm::vec3& center, const glm::vec3& size,
                                 uint64_t material = 0, float strength = 1.0f);
        void castSpellCylinder(const glm::vec3& center, float radius, float height,
                                     uint64_t material = 0, float strength = 1.0f);
        void castSpellCustom(const glm::vec3& center, float radius,
                                   uint64_t material, float strength,
                                   SDFFunction customSDF, void* userData = nullptr);

        void createPhysicsBodyForSpell(SpellEffect& spell);
        void destroyPhysicsBodyForSpell(SpellEffect& spell);
        void onSpellCollision(SpellEffect* spell,
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

        void createPhysicsMeshData(SpellEffect& spell,
                                         const std::vector<glm::vec3>& vertices,
                                         const std::vector<glm::vec3>& normals,
                                         const std::vector<glm::vec3>& colors);
        void removeFormationVoxels(const SpellEffect& spell);

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
        void debugComputeShaderState();

        void DisplayFPSCount();

        bool DebugMode1 = false;
        bool DebugMode2 = false;
        int activeDebugMode = 0;
        int FilledChunks = 0;

        ////Initialization-Steps
        void setupSSBOsAndTables();

        void setupInput();

        void setupCamera();

        void generateChunks();

        void setupVEffects();


        ////Simulation-Steps
        //Physics:
        void updatePhysics();

        void applyImpactAtPosition(const glm::vec3 &worldPos, float radius, float impulse, float mass);

        void updateDeltaTime();

        //Lighting:
        void updateLightSpatialHash();

        void rebuildChunkLights(const ChunkCoord &coord);

        void updateChunkLights(Chunk *chunk);

        void processEmissiveChunks();

        uint32_t makeLightID(int cx, int cy, int cz);


        ////Input-Steps
        void update();

        void handleCameraInput();

        glm::vec3 getCameraFront() const;


        ////Post-Prod Steps?


        ////Rendering-Steps::
        //General Rendering:
        void renderSkybox();
        void renderChunks();
        void renderAnimatedVoxels();
        void renderPhysicsFormations();
        void renderSpellPreview();
        void renderFluidPlanets();

        //marching cubes Shader:
        void generateChunkMesh(Chunk *chunk);

        void uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor = nullptr);

        void resetAtomicCounter();

        void setComputeUniforms(const glm::vec3 &position, Shader &computeShader);

        //vertex/frag Shader:
        void drawTriangles(Shader &voxelShader, Chunk *chunk);
        void ensurePreviewCube();
        void ensurePreviewSphereMesh();

        ////Basic-variables:
        GLFWwindow *window = nullptr;
        int windowWidth = 800, windowHeight = 600;
        SoLoud::Soloud audio;
        std::unique_ptr<SoLoud::Wav> backgroundMusic;
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
        const float LIGHT_RADIUS = 220.0f * CHUNK_SIZE*VOXEL_SIZE*2;
        uint64_t frameCounter = 29; // Frame counter for light update staggering
        const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;
        std::vector<const gl3::VoxelLight *> flatEmissiveLightList;
        robin_hood::unordered_map<ChunkCoord, std::vector<VoxelLight *>, ChunkCoordHash> lightSpatialHash;
        // After building chunk-local lights we create a small merged pool to avoid seams.
        // Merged pool holds stable VoxelLight objects we point to from flatEmissiveLightList.
        std::vector<gl3::VoxelLight> mergedEmissiveLightPool;
        static constexpr int LIGHT_UPDATE_INTERVAL = 15; // Update lights every 15 frames
        robin_hood::unordered_set<uint32_t> usedLightIDs;

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
        static constexpr int ChunkCount = 70; //Total size of the Game World
        static constexpr int RenderingRange = 15; //Range around Camera that is rendered

        //Marching-cubes Variables
        size_t maxVerts = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 5 * 3; //Max amount of vertices marching cubes can create
        const int MAX_CHUNKS_PER_FRAME = 9;
        //std::vector<Chunk> dirtyChunks;
        //SSBOs for marching cubes:
        GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0,
        ssboCounter = 0, ssboTriangles = 0, particleSSBO = 0, fieldBitsSSBO = 0;

        //vEffects
        SunBillboard sunBillboards;
        std::vector<SunInstance> emissiveBillboards;
        int emissiveUpdateCounter=0;

        ////Input
        std::unique_ptr<CharacterController> characterController;
        InputManager input;
        InputActionMap actions;


        ////Shader:
        std::unique_ptr<Shader> skyboxShader;
        std::unique_ptr<Shader> voxelShader;
        std::unique_ptr<Shader> marchingCubesShader;
        std::unique_ptr<Shader> voxelSplatShader;
        std::unique_ptr<Shader> spellPreviewShader;



        ////helper functions:
        RayCastResult rayCastFromCamera(float maxDistance = 1000.0f);
        glm::vec3 calculateNormalAt(Chunk* chunk, const glm::ivec3& pos);
        float sampleDensityAtWorld(const glm::vec3 &worldPos) const;
        glm::vec3 sampleNormalAtWorld(const glm::vec3 &worldPos) const;
        void updateCamera();
        void createNoiseTexture();
        void setupSkybox();

        glm::vec2 getMouseDelta();
        glm::dvec2 previousMousePos = glm::dvec2(0.0, 0.0);
        bool hasPreviousMousePos = false;
    };
}