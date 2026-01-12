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
#include "rendering/VoxelStructures.h"
#include "rendering/SunBillboard.h"
#include "rendering/VoxelStructures.h"
#include "rendering/Chunk.h"
#include "../robin_hood.h"
#include "rendering/MultiGridChunkManager.h"

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

        int worldToChunk(float worldPos) const;

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

                    // If the positive vertex is outside (behind) the plane, the AABB is not visible
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
        std::vector<SpellEffect> activeSpells;
        std::vector<AnimatedVoxel> animatedVoxels;

        // Stable-id mapping for animated voxels
        std::unordered_map<uint64_t, size_t> animatedVoxelIndexMap;
        uint64_t nextAnimatedVoxelID = 1;


        // Spell system methods
        void castGravityWellSpell(const glm::vec3& center, float radius,
                                  uint64_t targetMaterial, float strength);

        void updateSpells(float deltaTime);

        void cleanupFinishedSpells();

        void findNearbyVoxelsForVisual(const glm::vec3& center, float radius,
                                       uint64_t targetMaterial,
                                       std::vector<AnimatedVoxel>& results,
                                       float strength);

        void createSpellFormation(const glm::vec3& center, float radius,
                                  float strength, uint64_t material,
                                  const glm::vec3& color,size_t collectedVoxels);

        void createPartialFormation(const SpellEffect& spell, float completionRatio);

        void createExteriorSmoothCrater(Chunk* chunk, const glm::ivec3& voxelPos,
                                        const glm::vec3& worldPos);

        void carveFormationIntoChunks(const WorldPlanet& formation, uint64_t material);

        float randomFloat(float min, float max);

        ////Debugging:
        void debugComputeShaderState();

        void DisplayFPSCount();

        bool DebugMode1 = false;
        bool DebugMode2 = false;
        int activeDebugMode = 0;
        int FilledChunks = 0;

        ////Initialization-Steps
        void setupSSBOsAndTables();

        void setupCamera();

        void generateChunks();

        void setupVEffects();


        ////Simulation-Steps
        //Physics:
        void updatePhysics();

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
        void renderChunks();
        void renderAnimatedVoxels();
        void renderFluidPlanets();

        //marching cubes Shader:
        void generateChunkMesh(Chunk *chunk);

        void uploadVoxelChunk(const Chunk &chunk, const glm::vec3 *overrideColor = nullptr);

        void resetAtomicCounter();

        void setComputeUniforms(const glm::vec3 &position, const glm::vec3 &objectScale, Shader &computeShader);

        //vertex/frag Shader:
        void drawTriangles(Shader &voxelShader, Chunk *chunk);


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
        const float LIGHT_RADIUS = 220.0f * CHUNK_SIZE;
        uint64_t frameCounter = 29; // Frame counter for light update staggering
        const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;
        std::vector<const gl3::VoxelLight *> flatEmissiveLightList;
        std::unordered_map<ChunkCoord, std::vector<VoxelLight *>, ChunkCoordHash> lightSpatialHash;
        // After building chunk-local lights we create a small merged pool to avoid seams.
        // Merged pool holds stable VoxelLight objects we point to from flatEmissiveLightList.
        std::vector<gl3::VoxelLight> mergedEmissiveLightPool;
        static constexpr int LIGHT_UPDATE_INTERVAL = 15; // Update lights every 30 frames
        std::unordered_set<uint32_t> usedLightIDs;


        ////Camera-Variables:
        glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
        glm::vec2 cameraRotation = glm::vec2(-90.0f, 0.0f); // pitch, yaw


        ////Rendering-Variables:
        //World-Variables:
        const int DIM = CHUNK_SIZE + 1; //Chunk Size with a bit off padding for marching cubes
        size_t voxelCount = DIM * DIM * DIM; //How many voxels can be in one Chunk
        static constexpr int ChunkCount = 60; //Total size of the Game World
        static constexpr int RenderingRange = 35; //Range around Camera that is rendered

        //Marching-cubes Variables
        size_t maxVerts =
                CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 5 * 3; //Max amount of vertices marching cubes can create
        const int MAX_CHUNKS_PER_FRAME = 20;
        //std::vector<Chunk> dirtyChunks;
        //SSBOs for marching cubes:
        GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0,
                ssboCounter = 0, ssboTriangles = 0, particleSSBO = 0, fieldBitsSSBO = 0;

        //vEffects
        SunBillboard sunBillboards;
        std::vector<SunInstance> emissiveBillboards;


        ////Lists (deprecated)
        std::vector<WorldPlanet> fluidPlanets;


        ////Shader:
        std::unique_ptr<Shader> voxelShader;
        std::unique_ptr<Shader> marchingCubesShader;
        std::unique_ptr<Shader> voxelSplatShader;


        ////helper functions:
        RayCastResult rayCastFromCamera(float maxDistance = 1000.0f);
        glm::vec3 calculateNormalAt(Chunk* chunk, const glm::ivec3& pos);
    };
}