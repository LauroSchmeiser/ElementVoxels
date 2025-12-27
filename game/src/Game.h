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
#include "entities//Entity.h"
#include "rendering/VoxelStructures.h"
#include "rendering/SunBillboard.h"
#include "rendering/VoxelStructures.h"
#include "rendering/Chunk.h"
#include "../robin_hood.h"
#include "rendering/MultiGridChunkManager.h"

namespace gl3 {

    static constexpr int ChunkCount=30;
    static constexpr int RenderingRange=20;

    class Game {
    public:
        Game(int width, int height, const std::string &title);
        virtual ~Game();
        void run();
        glm::mat4 calculateMvpMatrix(glm::vec3 position, float zRotationInDegrees, glm::vec3 scale);
        GLFWwindow *getWindow() { return window; }

    private:

        //std::unique_ptr<ChunkManager> chunkManager = std::make_unique<ChunkManager>(); // NEW

        // Or if using MultiGridChunkManager:
        std::unique_ptr<MultiGridChunkManager> chunkManager = std::make_unique<MultiGridChunkManager>();
        static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
        void update();
        void draw();
        void updateDeltaTime();
        void updatePhysics();
        bool hasSolidVoxels(const gl3::Chunk& chunk);

        glm::vec3 getChunkWorldPosition(const ChunkCoord &coord);
        ChunkCoord worldToChunkCoord(const glm::vec3 &worldPos) const;
        int worldToChunk(float worldPos) const;

        uint32_t makeLightID(int cx, int cy, int cz);


            // --- Generate planet transforms ---


        struct WorldPlanet {
            glm::vec3 worldPos;   // world-space center
            float radius;         // world-space radius
            glm::vec3 color;
            int type;             // 1=rock, 2=lava, 3=water
        };

        // Updated signatures
        void uploadVoxelChunk(const Chunk& chunk, const glm::vec3* overrideColor = nullptr);
        void resetAtomicCounter();
        void setComputeUniforms(const glm::vec3& position, const glm::vec3& objectScale, Shader& computeShader);
        void dispatchCompute();
        void drawTriangles(Shader& voxelShader);

        void handleCameraInput();
        float getVoxelPlanetRadius(const glm::vec3& scale, float baseChunkRadius);
        bool isOverlapping(const glm::vec3 &pos, float rad, const std::vector<gl3::Game::WorldPlanet> &others);


            //Initialization-Steps
        void setupSSBOsAndTables();
        void setupCamera();
        void generateChunks();
        void fillChunks();
        void setSimulationVariables();
        void setupVEffects();
        void findBestParent();


        //Simulation-Steps
        //void updateWorldLighting();
        void rebuildChunkLights(const ChunkCoord &coord);
        std::vector<VoxelLight*> collectNearbyLightsFast(
                const ChunkCoord& chunkCoord,
                const glm::vec3& chunkOrigin,
                const std::unordered_map<ChunkCoord, std::vector<VoxelLight*>, ChunkCoordHash>& hash);
        void buildLightSpatialHash(
                std::unordered_map<ChunkCoord, std::vector<VoxelLight*>, ChunkCoordHash>& hash);
        //Input-Steps

        //Post-Prod Steps?

        //Rendering-Steps
        void renderChunks();
        void processEmissiveChunks();
        void renderSuns();
        void renderPlanets();
        void renderFluidPlanets();

        glm::vec3 getCameraFront() const;
        GLFWwindow *window = nullptr;
        std::vector<std::unique_ptr<Entity>> entities;
        SoLoud::Soloud audio;
        std::unique_ptr<SoLoud::Wav> backgroundMusic;
        float lastFrameTime = 1.0f/60;
        float deltaTime = 1.0f/60;
        float accumulator = 0.1f;
        glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
        glm::vec2 cameraRotation = glm::vec2(-90.0f, 0.0f); // pitch, yaw
        int windowWidth = 800, windowHeight = 600;
        const int DIM = CHUNK_SIZE + 1;
        size_t voxelCount = DIM * DIM * DIM;
        size_t maxVerts = CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE * 5 * 3;

        //SSBOs for marching cubes
        GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0, ssboCounter = 0, ssboTriangles = 0, particleSSBO=0, fieldBitsSSBO=0;

        const int MAX_LIGHTS = 4;       // matches shader
        const float LIGHT_RADIUS = 10000.0f*CHUNK_SIZE;
        const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;

        std::unordered_set<uint32_t> usedLightIDs;


        //Chunks:
        //Chunk meteorChunk;
        //Chunk baseChunk;
        //Chunk sunChunk;
        Chunk fluidPlanetChunk;


        //std::vector<Planet> suns;
        //std::vector<Planet> planets;
        //std::vector<Planet> meteors;
        std::vector<WorldPlanet> fluidPlanets;

        std::vector<WorldPlanet> CollisionEntities;

        //vEffects
        SunBillboard sunBillboards;


        std::unique_ptr<Shader> voxelShader;
        std::unique_ptr<Shader> marchingCubesShader;

        // inside Game class:
        std::unique_ptr<Shader> voxelSplatShader;


        struct Particle {
            glm::vec3 position;
            glm::vec3 velocity;
            float lifetime;
            float radius;     // metaball influence
            unsigned int  type;       // 0=water,1=fire,2=smoke...
        };

        std::vector<SunInstance> emissiveBillboards;

        std::vector<Particle> particles;

        int maxParticles =100;
        //helper function
    };
}