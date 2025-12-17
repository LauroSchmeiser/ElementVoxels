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
#include "entities//Entity.h"
#include "rendering/VoxelRenderer.h"
#include "rendering/SunBillboard.h"

namespace gl3 {

    static constexpr int ChunkCount=8;


    struct GameData{
        std::array<std::array<std::array<Chunk, ChunkCount>,ChunkCount>,ChunkCount> gameWorld;
    };

    class Game {
    public:
        Game(int width, int height, const std::string &title);
        virtual ~Game();
        void run();
        glm::mat4 calculateMvpMatrix(glm::vec3 position, float zRotationInDegrees, glm::vec3 scale);
        GLFWwindow *getWindow() { return window; }

    private:
        static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
        void update();
        void draw();
        void updateDeltaTime();
        void updatePhysics();
        bool checkEmptyChunks(Chunk chunk);


        // --- Generate planet transforms ---
        struct Planet {
            glm::vec3 position;
            glm::vec3 scale;
            float rotationAngle;
            glm::vec3 rotationAxis;
            float rotationSpeed;
            glm::vec3 color;
            float orbitAngle;
            float orbitSpeed;
            float orbitRadius;
            float orbitInclination;
            float orbitOffset;
            Planet* parent = nullptr;  // <—— parent "sun"
        };

        void simulatePhysics(const std::vector<gl3::Game::Planet> &others);

        // Updated signatures
        void uploadVoxelChunk(const Chunk& chunk, const glm::vec3* overrideColor = nullptr);
        void resetAtomicCounter();
        void setComputeUniforms(const glm::vec3& position, const glm::vec3& objectScale, Shader& computeShader);
        void dispatchCompute();
        void drawTriangles(Shader& voxelShader);

        void UpdateRotation(std::vector<Planet>& planets);
        void handleCameraInput();
        float getVoxelPlanetRadius(const glm::vec3& scale, float baseChunkRadius);


        //Initialization-Steps
        void setupSSBOsAndTables();
        void setupCamera();
        void generateChunks();
        void fillChunks();
        void setSimulationVariables();
        void setupVEffects();
        void findBestParent();


        //Simulation-Steps
        bool isOverlapping(const glm::vec3& pos, float rad, const std::vector<Planet>& others);

        //Input-Steps

        //Post-Prod Steps?

        //Rendering-Steps
        void renderChunks();
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
        float accumulator = 0.f;
        glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
        glm::vec2 cameraRotation = glm::vec2(-90.0f, 0.0f); // pitch, yaw
        int windowWidth = 800, windowHeight = 600;
        const int DIM = CHUNK_SIZE + 1;
        size_t voxelCount = DIM * DIM * DIM;
        size_t maxVerts = CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE * 5 * 3;

        //SSBOs for marching cubes
        GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0, ssboCounter = 0, ssboTriangles = 0, particleSSBO=0, fieldBitsSSBO=0;



        //Chunks:
        //Chunk meteorChunk;
        //Chunk baseChunk;
        //Chunk sunChunk;
        Chunk fluidPlanetChunk;


        //std::vector<Planet> suns;
        //std::vector<Planet> planets;
        //std::vector<Planet> meteors;
        std::vector<Planet> fluidPlanets;

        std::vector<Planet> CollisionEntities;

        std::unique_ptr<GameData> data = std::make_unique<GameData>();

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