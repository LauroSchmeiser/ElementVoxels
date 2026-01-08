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
          bool hasSolidVoxels(const gl3::Chunk& chunk);
          int worldToChunk(float worldPos) const;
          void markChunkModified(const ChunkCoord& coord);
          void unloadChunk(const ChunkCoord& coord);


        ////Debugging:
          void debugComputeShaderState();
          bool DebugMode1=false;
          bool DebugMode2=false;
          int activeDebugMode=0;


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
          void updateChunkLights(Chunk* chunk);
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
          void renderFluidPlanets();

          //marching cubes Shader:
          void generateChunkMesh(Chunk* chunk);
          void uploadVoxelChunk(const Chunk& chunk, const glm::vec3* overrideColor = nullptr);
          void resetAtomicCounter();
          void setComputeUniforms(const glm::vec3& position, const glm::vec3& objectScale, Shader& computeShader);

          //vertex/frag Shader:
          void drawTriangles(Shader& voxelShader,Chunk* chunk);


        ////Structs::
        struct Particle {
            glm::vec3 position;
            glm::vec3 velocity;
            float lifetime;
            float radius;     // metaball influence
            unsigned int  type;       // 0=water,1=fire,2=smoke...
        };

        // --- Generate planet transforms ---
        struct WorldPlanet {
            glm::vec3 worldPos;   // world-space center
            float radius;         // world-space radius
            glm::vec3 color;
            int type;             // 1=rock, 2=lava, 3=water
        };


        ////Basic-variables:
          GLFWwindow *window = nullptr;
          int windowWidth = 800, windowHeight = 600;
          SoLoud::Soloud audio;
          std::unique_ptr<SoLoud::Wav> backgroundMusic;


        ////Simultation-Variables:
          //Physics-Variables:
          float lastFrameTime = 1.0f/60;
          float deltaTime = 1.0f/60;
          float accumulator = 0.1f;
          std::vector<WorldPlanet> CollisionEntities;
          std::vector<Particle> particles;
          int maxParticles =100;

        //Lighting-Variables:
          const int MAX_LIGHTS = 4; // has to match marching cubes shader
          const float LIGHT_RADIUS = 220.0f*CHUNK_SIZE;
          uint64_t frameCounter = 29; // Frame counter for light update staggering
          const float LIGHT_RADIUS_SQ = LIGHT_RADIUS * LIGHT_RADIUS;
          std::vector<const gl3::VoxelLight*> flatEmissiveLightList;
          std::unordered_map<ChunkCoord, std::vector<VoxelLight*>, ChunkCoordHash> lightSpatialHash;
          // After building chunk-local lights we create a small merged pool to avoid seams.
          // Merged pool holds stable VoxelLight objects we point to from flatEmissiveLightList.
          std::vector<gl3::VoxelLight> mergedEmissiveLightPool;
          static constexpr int LIGHT_UPDATE_INTERVAL = 15; // Update lights every 15 frames
          std::unordered_set<uint32_t> usedLightIDs;


        ////Camera-Variables:
          glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 50.0f);
          glm::vec2 cameraRotation = glm::vec2(-90.0f, 0.0f); // pitch, yaw


        ////Rendering-Variables:
          //World-Variables:
          const int DIM = CHUNK_SIZE + 1; //Chunk Size with a bit off padding for marching cubes
          size_t voxelCount = DIM * DIM * DIM; //How many voxels can be in one Chunk
          static constexpr int ChunkCount=100; //Total size of the Game World
          static constexpr int RenderingRange=50; //Range around Camera that is rendered

          //Marching-cubes Shader Variables:
          size_t maxVerts = CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE * 5 * 3; //Max amount of vertices marching cubes can create
          //SSBOs for marching cubes:
          GLuint ssboVoxels = 0, ssboEdgeTable = 0, ssboTriTable = 0,
          ssboCounter = 0, ssboTriangles = 0, particleSSBO=0, fieldBitsSSBO=0;

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
    };
}