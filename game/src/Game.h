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
#include "entities//Entity.h"
#include "rendering/VoxelRenderer.h"

namespace gl3 {
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


        // --- Generate planet transforms ---
        struct Planet {
            glm::vec3 position;
            glm::vec3 scale;
            float rotationAngle;
            glm::vec3 rotationAxis;
            float rotationSpeed;
            glm::vec3 color;
        };

        //TODO: New Methods
        void uploadVoxelChunk(Chunk chunk);
        void resetAtomicCounter();
        void setComputeUniforms(glm::vec3 position, Shader computeShader);
        void dispatchCompute();
        void drawTriangles(Shader voxelShader);

        void UpdateRotation(std::vector<Planet> planets);
        void handleCameraInput();
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
        int windowWidth, windowHeight;
        GLuint ssboVoxels, ssboEdgeTable, ssboTriTable , ssboCounter, ssboTriangles;

    };
}



