#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include "Assets.h"
#include "rendering/VoxelMesher.h"
#include "rendering/TestChunk.h"
#include "rendering/Shader.h"
#include "entities/VoxelEntity.h"

namespace gl3 {
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
        // --- Setup shader ---
        Shader voxelShader("shaders/voxel.vert", "shaders/voxel.frag");

        // --- Create one base voxel chunk (sphere) ---
        Chunk baseChunk;

        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - CHUNK_SIZE / 2.0f;
                    float dy = y - CHUNK_SIZE / 2.0f;
                    float dz = z - CHUNK_SIZE / 2.0f;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = baseChunk.voxels[x][y][z];
                    if (dist < CHUNK_SIZE / 2.0f) {
                        voxel.type = 1;
                        voxel.color = glm::vec3(
                                (float)x / CHUNK_SIZE,
                                (float)y / CHUNK_SIZE,
                                (float)z / CHUNK_SIZE
                        );
                    } else {
                        voxel.type = 0;
                    }
                }
            }
        }

        // --- Generate mesh from voxel chunk ---
        Mesh planetMesh = generateVoxelChunkMesh(baseChunk);

        // --- Generate planet transforms ---
        struct Planet {
            glm::vec3 position;
            glm::vec3 scale;
            float rotation;
            glm::vec3 rotationAxis;
        };
        std::vector<Planet> planets;

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distPos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distAxis(-1.0f, 1.0f);

        for (int i = 0; i < 20; ++i) {
            planets.push_back({
                                      glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                      glm::vec3(distScale(rng)),
                                      0.0f,
                                      glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)))
                              });
        }

        // --- Camera setup ---
        cameraPos = glm::vec3(0.0f, 0.0f, 80.0f);
        cameraRotation = glm::vec2(0.0f, -90.0f);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        // --- Main loop ---
        while (!glfwWindowShouldClose(window)) {
            update();
            updateDeltaTime();
            handleCameraInput();

            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.2f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            voxelShader.use();

            for (auto &planet : planets) {
                // Spin each planet slowly
                planet.rotation += deltaTime * 15.0f;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, planet.position);
                model = glm::rotate(model, glm::radians(planet.rotation), planet.rotationAxis);
                model = glm::scale(model, planet.scale);

                glm::mat4 mvp = calculateMvpMatrix(planet.position, planet.rotation, planet.scale);

                voxelShader.setMatrix("model", model);
                voxelShader.setMatrix("mvp", mvp);

                planetMesh.draw();
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
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



}
