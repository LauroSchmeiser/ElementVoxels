#include "Game.h"
#include <stdexcept>
#include <random>
#include <iostream>
#include <amp_short_vectors.h>
#include "Assets.h"
#include "rendering/VoxelMesher.h"
#include "rendering/TestChunk.h"
#include "rendering/Shader.h"
#include "entities/VoxelEntity.h"
#include "rendering/smoothMesher.h"

namespace gl3 {
    // CPU-side voxel format that matches the compute shader's Voxel { float density; vec4 color; }
    struct CpuVoxel { float density; glm::vec4 color; };

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
        // Prepare SSBOs and static tables
        size_t voxelCount = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        size_t maxVerts = CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE * 5 * 3;

        // 0: voxels SSBO
        glGenBuffers(1, &ssboVoxels);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxelCount * sizeof(CpuVoxel), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels); // bind to 0

        int edgeTableCPU[256]={
                0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
                0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
                0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
                0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
                0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
                0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
                0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
                0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
                0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
                0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
                0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
                0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
                0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
                0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
                0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
                0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
                0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
                0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
                0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
                0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
                0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
                0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
                0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
                0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
                0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
                0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
                0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
                0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
                0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
                0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
                0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
                0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };


        //TRI-Table
        int triTableCPU[256*16] =
                {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1,
                 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1,
                 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1,
                 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1,
                 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1,
                 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
                 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1,
                 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1,
                 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
                 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1,
                 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1,
                 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1,
                 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1,
                 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1,
                 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
                 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1,
                 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1,
                 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1,
                 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1,
                 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1,
                 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1,
                 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1,
                 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1,
                 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1,
                 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1,
                 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1,
                 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1,
                 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1,
                 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1,
                 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1,
                 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1,
                 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1,
                 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1,
                 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1,
                 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1,
                 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1,
                 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1,
                 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
                 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1,
                 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1,
                 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1,
                 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
                 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1,
                 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1,
                 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1,
                 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1,
                 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1,
                 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1,
                 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
                 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1,
                 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1,
                 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1,
                 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1,
                 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1,
                 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1,
                 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1,
                 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1,
                 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1,
                 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1,
                 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1,
                 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1,
                 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1,
                 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1,
                 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1,
                 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1,
                 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1,
                 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1,
                 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1,
                 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1,
                 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1,
                 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1,
                 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1,
                 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1,
                 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1,
                 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1,
                 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1,
                 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1,
                 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1,
                 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1,
                 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1,
                 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1,
                 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1,
                 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1,
                 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1,
                 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1,
                 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
                 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
                 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1,
                 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1,
                 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1,
                 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1,
                 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1,
                 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1,
                 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1,
                 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1,
                 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1,
                 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1,
                 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1,
                 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1,
                 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1,
                 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1,
                 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1,
                 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1,
                 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1,
                 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1,
                 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1,
                 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1,
                 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1,
                 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1,
                 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
                 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1,
                 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1,
                 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1,
                 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1,
                 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1,
                 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1,
                 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1,
                 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1,
                 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1,
                 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1,
                 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1,
                 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1,
                 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1,
                 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1,
                 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1,
                 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1,
                 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1,
                 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1,
                 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1,
                 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1,
                 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1,
                 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1,
                 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1,
                 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1,
                 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1,
                 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1,
                 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1,
                 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1,
                 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1,
                 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1,
                 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1,
                 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1,
                 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1,
                 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1,
                 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1,
                 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1,
                 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1,
                 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1,
                 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1,
                 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1,
                 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1,
                 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1,
                 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1,
                 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1,
                 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1,
                 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1,
                 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1,
                 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1,
                 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1,
                 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1,
                 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1,
                 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1,
                 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1,
                 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1,
                 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1,
                 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1,
                 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1,
                 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1,
                 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1,
                 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1,
                 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1,
                 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1,
                 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1,
                 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1,
                 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1,
                 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1,
                 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1,
                 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1,
                 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1,
                 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

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

        // 4: triangles SSBO (output)
        glGenBuffers(1, &ssboTriangles);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboTriangles);
        glBufferData(GL_SHADER_STORAGE_BUFFER, maxVerts * sizeof(OutVertex), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);

        // --- Setup shaders ---
        Shader voxelShader("shaders/voxel.vert", "shaders/voxel.frag");
        Shader computeShader("shaders/marching_cubes.comp");
        // Create chunks and set densities properly (density = radius - dist)
        Chunk baseChunk;
        Chunk sunChunk;
        const float radius = CHUNK_SIZE * 0.5f;

        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - CHUNK_SIZE / 2.0f;
                    float dy = y - CHUNK_SIZE / 2.0f;
                    float dz = z - CHUNK_SIZE / 2.0f;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = baseChunk.voxels[x][y][z];
                    if (dist < radius) {
                        voxel.type = 1;
                        voxel.density = radius - dist; // positive inside
                        voxel.color = glm::vec3((float)x / CHUNK_SIZE, (float)y / CHUNK_SIZE, (float)z / CHUNK_SIZE);
                    } else {
                        voxel.type = 0;
                        voxel.density = radius - dist; // negative outside
                        voxel.color = glm::vec3(0.0f);
                    }
                }
            }
        }

        // sunChunk similar (maybe different color/scale)
        for (int x = 0; x < CHUNK_SIZE; ++x) {
            for (int y = 0; y < CHUNK_SIZE; ++y) {
                for (int z = 0; z < CHUNK_SIZE; ++z) {
                    float dx = x - CHUNK_SIZE / 2.0f;
                    float dy = y - CHUNK_SIZE / 2.0f;
                    float dz = z - CHUNK_SIZE / 2.0f;
                    float dist = sqrt(dx * dx + dy * dy + dz * dz);

                    auto &voxel = sunChunk.voxels[x][y][z];
                    if (dist < radius) {
                        voxel.type = 1;
                        voxel.density = radius - dist;
                        voxel.color = glm::vec3((float)x / CHUNK_SIZE, (float)y / CHUNK_SIZE, (float)z / CHUNK_SIZE);
                    } else {
                        voxel.type = 0;
                        voxel.density = radius - dist;
                        voxel.color = glm::vec3(0.0f);
                    }
                }
            }
        }


        // --- Generate mesh from voxel chunk ---
        //Mesh planetMesh = generateVoxelChunkMesh(baseChunk);
        //Mesh sunMesh = generateVoxelChunkMesh(sunChunk);

        std::vector<Planet> suns;
        std::vector<Planet> planets;

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distPos(-100.0f, 100.0f);
        std::uniform_real_distribution<float> distScale(0.5f, 3.0f);
        std::uniform_real_distribution<float> distAxis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> distSpeed(5.0f, 10.0f);

        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);

        for (int j = 0; j < 2; ++j) {
                glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
                if (glm::length(axis) < 0.001f) axis = glm::vec3(0, 1, 0);

                suns.push_back({
                                       glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                       glm::vec3(3* distScale(rng)),
                                       0.0f,
                                       axis,
                                       distSpeed(rng),
                                       glm::vec3(1.0f, 1.0f, 0)
                               });
            }
        for (int i = 0; i < 10; ++i) {
            glm::vec3 axis = glm::normalize(glm::vec3(distAxis(rng), distAxis(rng), distAxis(rng)));
            if (glm::length(axis) < 0.001f) axis = glm::vec3(0, 1, 0);

            planets.push_back({
                                      glm::vec3(distPos(rng), distPos(rng), distPos(rng)),
                                      glm::vec3(distScale(rng)),
                                      0.0f,
                                      axis,
                                      distSpeed(rng),
                                      glm::vec3(distColor(rng), distColor(rng), distColor(rng))  // << ONE color per planet
                              });
        }


        // --- Camera setup ---
        cameraPos = glm::vec3(0.0f, 0.0f, 80.0f);
        cameraRotation = glm::vec2(0.0f, -90.0f);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


        /*
        // --- Main loop ---
        while (!glfwWindowShouldClose(window)) {
            update();
            updateDeltaTime();
            handleCameraInput();

            glEnable(GL_DEPTH_TEST);
            glClearColor(0.0f, 0.0f, 0.2f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            voxelShader.use();

            for(auto & sun : suns){
                // Each planet spins around its own random axis
                sun.rotationAngle += deltaTime * sun.rotationSpeed;
                if (sun.rotationAngle > 360.0f) sun.rotationAngle -= 360.0f;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, sun.position);
                model = glm::rotate(model, glm::radians(sun.rotationAngle), sun.rotationAxis);
                model = glm::scale(model, sun.scale);

                glm::mat4 mvp = calculateMvpMatrix(sun.position, sun.rotationAngle, sun.scale);

                voxelShader.setVec3("uniformColor", sun.color);
                float combinedLight=1.0f;
                glm::vec3 combinedDir= glm::vec3(0,0,0);
                combinedLight= 1.0f;
                combinedDir=glm::vec3(0,0,0);
                voxelShader.setFloat("emission", 1.0f);
                voxelShader.setFloat("lightIntensity",combinedLight);
                voxelShader.setMatrix("model", model);
                voxelShader.setVec3("lightPos",combinedDir);
                voxelShader.setMatrix("mvp", mvp);
                sunMesh.draw();
            }

            for (auto &planet : planets) {
                // Each planet spins around its own random axis
                planet.rotationAngle += deltaTime * planet.rotationSpeed;
                if (planet.rotationAngle > 360.0f) planet.rotationAngle -= 360.0f;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, planet.position);
                model = glm::rotate(model, glm::radians(planet.rotationAngle), planet.rotationAxis);
                model = glm::scale(model, planet.scale);

                glm::mat4 mvp = calculateMvpMatrix(planet.position, planet.rotationAngle, planet.scale);

                voxelShader.setVec3("uniformColor", planet.color);
                float combinedLight = 0.0f;
                glm::vec3 lightDir = glm::vec3(0);

                for (auto &sun : suns) {
                    float dist = glm::distance(planet.position, sun.position);

                    combinedLight += 100 / (dist * dist);
                    lightDir += glm::normalize(sun.position - planet.position);
                }

                lightDir = glm::normalize(lightDir);

                voxelShader.setFloat("emission", 0.0f);
                voxelShader.setFloat("lightIntensity",combinedLight);
                voxelShader.setMatrix("model", model);
                voxelShader.setVec3("lightPos",lightDir);
                voxelShader.setMatrix("mvp", mvp);
                planetMesh.draw();
            }

            //TODO new code
            mcShader.use(); // use your marching cubes compute shader

            // --- Render Suns ---
            for (auto &sun : suns) {
                uploadVoxelChunk(sunChunk);       // step 3
                resetAtomicCounter();              // step 4
                setComputeUniforms(sun.position); // step 5a
                dispatchCompute();                 // step 5b
                drawTriangles();                   // step 5c
            }



            glfwSwapBuffers(window);
            glfwPollEvents();
        }
         */

        while (!glfwWindowShouldClose(window)) {
            update();
            updateDeltaTime();
            handleCameraInput();

            glEnable(GL_DEPTH_TEST);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // --- Rotate suns and planets ---
            UpdateRotation(suns);
            UpdateRotation(planets);

            computeShader.use(); // use your marching cubes compute shader

            // compute PV once per frame (do this outside the loops)
            float aspect = (windowHeight == 0) ? (float)windowWidth / 1.0f : (float)windowWidth / (float)windowHeight;
            glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
            glm::mat4 view = glm::lookAt(cameraPos, cameraPos + getCameraFront(), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 pv = projection * view;
            glm::mat4 identityModel = glm::mat4(1.0f);


            // For each sun: upload voxels, reset counter, set uniforms, dispatch compute, read vertex count, draw
            for (auto &sun : suns) {
                uploadVoxelChunk(sunChunk, &sun.color);           // upload densities/colors, uses binding 0
                resetAtomicCounter();                     // zero counter in binding 3
                setComputeUniforms(sun.position, sun.scale, computeShader);
                dispatchCompute();

                // read debug vertex count
                unsigned int vertexCount = 0;
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
                std::cout << "[compute] sun vertexCount = " << vertexCount << std::endl;

                // --- Set voxel rendering shader uniforms for this sun BEFORE drawing ---
                // render
                voxelShader.use();
                voxelShader.setMatrix("model", identityModel);  // IMPORTANT: identity
                voxelShader.setMatrix("mvp", pv);               // PV only (positions are world-space)
                voxelShader.setVec3("viewPos", cameraPos);
                voxelShader.setVec3("lightPos", sun.position);
                voxelShader.setFloat("lightIntensity", 20.0f);
                voxelShader.setFloat("emission", 3.0f);
                voxelShader.setVec3("uniformColor", sun.color);

                // draw the vertices produced by the compute shader
                drawTriangles(voxelShader);
            }


            // Render planets
            for (auto &planet : planets) {
                uploadVoxelChunk(baseChunk, &planet.color);
                resetAtomicCounter();
                setComputeUniforms(planet.position, planet.scale, computeShader);
                dispatchCompute();

                unsigned int vertexCount = 0;
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);
                std::cout << "[compute] planet vertexCount = " << vertexCount << std::endl;

                voxelShader.use();
                voxelShader.setMatrix("model", identityModel);
                voxelShader.setMatrix("mvp", pv);
                voxelShader.setVec3("viewPos", cameraPos);

                // Choose a real world-space light position for illumination.
                // Use the first sun as the light (or compute a weighted average if you want multi-sun)
                glm::vec3 primaryLightPos = suns.empty() ? glm::vec3(20.0f, 20.0f, 20.0f) : suns[0].position;

                // Basic inverse-square brightness (you already computed combinedLight — clamp it and ensure a minimum)
                float combinedLight = 0.0f;
                for (auto &s : suns) {
                    float d = glm::distance(planet.position, s.position);
                    combinedLight += 100.0f / (d * d + 1e-6f);
                    // basic inverse-square attenuation
                }
                if(combinedLight<1.0f)
                {
                    combinedLight=1.0f;
                }
                float intensity = glm::clamp(combinedLight, 0.2f, 100.0f); // ensure at least 0.2 ambient-like lighting
                voxelShader.setVec3("lightPos", primaryLightPos);
                voxelShader.setFloat("lightIntensity", intensity);
                voxelShader.setFloat("emission", 0.0f);
                voxelShader.setVec3("uniformColor", planet.color);
                // draw the vertices produced by the compute shader
                drawTriangles(voxelShader);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

    }

    void Game::UpdateRotation(std::vector<Planet>& planets)
    {
        for(auto & planet : planets){
            planet.rotationAngle += deltaTime * planet.rotationSpeed;
            if (planet.rotationAngle > 360.0f) planet.rotationAngle -= 360.0f;
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


    void Game::uploadVoxelChunk(const Chunk& chunk, const glm::vec3* overrideColor)
    {
        struct CpuVoxel { float density; glm::vec4 color; };
        size_t voxelCount = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        std::vector<CpuVoxel> voxels(voxelCount);

        for(int x=0;x<CHUNK_SIZE;x++) {
            for(int y=0;y<CHUNK_SIZE;y++) {
                for(int z=0;z<CHUNK_SIZE;z++) {
                    auto &v = chunk.voxels[x][y][z];
                    int idx = x + y*CHUNK_SIZE + z*CHUNK_SIZE*CHUNK_SIZE;

                    voxels[idx].density = v.type ? v.density : -1.0f;

                    if (overrideColor) {
                        voxels[idx].color = glm::vec4(*overrideColor, 1.0f); // use uniform color
                    } else {
                        voxels[idx].color = glm::vec4(v.color, 1.0f);        // per-voxel color as before
                    }
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVoxels);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxels.size()*sizeof(CpuVoxel), voxels.data());

        // Debug (optional): print first and center voxel colors/density
        // int centerIdx = (CHUNK_SIZE/2) + (CHUNK_SIZE/2)*CHUNK_SIZE + (CHUNK_SIZE/2)*CHUNK_SIZE*CHUNK_SIZE;
        // std::cout << "uploadVoxelChunk: color0=" << voxels[0].color.r << ","<<voxels[0].color.g<<","<<voxels[0].color.b
        //           << " centerDensity=" << voxels[centerIdx].density << "\n";
    }

    void Game::resetAtomicCounter()
    {
        unsigned int zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &zero);

    }

    void Game::setComputeUniforms(const glm::vec3& position, const glm::vec3& objectScale, Shader& computeShader)
    {
        computeShader.use();

        // base voxel size (world units per voxel) you used previously
        const float baseVoxelSize = 1.0f;

        // If the object has a uniform scale, apply it by scaling voxelSize.
        // If objectScale is vec3, you can pick x (uniform scale) or average.
        float scale = (objectScale.x + objectScale.y + objectScale.z) / 3.0f;
        float effectiveVoxelSize = baseVoxelSize * scale;

        // We want the chunk centered at 'position'. The compute shader expects gridOrigin
        // to be the world position of voxel (0,0,0). So offset by half the grid extents:
        glm::vec3 halfExtents = (glm::vec3(CHUNK_SIZE) * 0.5f) * effectiveVoxelSize;
        glm::vec3 gridOrigin = position - halfExtents;

        // upload uniforms
        computeShader.setVec3("gridOrigin", gridOrigin);
        computeShader.setFloat("voxelSize", effectiveVoxelSize);
        // shader expects ivec3 voxelGridDim
        computeShader.setIVec3("voxelGridDim", glm::ivec3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE));
    }
    void Game::dispatchCompute()
    {
        int groups = (CHUNK_SIZE - 1 + 7) / 8; // local_size = 8 in compute shader

        // Bind SSBOs to match compute shader bindings:
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboVoxels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboEdgeTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboTriTable);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssboTriangles);

        glDispatchCompute(groups, groups, groups);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    }

    void Game::drawTriangles(Shader& voxelShader)
    {
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Use output SSBO as VBO
        glBindBuffer(GL_ARRAY_BUFFER, ssboTriangles);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)0);                   // position
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)(sizeof(glm::vec4))); // normal
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(OutVertex), (void*)(2 * sizeof(glm::vec4))); // color

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);

        unsigned int vertexCount = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCounter);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &vertexCount);

        voxelShader.use();

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);

        glDeleteVertexArrays(1, &vao);
    }



}
