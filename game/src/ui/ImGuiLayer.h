#pragma once

#include <GLFW/glfw3.h>

namespace gl3 {

    class ImGuiLayer {
    public:
        void init(GLFWwindow* window, const char* glslVersion = "#version 460");
        void shutdown();

        void beginFrame();
        void endFrame();

        bool isInitialized() const { return initialized; }

    private:
        bool initialized = false;
    };

} // namespace gl3