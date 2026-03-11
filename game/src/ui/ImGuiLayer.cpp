#include "ImGuiLayer.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

#include <imgui.h>

namespace gl3 {

    void ImGuiLayer::init(GLFWwindow* window, const char* glslVersion)
    {
        if (initialized) return;

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        (void)io;

        // optional default style
        ImGui::StyleColorsDark();

        // Init platform/renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, /*install_callbacks=*/true);
        ImGui_ImplOpenGL3_Init(glslVersion);

        initialized = true;
    }

    void ImGuiLayer::shutdown()
    {
        if (!initialized) return;

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        initialized = false;
    }

    void ImGuiLayer::beginFrame()
    {
        if (!initialized) return;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void ImGuiLayer::endFrame()
    {
        if (!initialized) return;

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

} // namespace gl3