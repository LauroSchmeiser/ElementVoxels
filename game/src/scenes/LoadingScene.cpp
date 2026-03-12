#include "LoadingScene.h"
#include "../Game.h"
#include "../SceneId.h"

#include <imgui.h>
#include <GLFW/glfw3.h>

namespace gl3 {

    void LoadingScene::onEnter(Game& game)
    {
        progress = 0.0f;
        t = 0.0;

        // show cursor (optional)
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        // start preload pipeline
        game.beginGameplayPreload();
    }

    void LoadingScene::onExit(Game& game)
    {
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    void LoadingScene::update(Game& game, float dt)
    {
        t += dt;

        // Do a bit of loading each frame
        progress = game.tickGameplayPreload();

        // When ready -> go to gameplay
        if (progress >= 1.0f) {
            game.requestSceneChange(SceneId::Gameplay);
            return;
        }
    }

    void LoadingScene::render(Game& game)
    {
        // Background
        game.renderSkybox();

        // UI
        game.imguiLayer.beginFrame(); // assumes your imguiLayer has beginFrame/endFrame API

        ImGuiIO& io = ImGui::GetIO();
        const ImVec2 center(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);

        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(520, 160), ImGuiCond_Always);

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoTitleBar;

        if (ImGui::Begin("Loading", nullptr, flags))
        {
            ImGui::TextUnformatted("Loading...");
            ImGui::Spacing();

            ImGui::TextUnformatted(game.getGameplayPreloadStageName().c_str());
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 22.0f));

            ImGui::Spacing();
            ImGui::Text("Please wait");
        }
        ImGui::End();

        game.imguiLayer.endFrame();
    }

} // namespace gl3