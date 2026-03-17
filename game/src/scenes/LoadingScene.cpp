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
        //glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

       // game.setupSkybox();
        //game.bakeNebulaCubemap(512);
        // start preload pipeline
        game.beginGameplayPreload(/*newRun=*/true);
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
        // Make sure the frame is cleared BEFORE drawing skybox
        glViewport(0, 0, game.getWindowWidth(), game.getWindowHeight()); // if you have getters; otherwise omit
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Ensure depth state is sane for skybox
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);           // typical for skybox
        glDepthFunc(GL_LEQUAL);          // typical for skybox

        game.renderSkybox();

        // Restore defaults for later rendering (and ImGui doesn’t care, but gameplay will)
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        // UI
        game.imguiLayer.beginFrame();

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
        }
        ImGui::End();

        game.imguiLayer.endFrame();
    }
} // namespace gl3