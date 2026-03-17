#include "../Game.h"
#include "../SceneId.h"
#include "MainMenuScene.h"

#include <imgui.h>
#include <GLFW/glfw3.h>

namespace gl3 {

    void MainMenuScene::onEnter(Game& game)
    {
        // show cursor for menu usage
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        open = true;
        game.setupSkybox();
        game.bakeNebulaCubemap(512);
    }

    void MainMenuScene::onExit(Game& game)
    {
        // optional: hide cursor when leaving menu
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    void MainMenuScene::update(Game& game, float /*dt*/)
    {
        // Typical: allow Alt+F4 / window close handled elsewhere.
        // No simulation needed here.
    }

    void MainMenuScene::render(Game& game)
    {
        setWindowTitle(game);
        // 1) Draw skybox as aesthetic background
        game.renderSkybox();

        // 2) Draw ImGui on top
        game.imgui().beginFrame();

        // Fullscreen center panel
        ImGuiIO& io = ImGui::GetIO();
        const ImVec2 display = io.DisplaySize;

        const float panelW = 420.0f;
        const float panelH = 220.0f;

        ImGui::SetNextWindowPos(ImVec2((display.x - panelW) * 0.5f, (display.y - panelH) * 0.5f));
        ImGui::SetNextWindowSize(ImVec2(panelW, panelH));

        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoTitleBar;

        if (ImGui::Begin("MainMenu", &open, flags))
        {
            ImGui::SetWindowFontScale(3);

            ImGui::SetCursorPosY(20.0f);

            ImGui::PushFont(nullptr);
            ImGui::SetCursorPosX((panelW - ImGui::CalcTextSize("Element Voxels").x) * 0.5f);
            ImGui::TextUnformatted("Element Voxels");
            ImGui::PopFont();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const ImVec2 btnSize(panelW * 0.75f, 44.0f);
            ImGui::SetCursorPosX((panelW - btnSize.x) * 0.5f);

            if (ImGui::Button("Start Game", btnSize))
            {
                game.requestSceneChange(SceneId::Loading);
            }

            ImGui::Spacing();
            ImGui::SetCursorPosX((panelW - btnSize.x) * 0.5f);

            if (ImGui::Button("Back to Desktop", btnSize))
            {
                glfwSetWindowShouldClose(game.getWindow(), true);
                //game.requestSceneChange(SceneId::GameOver);
            }
        }
        ImGui::End();

        game.imgui().endFrame();
    }

    void MainMenuScene::setWindowTitle(Game& game) {
            static double lastTime = 0.0;
            static int frames = 0;

            double currentTime = glfwGetTime();
            frames++;

            if (currentTime - lastTime >= 1.0) {
                double fps = frames / (currentTime - lastTime);
                frames = 0;
                lastTime = currentTime;

                std::string title = "Voxel Engine | FPS: " + std::to_string((int)fps);
                glfwSetWindowTitle(game.getWindow(), title.c_str());
            }
    }

} // namespace gl3