#include "../Game.h"
#include "../SceneId.h"
#include "MainMenuScene.h"
#include "../Assets.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include "../../../extern/stb_image.h"

namespace gl3 {

    void MainMenuScene::onEnter(Game& game)
    {
        // show cursor for menu usage
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        open = true;
        game.setupSkybox();
        game.bakeNebulaCubemap(512);

        loadTitleTexture(gl3::resolveAssetPath("textures/title.png").string());

        loadStartButtonTexture(gl3::resolveAssetPath("textures/cobble.jpg").string());
        loadSettingsButtonTexture(gl3::resolveAssetPath("textures/marble_rock_03_diff_4k.jpg").string());
        loadExitButtonTexture(gl3::resolveAssetPath("textures/aerial_rocks_02_diff_4k.jpg").string());
        game.applyAudioSettings();

    }

    void MainMenuScene::onExit(Game& game)
    {
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        destroyTitleTexture();
        destroyStartButtonTexture();
        destroySettingsButtonTexture();
        destroyExitButtonTexture();
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
        // after: ImGuiIO& io = ImGui::GetIO();
        const ImVec2 display = io.DisplaySize;

        // Draw title ABOVE panel (outside it), centered on screen
        if (titleTex != 0 && titleW > 0 && titleH > 0) {
            const float maxW = display.x * 0.65f;
            float drawW = (float)titleW;
            float drawH = (float)titleH;
            const float scale = std::min(1.0f, maxW / drawW);
            drawW *= scale;
            drawH *= scale;

            ImGui::SetNextWindowPos(ImVec2((display.x - drawW) * 0.5f, 40.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(drawW, drawH), ImGuiCond_Always);

            ImGuiWindowFlags logoFlags =
                    ImGuiWindowFlags_NoDecoration |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoInputs |
                    ImGuiWindowFlags_NoBackground;

            ImGui::Begin("MainMenuTitleImage", nullptr, logoFlags);
            ImGui::Image((ImTextureID)(intptr_t)titleTex, ImVec2(drawW, drawH));
            ImGui::End();
        }


        const float panelW = 960.0f;
        const float panelH = 620.0f;

        ImGui::SetNextWindowPos(ImVec2((display.x - panelW) * 0.5f, (display.y - panelH) * 0.72f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Always);

        // remove black background and border outline
        ImGui::SetNextWindowBgAlpha(0.0f);

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 18.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 18.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);


        ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoTitleBar;

        static bool showSettings = false;

        if (ImGui::Begin("MainMenu", &open, flags)) {
            const ImVec2 btnSize(panelW * 0.82f, 88.0f);
            ImGui::SetWindowFontScale(1.25f);

            if (!showSettings) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 24.0f);

                ImGui::SetCursorPosY(120.0f);

                // START BUTTON - The DrawTexturedMenuButton now handles hover sound internally
                ImGui::SetCursorPosX((panelW - btnSize.x) * 0.5f);
                if (DrawTexturedMenuButton("btn_start", "Start Game",
                                           (ImTextureID)(intptr_t)startBtnTex, btnSize)) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    game.requestSceneChange(SceneId::Loading);
                }

                ImGui::Dummy(ImVec2(0.0f, 18.0f));

                // SETTINGS BUTTON
                ImGui::SetCursorPosX((panelW - btnSize.x) * 0.5f);
                if (DrawTexturedMenuButton("btn_settings", "Settings",
                                           (ImTextureID)(intptr_t)settingsBtnTex, btnSize)) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    showSettings = true;
                }

                ImGui::Dummy(ImVec2(0.0f, 18.0f));

                // EXIT BUTTON
                ImGui::SetCursorPosX((panelW - btnSize.x) * 0.5f);
                if (DrawTexturedMenuButton("btn_desktop", "Back to desktop",
                                           (ImTextureID)(intptr_t)exitBtnTex, btnSize)) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    glfwSetWindowShouldClose(game.getWindow(), true);
                }

                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(4);
            } else
            {
                ImGui::SetWindowFontScale(3.5f);
                ImGui::SetCursorPosY(28.0f);
                ImGui::SetCursorPosX(28.0f);

                ImGui::TextUnformatted("Settings");
                ImGui::Dummy(ImVec2(0.0f, 12.0f));
                ImGui::Separator();
                ImGui::Dummy(ImVec2(0.0f, 18.0f));

                bool changed = false;

                ImGui::PushItemWidth(panelW * 0.42f);

                changed |= ImGui::SliderFloat(" Mouse Sensitivity", &game.settings.sensitivity, 0.01f, 1.0f, "%.3f");
                const char* sensitivityBtnId = "settings_sensitivity";
                if (ImGui::IsItemHovered() && lastHoveredButton != sensitivityBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = sensitivityBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                changed |= ImGui::SliderFloat(" Master Volume", &game.settings.masterVolume, 0.0f, 1.0f, "%.2f");
                const char* masterBtnId = "settings_master";
                if (ImGui::IsItemHovered() && lastHoveredButton != masterBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = masterBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                changed |= ImGui::SliderFloat(" SFX Volume", &game.settings.sfxVolume, 0.0f, 1.0f, "%.2f");
                const char* sfxBtnId = "settings_sfx";
                if (ImGui::IsItemHovered() && lastHoveredButton != sfxBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = sfxBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                changed |= ImGui::SliderFloat(" Music Volume", &game.settings.musicVolume, 0.0f, 1.0f, "%.2f");
                const char* musicBtnId = "settings_music";
                if (ImGui::IsItemHovered() && lastHoveredButton != musicBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = musicBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                changed |= ImGui::SliderFloat(" Gamma", &game.settings.gamma, 1.6f, 3.0f, "%.2f");
                const char* gammaBtnId = "settings_gamma";
                if (ImGui::IsItemHovered() && lastHoveredButton != gammaBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = gammaBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                changed |= ImGui::SliderFloat(" Brightness", &game.settings.brightness, 0.5f, 1.5f, "%.2f");
                const char* brightnessBtnId = "settings_brightness";
                if (ImGui::IsItemHovered() && lastHoveredButton != brightnessBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = brightnessBtnId;
                }
                ImGui::Dummy(ImVec2(0.0f, 14.0f));

                const char *modeLabels[] = {"Fullscreen", "Windowed", "Borderless"};
                int mode = static_cast<int>(game.settings.displayMode);
                if (ImGui::Combo(" Display Mode", &mode, modeLabels, IM_ARRAYSIZE(modeLabels))) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    game.settings.displayMode = static_cast<Game::DisplayMode>(mode);
                    changed = true;
                }
                const char* displayModeBtnId = "settings_displayMode";
                if (ImGui::IsItemHovered() && lastHoveredButton != displayModeBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = displayModeBtnId;
                }

                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                std::vector<const char *> resLabels;
                resLabels.reserve(game.commonResolutions.size());
                for (auto &r: game.commonResolutions) resLabels.push_back(r.label);

                if (ImGui::Combo(" Resolution", &game.settings.resolutionIndex, resLabels.data(),
                                 (int) resLabels.size())) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    changed = true;
                }
                const char* pickResolutionBtnId = "settings_pickResolution";
                if (ImGui::IsItemHovered() && lastHoveredButton != pickResolutionBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = pickResolutionBtnId;
                }

                ImGui::PopItemWidth();

                ImGui::Dummy(ImVec2(0.0f, 22.0f));

                const ImVec2 bigBtn1(860.0f, 52.0f);
                const ImVec2 bigBtn2(419.0f, 52.0f);

                if (ImGui::Button("Set Native Resolution", bigBtn1)) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    game.pickResolutionFromNativeMonitor(false);
                }
                const char* setNativeBtnId = "settings_setNative";
                if (ImGui::IsItemHovered() && lastHoveredButton != setNativeBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = setNativeBtnId;
                }

                ImGui::Dummy(ImVec2(0.0f, 22.0f));

                if (ImGui::Button("Back", bigBtn2)) {
                    g_SoundManager.playSound(SoundID::MenuClose, 1.0f, 0.5f);
                    showSettings = false;
                }
                const char* backBtnId = "settings_back";
                if (ImGui::IsItemHovered() && lastHoveredButton != backBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = backBtnId;
                }

                ImGui::SameLine();
                ImGui::Dummy(ImVec2(11.0f, 0.0f));
                ImGui::SameLine();
                if (ImGui::Button("Apply Display", bigBtn2)) {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    game.applyDisplaySettings();
                }
                const char* applyBtnId = "settings_apply";
                if (ImGui::IsItemHovered() && lastHoveredButton != applyBtnId) {
                    g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                    lastHoveredButton = applyBtnId;
                }
                if(changed)
                {
                    g_SoundManager.playSound(SoundID::ButtonClick, 1.0f, 1.0f);
                    game.applyAudioSettings();
                }

            }
        }
        ImGui::End();

        ImGui::PopStyleVar(4);
        game.imgui().endFrame();
    }

    void MainMenuScene::setWindowTitle(Game& game) {
            static double lastTime = 0.0;
            static int frames = 0;

            double currentTime = glfwGetTime();
            frames++;

           /* if (currentTime - lastTime >= 1.0) {
                double fps = frames / (currentTime - lastTime);
                frames = 0;
                lastTime = currentTime;

                std::string title = "Voxel Engine | FPS: " + std::to_string((int)fps);
                glfwSetWindowTitle(game.getWindow(), title.c_str());
            }*/
    }

    bool MainMenuScene::loadTitleTexture(const std::string& path)
    {
        destroyTitleTexture();

        stbi_set_flip_vertically_on_load(0);
        int comp = 0;
        unsigned char* data = stbi_load(path.c_str(), &titleW, &titleH, &comp, STBI_rgb_alpha);
        if (!data) return false;

        glGenTextures(1, &titleTex);
        glBindTexture(GL_TEXTURE_2D, titleTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, titleW, titleH, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return true;
    }

    void MainMenuScene::destroyTitleTexture()
    {
        if (titleTex != 0) {
            glDeleteTextures(1, &titleTex);
            titleTex = 0;
        }
        titleW = titleH = 0;
    }

    bool MainMenuScene::loadStartButtonTexture(const std::string& path)
    {
        destroyStartButtonTexture();

        stbi_set_flip_vertically_on_load(0);
        int w=0, h=0, comp=0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, STBI_rgb_alpha);
        if (!data) return false;

        glGenTextures(1, &startBtnTex);
        glBindTexture(GL_TEXTURE_2D, startBtnTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return true;
    }

    void MainMenuScene::destroyStartButtonTexture()
    {
        if (startBtnTex != 0) {
            glDeleteTextures(1, &startBtnTex);
            startBtnTex = 0;
        }
    }

    bool MainMenuScene::loadSettingsButtonTexture(const std::string& path)
    {
        destroySettingsButtonTexture();

        stbi_set_flip_vertically_on_load(0);
        int w=0, h=0, comp=0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, STBI_rgb_alpha);
        if (!data) return false;

        glGenTextures(1, &settingsBtnTex);
        glBindTexture(GL_TEXTURE_2D, settingsBtnTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return true;
    }

    void MainMenuScene::destroySettingsButtonTexture()
    {
        if (settingsBtnTex != 0) {
            glDeleteTextures(1, &settingsBtnTex);
            settingsBtnTex = 0;
        }
    }

    bool MainMenuScene::loadExitButtonTexture(const std::string& path)
    {
        destroyExitButtonTexture();

        stbi_set_flip_vertically_on_load(0);
        int w=0, h=0, comp=0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, STBI_rgb_alpha);
        if (!data) return false;

        glGenTextures(1, &exitBtnTex);
        glBindTexture(GL_TEXTURE_2D, exitBtnTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return true;
    }

    void MainMenuScene::destroyExitButtonTexture()
    {
        if (exitBtnTex != 0) {
            glDeleteTextures(1, &exitBtnTex);
            exitBtnTex = 0;
        }
    }

    bool MainMenuScene::DrawTexturedMenuButton(const char* id,
                                               const char* label,
                                               ImTextureID tex,
                                               const ImVec2& size)
    {
        if (!tex) {
            // Check hover state for regular button
            bool hovered = ImGui::IsItemHovered();
            if (hovered && lastHoveredButton != id) {
                g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
                lastHoveredButton = id;
            }
            if (!hovered && lastHoveredButton == id) {
                lastHoveredButton = nullptr;
            }
            return ImGui::Button(label, size);
        }

        ImGui::SetWindowFontScale(5.0);

        ImVec2 basePos = ImGui::GetCursorScreenPos();

        // Create the invisible button - this is where the interaction happens
        ImGui::InvisibleButton(id, size);

        // Check hover state IMMEDIATELY after creating the button
        const bool hovered = ImGui::IsItemHovered();
        const bool held = ImGui::IsItemActive();
        const bool clicked = ImGui::IsItemClicked();

        // Play hover sound if this is a new button being hovered
        if (hovered && lastHoveredButton != id) {
            g_SoundManager.playSound(SoundID::ButtonHover, 1.0f, 1.0f);
            lastHoveredButton = id;
        }
        // Reset if not hovering
        if (!hovered && lastHoveredButton == id) {
            lastHoveredButton = nullptr;
        }

        // Hover / press effects
        float scale = hovered ? 1.035f : 1.0f;
        if (held) scale = 0.985f;

        ImVec2 drawSize(size.x * scale, size.y * scale);
        ImVec2 drawPos(
                basePos.x + (size.x - drawSize.x) * 0.5f,
                basePos.y + (size.y - drawSize.y) * 0.5f
        );

        if (held) {
            drawPos.y += 3.0f;
        }

        ImVec4 tint(1.0f, 1.0f, 1.0f, 1.0f);
        if (hovered) {
            tint = ImVec4(1.10f, 1.05f, 1.15f, 1.0f);
        }
        if (held) {
            tint = ImVec4(0.82f, 0.82f, 0.90f, 1.0f);
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();

        if (hovered) {
            ImVec2 glowMin(drawPos.x - 8.0f, drawPos.y - 6.0f);
            ImVec2 glowMax(drawPos.x + drawSize.x + 8.0f, drawPos.y + drawSize.y + 6.0f);
            dl->AddRectFilled(
                    glowMin,
                    glowMax,
                    IM_COL32(120, 180, 255, held ? 40 : 55),
                    28.0f
            );
        }

        dl->AddImageRounded(
                tex,
                drawPos,
                ImVec2(drawPos.x + drawSize.x, drawPos.y + drawSize.y),
                ImVec2(0, 0),
                ImVec2(1, 1),
                ImGui::ColorConvertFloat4ToU32(tint),
                24.0f
        );

        if (hovered) {
            dl->AddRect(
                    drawPos,
                    ImVec2(drawPos.x + drawSize.x, drawPos.y + drawSize.y),
                    IM_COL32(255, 255, 255, held ? 70 : 110),
                    24.0f,
                    0,
                    2.0f
            );
        }

        // Text rendering
        ImVec2 textSize = ImGui::CalcTextSize(label);
        ImVec2 textPos(
                drawPos.x + (drawSize.x - textSize.x) * 0.5f,
                drawPos.y + (drawSize.y - textSize.y) * 0.5f
        );

        if (held) {
            textPos.y += 1.0f;
        }

        const ImU32 outlineCol = IM_COL32(0, 0, 0, 220);
        const float o = 2.0f;

        dl->AddText(ImVec2(textPos.x - o, textPos.y), outlineCol, label);
        dl->AddText(ImVec2(textPos.x + o, textPos.y), outlineCol, label);
        dl->AddText(ImVec2(textPos.x, textPos.y - o), outlineCol, label);
        dl->AddText(ImVec2(textPos.x, textPos.y + o), outlineCol, label);
        dl->AddText(ImVec2(textPos.x - o, textPos.y - o), outlineCol, label);
        dl->AddText(ImVec2(textPos.x + o, textPos.y - o), outlineCol, label);
        dl->AddText(ImVec2(textPos.x - o, textPos.y + o), outlineCol, label);
        dl->AddText(ImVec2(textPos.x + o, textPos.y + o), outlineCol, label);
        dl->AddText(textPos, IM_COL32(255, 255, 255, 255), label);

        return clicked;
    }

}