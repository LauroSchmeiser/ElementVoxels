#pragma once
#include "../IScene.h"
#include "imgui.h"
#include <string>


namespace gl3 {

// Minimal immediate-mode UI (no imgui): draw simple text buttons in the window title,
// and use keyboard input to activate.
// You can later swap render() to a real GUI.
    class MainMenuScene final : public IScene {
    public:
        void onEnter(Game& game) override;
        void onExit(Game& game) override;

        void update(Game& game, float dt) override;
        void render(Game& game) override;

    private:
        int selected = 0; // 0 = Start, 1 = Exit
        double blinkT = 0.0;
        bool open = true;

        GLuint titleTex = 0;
        int titleW = 0;
        int titleH = 0;

        GLuint startBtnTex = 0;
        GLuint settingsBtnTex = 0;
        GLuint exitBtnTex = 0;

        bool loadStartButtonTexture(const std::string& path);
        bool loadSettingsButtonTexture(const std::string& path);
        bool loadExitButtonTexture(const std::string& path);

        void destroyStartButtonTexture();
        void destroySettingsButtonTexture();
        void destroyExitButtonTexture();


        void setWindowTitle(Game& game);
        bool loadTitleTexture(const std::string& path);
        void destroyTitleTexture();

        bool DrawTexturedMenuButton(const char *id, const char *label, void *tex, const ImVec2 &size);
    };

} // namespace gl3