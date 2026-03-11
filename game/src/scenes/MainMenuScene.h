#pragma once
#include "../IScene.h"
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


        void setWindowTitle(Game& game);
    };

} // namespace gl3