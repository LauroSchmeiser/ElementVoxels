//
// Created by lauro on 17/03/2026.
//

#pragma once

#include "../Game.h"

namespace gl3 {
    class GameOverScene final : public IScene {
    public:
        void onEnter(Game &game) override;

        void onExit(Game &game) override;

        void update(Game &game, float dt) override;

        void render(Game &game) override;

    private:
        int selected = 0; // 0 = Start, 1 = Exit
        double blinkT = 0.0;
        bool open = true;


        void setWindowTitle(Game &game);
    };
}
