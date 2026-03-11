#pragma once
#include "../IScene.h"

namespace gl3 {

// Minimal wrapper scene: calls the existing "gameplay loop pieces" that are currently in Game::run().
// We'll refactor Game::run() to use scenes, but we keep it simple: Game provides a "frameStepGameplay()".
    class GameplayScene final : public IScene {
    public:
        void onEnter(Game& game) override;
        void onExit(Game& game) override;

        void update(Game& game, float dt) override;
        void render(Game& game) override;

    private:
        bool initialized = false;
    };

} // namespace gl3