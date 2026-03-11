#pragma once
#include "../IScene.h"

namespace gl3 {

    class LoadingScene final : public IScene {
    public:
        void onEnter(Game& game) override;
        void onExit(Game& game) override;

        void update(Game& game, float dt) override;
        void render(Game& game) override;

    private:
        float progress = 0.0f;
        double t = 0.0;
    };

} // namespace gl3