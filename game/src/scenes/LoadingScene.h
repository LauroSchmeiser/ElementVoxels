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
        int selectedSlot = 0;
        int selectedCategory = -1; // 0 material,1 form,2 size,3 range,4 type
    };

} // namespace gl3