#pragma once
namespace gl3 {
    class Game;

    class IScene {
    public:
        virtual ~IScene() = default;

        virtual void onEnter(Game& game) = 0;
        virtual void onExit(Game& game) = 0;

        virtual void update(Game& game, float dt) = 0;
        virtual void render(Game& game) = 0;
    };
}