#pragma once
#include <memory>
#include <unordered_map>
#include "SceneId.h"
#include "IScene.h"

namespace gl3 {
    class Game;

    class SceneManager {
    public:
        explicit SceneManager(Game& game) : game(game) {}

        void registerScene(SceneId id, std::unique_ptr<IScene> scene);
        void requestChange(SceneId id);
        void applyPendingChange(); // call once per frame (after update)

        IScene* current() const { return currentScene; }
        SceneId currentId() const { return currentIdValue; }

    private:
        Game& game;

        std::unordered_map<SceneId, std::unique_ptr<IScene>> scenes;

        IScene* currentScene = nullptr;
        SceneId currentIdValue = SceneId::MainMenu;

        bool hasPending = false;
        SceneId pendingId = SceneId::MainMenu;
    };
}