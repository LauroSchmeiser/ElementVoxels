#include "SceneManager.h"
#include "Game.h"
#include <stdexcept>

namespace gl3 {

    void SceneManager::registerScene(SceneId id, std::unique_ptr<IScene> scene) {
        if (!scene) return;
        scenes[id] = std::move(scene);
    }

    void SceneManager::requestChange(SceneId id) {
        hasPending = true;
        pendingId = id;
    }

    void SceneManager::applyPendingChange() {
        if (!hasPending) return;

        // exit old
        if (currentScene) {
            currentScene->onExit(game);
        }

        // enter new
        auto it = scenes.find(pendingId);
        if (it == scenes.end() || !it->second) {
            throw std::runtime_error("SceneManager: requested scene not registered");
        }

        currentScene = it->second.get();
        currentIdValue = pendingId;
        hasPending = false;

        currentScene->onEnter(game);
    }

} // namespace gl3