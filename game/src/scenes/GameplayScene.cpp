#include "GameplayScene.h"
#include "../SceneId.h"
#include "../Game.h"
#include <GLFW/glfw3.h>

namespace gl3 {

    void GameplayScene::onEnter(Game& game) {
        // If you want, you can make gameplay init lazy here.
        // For now we just mark cursor disabled and let Game handle init once.
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        if (!initialized) {
            // Let Game run its existing initialization, but moved to a public helper.
            game.initGameplayIfNeeded();
            initialized = true;
        }
    }

    void GameplayScene::onExit(Game& /*game*/) {
    }

    void GameplayScene::update(Game& game, float /*dt*/) {
        // Escape to main menu (optional convenience)
        if (glfwGetKey(game.getWindow(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            game.requestSceneChange(SceneId::MainMenu);
            return;
        }

        game.updateGameplayFrame(); // physics + input + simulation
    }

    void GameplayScene::render(Game& game) {
        game.renderGameplayFrame();
    }

} // namespace gl3