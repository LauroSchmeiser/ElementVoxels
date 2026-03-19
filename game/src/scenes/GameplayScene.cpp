#include "GameplayScene.h"
#include "../SceneId.h"
#include "../Game.h"
#include "imgui.h"
#include <GLFW/glfw3.h>

namespace gl3 {

    void GameplayScene::onEnter(Game& game) {
        glfwSetInputMode(game.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        escWasDown = false;
        game.setPaused(false);
    }

    void GameplayScene::onExit(Game& game) {
        game.setPaused(false);
    }

    void GameplayScene::update(Game& game, float /*dt*/) {
        // Toggle pause on ESC press (edge)
        const bool escDown = glfwGetKey(game.getWindow(), GLFW_KEY_ESCAPE) == GLFW_PRESS;
        if (escDown && !escWasDown) {
            game.togglePaused();
        }
        escWasDown = escDown;

        // If paused: do NOT advance gameplay simulation/time
        if (game.isPaused()) {
            // No updateGameplayFrame()
            // (Still allow sceneManager changes from pause menu buttons)
            return;
        }

        game.updateGameplayFrame();
    }

    void GameplayScene::render(Game& game) {
        game.renderGameplayFrame();
    }

} // namespace gl3