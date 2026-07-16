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
        game.applyAudioSettings();
        game.audio.setPauseAll(true);
        game.musicHandle = game.audio.play(*game.backgroundMusic);
    }

    void GameplayScene::onExit(Game& game) {
        game.setPaused(false);
    }

    void GameplayScene::update(Game& game, float /*dt*/) {
        const bool escDown = glfwGetKey(game.getWindow(), GLFW_KEY_ESCAPE) == GLFW_PRESS;
        if (escDown && !escWasDown) {
            game.togglePaused();
        }
        escWasDown = escDown;

        if (game.isPaused()) {
            return;
        }

        game.updateGameplayFrame();
    }

    void GameplayScene::render(Game& game) {
        game.renderGameplayFrame();
    }

} // namespace gl3