#include <iostream>
#include "game/src/Game.h"

int main() {
    // Create the game instance
    gl3::Game game(1920, 1200, "Voxel Test");

    try {
        game.run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
