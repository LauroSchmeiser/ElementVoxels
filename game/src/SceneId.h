#pragma once
#include <cstdint>

namespace gl3 {
    enum class SceneId : uint8_t {
        MainMenu = 0,
        Loading = 1,
        Gameplay = 2,
    };
}