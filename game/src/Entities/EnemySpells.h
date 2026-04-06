#pragma once
#include <glm/glm.hpp>
#include "Enemy.h"

namespace gl3 {

    struct EnemySpellContext {
        glm::vec3 casterPos;
        glm::vec3 targetPos;
        glm::vec3 aimDir; // normalized
    };

    // hook these into Game (spawn particles / physics bodies / whatever)
    void castEnemySpell(EnemySpellId id, const EnemySpellContext& ctx);

}