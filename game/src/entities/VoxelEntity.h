#pragma once

#include <iostream>
#include "Entity.h"
#include "../rendering/VoxelMesher.h"
#include "../rendering/Shader.h"
#include "../rendering/Mesh.h"
#include "../Game.h"

namespace gl3 {

    class VoxelEntity : public Entity {
    public:
        VoxelEntity(Shader shader, Mesh mesh, glm::vec3 position)
                : Entity(shader, std::move(mesh), position, 0.0f, glm::vec3(1.0f), glm::vec4(1.0f))
        {}

    void update(Game* game, float dt) override {
            // Nothing yet — static voxel chunk
        }

        void draw(Game *game) override;


    protected:
        void createPhysicsBody() override;
    };

}
