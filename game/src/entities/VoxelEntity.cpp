#include "VoxelEntity.h"
#include "../Game.h"
#include <iostream>

namespace gl3 {
    void VoxelEntity::draw(gl3::Game* game) {
        shader.use();

        // Model matrix for this entity
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        modelMatrix = glm::translate(modelMatrix, position);
        modelMatrix = glm::scale(modelMatrix, scale);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(zRotation), glm::vec3(0,1,0));

        // MVP = projection * view * model
        glm::mat4 mvpMatrix = game->calculateMvpMatrix(position, zRotation, scale);

        // Send matrices to shader
        shader.setMatrix("model", modelMatrix);
        shader.setMatrix("mvp", mvpMatrix);

        // Optional: set entity color
        shader.setVector("vertexColor", color);

        mesh.draw();
    }


    void VoxelEntity::createPhysicsBody() {
        // Empty for now
    }
}
