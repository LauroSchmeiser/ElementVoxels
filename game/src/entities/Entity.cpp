#include "Entity.h"

#include <iostream>

#include "../Game.h"

namespace gl3 {
    Entity::Entity(Shader &shader, Mesh mesh, glm::vec3 position, float zRotation, glm::vec3 scale, glm::vec4 color) :
            shader(shader),
            mesh(std::move(mesh)),
            position(position),
            zRotation(zRotation),
            scale(scale),
            color(color)
    {}

    Entity::~Entity()
    {
    }




    void Entity::startContact() {

    }
}
