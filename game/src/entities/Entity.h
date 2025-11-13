#pragma once

#include "glm/vec3.hpp"
#include "../rendering/Mesh.h"
#include "../rendering/Shader.h"
#include "GLFW/glfw3.h"

namespace gl3{
    class Game;

    class Entity {
    public:
        Entity(Shader &shader,
               Mesh mesh,
               glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
               float zRotation = 0.0f,
               glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f),
               glm::vec4 color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)
            );

        virtual ~Entity();

        virtual void update(Game *game, float deltaTime) {}

        virtual void draw(Game *game)=0;

        [[nodiscard]] const glm::vec3 &getPosition() const {return position; }
        [[nodiscard]] float getZRotation() const { return zRotation; }
        [[nodiscard]] const glm::vec3 &getScale() const { return scale; }
        void setPosition(const glm::vec3 &position) { Entity::position = position; }
        void setZRotation(float zRotation) { Entity::zRotation = zRotation; }
        void setScale(const glm::vec3 &scale) { Entity::scale = scale; }
        virtual void startContact();

        glm::vec4 color;
        glm::vec3 scale;
        float zRotation;
    protected:
        virtual void createPhysicsBody() = 0;
        glm::vec3 position;

    public:
        Shader &shader;
        Mesh mesh;
    };
}



