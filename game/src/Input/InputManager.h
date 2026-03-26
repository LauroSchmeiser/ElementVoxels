#include <unordered_map>
#include "GLFW/glfw3.h"
#include "glm/vec2.hpp"
#include "../../../extern/robin_hood.h"

class InputManager {
public:
    enum class KeyState {
        RELEASED = 0,
        PRESSED,
        JUST_PRESSED,  // Pressed this frame
        JUST_RELEASED  // Released this frame
    };

    void trackKey(int glfwKey) {
        keyStates[glfwKey] = KeyState::RELEASED;
    }

    void trackKeys(const std::vector<int>& keys) {
        for (int key : keys) {
            trackKey(key);
        }
    }

    void trackMouseButton(int button) {
        keyStates[button] = KeyState::RELEASED;
    }

    void update(GLFWwindow* window) {
        for (auto& [key, state] : keyStates) {
            bool isPressed = glfwGetKey(window, key) == GLFW_PRESS;

            if (isPressed) {
                state = (state == KeyState::PRESSED || state == KeyState::JUST_PRESSED)
                        ? KeyState::PRESSED
                        : KeyState::JUST_PRESSED;
            } else {
                state = (state == KeyState::RELEASED || state == KeyState::JUST_RELEASED)
                        ? KeyState::RELEASED
                        : KeyState::JUST_RELEASED;
            }
        }

        trackMouseButton(GLFW_MOUSE_BUTTON_RIGHT);

        double x, y;
        glfwGetCursorPos(window, &x, &y);
        mouseDelta = glm::vec2(x - lastMousePos.x, y - lastMousePos.y);
        lastMousePos = glm::vec2(x, y);
    }

    KeyState getKey(int glfwKey) const {
        auto it = keyStates.find(glfwKey);
        return it != keyStates.end() ? it->second : KeyState::RELEASED;
    }

    bool isKeyPressed(int glfwKey) const {
        auto state = getKey(glfwKey);
        return state == KeyState::PRESSED || state == KeyState::JUST_PRESSED;
    }

    bool isKeyJustPressed(int glfwKey) const {
        return getKey(glfwKey) == KeyState::JUST_PRESSED;
    }

private:
    robin_hood::unordered_map<int, KeyState> keyStates;
    glm::vec2 lastMousePos;
    glm::vec2 mouseDelta;
};