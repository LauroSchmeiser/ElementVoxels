#include <string>
#include "../../../extern/robin_hood.h"
#include "InputManager.h"

struct InputAction {
    std::string name;
    std::vector<int> keys;  // Multiple bindings for same action
    bool isPressed = false;
    bool wasJustPressed = false;
    float value = 0.0f;  // For analog input

    void update(const InputManager& input) {
        wasJustPressed = false;
        isPressed = false;

        for (int key : keys) {
            if (input.isKeyPressed(key)) {
                isPressed = true;
                if (input.isKeyJustPressed(key)) {
                    wasJustPressed = true;
                }
                value = 1.0f;
                break;
            }
        }
    }
};

class InputActionMap {
public:
    InputAction& addAction(const std::string& name, std::vector<int> keys) {
        return actions[name] = {name, keys};
    }

    void update(const InputManager& input) {
        for (auto& [name, action] : actions) {
            action.update(input);
        }
    }

    InputAction& operator[](const std::string& name) {
        return actions[name];
    }

private:
    robin_hood::unordered_map<std::string, InputAction> actions;
};