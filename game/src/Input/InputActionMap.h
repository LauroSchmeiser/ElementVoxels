#include <string>
#include "../../../extern/robin_hood.h"
#include "InputManager.h"

struct InputAction {
    std::string name;
    std::vector<int> keys;  // Multiple bindings for same action

    // State tracking
    bool isPressed = false;      // Currently pressed (including hold)
    bool wasJustPressed = false; // Pressed this frame
    bool wasJustReleased = false; // Released this frame
    bool isHeld = false;         // Being held down (after initial press)
    float holdTime = 0.0f;       // How long the key has been held (in frames or seconds)
    float value = 0.0f;          // For analog input

    void update(const InputManager& input, float deltaTime = 0.0f) {
        // Reset frame-specific states
        wasJustPressed = false;
        wasJustReleased = false;

        bool previousPressed = isPressed;
        isPressed = false;

        // Check all bound keys
        for (int key : keys) {
            if (input.isKeyPressed(key)) {
                isPressed = true;

                if (input.isKeyJustPressed(key)) {
                    wasJustPressed = true;
                    isHeld = false;  // Reset hold state on new press
                    holdTime = 0.0f;
                }

                value = 1.0f;  // You could modify this for analog input
                break;
            }
        }

        // Handle hold state
        if (isPressed) {
            if (!wasJustPressed) {
                isHeld = true;  // Key is being held (not just pressed this frame)
                holdTime += deltaTime;  // Accumulate hold time
            }
        } else {
            // Key is not pressed
            isHeld = false;
            holdTime = 0.0f;

            // Check if it was just released
            if (previousPressed && !isPressed) {
                wasJustReleased = true;
            }
        }
    }

    // Helper methods for common queries
    bool isActivelyPressed() const { return isPressed; }
    bool isJustPressed() const { return wasJustPressed; }
    bool isJustReleased() const { return wasJustReleased; }
    bool isHolding() const { return isHeld; }
    float getHoldTime() const { return holdTime; }
    float getNormalizedHoldTime(float maxHoldTime) const {
        return glm::clamp(holdTime / maxHoldTime, 0.0f, 1.0f);
    }
};

class InputActionMap {
public:
    InputAction& addAction(const std::string& name, std::vector<int> keys) {
        return actions[name] = {name, keys};
    }

    void update(const InputManager& input, float deltaTime = 0.0f) {
        for (auto& [name, action] : actions) {
            action.update(input, deltaTime);
        }
    }

    InputAction& operator[](const std::string& name) {
        return actions[name];
    }

    // Get all currently held actions
    std::vector<InputAction*> getHeldActions() {
        std::vector<InputAction*> heldActions;
        for (auto& [name, action] : actions) {
            if (action.isHeld) {
                heldActions.push_back(&action);
            }
        }
        return heldActions;
    }

private:
    robin_hood::unordered_map<std::string, InputAction> actions;
};