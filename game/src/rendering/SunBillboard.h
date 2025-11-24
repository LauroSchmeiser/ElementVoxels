#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "Shader.h"

struct SunInstance {
    glm::vec3 position;
    float scale;
    glm::vec3 color;
};

class SunBillboard {
public:
    SunBillboard();
    ~SunBillboard();

    // call once at init (maxInstances can be small; dynamic is supported)
    void init(int maxInstances);

    // call every frame with current view/proj/time
    // instances: vector of SunInstance
    void render(const std::vector<SunInstance>& instances, const glm::mat4& view, const glm::mat4& proj, float time);

    void shutdown();

private:
    unsigned int quadVBO = 0;
    unsigned int vao = 0;
    unsigned int instanceVBO = 0;
    unsigned int colorVBO = 0;
    int maxInstances = 0;
    gl3::Shader* shader = nullptr;
};