#include "SunBillboard.h"
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

SunBillboard::SunBillboard() {}
SunBillboard::~SunBillboard() { shutdown(); }

void SunBillboard::init(int maxInst) {
    maxInstances = maxInst;

    // load shader (paths relative to your executable working dir)
    shader = new gl3::Shader("shaders/sun_billboard.vert", "shaders/sun_billboard.frag");

    // quad positions (two triangles) in [-0.5..0.5] space
    float quadVerts[] = {
            -0.5f, -0.5f,
            0.5f, -0.5f,
            -0.5f,  0.5f,
            0.5f, -0.5f,
            0.5f,  0.5f,
            -0.5f,  0.5f
    };

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glVertexAttribDivisor(0, 0); // per-vertex

    // instance buffer: vec4 posScale (xyz pos, w scale)
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, maxInstances * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);

    // instance color buffer: vec4 (rgb, a)
    glGenBuffers(1, &colorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, maxInstances * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glVertexAttribDivisor(2, 1);

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SunBillboard::render(const std::vector<SunInstance>& instances, const glm::mat4& view, const glm::mat4& proj, float time) {
    if (instances.empty()) return;

    int count = (int)instances.size();
    if (count > maxInstances) {
        // reallocate instance buffers to fit more instances
        maxInstances = count * 2;
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, maxInstances * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
        glBufferData(GL_ARRAY_BUFFER, maxInstances * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    }

    // fill instance data
    std::vector<float> posScaleData(count * 4);
    std::vector<float> colorData(count * 4);
    for (int i = 0; i < count; ++i) {
        const auto &s = instances[i];
        posScaleData[i*4 + 0] = s.position.x;
        posScaleData[i*4 + 1] = s.position.y;
        posScaleData[i*4 + 2] = s.position.z;
        posScaleData[i*4 + 3] = s.scale;

        colorData[i*4 + 0] = s.color.r;
        colorData[i*4 + 1] = s.color.g;
        colorData[i*4 + 2] = s.color.b;
        colorData[i*4 + 3] = 1.0f;
    }

    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, posScaleData.size() * sizeof(float), posScaleData.data());
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, colorData.size() * sizeof(float), colorData.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // setup GL state for additive billboard rendering
    // setup GL state for additive billboard rendering
    glDepthFunc(GL_LESS);
    glBlendFunc(GL_ONE, GL_ONE); // additive glow
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);           // still test depth so the billboard is occluded by nearer geometry
    glDepthMask(GL_FALSE);             // but don't write depth -> avoids occluding other things            // don't write depth (still test)

    shader->use();
    shader->setMatrix("view", view);
    shader->setMatrix("proj", proj);
    shader->setFloat("time", time);

    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
    glBindVertexArray(0);

    // restore GL state
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void SunBillboard::shutdown() {
    if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
    if (instanceVBO) { glDeleteBuffers(1, &instanceVBO); instanceVBO = 0; }
    if (colorVBO) { glDeleteBuffers(1, &colorVBO); colorVBO = 0; }
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (shader) { delete shader; shader = nullptr; }
}