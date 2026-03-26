#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 pv;
uniform mat4 model;

out vec3 vWorldPos;
out vec3 vLocalPos;

uniform vec3 uCenter;

void main() {
    vec4 wp = model * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vLocalPos = wp.xyz - uCenter;

    gl_Position = pv * wp;
}