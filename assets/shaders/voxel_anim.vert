#version 330 core

layout(location = 0) in vec3 aPos;

layout(location = 1) in vec3 aInstanceNormal;
layout(location = 2) in vec3 aInstanceColor;
layout(location = 3) in vec4 aInstancePosScale;

out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;

uniform mat4 pv;

void main() {
    vec3 worldPos = aInstancePosScale.xyz + aPos * aInstancePosScale.w;
    fragPos = worldPos;
    vertexColor = aInstanceColor;
    normal = normalize(aInstanceNormal);

    gl_Position = pv * vec4(worldPos, 1.0);
}