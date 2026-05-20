#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

uniform mat4 view;
uniform mat4 projection;

uniform vec3 uWorldPos;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uSize;
uniform float uRotation;

out vec2 vUV;
out vec2 vLocal;

void main()
{
    float c = cos(uRotation);
    float s = sin(uRotation);

    vec2 rp = vec2(
    aPos.x * c - aPos.y * s,
    aPos.x * s + aPos.y * c
    );

    vec3 worldPos =
    uWorldPos +
    uCameraRight * (rp.x * uSize) +
    uCameraUp    * (rp.y * uSize);

    vUV = aUV;
    vLocal = aPos;

    gl_Position = projection * view * vec4(worldPos, 1.0);
}