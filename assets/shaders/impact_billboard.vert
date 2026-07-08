#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

// per-instance
layout(location = 2) in vec4 iPosSize;      // xyz pos, w size
layout(location = 3) in vec4 iColor;        // rgba
layout(location = 4) in vec4 iRotLifeKind;  // x rot, y life01, z kind

uniform mat4 view;
uniform mat4 projection;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;

out vec2 vUV;
out vec2 vLocal;
out vec4 vColor;
flat out int vKind;
out float vLife01;

void main()
{
    float rot = iRotLifeKind.x;
    float c = cos(rot);
    float s = sin(rot);

    vec2 rp = vec2(
    aPos.x * c - aPos.y * s,
    aPos.x * s + aPos.y * c
    );

    vec3 worldPos =
    iPosSize.xyz +
    uCameraRight * (rp.x * iPosSize.w) +
    uCameraUp    * (rp.y * iPosSize.w);

    vUV = aUV;
    vLocal = aPos;
    vColor = iColor;
    vLife01 = iRotLifeKind.y;
    vKind = int(iRotLifeKind.z + 0.5);

    gl_Position = projection * view * vec4(worldPos, 1.0);
}