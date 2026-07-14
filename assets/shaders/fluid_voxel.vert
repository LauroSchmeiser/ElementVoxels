#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aUV;
layout(location = 4) in uint aFlags;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;

uniform mat4 model;
uniform mat4 mvp;
uniform float uTime;

void main()
{
    vec3 pos = aPos;

    float wobble =
    sin(pos.x * 1.7 + uTime * 2.5) * 0.06 +
    cos(pos.z * 1.3 + uTime * 2.0) * 0.04;

    pos.y += wobble;

    vec4 world = model * vec4(pos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(transpose(inverse(model))) * aNormal);
    vColor = aColor;

    gl_Position = mvp * vec4(pos, 1.0);
}