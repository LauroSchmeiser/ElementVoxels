#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aUV;
layout(location = 4) in uint aFlags;

flat out uint vFlags;
out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;
out vec2 vUV;

uniform mat4 mvp;
uniform mat4 model;

void main() {
    vec4 worldPos = model * vec4(aPos, 1.0);
    fragPos = worldPos.xyz;

    vertexColor = aColor;
    vFlags = aFlags;
    vUV = aUV;


    mat3 normalMat = transpose(inverse(mat3(model)));
    normal = normalize(normalMat * aNormal);

    gl_Position = mvp * vec4(aPos, 1.0);
}