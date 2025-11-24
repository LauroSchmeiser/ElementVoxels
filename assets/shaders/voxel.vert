#version 330 core

// Attribute layout MUST match how you upload the OutVertex SSBO:
layout(location = 0) in vec3 aPos;      // from OutVertex.pos.xyz
layout(location = 1) in vec3 aNormal;   // from OutVertex.normal.xyz
layout(location = 2) in vec3 aColor;    // from OutVertex.color.xyz

out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;

uniform mat4 mvp;
uniform mat4 model;

void main() {
    // world-space position
    vec4 worldPos = model * vec4(aPos, 1.0);
    fragPos = worldPos.xyz;

    // per-vertex color from SSBO
    vertexColor = aColor;

    // transform normal by normal matrix (model's inverse-transpose)
    mat3 normalMat = transpose(inverse(mat3(model)));
    normal = normalize(normalMat * aNormal);

    gl_Position = mvp * vec4(aPos, 1.0);
}