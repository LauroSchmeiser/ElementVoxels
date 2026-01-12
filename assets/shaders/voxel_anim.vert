#version 330 core

// per-vertex cube vertex
layout(location = 0) in vec3 aPos;

// per-instance attributes
layout(location = 1) in vec3 aInstanceNormal;   // per-instance normal
layout(location = 2) in vec3 aInstanceColor;    // per-instance color (albedo)
layout(location = 3) in vec4 aInstancePosScale; // xyz = world pos, w = scale

out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;

uniform mat4 pv;

void main() {
    // Build world-space position for this vertex
    vec3 worldPos = aInstancePosScale.xyz + aPos * aInstancePosScale.w;
    fragPos = worldPos;
    vertexColor = aInstanceColor;
    // Transform normal by view (we don't have a model matrix other than translate+scale).
    // Since we only do uniform scale and translation, the normal is unchanged by scale sign; normalize anyway.
    normal = normalize(aInstanceNormal);

    gl_Position = pv * vec4(worldPos, 1.0);
}