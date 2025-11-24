#version 330 core
layout(location = 0) in vec2 aQuadPos;       // [-0.5..0.5] quad coords
layout(location = 1) in vec4 aInstancePosScale; // xyz = world pos, w = scale
layout(location = 2) in vec4 aInstanceColor;    // rgb = color, a unused

out vec2 vUv;
out vec3 vColor;

uniform mat4 view;
uniform mat4 proj;

void main() {
    // camera right and up vectors from view matrix (assuming view is world->view)
    vec3 right = normalize(vec3(view[0][0], view[1][0], view[2][0]));
    vec3 up    = normalize(vec3(view[0][1], view[1][1], view[2][1]));


    vec3 worldPos = aInstancePosScale.xyz;
    float scale = aInstancePosScale.w;

    vec3 corner = worldPos + (right * aQuadPos.x + up * aQuadPos.y) * scale;

    gl_Position = proj * view * vec4(corner, 1.0);

    vUv = aQuadPos;

    vColor = aInstanceColor.rgb;
}