#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;

uniform mat4 mvp;
uniform mat4 model;
uniform vec3 uniformColor;


void main() {
    fragPos = vec3(model * vec4(aPos, 1.0));
    vertexColor = uniformColor;
    normal = aPos == vec3(0.0) ? vec3(0,0,1) : normalize(aPos);
    gl_Position = mvp * vec4(aPos, 1.0);
}
