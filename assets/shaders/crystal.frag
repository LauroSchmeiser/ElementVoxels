#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;
flat in uint vFlags;


out vec4 FragColor;

void main() {
    if ((vFlags & 1u) != 0u) discard;

    uint mat = (vFlags >> 1u) & 63u;

    if(mat==7u)
    {
        FragColor = vec4(1.0,0,0,0);
    }
    else if(mat==6u)
    {
        FragColor = vec4(0.2,0,0.4,0);
    }
    else
    {
        FragColor = vec4(0.85, 0.0, 0.82, 1.0);
    }
}