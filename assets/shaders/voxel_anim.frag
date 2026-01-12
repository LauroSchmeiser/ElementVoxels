#version 450 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    // simple unlit color; you can add glow/pulse in CPU-supplied color or via additional varying
    FragColor = vec4(vColor, 1.0);
}