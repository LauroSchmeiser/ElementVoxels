#version 450 core
uniform vec3 color;

out vec4 FragColor;

void main() {
    // Magical glow effect
    vec3 finalColor = color;

    FragColor = vec4(finalColor, 1.0f)
    ;}