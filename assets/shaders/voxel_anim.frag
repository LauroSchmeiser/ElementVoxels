#version 450 core
uniform vec3 color;
uniform float alpha = 1.0;

out vec4 FragColor;

void main() {
    // Magical glow effect
    vec3 finalColor = color;

    FragColor = vec4(finalColor, 0.35f * alpha);}