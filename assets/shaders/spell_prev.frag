#version 330 core

in vec3 vWorldPos;
in vec3 vLocalPos;

out vec4 FragColor;

uniform int uPreviewMode;

// wall
uniform vec3 uWallSize;

// material availability feedback
uniform float uFillRatio;
uniform vec3  uLowColor;
uniform vec3  uHighColor;

// alpha for formation shell
uniform float uFormationAlpha;

float sdfBox(vec3 p, vec3 halfSize) {
    vec3 q = abs(p) - halfSize;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// negative inside, positive outside
float formationSdf() {
    if (uPreviewMode == 0) {

        return length(vLocalPos) - 1.0;
    } else {
        return sdfBox(vLocalPos, vec3(0.5));
    }
}

void main() {
    float dForm = formationSdf();

    float shell = 0.08;
    float formMask = 1.0 - smoothstep(shell, shell * 1.8, abs(dForm));
    float a = uFormationAlpha * formMask;

    if (a < 0.01) discard;

    // Hologram grid in local space
    float gridScale = 0.25;
    vec3 g = abs(fract(vLocalPos / gridScale) - 0.5);
    float gridLine = min(min(g.x, g.y), g.z);
    float grid = 1.0 - smoothstep(0.06, 0.12, gridLine);

    float r = clamp(uFillRatio, 0.0, 1.0);
    r = smoothstep(0.0, 1.0, r);

    vec3 formationCol = mix(uLowColor, uHighColor, r);
    formationCol *= mix(0.75, 1.25, grid);

    FragColor = vec4(formationCol, clamp(a, 0.0, 0.85));
}