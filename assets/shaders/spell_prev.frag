#version 330 core

in vec3 vWorldPos;
in vec3 vLocalPos;

out vec4 FragColor;

uniform int uPreviewMode;

// shared
uniform vec3  uCenter;
uniform float uVoxelSize;

// sphere
uniform float uFormationRadius;

// wall
uniform vec3  uWallNormal;
uniform vec3  uWallUp;
uniform vec3  uWallSize;

// material availability feedback
uniform float uFillRatio;
uniform vec3  uLowColor;
uniform vec3  uHighColor;

// alpha for formation shell
uniform float uFormationAlpha;

mat3 buildBasis(vec3 forward, vec3 up) {
    vec3 f = normalize(forward);
    vec3 u = normalize(up);
    vec3 r = normalize(cross(f, u));
    vec3 cu = normalize(cross(r, f));
    return mat3(r, cu, f);
}

float sdfBox(vec3 p, vec3 halfSize) {
    vec3 q = abs(p) - halfSize;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// negative inside, positive outside
float formationSdf() {
    if (uPreviewMode == 0) {
        return length(vLocalPos) - uFormationRadius;
    } else {
        mat3 B = buildBasis(uWallNormal, uWallUp);
        vec3 rp = transpose(B) * vLocalPos;
        return sdfBox(rp, 0.5 * uWallSize);
    }
}

void main() {
    float dForm = formationSdf();
    float shell = 0.6 * uVoxelSize;
    float formMask = 1.0 - smoothstep(shell, shell * 1.8, abs(dForm));
    float a = uFormationAlpha * formMask;

    if (a < 0.01) discard;

    // Hologram grid
    float gridScale = 2.0 * uVoxelSize;
    vec3 g = abs(fract(vLocalPos / gridScale) - 0.5);
    float gridLine = min(min(g.x, g.y), g.z);
    float grid = 1.0 - smoothstep(0.06, 0.12, gridLine);

    //  color blend (red -> green)
    float r = clamp(uFillRatio, 0.0, 1.0);
    r = smoothstep(0.0, 1.0, r);

    vec3 formationCol = mix(uLowColor, uHighColor, r);

    formationCol *= mix(0.75, 1.25, grid);

    FragColor = vec4(formationCol, clamp(a, 0.0, 0.85));
}