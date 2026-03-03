#version 330 core

in vec3 vWorldPos;
in vec3 vLocalPos;

out vec4 FragColor;

uniform int uPreviewMode; // keep for later (0 sphere, 1 wall)

// shared
uniform vec3  uCenter;
uniform float uVoxelSize;

// sphere
uniform float uFormationRadius;

// wall (unused for now but kept)
uniform vec3  uWallNormal;
uniform vec3  uWallUp;
uniform vec3  uWallSize;

// pull overlay
uniform float uPullRadius;

// colors / alpha
uniform vec3  uFormationColor;
uniform vec3  uPullColor;
uniform float uFormationAlpha;
uniform float uPullAlpha;

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
    // Pull overlay (soft sphere volume)
    float dPull = length(vLocalPos) - uPullRadius;
    float pullFade = 2.0 * uVoxelSize;
    float pullMask = 1.0 - smoothstep(0.0, pullFade, max(dPull, 0.0));
    float pullA = uPullAlpha * pullMask;

    // Formation overlay (thin shell)
    float dForm = formationSdf();
    float shell = 0.6 * uVoxelSize;
    float formMask = 1.0 - smoothstep(shell, shell * 1.8, abs(dForm));
    float formA = uFormationAlpha * formMask;

    // Optional hologram grid
    float gridScale = 2.0 * uVoxelSize;
    vec3 g = abs(fract(vLocalPos / gridScale) - 0.5);
    float gridLine = min(min(g.x, g.y), g.z);
    float grid = 1.0 - smoothstep(0.06, 0.12, gridLine);

    vec3 formationCol = uFormationColor * mix(0.75, 1.25, grid);
    vec3 pullCol      = uPullColor;

    float a = clamp(pullA + formA, 0.0, 0.85);
    if (a < 0.01) discard;

    vec3 col = pullCol * pullA + formationCol * formA;
    col /= max(a, 1e-5);

    FragColor = vec4(col, a);
}