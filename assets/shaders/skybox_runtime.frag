#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform float time;
uniform samplerCube nebulaCube;

// cheap hash
float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// rotate around Y (example)
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
    c, 0, -s,
    0, 1,  0,
    s, 0,  c
    );
}

void main() {
    vec3 dir = normalize(TexCoords);

    // --- nebula "movement" by rotating lookup direction (cheap) ---
    float t = time * 0.035;             // slow
    vec3 dNeb = rotY(t) * dir;

    vec3 nebula = texture(nebulaCube, dNeb).rgb;

    // --- twinkling stars (cheap, no textures) ---
    // Use a high-frequency grid in direction space (good enough for skybox)
    vec3 starP = dir * 75.0;
    vec3 cell = floor(starP);
    vec3 f = fract(starP) - 0.5;

    float r = hash13(cell);
    // star density:
    float starMask = step(r, 0.015);   // ~1.5% of cells become stars

    float dist = length(f)*2;
    float star = smoothstep(0.25, 0.0, dist); // soft star blob

    // twinkle
    float tw = 0.6 + 0.4 * sin(time * 3.0 + r * 50.0);

    vec3 starCol = mix(vec3(0.8,0.9,1.0), vec3(1.0,0.9,0.8), hash13(cell + 7.13));
    vec3 stars = starCol * starMask * star * tw * 2.0;

    vec3 color = nebula + stars;

    // optional tone map / gamma depending on your pipeline
    color = pow(color, vec3(1.0/0.25));

    FragColor = vec4(color, 1.0);
}