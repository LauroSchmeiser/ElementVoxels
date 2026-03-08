// Fix: your repo almost certainly already has a function named noise3(...)
// with a DIFFERENT return type (commonly vec3), so GLSL sees an overload clash.
//
// Rename the functions to avoid colliding with existing ones.

#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;
in vec2 vUV;
flat in uint vFlags;

out vec4 FragColor;

uniform int numLights;
uniform vec3 lightPos[4];
uniform vec3 lightColor[4];
uniform float lightIntensity[4];
uniform vec3 ambientColor;
uniform vec3 viewPos;

uniform int  uOverlayEnabled;
uniform vec3 uOverlayCenter;
uniform float uOverlayRadius;
uniform uint uOverlayMaterial;
uniform vec3 uOverlayColor;
uniform float uOverlayAlpha;

uniform float emission;
uniform vec3 emissionColor;

// Burn
uniform int   uBurnEnabled;
uniform float uBurn;
uniform vec3  uBurnCenter;
uniform float uBurnRadius;
uniform float uBurnNoiseScale;
uniform float uBurnEdgeWidth;
uniform vec3  uBurnEmberColor;
uniform float uBurnCharStrength;

const float PI = 3.14159265;
const float MIN_ALBEDO = 0.1;

// ---- renamed: burnHash / burnNoise ----
float burnHash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float burnNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = burnHash13(i + vec3(0,0,0));
    float n100 = burnHash13(i + vec3(1,0,0));
    float n010 = burnHash13(i + vec3(0,1,0));
    float n110 = burnHash13(i + vec3(1,1,0));
    float n001 = burnHash13(i + vec3(0,0,1));
    float n101 = burnHash13(i + vec3(1,0,1));
    float n011 = burnHash13(i + vec3(0,1,1));
    float n111 = burnHash13(i + vec3(1,1,1));

    float n00 = mix(n000, n100, f.x);
    float n10 = mix(n010, n110, f.x);
    float n01 = mix(n001, n101, f.x);
    float n11 = mix(n011, n111, f.x);

    float n0 = mix(n00, n10, f.y);
    float n1 = mix(n01, n11, f.y);

    return mix(n0, n1, f.z);
}

// 4x4 Bayer dither
float burnBayer4(vec2 p) {
    int x = int(mod(p.x, 4.0));
    int y = int(mod(p.y, 4.0));
    int idx = x + y * 4;

    int m[16] = int[16](
    0,  8,  2, 10,
    12, 4, 14, 6,
    3, 11, 1,  9,
    15, 7, 13, 5
    );

    return (float(m[idx]) + 0.5) / 16.0;
}

void main() {
    if ((vFlags & 1u) != 0u) discard;

    uint mat = (vFlags >> 1u) & 63u;

    vec3 albedo = max(vertexColor, vec3(MIN_ALBEDO));
    vec3 N = normalize(normal);

    vec3 lightAccum = vec3(0.0);
    for (int i = 0; i < numLights; ++i) {
        vec3 toLight = lightPos[i] - fragPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 L = toLight / max(dist, 0.001);

        float NdotL = max(dot(N, L), 0.0);
        float attenuation = 1.0 / (distSq + 1.0);
        float intensity = lightIntensity[i] * attenuation;
        lightAccum += lightColor[i] * intensity * NdotL;
    }

    vec3 diffuse = lightAccum * (albedo / PI);
    vec3 ambient = ambientColor * albedo;
    vec3 emiss = emission * emissionColor;

    // Burn / disintegrate
    if (uBurnEnabled != 0) {
        float burn = clamp(uBurn, 0.0, 1.0);

        float n = burnNoise3D(fragPos * uBurnNoiseScale);

        float dist01 = 0.0;
        if (uBurnRadius > 0.0) {
            dist01 = clamp(length(fragPos - uBurnCenter) / uBurnRadius, 0.0, 1.0);
        }

        float field = (n * 0.85 + (1.0 - dist01) * 0.15) - burn;

        float w = max(uBurnEdgeWidth, 1e-4);
        float alpha = smoothstep(-w, w, field);

        float d = burnBayer4(gl_FragCoord.xy);
        if (alpha <= d) discard;

        float charMask = 1.0 - smoothstep(-w, w * 2.5, field);
        albedo = mix(albedo, albedo * (1.0 - clamp(uBurnCharStrength, 0.0, 1.0)), charMask);

        float edge = 1.0 - smoothstep(0.0, w, abs(field));
        emiss += edge * uBurnEmberColor;
    }

    vec3 hdr = ambient + diffuse + emiss;
    vec3 color = hdr / (hdr + vec3(1.0));

    // Overlay
    if (uOverlayEnabled != 0) {
        float d = length(fragPos - uOverlayCenter);
        float fade = 2.0;
        float mask = 1.0 - smoothstep(uOverlayRadius - fade, uOverlayRadius, d);
        float m = (mat == (uOverlayMaterial & 63u)) ? 1.0 : 0.0;
        float a = clamp(uOverlayAlpha * mask * m, 0.0, 1.0);
        color = mix(color, uOverlayColor, a);
    }

    FragColor = vec4(color, 1.0);
}