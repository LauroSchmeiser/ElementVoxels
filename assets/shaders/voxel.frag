#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;
flat in uint vFlags;

out vec4 FragColor;

uniform int numLights;
uniform vec3 lightPos[4];
uniform vec3 lightColor[4];
uniform float lightIntensity[4];
uniform vec3 ambientColor;
uniform vec3 viewPos;

uniform int  uOverlayEnabled;   // 0/1
uniform vec3 uOverlayCenter;    // spell preview center (world)
uniform float uOverlayRadius;   // search radius (world)
uniform uint uOverlayMaterial;  // 0..63
uniform vec3 uOverlayColor;     // e.g. vec3(0.2, 1.0, 0.2)
uniform float uOverlayAlpha;    // e.g. 0.35

uniform float emission;
uniform vec3 emissionColor;

const float PI = 3.14159265;
const float MIN_ALBEDO = 0.1;

void main() {
    // bit0 = emissive discard flag (your existing behavior)
    if ((vFlags & 1u) != 0u) discard;

    // bits1..6 = material id (0..63)
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

    vec3 hdr = ambient + diffuse + emiss;
    vec3 color = hdr / (hdr + vec3(1.0));

    // ----------------------------
    // Overlay: show pull region material in-range
    // ----------------------------
    if (uOverlayEnabled != 0) {
        float d = length(fragPos - uOverlayCenter);

        // Soft fade at edge (so it looks nice)
        float fade = 2.0; // world units; tweak or scale by VOXEL_SIZE if you want
        float mask = 1.0 - smoothstep(uOverlayRadius - fade, uOverlayRadius, d);

        // only tint matching material
        float m = (mat == (uOverlayMaterial & 63u)) ? 1.0 : 0.0;

        float a = clamp(uOverlayAlpha * mask * m, 0.0, 1.0);

        // Alpha blend over lit color
        color = mix(color, uOverlayColor, a);
    }

    FragColor = vec4(color, 1.0);
}