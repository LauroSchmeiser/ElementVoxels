#version 430 core

struct VoxelLightGpu {
    vec4 posIntensity; // xyz pos, w intensity
    vec4 color;        // xyz color, w unused
};

const uint MAX_LIGHTS = 4u;

struct ChunkLightIndexGpu {
    uint count;
    uint indices[4];
    uint pad[3];
};

layout(std430, binding = 10) readonly buffer LightsSSBO {
    VoxelLightGpu lights[];
};

layout(std430, binding = 11) readonly buffer ChunkIdxSSBO {
    ChunkLightIndexGpu chunkLights[];
};

in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;
in vec2 vUV;
flat in uint vFlags;
flat in uint vChunkSlot;

out vec4 FragColor;

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

// Burns
uniform int   uBurnEnabled;
uniform float uBurn;
uniform vec3  uBurnCenter;
uniform float uBurnRadius;
uniform float uBurnNoiseScale;
uniform float uBurnEdgeWidth;
uniform vec3  uBurnEmberColor;
uniform float uBurnCharStrength;

// --- MATERIALS / TEXTURES ---
uniform sampler2DArray uAlbedoArray;
uniform float uMatRoughness[64];
uniform float uMatSpecular[64];
uniform float uUVScale[64];

const float PI = 3.14159265;
const float MIN_ALBEDO = 0.0;

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

vec3 sampleTriplanarAlbedo(uint mat, vec3 worldPos, vec3 N)
{
    vec3 w = abs(normalize(N));
    w = pow(w, vec3(4.0));
    w /= (w.x + w.y + w.z + 1e-6);

    float s = uUVScale[mat];

    vec2 uvX = worldPos.yz * s;
    vec2 uvY = worldPos.xz * s;
    vec2 uvZ = worldPos.xy * s;

    vec3 tx = texture(uAlbedoArray, vec3(uvX, float(mat))).rgb;
    vec3 ty = texture(uAlbedoArray, vec3(uvY, float(mat))).rgb;
    vec3 tz = texture(uAlbedoArray, vec3(uvZ, float(mat))).rgb;

    return tx * w.x + ty * w.y + tz * w.z;
}

void main() {
    if ((vFlags & 1u) != 0u) discard;

    uint mat = (vFlags >> 1u) & 63u;
    vec3 N = normalize(normal);

    vec3 texAlbedo = sampleTriplanarAlbedo(mat, fragPos, N);

    float tintStrength = 0.65;
    vec3 albedo = mix(texAlbedo, texAlbedo * vertexColor, tintStrength);
    albedo = max(albedo, vec3(MIN_ALBEDO));

    vec3 V = normalize(viewPos - fragPos);

    vec3 lightAccum = vec3(0.0);
    vec3 specAccum  = vec3(0.0);

    float rough = clamp(uMatRoughness[mat], 0.02, 1.0);
    float specStrength = clamp(uMatSpecular[mat], 0.0, 1.0);
    float shininess = mix(256.0, 8.0, rough);

    ChunkLightIndexGpu info = chunkLights[vChunkSlot];
    uint n = min(info.count, MAX_LIGHTS);

    for (uint i = 0u; i < n; ++i) {
        VoxelLightGpu Lg = lights[info.indices[i]];

        vec3 lightPos = Lg.posIntensity.xyz;
        float lightIntensity = Lg.posIntensity.w;
        vec3 lightColor = Lg.color.xyz;

        vec3 toLight = lightPos - fragPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 L = toLight / max(dist, 0.001);

        float NdotL = max(dot(N, L), 0.0);
        float attenuation = 7.5 / (distSq + 1.0);
        float intensity = lightIntensity * attenuation;

        vec3 radiance = lightColor * intensity;

        lightAccum += radiance * NdotL;

        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec = pow(NdotH, shininess) * specStrength * step(0.0, NdotL);
        specAccum += radiance * spec;
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

    vec3 hdr = ambient + diffuse + specAccum + emiss;
    hdr= pow(hdr, vec3(1.0/0.5));
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