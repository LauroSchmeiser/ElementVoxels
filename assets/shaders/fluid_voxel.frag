#version 430 core

struct VoxelLightGpu {
    vec4 posIntensity;
    vec4 color;
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

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;
flat in uint vFlags;
flat in uint vChunkSlot;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outThickness;

uniform vec3 viewPos;
uniform vec3 ambientColor;
uniform int uPass;

uniform float uBrightness;
uniform float uGamma;

uniform sampler2DArray uAlbedoArray;
uniform float uMatRoughness[64];
uniform float uMatSpecular[64];
uniform float uUVScale[64];

uniform sampler2DArray uNormalArray;
uniform sampler2DArray uRoughArray;
uniform sampler2DArray uAOArray;
uniform sampler2DArray uHeightArray;

uniform float uNormalStrength;
uniform float uHeightScale;
uniform float uAOStrength;
uniform int uNormalYFlip;

uniform int uFluidDebugMode;
uniform float uTime;

const float PI = 3.14159265;
const float BASE_HEIGHT = 0.5;
const float BASE_ROUGHNESS = 0.22;
const float BASE_AO = 1.0;

vec3 materialDebugColor(uint mat)
{
    float x = float(mat);
    return fract(vec3(
    x * 0.1031,
    x * 0.11369 + 0.317,
    x * 0.13787 + 0.731
    ));
}

vec3 triplanarWeights(vec3 N)
{
    vec3 w = abs(normalize(N));
    w = pow(w, vec3(4.0));
    return w / (w.x + w.y + w.z + 1e-6);
}

vec3 sampleTriplanarAlbedo(uint mat, vec3 p, vec3 N)
{
    vec3 w = triplanarWeights(N);
    float s = uUVScale[mat];

    vec3 tx = texture(uAlbedoArray, vec3(p.yz * s, float(mat))).rgb;
    vec3 ty = texture(uAlbedoArray, vec3(p.xz * s, float(mat))).rgb;
    vec3 tz = texture(uAlbedoArray, vec3(p.xy * s, float(mat))).rgb;

    return tx * w.x + ty * w.y + tz * w.z;
}

float sampleScalarWithFallback(
sampler2DArray texArr,
uint mat,
vec2 uv,
float baseValue,
out float filled)
{
    vec4 t = texture(texArr, vec3(uv, float(mat)));
    float hasA = step(0.001, t.a);
    float hasRGB = step(0.001, dot(abs(t.rgb), vec3(1.0)));
    filled = max(hasA, hasRGB);

    return mix(baseValue, t.r, filled);
}

float sampleTriplanarScalarFallback(
sampler2DArray texArr,
uint mat,
vec3 p,
vec3 N,
float baseValue,
out float filledOut)
{
    vec3 w = triplanarWeights(N);
    float s = uUVScale[mat];

    float fx, fy, fz;
    float x = sampleScalarWithFallback(texArr, mat, p.yz * s, baseValue, fx);
    float y = sampleScalarWithFallback(texArr, mat, p.xz * s, baseValue, fy);
    float z = sampleScalarWithFallback(texArr, mat, p.xy * s, baseValue, fz);

    filledOut = fx * w.x + fy * w.y + fz * w.z;
    return x * w.x + y * w.y + z * w.z;
}

vec3 sampleNormalWithFallback(
sampler2DArray texArr,
uint mat,
vec2 uv,
vec3 baseN,
out float filled)
{
    vec4 t = texture(texArr, vec3(uv, float(mat)));

    float hasA = step(0.001, t.a);
    float hasRGB = step(0.001, dot(abs(t.rgb), vec3(1.0)));
    filled = max(hasA, hasRGB);

    vec3 nTex = t.xyz * 2.0 - 1.0;
    if (uNormalYFlip != 0) nTex.y = -nTex.y;

    return normalize(mix(baseN, nTex, filled));
}

vec3 sampleTriplanarNormalFallback(
sampler2DArray texArr,
uint mat,
vec3 p,
vec3 Ngeom,
out float filledOut)
{
    vec3 w = triplanarWeights(Ngeom);
    float s = uUVScale[mat];

    float fx, fy, fz;
    vec3 baseLocal = vec3(0.0, 0.0, 1.0);

    vec3 nx = sampleNormalWithFallback(texArr, mat, p.yz * s, baseLocal, fx);
    vec3 ny = sampleNormalWithFallback(texArr, mat, p.xz * s, baseLocal, fy);
    vec3 nz = sampleNormalWithFallback(texArr, mat, p.xy * s, baseLocal, fz);

    vec3 nX = normalize(vec3(nx.z, nx.x, nx.y));
    vec3 nY = normalize(vec3(ny.x, ny.z, ny.y));
    vec3 nZ = normalize(vec3(nz.x, nz.y, nz.z));

    filledOut = fx * w.x + fy * w.y + fz * w.z;
    return normalize(nX * w.x + nY * w.y + nZ * w.z);
}

void main()
{
    uint sourceMaterial = (vFlags >> 1u) & 63u;
    uint mat = min(sourceMaterial, 9u);
    uint id = mat + 10u;

    vec3 Ngeom = normalize(vNormal);
    vec3 V = normalize(viewPos - vWorldPos);

    vec3 flowOffset = vec3(uTime * 0.10, uTime * 0.06, uTime * 0.08);
    vec3 samplePos = vWorldPos + flowOffset;

    float heightFilled = 0.0;
    float height = sampleTriplanarScalarFallback(
    uHeightArray, id, samplePos, Ngeom, BASE_HEIGHT, heightFilled
    );
    samplePos += Ngeom * ((height - 0.5) * uHeightScale * 0.35);

    vec3 texAlbedo = sampleTriplanarAlbedo(id, samplePos, Ngeom);
    vec3 albedo = mix(texAlbedo,  vColor, 0.5);

    float normalFilled = 0.0;
    vec3 Nmap = sampleTriplanarNormalFallback(
    uNormalArray, id, samplePos, Ngeom, normalFilled
    );
    float fluidNormalStrength = clamp(uNormalStrength * 0.45, 0.0, 0.65);
    vec3 N = normalize(mix(Ngeom, Nmap, fluidNormalStrength));

    float roughFilled = 0.0;
    float roughTex = sampleTriplanarScalarFallback(
    uRoughArray, id, samplePos, Ngeom, BASE_ROUGHNESS, roughFilled
    );

    float aoFilled = 0.0;
    float aoTex = sampleTriplanarScalarFallback(
    uAOArray, id, samplePos, Ngeom, BASE_AO, aoFilled
    );

    float baseRough = clamp(roughTex * uMatRoughness[id], 0.03, 0.55);
    float fluidBody = clamp(1.0 - roughTex, 0.0, 1.0);
    float roughness = mix(baseRough, baseRough * 0.45, fluidBody * 0.7);
    float ao = mix(1.0, clamp(aoTex, 0.0, 1.0), clamp(uAOStrength, 0.0, 1.0));

    if (uFluidDebugMode != 0) {
        vec3 debugColor = vec3(0.0);

        if (uFluidDebugMode == 1) debugColor = texAlbedo;
        else if (uFluidDebugMode == 2) debugColor = N * 0.5 + 0.5;
        else if (uFluidDebugMode == 3) debugColor = vec3(roughness);
        else if (uFluidDebugMode == 4) debugColor = vec3(ao);
        else if (uFluidDebugMode == 5) debugColor = materialDebugColor(sourceMaterial);
        else if (uFluidDebugMode == 6) debugColor = materialDebugColor(id);
        else if (uFluidDebugMode == 7) debugColor = vec3(float(min(sourceMaterial, 9u)) / 9.0);
        else if (uFluidDebugMode == 8) debugColor = vec3(fluidBody);

        outColor = vec4(debugColor, 1.0);
        outThickness = 1.0;
        return;
    }

    float NdotV = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - NdotV, 4.0);

    vec3 lightAccum = vec3(0.0);
    vec3 specAccum = vec3(0.0);

    ChunkLightIndexGpu info = chunkLights[vChunkSlot];
    uint n = min(info.count, MAX_LIGHTS);

    for (uint i = 0u; i < n; ++i) {
        VoxelLightGpu Lg = lights[info.indices[i]];
        vec3 toLight = Lg.posIntensity.xyz - vWorldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 L = toLight / max(dist, 1e-4);

        float NdotL = max(dot(N, L), 0.0);
        float attenuation = 7.5 / (distSq + 1.0);
        vec3 radiance = Lg.color.xyz * (Lg.posIntensity.w * attenuation);

        lightAccum += radiance * NdotL;

        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float shininess = mix(320.0, 28.0, roughness);
        float specStrength = mix(0.35, 1.2, fluidBody) * mix(0.2, 1.0, clamp(uMatSpecular[id], 0.0, 1.0));
        float spec = pow(NdotH, shininess) * specStrength * step(0.0, NdotL);
        specAccum += radiance * spec;
    }

    vec3 ambient = ambientColor * albedo * ao * mix(0.35, 0.90, fluidBody);
    vec3 diffuse = lightAccum * (albedo / PI) * mix(0.35, 0.75, fluidBody);

    vec3 reflectTint = mix(albedo * 0.75, vec3(1.0), 0.25);
    vec3 fresnelReflect = reflectTint * fresnel * mix(0.4, 1.15, fluidBody);

    vec3 R = reflect(-V, N);

    vec3 spaceDeep    = vec3(0.015, 0.020, 0.035);
    vec3 spaceMid     = vec3(0.040, 0.070, 0.140);
    vec3 spaceGlow    = vec3(0.160, 0.280, 0.520);

    float horizonAmt  = pow(1.0 - abs(R.y), 1.6);
    float zenithAmt   = pow(clamp(R.y * 0.5 + 0.5, 0.0, 1.0), 1.9);
    vec3 envColor     = mix(spaceDeep, spaceMid, horizonAmt);
    envColor          = mix(envColor, spaceGlow, zenithAmt * 0.25);

    float rim = pow(1.0 - max(dot(N, V), 0.0), 1.25);
    float spaceReflectStrength = mix(0.25, 1.10, fluidBody) * (0.25 + fresnel * 0.80);
    vec3 spaceReflect = envColor * (spaceReflectStrength + rim * 0.55);

    vec3 deepColor = albedo * ao * mix(0.30, 0.60, fluidBody);
    vec3 surfaceColor = ambient + diffuse + specAccum + fresnelReflect + spaceReflect;

    if (uPass == 1) {
        vec3 color = deepColor + ambient * 0.55;
        color = pow(max(color, vec3(0.0)), vec3(1.0 / max(uGamma * 0.5, 0.001)));
        color *= uBrightness * 1.15;

        outColor = vec4(color, mix(0.78, 0.92, fluidBody));
        outThickness = mix(0.75, 1.0, fluidBody);
        return;
    }

    float depthFactor = mix(0.82, 0.98, NdotV);
    vec3 color = surfaceColor * depthFactor;
    color = pow(max(color, vec3(0.0)), vec3(1.0 / max(uGamma * 0.5, 0.001)));
    color *= uBrightness * 1.2;

    float alpha = clamp(mix(0.7, 0.82, fluidBody) + fresnel * 0.16 - roughness * 0.08, 0.80, 0.90);

    outColor = vec4(color, alpha);
    outThickness = mix(0.45, 1.0, fluidBody) + fresnel * 0.35;
}