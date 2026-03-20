#version 330 core
out vec4 FragColor;
in vec2 vUV;

uniform sampler2D uSceneColor;
uniform sampler2D uSceneDepth;

uniform float uNear;
uniform float uFar;

uniform float uFogStrength;   // 0..1
uniform float uFogStart;      // linear depth units
uniform float uFogEnd;        // linear depth units

uniform float uGlowThreshold; // >1 if HDR scene
uniform float uGlowStrength;  // 0..2
uniform vec2  uInvResolution; // 1/width, 1/height

uniform float uExposure;      // 1.0
uniform float uGamma;         // 2.2

float linearizeDepth(float z01, float near, float far) {
    float z = z01 * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

vec3 tonemapReinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// small kernel, “blur-ish” glow
vec3 sampleGlow(vec2 uv, float centerDepth01)
{
    // 9-tap kernel
    vec2 o = uInvResolution * 2.0;

    vec2 taps[9] = vec2[](
    vec2( 0,  0),
    vec2( 1,  0), vec2(-1,  0),
    vec2( 0,  1), vec2( 0, -1),
    vec2( 1,  1), vec2(-1,  1),
    vec2( 1, -1), vec2(-1, -1)
    );

    float w[9] = float[](0.22, 0.12,0.12, 0.12,0.12, 0.075,0.075, 0.075,0.075);

    vec3 acc = vec3(0.0);
    float sum = 0.0;

    for (int i = 0; i < 9; ++i) {
        vec2 uv2 = uv + taps[i] * o;

        vec3 c = texture(uSceneColor, uv2).rgb;
        float d = texture(uSceneDepth, uv2).r;

        // Depth-aware: if depth differs a lot, reduce contribution (prevents halos across edges)
        float dd = abs(d - centerDepth01);
        float edgeKill = 1.0 - smoothstep(0.002, 0.02, dd);

        float l = luminance(c);
        float m = smoothstep(uGlowThreshold, uGlowThreshold * 1.5, l); // bright-pass

        float wi = w[i] * edgeKill * m;
        acc += c * wi;
        sum += wi;
    }

    if (sum < 1e-5) return vec3(0.0);
    return acc / sum;
}

void main()
{
    vec3 scene = texture(uSceneColor, vUV).rgb;
    float d01  = texture(uSceneDepth, vUV).r;

    float isSky = step(0.9999, d01);

    // Pick a “sky fog target” color.
    // If you want better matching, pass a uniform with your average nebula tint.
    vec3 skyFogColor = vec3(0.08, 0.10, 0.14);

    // Fog factor by distance (only on geometry pixels)
    float dLin = linearizeDepth(d01, uNear, uFar);
    float fog = clamp((dLin - uFogStart) / max(uFogEnd - uFogStart, 1e-3), 0.0, 1.0);
    fog *= uFogStrength * (1.0 - isSky);

    vec3 color = mix(scene, skyFogColor, fog);

    // Glow computed before tonemap (important)
    vec3 glow = sampleGlow(vUV, d01) * uGlowStrength;

    // Reduce glow on pure sky pixels so stars don’t blow out too hard (tunable)
    glow *= (1.0 - isSky * 0.65);

    // Composite
    color += glow;

    // Final unified output transform
    //color *= uExposure;
    //color = tonemapReinhard(color);
    //color = pow(max(color, vec3(0.0)), vec3(1.0 / uGamma));

    FragColor = vec4(color, 1.0);
}