#version 430 core

#ifdef GL_ARB_shader_draw_parameters
#extension GL_ARB_shader_draw_parameters : enable
#endif

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aUV;
layout(location = 4) in uint aFlags;

flat out uint vFlags;
flat out uint vChunkSlot;
out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;
out vec2 vUV;
out vec3 localPos;
out float localScale;
out float time;

uniform mat4 mvp;
uniform mat4 model;
uniform float scale;
uniform float uTime;



uint getBaseInstance()
{
    #ifdef GL_ARB_shader_draw_parameters
    return uint(gl_BaseInstanceARB);
    #else
    return uint(gl_BaseInstance);
    #endif
}

void main() {
    localScale = (scale <= 0.0) ? 1.0 : scale;
    vChunkSlot = getBaseInstance();

    // Decode material from packed flags: packedFlags = (material << 1u)
    uint material = (aFlags >> 1u) & 63u;

    vec3 pos = aPos;
    vec3 nrm = aNormal;

    // VERY obvious tentacle animation
    if (material == 7u) {
        float t = uTime * 6.0;

        float w1 = sin(t + aPos.y * 1.4 + aPos.z * 1.1);
        float w2 = cos(t * 0.85 + aPos.x * 1.2 - aPos.y * 0.9);
        float w3 = sin(t * 1.3 + aPos.z * 1.7);

        // Distance-from-center mask (local space)
        vec3 centerLocal = vec3(16.0, 16.0, 16.0) * localScale;
        float distFromCenter = length(aPos - centerLocal);

        // Ramps from near 0 at center to 1 far out (tune 2nd/3rd args as needed)
        float radialMask = smoothstep(3.0 * localScale, 14.0 * localScale, distFromCenter);

        // Keep a minimum so inner parts still move a bit
        float ampMask = mix(0.15, 1.0, radialMask);

        vec3 wobble = vec3(
        w3 * 0.22,
        w1 * 0.30,
        w2 * 0.26
        ) * ampMask;

        pos += wobble;
        nrm = normalize(nrm + vec3(w3 * 0.35, w1 * 0.45, w2 * 0.40) * ampMask);
    }

    vec4 worldPos = model * vec4(pos, 1.0);
    fragPos = worldPos.xyz;

    vertexColor = aColor;
    vFlags = aFlags;
    vUV = aUV;

    mat3 normalMat = transpose(inverse(mat3(model)));
    normal = normalize(normalMat * nrm);
    time=uTime;
    localPos = pos;
    gl_Position = mvp * vec4(pos, 1.0);
}