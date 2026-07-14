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

uniform bool uHasPlayerContact;
uniform vec3 uPlayerContactPoint;

const float uContactReachRadius = 4.5;
const float uContactReachStrength = 50.5;

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

    uint material = (aFlags >> 1u) & 63u;

    vec3 pos = aPos;
    vec3 nrm = aNormal;
    vec3 color =aColor;

    // Base flesh wobble
    if (material == 7u) {
        float t = uTime * 6.0;

        float w1 = sin(t + aPos.y * 1.4 + aPos.z * 1.1);
        float w2 = cos(t * 0.85 + aPos.x * 1.2 - aPos.y * 0.9);
        float w3 = sin(t * 1.3 + aPos.z * 1.7);

        vec3 centerLocal = vec3(16.0, 16.0, 16.0) * localScale;
        float distFromCenter = length(aPos - centerLocal);

        float radialMask = smoothstep(3.0 * localScale, 14.0 * localScale, distFromCenter);
        float ampMask = mix(0.15, 1.0, radialMask);

        vec3 wobble = vec3(
        w3 * 0.08,
        w1 * 0.12,
        w2 * 0.10
        ) * ampMask;

        pos += wobble;
        nrm = normalize(nrm + vec3(w3 * 0.12, w1 * 0.16, w2 * 0.14) * ampMask);
    }

    // -----------------------------------------------------------------------------
    // Localized "consume the player" tendril effect
    // -----------------------------------------------------------------------------
    if (material == 7u && uHasPlayerContact)
    {
       /* mat4 invModel = inverse(model);
        vec3 contactLocal = (invModel * vec4(uPlayerContactPoint, 1.0)).xyz;

        vec3 toContact = contactLocal - pos;
        float d = length(toContact);

        float reach = max(uContactReachRadius / max(localScale, 0.0001), 0.001);

        if (d < reach)
        {
            //---------------------------------------------------------
            // Contact mask
            //---------------------------------------------------------
            float mask = 1.0 - d / reach;
            mask = smoothstep(0.0, 1.0, mask);

            // Favor a ring instead of a solid mound
            float ring =
            smoothstep(0.15, 0.45, mask) *
            (1.0 - smoothstep(0.55, 0.95, mask));

            //---------------------------------------------------------
            // Outer shell only
            //---------------------------------------------------------
            vec3 centerLocal = vec3(16.0) * localScale;
            float distFromCenter = length(pos - centerLocal);

            float shell =
            smoothstep(6.0 * localScale,
            15.0 * localScale,
            distFromCenter);

            //---------------------------------------------------------
            // Procedural tendril mask
            //---------------------------------------------------------
            float t1 = sin(pos.x * 1.8 + uTime * 10.035);
            float t2 = sin(pos.z * 2.4 - uTime * 10.040);
            float t3 = sin((pos.x + pos.z) * 1.5 + uTime * 10.028);

            float tendrilMask = max(t1 * t2 * t3, 0.0);
            tendrilMask = pow(tendrilMask, 5.0);

            //---------------------------------------------------------
            // Independent pulse
            //---------------------------------------------------------
            float phase =
            pos.x * 0.9 +
            pos.z * 1.3 +
            pos.y * 0.5;

            float pulse =
            0.5 +
            0.5 *
            sin(uTime * 1.040 + phase);

            //---------------------------------------------------------
            // Growth amount
            //---------------------------------------------------------
            float grow =
            uContactReachStrength *
            ring *
            shell *
            tendrilMask *
            (0.4 + pulse);

            //---------------------------------------------------------
            // Growth direction
            //---------------------------------------------------------
            vec3 growDir = vec3(0.0, 1.0, 0.0);

            vec3 towardPlayer = normalize(toContact);

            float ang =
            fract(
            sin(dot(pos.xz,
            vec2(27.13, 91.73)))
            * 43758.5453)
            * 6.28318;

            vec3 randomDir =
            normalize(vec3(
            cos(ang),
            1.2,
            sin(ang)));

            vec3 finalDir =
            normalize(
            mix(randomDir,
            towardPlayer,
            0.45));


            //---------------------------------------------------------
            // Pull surrounding flesh inward
            //---------------------------------------------------------
            float sink =
            (1.0 - tendrilMask) *
            ring *
            0.6;

            pos += towardPlayer * sink;

            //---------------------------------------------------------
            // Raise tendrils
            //---------------------------------------------------------
            pos += finalDir * grow;

            //---------------------------------------------------------
            // Curl the tips
            //---------------------------------------------------------
            pos +=   grow * 0.25 ;

            //---------------------------------------------------------
            // Bend normals
            //---------------------------------------------------------
            nrm =
            normalize(
            mix(
            nrm,
            finalDir,
            0.5 * grow / uContactReachStrength));
            color=vec3(0,0,0);
        }*/
}

    vec4 worldPos = model * vec4(pos, 1.0);
    fragPos = worldPos.xyz;

    vertexColor = color;
    vFlags = aFlags;
    vUV = aUV;

    mat3 normalMat = transpose(inverse(mat3(model)));
    normal = normalize(normalMat * nrm);
    time = uTime;
    localPos = pos;

    gl_Position = mvp * vec4(pos, 1.0);
}