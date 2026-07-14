#version 430 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aUV;
layout(location = 4) in uint aFlags;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vColor;

uniform mat4 model;
uniform mat4 mvp;
uniform float uTime;

void main()
{
    vec3 pos = aPos;
    float t = uTime * 1.2;

    // Gentle surface wave - only affects Y slightly
    float wave1 = sin(pos.x * 0.8 + t * 1.5) * 0.04;
    float wave2 = cos(pos.z * 0.7 + t * 1.2) * 0.035;
    float wave3 = sin((pos.x + pos.z) * 0.5 + t * 1.8) * 0.025;

    // Apply wave only to Y (up direction) for a water surface effect
    pos.y += wave1 + wave2 + wave3;

    // Very subtle X/Z displacement for realism (much smaller)
    float waveX = sin(pos.z * 0.6 + t * 0.9) * 0.015;
    float waveZ = cos(pos.x * 0.6 + t * 0.9) * 0.015;
    pos.x += waveX;
    pos.z += waveZ;

    vec4 world = model * vec4(pos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(transpose(inverse(model))) * aNormal);
    vColor = aColor;

    gl_Position = mvp * vec4(pos, 1.0);
}