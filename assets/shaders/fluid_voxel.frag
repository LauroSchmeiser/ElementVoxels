#version 430 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outThickness;

uniform vec3 viewPos;

void main()
{
    vec3 N = normalize(vNormal);
    vec3 V = normalize(viewPos - vWorldPos);

    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    vec3 base = vColor;
    vec3 surface = mix(base * 0.7, vec3(1.0), fresnel * 0.25);

    outColor = vec4(surface, 0.35 + fresnel * 0.25);

    // cheap thickness proxy
    outThickness = 1.0;

}