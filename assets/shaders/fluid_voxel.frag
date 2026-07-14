#version 430 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outThickness;

uniform vec3 viewPos;
uniform int uPass; // 0 = front, 1 = back

void main()
{
    vec3 N = normalize(vNormal);
    vec3 V = normalize(viewPos - vWorldPos);

    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 base = vColor;

    if (uPass == 1) {
        // Back faces (inside water) - dark, highly opaque
        vec3 darkWater = base * 0.3;
        float alpha = 0.9;
        outColor = vec4(darkWater, alpha);
        outThickness = 1.0;
        return;
    }

    // Front faces (surface water)
    vec3 surfaceColor = mix(base * 1.2, vec3(1.0), fresnel * 0.3);
    float depthFactor = 1.0 - max(dot(N, V), 0.0) * 0.2;
    vec3 finalColor = surfaceColor * depthFactor;

    float alpha = 0.75 + fresnel * 0.2;
    alpha = clamp(alpha, 0.7, 0.95);

    outColor = vec4(finalColor, alpha);
    outThickness = 0.5 + fresnel * 0.5;
}