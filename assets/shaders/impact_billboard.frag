#version 330 core

in vec2 vUV;
in vec2 vLocal;
in vec4 vColor;
flat in int vKind;
in float vLife01;

out vec4 FragColor;

uniform sampler2D uNoiseTex;

void main()
{
    vec2 p = vLocal * 2.0;
    float r = length(p);

    if (vKind == 0)
    {
        float baseMask = 1.0 - smoothstep(0.25, 1.05, r);

        // 1-2 cheap texture samples instead of FBM
        vec2 uv1 = vUV * 3.5 + vec2(vLife01 * 0.45, -vLife01 * 0.65);
        vec2 uv2 = vUV * 6.0 + vec2(-vLife01 * 0.25, vLife01 * 0.35);

        float n1 = texture(uNoiseTex, uv1).r;
        float n2 = texture(uNoiseTex, uv2).r;

        float breakup = mix(0.70, 1.20, n1) * mix(0.85, 1.10, n2);
        float centerFade = mix(1.0, 0.7, vLife01);

        float alpha = baseMask * breakup * centerFade;
        alpha *= smoothstep(1.0, 0.15, vLife01);

        if (alpha < 0.02) discard;

        vec3 warm = vColor.rgb * vec3(1.15, 1.0, 0.9);
        vec3 soot = vColor.rgb * vec3(0.55, 0.55, 0.58);
        vec3 smokeCol = mix(warm, soot, vLife01);
        smokeCol *= mix(0.9, 1.1, n1);

        FragColor = vec4(smokeCol, alpha * vColor.a);
    }
    else
    {
        vec2 q = vLocal * 2.0;
        float rr = length(q);
        float ang = atan(q.y, q.x);

        float spikes = abs(cos(ang * 4.0));
        spikes = pow(spikes, 10.0);

        float burstRadius = mix(0.35, 1.0, spikes);
        float starMask = 1.0 - smoothstep(burstRadius * 0.65, burstRadius, rr);

        float core = 1.0 - smoothstep(0.0, 0.35, rr);
        core = pow(core, 2.5);

        float ring = 1.0 - smoothstep(0.04, 0.18, abs(rr - 0.55));
        ring *= (1.0 - smoothstep(0.55, 1.0, rr));

        float alpha = max(starMask, core * 1.2);
        alpha += ring * 0.35;
        alpha *= (1.0 - vLife01);
        alpha *= (1.0 - vLife01);
        alpha *= 1.4;

        if (alpha < 0.01) discard;

        vec3 hotCore = vec3(1.8, 1.45, 0.75);
        vec3 hotMid  = vec3(1.4, 0.85, 0.25);
        vec3 hotEdge = vec3(0.95, 0.45, 0.10);

        float centerT = clamp(core, 0.0, 1.0);
        float edgeT = clamp(rr, 0.0, 1.0);

        vec3 flashCol = mix(hotEdge, hotMid, 1.0 - edgeT);
        flashCol = mix(flashCol, hotCore, centerT);

        FragColor = vec4(flashCol, alpha * vColor.a);
    }
}