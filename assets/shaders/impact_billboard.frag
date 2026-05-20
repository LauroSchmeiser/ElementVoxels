#version 330 core

in vec2 vUV;
in vec2 vLocal;

out vec4 FragColor;

uniform vec4 uColor;
uniform int uKind;      // 0 smoke, 1 flash
uniform float uLife01;  // 0..1

float hash21(vec2 p)
{
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float valueNoise2(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
    (c - a) * u.y * (1.0 - u.x) +
    (d - b) * u.x * u.y;
}

float fbm(vec2 p)
{
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += valueNoise2(p) * a;
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

void main()
{
    // centered coordinates
    vec2 p = vLocal * 2.0;
    float r = length(p);

    if (uKind == 0)
    {
        // -----------------------
        // Smoke / dust
        // -----------------------

        // Radial body with stronger readable center
        float baseMask = 1.0 - smoothstep(0.25, 1.05, r);

        // Breakup noise
        vec2 noiseUV = vUV * 5.0 + vec2(uLife01 * 0.7, -uLife01 * 1.1);
        float n1 = fbm(noiseUV);
        float n2 = fbm(noiseUV * 1.8 + 12.7);

        float breakup = mix(0.65, 1.25, n1);
        breakup *= mix(0.8, 1.15, n2);

        // Slight hollowing as it ages so it feels more like a puff
        float centerFade = mix(1.0, 0.7, uLife01);
        float alpha = baseMask * breakup * centerFade;

        // Fade over lifetime, but stay readable longer
        alpha *= smoothstep(1.0, 0.15, uLife01);
        alpha *= 0.95;
        alpha=clamp(alpha,0.5,1.0);


        if (alpha < 0.02)
        discard;

        // Smoke color:
        // a little warm near birth, cooler/sootier later
        vec3 warm = uColor.rgb * vec3(1.15, 1.0, 0.9);
        vec3 soot = uColor.rgb * vec3(0.55, 0.55, 0.58);
        vec3 smokeCol = mix(warm, soot, uLife01);

        // Slight brighter edges / turbulence variation
        smokeCol *= mix(0.85, 1.15, n1);

        FragColor = vec4(smokeCol, alpha * uColor.a);
    }
    else
    {
        // -----------------------
        // Flash / cartoon burst
        // -----------------------

        vec2 q = vLocal * 2.0;
        float r = length(q);

        // angle in -PI..PI
        float ang = atan(q.y, q.x);

        // Star spikes
        // Increase frequency for more points
        float spikes = abs(cos(ang * 4.0));
        spikes = pow(spikes, 10.0);

        // Base burst radius varies by angle
        float burstRadius = mix(0.35, 1.0, spikes);

        // Main star silhouette
        float starMask = 1.0 - smoothstep(burstRadius * 0.65, burstRadius, r);

        // Bright center
        float core = 1.0 - smoothstep(0.0, 0.35, r);
        core = pow(core, 2.5);

        // Shock ring
        float ring = 1.0 - smoothstep(0.04, 0.18, abs(r - 0.55));
        ring *= (1.0 - smoothstep(0.55, 1.0, r));

        // Combine
        float alpha = max(starMask, core * 1.2);
        alpha += ring * 0.35;

        // Fast lifetime decay
        alpha *= (1.0 - uLife01);
        alpha *= (1.0 - uLife01);
        alpha *= 1.4;

        if (alpha < 0.01)
        discard;

        // Color blend: hot center, orange outer burst
        vec3 hotCore = vec3(1.8, 1.45, 0.75);
        vec3 hotMid  = vec3(1.4, 0.85, 0.25);
        vec3 hotEdge = vec3(0.95, 0.45, 0.10);

        float centerT = clamp(core, 0.0, 1.0);
        float edgeT = clamp(r, 0.0, 1.0);

        vec3 flashCol = mix(hotEdge, hotMid, 1.0 - edgeT);
        flashCol = mix(flashCol, hotCore, centerT);

        FragColor = vec4(flashCol, alpha * uColor.a);
    }
}