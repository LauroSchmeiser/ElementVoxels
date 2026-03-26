#version 330 core
in vec2 vUv;
in vec3 vColor;
out vec4 FragColor;

uniform float time;

float hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
    dot(p, vec2(269.5, 183.3)));
    return fract(sin(p.x + p.y) * 43758.5453123);
}
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i + vec2(0.0,0.0));
    float b = hash(i + vec2(1.0,0.0));
    float c = hash(i + vec2(0.0,1.0));
    float d = hash(i + vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a, b, u.x) + (c - a)*u.y*(1.0 - u.x) + (d - b)*u.x*u.y;
}
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += a * noise(p);
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 uv = vUv;
    float r = length(uv * 2.0);


    if (r > 1.0) discard;

    float core = smoothstep(0.35, 0.0, r);
    float corona = smoothstep(0.9, 0.45, r) * (1.0 - core);

    float t = fbm(uv * 5.0 + time * 0.7);
    float t2 = fbm(uv * 11.0 - time * 0.9 + vec2(3.1, 4.2));
    float turb = mix(t, t2, 0.5);

    float intensity = clamp(core * (1.0 + 0.8 * turb) + corona * (0.6 + 0.6 * turb), 0.0, 2.0);

    vec3 col = vColor * (0.6 + 0.8 * intensity);

    float edge0 = 0.92;
    float edge1 = 1.00;
    float circleMask = smoothstep(edge1, edge0, r);

    float alpha = clamp((intensity * 0.85) * circleMask, 0.0, 1.0);

    if (alpha < 0.001) discard;

    FragColor = vec4(col * alpha, alpha);
}