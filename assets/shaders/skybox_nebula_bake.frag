#version 330 core
out vec4 FragColor;

in vec3 TexCoords;
uniform float bakeTime;
uniform sampler2D noiseTexture;


float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 p) {
    return texture(noiseTexture, p).r;
}

float noise3D(vec3 p) {
    float xy = texture(noiseTexture, p.xy * 0.2 + p.z * 0.1).r;
    float yz = texture(noiseTexture, p.yz * 0.2 + p.x * 0.1).g;
    float xz = texture(noiseTexture, p.xz * 0.2 + p.y * 0.1).b;
    return (xy + yz + xz) / 3.0;
}

float smoothNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = noise3D(i);
    float b = noise3D(i + vec3(1,0,0));
    float c = noise3D(i + vec3(0,1,0));
    float d = noise3D(i + vec3(1,1,0));
    float e = noise3D(i + vec3(0,0,1));
    float f_ = noise3D(i + vec3(1,0,1));
    float g = noise3D(i + vec3(0,1,1));
    float h = noise3D(i + vec3(1,1,1));

    float mixed1 = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    float mixed2 = mix(mix(e, f_, f.x), mix(g, h, f.x), f.y);
    return mix(mixed1, mixed2, f.z);
}

float fbm3D(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 3.5;
    float frequency = 1.3;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * smoothNoise3D(p * frequency);
        maxValue += amplitude;
        amplitude *= 0.75;
        frequency *= 1.95;
    }

    return value / maxValue;
}

void main() {
    vec3 dir = normalize(TexCoords);
    float slowTime = bakeTime * 0.15;

    vec3 p1 = dir * 2.0 + vec3(slowTime * 0.3, 0.0, 0.0);
    vec3 p2 = dir * 1.5 + vec3(0.0, slowTime * 0.2, slowTime * 0.1);
    vec3 p3 = dir * 3.0 + vec3(slowTime * -0.1, slowTime * 0.15, 0.0);
    vec3 p4 = dir * 1.0 + vec3(0.0, 0.0, slowTime * 0.25);

    float mask1 = fbm3D(p1, 6);
    float mask2 = fbm3D(p2, 5);
    float mask3 = fbm3D(p3, 4);
    float mask4 = fbm3D(p4, 5);

    mask1 = pow(mask1 * 1.3, 2.5);
    mask2 = pow(mask2 * 1.2, 2.0);
    mask3 = pow(mask3 * 1.4, 2.2);
    mask4 = pow(mask4 * 1.1, 1.8);

    vec3 cyan    = vec3(0.2, 0.8, 1.0);
    vec3 magenta = vec3(1.0, 0.2, 0.8);
    vec3 yellow  = vec3(1.0, 0.9, 0.3);
    vec3 green   = vec3(0.3, 1.0, 0.4);
    vec3 orange  = vec3(1.0, 0.5, 0.1);
    vec3 purple  = vec3(0.7, 0.2, 1.0);
    vec3 blue    = vec3(0.2, 0.4, 1.0);
    vec3 red     = vec3(1.0, 0.2, 0.3);

    vec3 nebulaColor = vec3(0.0);

    vec3 color1 = mix(cyan,   purple, smoothNoise3D(dir * 1.5));
    vec3 color2 = mix(magenta, blue,  smoothNoise3D(dir * 2.0 + vec3(10.0)));
    vec3 color3 = mix(yellow,  red,   smoothNoise3D(dir * 1.8 + vec3(20.0)));
    vec3 color4 = mix(green,   orange,smoothNoise3D(dir * 2.5 + vec3(30.0)));

    nebulaColor += color1 * mask1 * 0.75;
    nebulaColor += color2 * mask2 * 0.65;
    nebulaColor += color3 * mask3 * 0.55;
    nebulaColor += color4 * mask4 * 0.65;

    vec3 finalColor = nebulaColor * 3.5;

  //  finalColor = max(finalColor, vec3(0.0));

    FragColor = vec4(finalColor, 1.0);
}