#version 330 core
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform sampler2D noiseTexture;
uniform mat4 invProjection;
uniform mat4 invView;
uniform mat4 prevInvView;  // NEW: Previous frame's inverse view matrix

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

// 3D noise using texture
float noise3D(vec3 p) {
    float xy = texture(noiseTexture, p.xy * 0.3 + p.z * 0.1).r;
    float yz = texture(noiseTexture, p.yz * 0.3 + p.x * 0.1).g;
    float xz = texture(noiseTexture, p.xz * 0.3 + p.y * 0.1).b;
    return (xy + yz + xz) / 3.0;
}

// Smooth 3D noise
float smoothNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    return mix(
    mix(mix(noise3D(i), noise3D(i + vec3(1,0,0)), f.x),
    mix(noise3D(i + vec3(0,1,0)), noise3D(i + vec3(1,1,0)), f.x), f.y),
    mix(mix(noise3D(i + vec3(0,0,1)), noise3D(i + vec3(1,0,1)), f.x),
    mix(noise3D(i + vec3(0,1,1)), noise3D(i + vec3(1,1,1)), f.x), f.y),
    f.z
    );
}

// 3D Fractal Brownian Motion
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
    // Convert screen coordinates to NDC
    vec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - 1.0;
    uv.x *= resolution.x / resolution.y;

    // Reconstruct view direction in world space
    vec4 clip = vec4(uv, 1.0, 1.0);
    vec4 eye = invProjection * clip;
    eye = vec4(eye.xy, 1.0, 0.0);

    // CHANGED: Get both current and previous directions
    vec3 currentDir = normalize((invView * eye).xyz);
    vec3 prevDir = normalize((prevInvView * eye).xyz);

    // CHANGED: Smooth interpolation between frames (adjust 0.15 for more/less smoothing)
    // Lower value = smoother but more lag, higher = more responsive but less smooth
    vec3 worldDir = mix(prevDir, currentDir, 0.15);  // 0.15 = gentle smoothing

    // Keep original scaling
    vec3 dir = worldDir * 0.5;

    // Slow time for smooth animation
    float slowTime = time * 0.25;

    // Create multiple nebula layers with different scales
    vec3 p1 = dir * 2.0 + vec3(slowTime * 0.3, 0.0, 0.0);
    vec3 p2 = dir * 1.5 + vec3(0.0, slowTime * 0.2, slowTime * 0.1);
    vec3 p3 = dir * 3.0 + vec3(slowTime * -0.1, slowTime * 0.15, 0.0);
    vec3 p4 = dir * 1.0 + vec3(0.0, 0.0, slowTime * 0.25);

    // Get nebula density masks
    float mask1 = fbm3D(p1, 6);
    float mask2 = fbm3D(p2, 5);
    float mask3 = fbm3D(p3, 4);
    float mask4 = fbm3D(p4, 5);

    // Reshape masks for more defined nebulas
    mask1 = pow(mask1 * 1.3, 2.5);
    mask2 = pow(mask2 * 1.2, 2.0);
    mask3 = pow(mask3 * 1.4, 2.2);
    mask4 = pow(mask4 * 1.1, 1.8);

    // Vibrant nebula colors
    vec3 cyan = vec3(0.2, 0.8, 1.0);
    vec3 magenta = vec3(1.0, 0.2, 0.8);
    vec3 yellow = vec3(1.0, 0.9, 0.3);
    vec3 green = vec3(0.3, 1.0, 0.4);
    vec3 orange = vec3(1.0, 0.5, 0.1);
    vec3 purple = vec3(0.7, 0.2, 1.0);
    vec3 blue = vec3(0.2, 0.4, 1.0);
    vec3 red = vec3(1.0, 0.2, 0.3);

    vec3 nebulaColor = vec3(0.0);

    // Layer 1: Cyan to purple gradient
    vec3 color1 = mix(cyan, purple, smoothNoise3D(dir * 1.5));
    nebulaColor += color1 * mask1 * 0.75;

    // Layer 2: Magenta to blue
    vec3 color2 = mix(magenta, blue, smoothNoise3D(dir * 2.0 + vec3(10.0)));
    nebulaColor += color2 * mask2 * 0.65;

    // Layer 3: Yellow to red
    vec3 color3 = mix(yellow, red, smoothNoise3D(dir * 1.8 + vec3(20.0)));
    nebulaColor += color3 * mask3 * 0.55;

    // Layer 4: Green to orange
    vec3 color4 = mix(green, orange, smoothNoise3D(dir * 2.5 + vec3(30.0)));
    nebulaColor += color4 * mask4 * 0.65;

    // ===== STARS =====
    vec3 starColor = vec3(0.0);

    // Dense star field using 3D noise
    float stars1 = noise3D(dir * 50.0);
    stars1 = pow(stars1, 15.0) * 1.5;
    starColor += vec3(1.0, 0.95, 0.9) * stars1;

    float stars2 = noise3D(dir * 35.0 + vec3(100.0));
    stars2 = pow(stars2, 12.0) * 1.0;
    starColor += vec3(0.9, 0.95, 1.0) * stars2;

    // Bright twinkling stars
    float stars3 = noise3D(dir * 25.0 + vec3(200.0));
    stars3 = pow(stars3, 8.0);
    stars3 *= 0.7 + 0.3 * sin(time * 2.0 + stars3 * 100.0);
    starColor += vec3(1.0, 0.95, 0.85) * stars3 * 1.2;

    // Individual bright stars
    vec3 gridPos = dir * 30.0;
    vec3 grid = floor(gridPos);
    vec3 localPos = fract(gridPos) - 0.5;

    float cellRand = random(grid.xy + grid.z);
    if (cellRand < 0.08) {
        vec3 starOffset = vec3(
        random(grid.xy + vec2(10.0, 20.0)),
        random(grid.yz + vec2(30.0, 40.0)),
        random(grid.xz + vec2(50.0, 60.0))
        ) - 0.5;

        float dist = length(localPos - starOffset * 0.7);
        if (dist < 0.05) {
            float brightness = (1.0 - dist * 20.0);
            brightness *= 0.6 + 0.4 * sin(time * 3.0 + cellRand * 100.0);

            // Colorful stars
            float starColorRand = random(grid.xy + vec2(70.0, 80.0));
            vec3 thisStarColor;
            if (starColorRand < 0.5) thisStarColor = vec3(1.0, 1.0, 1.0);
            else if (starColorRand < 0.65) thisStarColor = vec3(1.0, 0.8, 0.6);
            else if (starColorRand < 0.8) thisStarColor = vec3(0.6, 0.8, 1.0);
            else if (starColorRand < 0.9) thisStarColor = vec3(1.0, 0.7, 0.8);
            else thisStarColor = vec3(0.8, 1.0, 0.8);

            starColor += thisStarColor * brightness * 1.5;
        }
    }

    // ===== FINAL COMPOSITION =====
    vec3 finalColor = vec3(0.0);

    // Add nebulas
    finalColor += nebulaColor * 2.15;

    // Add stars on top
    finalColor += starColor;

    // Subtle vignette
    vec2 screenUV = gl_FragCoord.xy / resolution.xy;
    float vignette = 1.0 - length(screenUV - 0.5) * 0.5;
    vignette = smoothstep(0.3, 1.0, vignette);
    finalColor *= vignette;

    // Ensure no over-bright values
    finalColor = clamp(finalColor, 0.0, 1.0);

    // Gamma correction
    finalColor = pow(finalColor, vec3(1.0/0.45));

    FragColor = vec4(finalColor, 1.0);
}