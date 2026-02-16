// skybox.frag (procedural space cubemap)
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;  // 3D direction vector from cube center

uniform float time;
uniform sampler2D noiseTexture;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 p) {
    return texture(noiseTexture, p).r;
}

// 3D noise using texture
float noise3D(vec3 p) {
    // Sample different slices of the 2D noise texture for 3D effect
    float xy = texture(noiseTexture, p.xy * 0.2 + p.z * 0.1).r;
    float yz = texture(noiseTexture, p.yz * 0.2 + p.x * 0.1).g;
    float xz = texture(noiseTexture, p.xz * 0.2 + p.y * 0.1).b;
    return (xy + yz + xz) / 3.0;
}

// Smooth 3D noise
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

    // Trilinear interpolation
    float mixed1 = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    float mixed2 = mix(mix(e, f_, f.x), mix(g, h, f.x), f.y);
    return mix(mixed1, mixed2, f.z);
}

// Fractal Brownian Motion for 3D
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
    // Normalize the direction vector to get a consistent direction
    vec3 dir = normalize(TexCoords);

    // Slow time for smooth animation
    float slowTime = time * 0.15;

    // ===== NEBULA GENERATION =====
    // Use spherical coordinates to create smooth transitions
    float theta = atan(dir.z, dir.x); // azimuth
    float phi = acos(dir.y);           // inclination

    // Create seamless spherical coordinates
    vec3 sphericalDir = vec3(
    sin(phi) * cos(theta),
    cos(phi),
    sin(phi) * sin(theta)
    );

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

    // Combine colors with masks
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
    float stars1 = noise3D(dir * 30.0 + slowTime * 0.5);
    stars1 = pow(stars1, 12.0) * 0.25;
    starColor += vec3(1.0, 0.95, 0.9) * stars1*2.5;

    float stars2 = noise3D(dir * 25.0 + vec3(50.0) + slowTime * 0.3);
    stars2 = pow(stars2, 10.0) * 0.125;
    starColor += vec3(0.9, 0.95, 1.0) * stars2*3.0;

    // Twinkling bright stars
    float stars3 = noise3D(dir * 20.0 + vec3(100.0) + slowTime);
    stars3 = pow(stars3, 8.0);
    stars3 *= 0.6 + 0.4 * sin(time * 2.0 + stars3 * 50.0);
    starColor += vec3(1.0, 0.9, 0.8) * stars3 * 4.5;

    // Individual bright stars (grid-based in spherical space)
    vec3 gridPos = dir * 40.0;
    vec3 grid = floor(gridPos);
    vec3 localPos = fract(gridPos) - 0.5;

    float cellRand = random(grid.xy + grid.z);
    if (cellRand < 0.1) {
        vec3 offset = vec3(
        random(grid.xy + vec2(10.0, 20.0)),
        random(grid.yz + vec2(30.0, 40.0)),
        random(grid.xz + vec2(50.0, 60.0))
        ) - 0.5;

        float dist = length(localPos - offset * 0.7);
        if (dist < 0.12) {
            float brightness = (1.0 - dist * 10.0);
            brightness *= 0.5 + 0.5 * sin(time * 3.0 + cellRand * 100.0);

            // Colored stars
            float colorRand = random(grid.xy + vec2(70.0, 80.0));
            vec3 thisStar;
            if (colorRand < 0.5) thisStar = vec3(1.0);
            else if (colorRand < 0.7) thisStar = vec3(1.0, 0.8, 0.6);
            else if (colorRand < 0.9) thisStar = vec3(0.6, 0.8, 1.0);
            else thisStar = vec3(1.0, 0.6, 0.8);

            starColor += thisStar * brightness * 2.0;
        }
    }

    // ===== FINAL COMPOSITION =====
    vec3 finalColor = vec3(0.0);

    // Add nebula (slightly dimmer to keep dark areas)
    finalColor += nebulaColor * 3.5;

    // Add stars on top
    finalColor += starColor;

    // Subtle vignette based on direction length
    float vignette = 1.0 - length(dir) * 0.01;
    finalColor *= vignette;

    // Ensure pure blacks
    finalColor = max(finalColor, vec3(0.0));

    // Gamma correction
    finalColor = pow(finalColor, vec3(1.0/0.25));

    FragColor = vec4(finalColor, 1.0);
}