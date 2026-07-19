#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

// Scene texture (your rendered game)
uniform sampler2D uSceneTexture;

// Movement parameters
uniform vec3 uVelocity3D;         // 3D velocity vector
uniform float uSpeed;             // Overall speed intensity (0.0 to 1.0+)
uniform float uTime;              // Time for animation

// Customization parameters
uniform float uLineCount = 24.0;
uniform float uLineWidth = 0.008;
uniform float uLineSharpness = 0.95;
uniform float uInnerRadius = 0.15;      // Clear center zone
uniform float uOuterFade = 0.72;        // Where lines start fading at edges
uniform float uVignetteStrength = 0.4;
uniform vec3 uLineColor = vec3(1.0, 1.0, 1.0);
uniform float uLineOpacity = 0.7;

// Camera matrices for 3D projection
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCameraPos;

float hash(float n)
{
    return fract(sin(n) * 43758.5453123);
}

void main()
{
    vec4 sceneColor = texture(uSceneTexture, TexCoord);

    float speedIntensity = clamp(uSpeed, 0.0, 1.0);

    if (speedIntensity < 0.05) {
        FragColor = sceneColor;
        return;
    }

    // Center coordinates (-0.5 to 0.5)
    vec2 centeredUV = TexCoord - 0.5;

    // Convert to polar coordinates
    float dist = length(centeredUV);
    float angle = atan(centeredUV.y, centeredUV.x);

    // Clear center zone (no lines near crosshair)
    if (dist < uInnerRadius) {
        FragColor = sceneColor;
        return;
    }

    // Create STATIC radial lines (no rotation/animation)
    float angleNormalized = (angle + 3.14159265) / (2.0 * 3.14159265);
    float stripeCoord = angleNormalized * uLineCount;
    float stripePhase = fract(stripeCoord);

    // distance from center of stripe, 0 at stripe center
    float stripeDist = abs(stripePhase - 0.5);

    // derivative-based AA width
    float aa = fwidth(stripeCoord) * 0.5;

    // stripe width should be interpreted in 0..0.5 range
    float halfWidth = clamp(uLineWidth, 0.001, 0.49);

    // anti-aliased line
    float line = 1.0 - smoothstep(halfWidth - aa, halfWidth + aa, stripeDist);

    // Add variation to line intensity (some lines brighter/darker)
    float lineIndex = floor(angleNormalized * uLineCount);
    float lineVariation = hash(lineIndex) * 0.3 + 0.7; // 0.7 to 1.0
    line *= lineVariation;

    // Sharpen lines based on sharpness parameter
    line = pow(line, 1.0 / max(uLineSharpness, 0.1));

    // Fade lines at outer edges
    float edgeFade = 1.0 - smoothstep(uOuterFade, 1.0, dist);
    line *= edgeFade;
    float outerSoftFade = 1.0 - smoothstep(0.75, 1.0, dist);
    line *= outerSoftFade;

    // Fade based on distance from center (stronger at edges)
    float distanceFactor = smoothstep(uInnerRadius, uOuterFade, dist);
    line *= distanceFactor;
    // Instead of rotation, add subtle flicker to line opacity
    float flicker = 0.85 + 0.15 * sin(uTime * 8.0 + lineIndex * 2.0);
    line *= flicker;

    // Vignette effect
    float vignette = 1.0 - (dist * dist * uVignetteStrength * speedIntensity);

    // Optional: subtle radial blur
    vec3 blurredColor = sceneColor.rgb;
    const int samples = 12;
    float blurAmount = speedIntensity * 0.015;

    vec2 blurDir = normalize(centeredUV);
    for (int i = 1; i <= samples; i++) {
        float t = float(i) / float(samples);
        vec2 offset = blurDir * t * blurAmount;
        blurredColor += texture(uSceneTexture, TexCoord - offset).rgb;
    }
    blurredColor /= float(samples + 1);

    // Combine effects
    vec3 finalColor = mix(sceneColor.rgb, blurredColor, speedIntensity * 0.5);
    finalColor *= vignette;

    // Add the radial speed lines
    finalColor += uLineColor * line * speedIntensity * uLineOpacity;

    FragColor = vec4(finalColor, 1.0);
}