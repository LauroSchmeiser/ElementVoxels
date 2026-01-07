#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;

out vec4 FragColor;

uniform int numLights;
uniform vec3 lightPos[4];
uniform vec3 lightColor[4];
uniform float lightIntensity[4];

uniform int debugMode;        // 0=normals color, 1=signed N·L for debugLightIndex, 2=signed sum N·L, 3=normal length
uniform int debugLightIndex;  // index to use for mode 1

// Helper to map signed dot to color: negative -> red, positive -> green
vec3 signedDotToColor(float s) {
    float pos = max(s, 0.0);
    float neg = max(-s, 0.0);
    // green for positive, red for negative
    return vec3(neg, pos, 0.0);
}

void main() {
    // normalized (safeguarded) normal
    vec3 N = normal;
    float nlen = length(N);
    if (nlen < 1e-6) {
        // fallback color for degenerate normals
        FragColor = vec4(1.0, 0.0, 1.0, 1.0);
        return;
    }
    N = N / nlen;

    if (debugMode == 0) {
        // visualize normal direction mapped to RGB
        FragColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }

    // compute signed N·L(s) (not clamped) to surface-visible differences
    if (debugMode == 1) {
        if (numLights <= 0) {
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        int idx = clamp(debugLightIndex, 0, max(0, numLights-1));
        vec3 L = normalize(lightPos[idx] - fragPos);
        float s = dot(N, L); // signed
        vec3 col = signedDotToColor(s);
        // also modulate brightness by distance-based attenuation (optional, for clarity)
        float d2 = dot(lightPos[idx] - fragPos, lightPos[idx] - fragPos);
        float atten = 1.0 / (d2 + 1.0);
        col *= (0.2 + 0.8 * atten); // keep some base contrast
        FragColor = vec4(col, 1.0);
        return;
    }

    if (debugMode == 2) {
        if (numLights <= 0) {
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        // sum signed N·L across lights (weighted by intensity)
        float sum = 0.0;
        float totW = 0.0;
        for (int i = 0; i < numLights; ++i) {
            vec3 L = normalize(lightPos[i] - fragPos);
            float s = dot(N, L); // signed
            float d2 = dot(lightPos[i] - fragPos, lightPos[i] - fragPos);
            float atten = 1.0 / (d2 + 1.0);
            float w = lightIntensity[i] * atten;
            sum += s * w;
            totW += w;
        }
        float avg = (totW > 0.0) ? (sum / totW) : sum;
        // clamp to [-1,1] then map to color
        avg = clamp(avg, -1.0, 1.0);
        vec3 col = signedDotToColor(avg);
        // boost contrast for visualization
        col = pow(col, vec3(0.6));
        FragColor = vec4(col, 1.0);
        return;
    }

    if (debugMode == 3) {
        // visualize normal length (should be ~1)
        float v = clamp(nlen, 0.0, 2.0);
        FragColor = vec4(vec3(v), 1.0);
        return;
    }

    if(debugMode >= 4)
    {
        FragColor=vec4(fragPos.x,fragPos.y,fragPos.z,1.0);
        return;
    }
    // fallback: show normalized normal
    FragColor = vec4(N * 0.5 + 0.5, 1.0);
}