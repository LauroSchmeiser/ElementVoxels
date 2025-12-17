#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;

out vec4 FragColor;

uniform int numLights;
uniform vec3 lightPos[4];
uniform vec3 lightColor[4];
uniform float lightIntensity[4]; // intensity already includes inverse-square (or use as scale)
uniform vec3 ambientColor; // small ambient term
uniform vec3 viewPos;

uniform float emission;
uniform vec3 emissionColor;

const float PI = 3.14159265;
const float MIN_ALBEDO = 0.0; // set >0 for debug visualizing lights on black planets

void main() {
    // surface albedo
    vec3 albedo = max(vertexColor, vec3(MIN_ALBEDO)); // use MIN_ALBEDO=0 for physical correctness

    vec3 N = normalize(normal);

    // accumulate light contributions
    vec3 lightAccum = vec3(0.02);



    for (int i = 0; i < numLights; ++i) {
        // direction from fragment to light
        vec3 L = normalize(lightPos[i] - fragPos);

        // Lambert diffuse
        float NdotL = max(dot(N, L), 0.01);

        // If you didn't bake inverse-square on CPU, compute attenuation here:
        // float dist = length(lightPos[i] - fragPos);
        // float att = 1.0 / (dist*dist + 1e-5);
        // float intensity = lightIntensity[i] * att;
        // But in this example we assume lightIntensity[i] already has attenuation baked.

        float intensity = lightIntensity[i];

        // Add color * intensity * lambert
        lightAccum += lightColor[i] * intensity * NdotL;
    }

    // convert accumulated light to reflected diffuse (energy-conserving Lambert)
    // diffuse reflectance = albedo / PI * incoming_radiance
    vec3 diffuse = (albedo / PI) * lightAccum;

    // ambient (very small)
    vec3 ambient = ambientColor * albedo;

    // emission (self-emissive)
    vec3 emiss = emission * emissionColor;

    // final HDR color
    vec3 hdr = ambient + diffuse + emiss;

    // simple Reinhard tone mapping
    vec3 color = hdr / (hdr + vec3(1.0));

    FragColor = vec4(color, 1.0);
}
