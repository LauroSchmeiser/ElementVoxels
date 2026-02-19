#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;

out vec4 FragColor;

uniform int numLights;
uniform vec3 lightPos[2];
uniform vec3 lightColor[2];
uniform float lightIntensity[2]; // intensity already includes inverse-square (or use as scale)
uniform vec3 ambientColor; // small ambient term
uniform vec3 viewPos;

uniform float emission;
uniform vec3 emissionColor;

const float PI = 3.14159265;
const float MIN_ALBEDO = 0.25; // set >0 for debug visualizing lights on black planets

void main() {
    // surface albedo
    vec3 albedo = max(vertexColor, vec3(MIN_ALBEDO)); // use MIN_ALBEDO=0 for physical correctness

    vec3 N = normalize(normal);

    // accumulate light contributions
    vec3 lightAccum = vec3(0.0);



    for (int i = 0; i < numLights; ++i) {
        vec3 toLight = lightPos[i] - fragPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 L = toLight / max(dist, 0.001);

        float NdotL = max(dot(N, L), 0.0);

        // inverse square attenuation
        float attenuation = 1 / (distSq + 1.0);

        float intensity = lightIntensity[i] * attenuation;

        lightAccum += lightColor[i] * intensity * NdotL;
    }


    // convert accumulated light to reflected diffuse (energy-conserving Lambert)
    // diffuse reflectance = albedo / PI * incoming_radiance
    vec3 diffuse = albedo * lightAccum;


    // ambient (very small)
    vec3 ambient = ambientColor * albedo;

    // emission (self-emissive)
    vec3 emiss = emission * emissionColor;

    // final HDR color
    vec3 hdr = ambient + diffuse + emiss;

    // simple Reinhard tone mapping
    vec3 color = hdr / (hdr + vec3(1.0));

    FragColor = vec4(hdr, 1.0);
    //FragColor = vec4(diffuse,1.0);
    // debug: show normals as color
    //FragColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0);
}
