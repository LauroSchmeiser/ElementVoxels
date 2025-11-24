#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;

out vec4 FragColor;

uniform vec3 lightPos = vec3(20.0, 20.0, 20.0);
uniform float lightIntensity = 1.0;
uniform vec3 viewPos = vec3(0.0, 0.0, 90.0);
uniform float emission = 0.0;
uniform vec3 emissionColor = vec3(0.0); // set per-object if needed
uniform vec3 lightColor = vec3(1.0);

void main() {
    vec3 n = normalize(normal);
    vec3 L = normalize(lightPos - fragPos);
    float diff = max(dot(n, L), 0.0);

    vec3 ambient = 0.06 * vertexColor;
    vec3 diffuse = diff * vertexColor * lightColor * lightIntensity;
    vec3 emiss = emission * emissionColor; // can be vertexColor or separate

    vec3 hdr = ambient + diffuse + emiss;
    // Reinhard tone map
    vec3 color = hdr / (hdr + vec3(1.0));

    FragColor = vec4(color, 1.0);
}