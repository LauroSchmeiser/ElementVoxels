#version 330 core
in vec3 fragPos;
in vec3 vertexColor;
in vec3 normal;

out vec4 FragColor;

uniform vec3 lightPos = vec3(20.0, 20.0, 20.0);
uniform float lightIntensity= 0.1f;
uniform vec3 viewPos = vec3(0.0, 0.0, 90.0);
uniform float emission = 0.0;


void main() {
    // simple diffuse lighting
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0);

    vec3 diffuse = diff * vertexColor;

    // ambient
    vec3 ambient = lightIntensity * vertexColor;

    vec3 color =  ambient + diffuse + emission * vertexColor;

    FragColor = vec4(color, 1.0);
}
