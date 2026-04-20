#version 430 core

#ifdef GL_ARB_shader_draw_parameters
#extension GL_ARB_shader_draw_parameters : enable
#endif

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec2 aUV;
layout(location = 4) in uint aFlags;

flat out uint vFlags;
flat out uint vChunkSlot;
out vec3 fragPos;
out vec3 vertexColor;
out vec3 normal;
out vec2 vUV;
out vec3 localPos;
out float localScale;

uniform mat4 mvp;
uniform mat4 model;
uniform float scale;

uint getBaseInstance()
{
    #ifdef GL_ARB_shader_draw_parameters
    return uint(gl_BaseInstanceARB);
    #else
    return uint(gl_BaseInstance);
    #endif
}

void main() {
    if(scale<=0.0)
    {
        localScale=1.0;
    }
    else
    {
        localScale=scale;
    }
    vChunkSlot = getBaseInstance();

    vec4 worldPos = model * vec4(aPos, 1.0);
    fragPos = worldPos.xyz;

    vertexColor = aColor;
    vFlags = aFlags;
    vUV = aUV;

    mat3 normalMat = transpose(inverse(mat3(model)));
    normal = normalize(normalMat * aNormal);
    localPos = aPos;
    gl_Position = gl_Position = mvp * vec4(aPos, 1.0);;
}