#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uSceneTexture;

uniform float uTime;
uniform float uDamageIntensity;

// Screen-space damage direction
uniform vec2 uDamageDirection;

uniform vec3 uDamageColor = vec3(1.0,0.05,0.02);


float rand(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453);
}


void main()
{
    vec2 uv = TexCoord;

    vec4 scene = texture(uSceneTexture, uv);

    // directional damage flash

    vec2 center = uv - 0.5;

    float angle = atan(center.y, center.x);

    float hitAngle = atan(
    uDamageDirection.y,
    uDamageDirection.x
    );


    float angleDiff = abs(
    atan(
    sin(angle-hitAngle),
    cos(angle-hitAngle)
    )
    );


    float slash =
    1.0 - smoothstep(
    0.0,
    0.7,
    angleDiff
    );


    slash *= length(center);


    // screen distortion

    float shake =
    sin(uTime*40.0) *
    0.002 *
    uDamageIntensity;


    vec2 distortionDir = vec2(0.0);

    float centerLength = length(center);

    if(centerLength > 0.001)
    {
        distortionDir = center / centerLength;
    }

    vec2 distortedUV =
    uv + distortionDir * shake;

    vec3 distorted =
    texture(
    uSceneTexture,
    distortedUV
    ).rgb;


    // chromatic hit effect

    float aberration =
    uDamageIntensity*0.006;


    float r =
    texture(
    uSceneTexture,
    uv + vec2(aberration,0)
    ).r;


    float b =
    texture(
    uSceneTexture,
    uv - vec2(aberration,0)
    ).b;


    vec3 chroma =
    vec3(
    r,
    distorted.g,
    b
    );


    // red damage overlay

    float vignette =
    smoothstep(
    0.2,
    0.8,
    length(center)
    );


    vec3 damage =
    uDamageColor *
    (
    slash*1.5 +
    vignette*0.5
    );


    float intensity =
    clamp(
    uDamageIntensity,
    0,
    1
    );


    vec3 finalColor =
    mix(
    scene.rgb,
    chroma,
    intensity*0.5
    );


    finalColor += damage*intensity;


    FragColor =
    vec4(finalColor,1.0);
}