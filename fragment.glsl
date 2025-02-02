#version 330 core

in vec2 TexCoord;
in vec4 vertexColor;
in vec4 FragPosLightSpace;
flat in float texID;
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform sampler2D texSampler[2]; // Texture sampler
uniform sampler2D shadowMap;
uniform vec3 lightPos;
uniform bool isOutline; // Distinguish between normal and outline rendering

float ShadowCalculation(vec4 fragPosLightSpace, vec3 fragNormal, vec3 lightDir) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // Early out if outside shadow map boundaries
    if (projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 0.0; // No shadow
    }

    fragNormal = normalize(fragNormal);

    float bias = max(0.001 * (1.0 - dot(fragNormal, lightDir)), 0.0005);
    float shadow = 0.0;

    // PCF sampling
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0); // Size of one texel in shadow map
    int samples = 3; // Kernel size: (2*samples+1)x(2*samples+1)

    for (int x = -samples; x <= samples; ++x) {
        for (int y = -samples; y <= samples; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            float closestDepth = texture(shadowMap, projCoords.xy + offset).r;
            if (projCoords.z - bias > closestDepth) {
                shadow += 1.0;
            }
        }
    }

    // Normalize shadow value to [0, 1]
    float kernelSize = float((2 * samples + 1) * (2 * samples + 1));
    shadow /= kernelSize;

    if (shadow <= 0.6){
        shadow = 0.0;
    }

    return shadow;
}

void main()
{
    // If rendering an outline use a single color like black
    if(isOutline){
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black outline
        return;
    }

    // Determine base color based on texture ID
    vec4 baseColor;
        if (int(texID) == 1) {
            baseColor = texture(texSampler[0], TexCoord);
        } else if (int(texID) == 2) {
            baseColor = texture(texSampler[1], TexCoord);
        } else {
            baseColor = vec4(1.0, 1.0, 1.0, 1.0); // White for invalid texture ID
        }

    vec3 lightDir = normalize(lightPos - FragPos);

    float shadow = ShadowCalculation(FragPosLightSpace, Normal, lightDir);

    FragColor = vec4(baseColor.rgb * (1.0 - shadow * 0.8), baseColor.a);
}
