#version 330 core

layout (location = 0) in vec3 aPos; // Cube vertex position
layout (location = 1) in vec2 aTexCoord; // Texture coordinate
layout (location = 2) in vec4 instancePos; // Instance position
layout (location = 3) in vec4 instanceColor; // Instance color
layout (location = 4) in float instanceTexture; // Instance texture index
layout (location = 5) in vec3 aNormal;

out vec4 vertexColor;
out vec4 FragPosLightSpace;
out vec2 TexCoord; // Pass the texture coordinate to the fragment shader
flat out float texID; // Texture ID to pass to fragment shader
out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main()
{
    vec4 worldPosition = model * vec4(aPos + instancePos.xyz, 1.0);
    gl_Position = projection * view * worldPosition;
    FragPosLightSpace = lightSpaceMatrix * worldPosition;
    vertexColor = instanceColor; // Pass color to fragment shader
    TexCoord = aTexCoord; // Pass the texture coordinate to the fragment shader
    texID = instanceTexture; // Pass the texture ID
    Normal = mat3(transpose(inverse(model))) * aNormal;
    FragPos = worldPosition.xyz;
}