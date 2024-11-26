#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec4 instancePos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main(){
    vec4 worldPosition = model * vec4(aPos, 1.0) + vec4(instancePos.xyz, 0.0);
    gl_Position = lightSpaceMatrix * worldPosition;
}