#version 400 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormals;

out vec3 normal;

uniform mat4 NMatrix;
uniform mat4 MMatrix;
uniform mat4 VMatrix;
uniform mat4 PMatrix;

void main() {
    gl_Position = PMatrix * VMatrix * MMatrix * vec4(aPos, 1.0);
    normal = vec3(NMatrix * vec4(aNormals, 1.0));
}