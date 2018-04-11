#version 400 core

layout (location = 0) in vec3 aPos;

out float eyeDepth;

uniform mat4 MMatrix;
uniform mat4 VMatrix;
uniform mat4 PMatrix;

void main() {
    gl_Position = PMatrix * VMatrix * MMatrix * vec4(aPos, 1.0);
    vec3 vEyePos = (VMatrix * MMatrix * vec4(aPos, 1.0)).xyz;

    // OpenGL Z axis goes out of the screen, so depths are negative
    eyeDepth = -vEyePos.z;
}