#version 400 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormals;

out vec3 normal;
out float depth;

uniform mat4 NMatrix;
uniform mat4 MVMatrix;
uniform mat4 MVPMatrix;

void main() {
    gl_Position = MVPMatrix * vec4(aPos, 1.0);
    normal = vec3(NMatrix * vec4(aNormals, 1.0));
    vec3 vEyePos = (MVMatrix * vec4(aPos, 1.0)).xyz;

    // OpenGL Z axis goes out of the screen, so depths are negative
    depth = -vEyePos.z;
}