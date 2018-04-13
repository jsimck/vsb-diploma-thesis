#version 400 core

in vec3 normal;
out vec4 fragColor;

void main() {
    fragColor = vec4(normal, 1.0);
}