#version 400 core

in vec3 normal;
in float depth;
out vec4 fragColor;

void main() {
    // 10.08091 -> scale factor to aproximately match scene depth
    fragColor = vec4(normal, depth * 10.08091);
}