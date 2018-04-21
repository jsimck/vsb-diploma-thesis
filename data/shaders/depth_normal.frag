#version 400 core

uniform float scale;

in vec3 normal;
in float depth;
out vec4 fragColor;

void main() {
    // 33.57835 -> scale factor to aproximately match scene depth
    fragColor = vec4(normal, depth * 33.57835 * scale);
}