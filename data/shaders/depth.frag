#version 400 core

in float depth;
out vec4 fragColor;

void main() {
    // 33.57835 -> scale factor to aproximately match scene depth
    fragColor = vec4(depth * 33.57835, 0.0, 0.0, 1.0);
}