#version 400 core

in float depth;
out vec4 fragColor;

void main() {
    // 10.08091 -> scale factor to aproximately match scene depth
    fragColor = vec4(depth * 10.08091, 0.0, 0.0, 1.0);
}