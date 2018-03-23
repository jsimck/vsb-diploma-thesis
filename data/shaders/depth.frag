#version 400 core

in float eyeDepth;
out vec4 fragColor;

void main() {
    fragColor = vec4(eyeDepth, 0.0, 0.0, 1.0);
}