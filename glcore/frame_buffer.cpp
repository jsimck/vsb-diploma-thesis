#include "frame_buffer.h"
#include <opencv2/core/types.hpp>
#include <iostream>

void FrameBuffer::init() {
    // Init frame buffer
    glGenFramebuffers(1, &id);
    glBindFramebuffer(GL_FRAMEBUFFER, id);

    // TODO SET FBO TEXTURE SIZE
    cv::Size winSize(108, 108);

    // FB texture
    glGenTextures(1, &Texture);
    glBindTexture(GL_TEXTURE_2D, Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, winSize.width, winSize.height, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Texture, 0); // Bind to frame buffer

    // The depth buffer
    glGenRenderbuffers(1, &RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, winSize.width, winSize.height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << glCheckFramebufferStatus(GL_FRAMEBUFFER) << std::endl;
        return;
    }

    // Unbind
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBuffer::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void FrameBuffer::unbind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}