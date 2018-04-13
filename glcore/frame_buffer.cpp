#include "frame_buffer.h"
#include "../objdetect/fine_pose.h"
#include "../utils/timer.h"
#include <opencv2/core/types.hpp>
#include <iostream>

namespace tless {
    FrameBuffer::FrameBuffer(int width, int height) : width(width), height(height) {
        // Set new viewport size
        glViewport(0, 0, width, height);

        // Init frame buffer
        glGenFramebuffers(1, &id);
        glBindFramebuffer(GL_FRAMEBUFFER, id);

        // FB texture
        glGenTextures(1, &Texture);
        glBindTexture(GL_TEXTURE_2D, Texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, Texture, 0); // Bind to frame buffer

        // The depth buffer
        glGenRenderbuffers(1, &RBO);
        glBindRenderbuffer(GL_RENDERBUFFER, RBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

        // Always check that our framebuffer is ok
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
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

    FrameBuffer::~FrameBuffer() {
        glDeleteTextures(1, &Texture);
        glDeleteRenderbuffers(1, &RBO);
        glDeleteFramebuffers(1, &id);
    }
}