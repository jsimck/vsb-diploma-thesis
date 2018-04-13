#ifndef VSB_SEMESTRAL_PROJECT_FRAME_BUFFER_H
#define VSB_SEMESTRAL_PROJECT_FRAME_BUFFER_H

#include <GL/glew.h>

namespace tless {
    /**
     * @brief Helper class which initializes frame buffer in given size and attaches texture to it.
     */
    class FrameBuffer {
    private:
        GLuint RBO, Texture;

    public:
        GLuint id;
        int width, height;

        explicit FrameBuffer(int width, int height);
        ~FrameBuffer();

        /**
         * @brief Use to bind frame buffer
         */
        void bind() const;

        /**
         * @brief Use to ubind frame buffer
         */
        void unbind() const;
    };
}

#endif
