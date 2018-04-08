#include "glutils.h"
#include "../glcore/shader.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace tless {
    glm::mat4 pMat(const cv::Mat &K, int x0, int y0, int w, int h, float nc, float fc, WindowCoords coords) {
        float depth = fc - nc;
        float q = -(fc + nc) / depth;
        float qn = -2 * (fc * nc) / depth;
        const auto *Kptr = K.ptr<float>();

        std::cout << K <<std::endl;

        std::cout << Kptr[0] << std::endl;
        std::cout << Kptr[1] << std::endl;
        std::cout << Kptr[2] << std::endl;
        std::cout << Kptr[3] << std::endl;
        std::cout << Kptr[4] << std::endl;
        std::cout << Kptr[5] << std::endl;

        if (coords == Y_UP) {
            return glm::transpose(glm::mat4(
                    2 * Kptr[0] / w, -2 * Kptr[1] / w , (-2 * Kptr[2] + w + 2 * x0) / w , 0,
                    0              , -2 * Kptr[4] / h , (-2 * Kptr[5] + h + 2 * y0) / h , 0,
                    0              , 0                , q                               , qn,
                    0              , 0                , -1                              , 0
            ));
        } else {
            assert(coords == Y_DOWN);
            return glm::transpose(glm::mat4(
                    2 * Kptr[0] / w, -2 * Kptr[1] / w , (-2 * Kptr[2] + w + 2 * x0) / w , 0,
                    0              , 2 * Kptr[4] / h  , (2 * Kptr[5] - h + 2 * y0) / h  , 0,
                    0              , 0                , q                               , qn,
                    0              , 0                , -1                              , 0
            ));
        }
    }

    glm::mat4 vMat(const cv::Mat &R, const cv::Mat &t) {
        glm::mat4 VMatrix;
        const auto *Rptr = R.ptr<float>();
        const auto *Tptr = t.ptr<float>();

        // Fill R matrix
        VMatrix[0][0] = Rptr[0];
        VMatrix[0][1] = Rptr[1];
        VMatrix[0][2] = Rptr[2];

        VMatrix[1][0] = Rptr[3];
        VMatrix[1][1] = Rptr[4];
        VMatrix[1][2] = Rptr[5];

        VMatrix[2][0] = Rptr[6];
        VMatrix[2][1] = Rptr[7];
        VMatrix[2][2] = Rptr[8];

        // Fill translation vector
        VMatrix[0][3] = Tptr[0];
        VMatrix[1][3] = Tptr[1];
        VMatrix[2][3] = Tptr[2];

        // Convert OpenCV to OpenGL camera system
        glm::mat4 yzFlip; // Create flip matrix for coordinate system conversion
        yzFlip[1][1] = -1;
        yzFlip[2][2] = -1;
        VMatrix *= yzFlip;

        return glm::transpose(VMatrix); // OpenGL expects column-wise matrix format
    }

    glm::mat4 mvMat(const glm::mat4 &model, const glm::mat4 &view) {
        return view * model;
    }

    glm::mat4 mvpMat(const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection) {
        return projection * view * model;
    }

    glm::mat4 nMat(const glm::mat4 &model, const glm::mat4 &view) {
        return glm::inverseTranspose(view * model);
    }

    void drawDepth(const Template &tpl, cv::Mat &dst, float clipNear, float clipFar) {
        // GLFW init and config
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

        // GLFW window creation
        cv::Size winSize = tpl.objBB.size();
        GLFWwindow *window = glfwCreateWindow(winSize.width, winSize.height, "DrawDepth", NULL, NULL);

        if (window == NULL) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwHideWindow(window);
        glViewport(0, 0, winSize.width, winSize.height);

        // Init Glew after GLFW init
        if (glewInit()) {
            std::cerr << "Failed to initialize GLXW" << std::endl;
            return;
        }

        // Load shader and mesh
        Shader shader("data/shaders/depth.vert", "data/shaders/depth.frag");
        Mesh mesh("data/models/obj_07.ply");

        // PMV data
        glm::mat3 K(
                1076.74064739f, 0, 203.98264967f,
                0, 1075.17825536f, 239.59181836f,
                0, 0, 1
        );
        glm::vec3 t(-4.06668495f, -18.75499854f, 634.87406861f);
        glm::mat3 R(
                -0.93424870f, -0.35660872f, -0.00292517f,
                -0.19815880f, 0.52592294f, -0.82712632f,
                0.29649935f, -0.77216270f, -0.56200858f
        );
        glm::mat4 Model;
        Model = glm::mat4(

        0.9995354844300639f, -0.01688384781014621f, 0.02537224957076902f, -0.02319064114524444f,
        0.02040765897149304f, 0.9891098183854488f, -0.1457576571875679f, 0.1084800211632654f,
        -0.02263499106385527f, 0.1462077387029465f, 0.9889949212826671f, 0.02965420771128631f,
        0,0,0,1
                );
        Model = glm::transpose(Model);

        // PVM initialization
        glm::mat4 VMatrix = vMat(tpl.camera.R, tpl.camera.t);
        glm::mat4 PMatrix = pMat(tpl.camera.K, 0, 0, winSize.width, winSize.height, clipNear, clipFar, WindowCoords::Y_UP);
        glm::mat4 MVMatrix = mvMat(Model, VMatrix);
        glm::mat4 MVPMatrix = mvpMat(Model, VMatrix, PMatrix);

        // OpenGL settings
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);

        /// Init frame buffer
        GLuint frameBuffer;
        glGenFramebuffers(1, &frameBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

        // FB texture
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, winSize.width, winSize.height, 0, GL_RGB, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0); // Bind to frame buffer

        // The depth buffer
        GLuint rbo;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, winSize.width, winSize.height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

        // Always check that our framebuffer is ok€
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cout << glCheckFramebufferStatus(GL_FRAMEBUFFER) << std::endl;
            return;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0); /// unbind

        // Render
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
        glEnable(GL_DEPTH_TEST);

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        shader.use();
        shader.setMat4("MVPMatrix", MVPMatrix);
        shader.setMat4("MVMatrix", MVMatrix);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        dst = cv::Mat::zeros(winSize.height, winSize.width, CV_32FC3);
        glReadPixels(0, 0, winSize.width, winSize.height, GL_BGR, GL_FLOAT, dst.data);

        // Unbind frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        // Cleanup
        glfwDestroyWindow(window);
        glfwTerminate();

        // Convert to 1-channel
        cv::cvtColor(dst, dst, CV_BGR2GRAY);
        cv::normalize(dst, dst, 0, 1, CV_MINMAX);
    }
}