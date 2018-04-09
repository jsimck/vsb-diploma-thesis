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

    void drawDepth(const Template &tpl, const FrameBuffer &fbo, const Shader &shader, const Mesh &mesh, cv::Mat &dst, const glm::mat4 &modelView, const glm::mat4 &modelViewProjection) {
        // Render
        fbo.bind();
        glEnable(GL_DEPTH_TEST);

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        shader.use();
        shader.setMat4("MVMatrix", modelView);
        shader.setMat4("MVPMatrix", modelViewProjection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        // TODO define winSize
        cv::Size winSize(108, 108);
        dst = cv::Mat::zeros(winSize.height, winSize.width, CV_32FC3);
        glReadPixels(0, 0, winSize.width, winSize.height, GL_BGR, GL_FLOAT, dst.data);

        // Unbind frame buffer
        fbo.unbind();
        glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

        // Convert to 1-channel
        cv::cvtColor(dst, dst, CV_BGR2GRAY);
        cv::normalize(dst, dst, 0, 1, CV_MINMAX);
    }

    void render(const Template &tpl, const FrameBuffer &fbo, const Shader &depthShader, const Shader &normalShader, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals, const glm::mat4 &modelView, const glm::mat4 &modelViewProjection) {
        // TODO define winSize
        cv::Size winSize(108, 108);

        // Render
        fbo.bind();

        /// NORMALS
        // Clear buffer
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        normalShader.use();
        normalShader.setMat4("NMatrix", glm::inverseTranspose(modelView));
        normalShader.setMat4("MVPMatrix", modelViewProjection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        normals = cv::Mat::zeros(winSize.height, winSize.width, CV_32FC3);
        glReadPixels(0, 0, winSize.width, winSize.height, GL_BGR, GL_FLOAT, normals.data);

        /// DEPTH
        // Clear buffer
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        depthShader.use();
        depthShader.setMat4("MVMatrix", modelView);
        depthShader.setMat4("MVPMatrix", modelViewProjection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        depth = cv::Mat::zeros(winSize.height, winSize.width, CV_32FC3);
        glReadPixels(0, 0, winSize.width, winSize.height, GL_BGR, GL_FLOAT, depth.data);

        // Unbind frame buffer
        fbo.unbind();

        // Convert to 1-channel and normalize depth
        cv::cvtColor(depth, depth, CV_BGR2GRAY);
        cv::normalize(depth, depth, 0, 1, CV_MINMAX);
    }
}