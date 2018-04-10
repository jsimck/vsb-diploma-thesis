#include "glutils.h"
#include "../glcore/shader.h"
#include "../objdetect/classifier.h"
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

    glm::mat4 tless::nMat(const glm::mat4 &modelView) {
        return glm::inverseTranspose(modelView);
    }

    glm::mat4 nMat(const glm::mat4 &model, const glm::mat4 &view) {
        return nMat(view * model);
    }

    void tless::rescaleK(cv::Mat &K, const cv::Size &src, const cv::Size &dst) {
        // Compute resize ratio in X and Y
        float resX = dst.width / static_cast<float>(src.width);
        float resY = dst.height / static_cast<float>(src.height);
        float res, offsetX = 0, offsetY = 0;

        // Check which direction is larger
        if (resY > resX) {
            res = resY;
            offsetX = (src.width - dst.width) / 2;
        } else {
            res = resX;
            offsetY = (src.height - dst.height) / 2;
        }

        // Rescale
        K.at<float>(0, 0) *= res;
        K.at<float>(0, 2) *= res;
        K.at<float>(1, 1) *= res;
        K.at<float>(1, 2) *= res;

        // Shift cam center
        K.at<float>(0, 2) += offsetX;
        K.at<float>(1, 2) += offsetY;
    }
}