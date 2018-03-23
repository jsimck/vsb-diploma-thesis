#include "glutils.h"

namespace tless {
    glm::mat4 pMat(const glm::mat3 &K, int x0, int y0, int w, int h, float nc, float fc, WindowCoords coords = Y_UP) {
        float depth = fc - nc;
        float q = -(fc + nc) / depth;
        float qn = -2 * (fc * nc) / depth;

        if (coords == Y_UP) {
            return glm::transpose(glm::mat4(
                    2 * K[0][0] / w, -2 * K[0][1] / w , (-2 * K[0][2] + w + 2 * x0) / w , 0,
                    0              , -2 * K[1][1] / h , (-2 * K[1][2] + h + 2 * y0) / h , 0,
                    0              , 0                , q                               , qn,
                    0              , 0                , -1                              , 0
            ));
        } else {
            assert(coords == Y_DOWN);
            return glm::transpose(glm::mat4(
                    2 * K[0][0] / w, -2 * K[0][1] / w , (-2 * K[0][2] + w + 2 * x0) / w , 0,
                    0              , 2 * K[1][1] / h  , (2 * K[1][2] - h + 2 * y0) / h  , 0,
                    0              , 0                , q                               , qn,
                    0              , 0                , -1                              , 0
            ));
        }
    }

    glm::mat4 vMat(const glm::mat3 &R, const glm::vec3 &t) {
        // Fill view matrix
        glm::mat4 VMatrix(R);
        VMatrix[0][3] = t[0];
        VMatrix[1][3] = t[1];
        VMatrix[2][3] = t[2];

        // Convert OpenCV to OpenGL camera system
        glm::mat4 yzFlip; // Create flip matrix for coordinate system conversion
        yzFlip[1][1] = -1;
        yzFlip[2][2] = -1;
        VMatrix *= yzFlip;

        return glm::transpose(VMatrix); // OpenGL expects column-wise matrix format
    }

    glm::mat4 mvMat(const glm::mat4 &model, const glm::mat4 &view) {
        return model * view;
    }

    glm::mat4 mvpMat(const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection) {
        return model * view * projection;
    }

    glm::mat4 nMat(const glm::mat4 &model, const glm::mat4 &view) {
        return glm::inverseTranspose(view * model);
    }
}