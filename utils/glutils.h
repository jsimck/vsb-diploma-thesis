#ifndef VSB_SEMESTRAL_PROJECT_GLUTILS_H
#define VSB_SEMESTRAL_PROJECT_GLUTILS_H

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <opencv2/core/mat.hpp>
#include "../glcore/mesh.h"
#include "../core/template.h"
#include "../glcore/frame_buffer.h"
#include "../glcore/shader.h"

namespace tless {
    enum WindowCoords { Y_DOWN, Y_UP };

    /**
     * @brief Calculates projection matrix out of camera intristic parameters.
     *
     * @param[in] K      Camera K matrix 3x3 32-bit float
     * @param[in] x0     x-Camera image origin, usually 0
     * @param[in] y0     y-Camera image origin, usually 0
     * @param[in] w      Image width
     * @param[in] h      Image height
     * @param[in] nc     Near clipping plane
     * @param[in] fc     Far clipping plane
     * @param[in] coords Direction of Y Y_DOWN or Y_UP
     * @return
     */
    glm::mat4 pMat(const cv::Mat &K, int x0, int y0, int w, int h, float nc = 100, float fc = 2000, WindowCoords coords = Y_UP);

    /**
     * @brief Calculates view matrix out of R and t.
     *
     *
     * @param[in] R Rotation matrix
     * @param[in] t Translation vector
     * @return
     */
    glm::mat4 vMat(const cv::Mat &R, const cv::Mat &t);

    /**
     * @brief Calculates ModelView matrix.
     *
     * @param[in] model Model matrix
     * @param[in] view  View matrix
     * @return          ModelView matrix
     */
    glm::mat4 mvMat(const glm::mat4 &model, const glm::mat4 &view);

    /**
     * @brief Calculates ModelViewProjection matrix.
     *
     * @param[in] model      Model matrix
     * @param[in] view       View matrix
     * @param[in] projection Projection matrix
     * @return               ModelViewProjection matrix
     */
    glm::mat4 mvpMat(const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection);

    /**
     * @brief Calculates Normal matrix.
     *
     * @param[in] model Model matrix
     * @param[in] view  View matrix
     * @return          Normal matrix
     */
    glm::mat4 nMat(const glm::mat4 &model, const glm::mat4 &view);

    /**
     * @brief Calculates Normal matrix.
     *
     * @param[in] modelView ModelView matrix
     * @return              Normal matrix
     */
    glm::mat4 nMat(const glm::mat4 &modelView);

    /**
     * @brief Recalculates camera intristic params based on new size.
     *
     * @param[in,out] K   Camera K matrix
     * @param[in]     src Source img size
     * @param[in]     dst Destination img size
     */
    void rescaleK(cv::Mat &K, const cv::Size &src, const cv::Size &dst);

    /**
     * @brief Recalculates camera intristic params based on the differences between two new windows.
     * @param[in,out] K   Camera K matrix
     * @param[in]     src Source img bounding box
     * @param[in]     dst Destination img bounding box
     */
    void rescaleK(cv::Mat &K, const cv::Rect &src, const cv::Rect &dst);
}

#endif