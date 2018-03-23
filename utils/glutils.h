#ifndef VSB_SEMESTRAL_PROJECT_GLUTILS_H
#define VSB_SEMESTRAL_PROJECT_GLUTILS_H

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace tless {
    enum WindowCoords { Y_DOWN, Y_UP };

    /**
     * @brief Calculates projection matrix out of camera intristic parameters.
     *
     * @param[in] K      Camera K matrix
     * @param[in] x0     x-Camera image origin, usually 0
     * @param[in] y0     y-Camera image origin, usually 0
     * @param[in] w      Image width
     * @param[in] h      Image height
     * @param[in] nc     Near clipping plane
     * @param[in] fc     Far clipping plane
     * @param[in] coords Direction of Y Y_DOWN or Y_UP
     * @return
     */
    glm::mat4 pMat(const glm::mat3 &K, int x0, int y0, int w, int h, float nc, float fc, WindowCoords coords = Y_DOWN);

    /**
     * @brief Calculates view matrix out of R and t.
     *
     *
     * @param[in] R Rotation matrix
     * @param[in] t Translation vector
     * @return
     */
    glm::mat4 vMat(const glm::mat3 &R, const glm::vec3 &t);

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
}

#endif