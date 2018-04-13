#ifndef VSB_SEMESTRAL_PROJECT_SCENE_H
#define VSB_SEMESTRAL_PROJECT_SCENE_H

#include <opencv2/core/mat.hpp>
#include <ostream>
#include "camera.h"

namespace tless {
    struct ScenePyramid {
        float scale;  //!< Current scale of scale pyramid
        Camera camera; //!< Camera params for current matrix
        cv::Mat srcRGB, srcGray, srcHue, srcDepth; //!< Source scene in different
        cv::Mat srcGradients, srcNormals, srcNormals3D; //!< Matrix of quantized features
        cv::Mat spreadGradients, spreadNormals; //!< Matrix of quantized features

        ScenePyramid(float scale = 1.0f) : scale(scale) {}
    };

    /**
     * @brief Scene wrapper, holds all scene images/normals etc. throughout classification.
     */
    struct Scene {
    public:
        uint id = 0;
        std::vector<ScenePyramid> pyramid; //!< Scene image pyramid

        Scene() = default;

        friend std::ostream &operator<<(std::ostream &os, const Scene &scene);
    };
}

#endif
