#ifndef VSB_SEMESTRAL_PROJECT_SCENE_H
#define VSB_SEMESTRAL_PROJECT_SCENE_H

#include <opencv2/core/mat.hpp>
#include <ostream>
#include "camera.h"

namespace tless {
    /**
     * @brief Scene wrapper, holds all scene images/normals etc. throughout classification.
     */
    struct Scene {
    public:
        uint id = 0;
        Camera camera;
        float scale = 1.0f;  //!< Current scale of scale pyramid
        int elev = 0, mode = 0;

        cv::Mat srcRGB, srcGray, srcHue, srcDepth; //!< Source scene in different
        cv::Mat srcGradients, srcNormals; //!< Matrix of quantized features

        Scene() = default;

        friend std::ostream &operator<<(std::ostream &os, const Scene &scene);
    };
}

#endif
