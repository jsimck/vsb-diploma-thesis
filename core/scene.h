#ifndef VSB_SEMESTRAL_PROJECT_SCENE_H
#define VSB_SEMESTRAL_PROJECT_SCENE_H

#include <opencv2/core/mat.hpp>
#include <ostream>
#include "camera.h"

struct Scene {
public:
    Camera camera;
    float scale = 1.0f;  //!< Current scale of scale pyramid
    int elev = 0, mode = 0;

    cv::Mat srcRGB, srcGray, srcHSV, srcDepth; //!< Source scene in different
    cv::Mat gradients, normals, magnitudes; //!< Matrix of quantized features

    Scene() = default;

    friend std::ostream &operator<<(std::ostream &os, const Scene &scene);
};

#endif
