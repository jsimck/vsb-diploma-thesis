#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"
#include "../core/group.h"
#include "../core/window.h"
#include "../core/dataset_info.h"

/**
 * class Objetness
 *
 * Simple objectness detection algorithm, based on depth discontinuities of depth images.
 * (depth discontinuities => areas where pixel arise on the edges of objects). Sliding window
 * is used to slide through the scene (using size of a smallest template in dataset) and calculating
 * amount of depth pixels in the scene. Window is classified as containing object if it contains
 * at least 30% of edgels of the template containing least amount of them (info.minEdgels), extracted
 * throught extractMinEdgels method.
 */
class Objectness {
public:
    // Params
    struct {
        uint step; // Stepping for sliding window [5]
        float tEdgesMin; // Min threshold applied in sobel filtered image thresholding [0.01f]
        float tEdgesMax; // Max threshold applied in sobel filtered image thresholding [0.1f]
        float tMatch; // Factor of minEdgels window should contain to be classified as valid [30% -> 0.3f]
    } params;

    // Constructors
    Objectness();

    // Methods
    void extractMinEdgels(std::vector<Template> &templates, DataSetInfo &info);
    void objectness(cv::Mat &sceneDepthNorm, std::vector<Window> &windows, DataSetInfo &info);
};

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
