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
 * depth discontinuities => areas where pixel arise on the edges of objects. First we extract template
 * with minimum number of depth discontinuity edgels using extractMinEdgels() method and save it's bounding box.
 * Then scene is also first run through sobel filter, then thresholded and then using sliding window of saved BB of
 * smalles template, we slide through the thresholded image and look for edgels. We classify sliding window as containing object
 * if it contains at least 30% of edgels in a template containing least amount of them.
 */
class Objectness {
private:
    uint step; // Stepping for sliding window [5]
    float tMin; // Min threshold applied in sobel filtered image thresholding [0.01f]
    float tMax; // Max threshold applied in sobel filtered image thresholding [0.1f]
    float tMatch; // Factor of minEdgels window should contain to be classified as valid [30% -> 0.3f]
public:
    // Statics
    static void filterSobel(cv::Mat &src, cv::Mat &dst);
    static void thresholdMinMax(cv::Mat &src, cv::Mat &dst, float min, float max);

    // Constructors
    Objectness(uint step = 5, float tMin = 0.01f, float tMax = 0.1f, float tMatch = 0.3f)
        : step(step), tMin(tMin), tMax(tMax), tMatch(tMatch) {}

    // Methods
    void extractMinEdgels(std::vector<Group> &groups, DataSetInfo &info);
    void objectness(cv::Mat &sceneDepthNorm, std::vector<Window> &windows, DataSetInfo &info);

    // Getters
    uint getStep() const;
    float getTMin() const;
    float getTMax() const;
    float getTMatch() const;

    // Setters
    void setStep(uint step);
    void setTMin(float tMin);
    void setTMax(float tMax);
    void setTMatch(float tMatch);
};

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
