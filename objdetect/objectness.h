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
private:
    uint step; // Stepping for sliding window [5]
    float tMin; // Min threshold applied in sobel filtered image thresholding [0.01f]
    float tMax; // Max threshold applied in sobel filtered image thresholding [0.1f]
    float tMatch; // Factor of minEdgels window should contain to be classified as valid [30% -> 0.3f]
public:
    // Statics
    static void filterSobel(const cv::Mat &src, cv::Mat &dst);
    static void thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max);

    // Constructors
    explicit Objectness(uint step = 5, float tMin = 0.01f, float tMax = 0.1f, float tMatch = 0.3f)
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
