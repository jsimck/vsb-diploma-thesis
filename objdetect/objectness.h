#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"
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
    unsigned int step; // Stepping for sliding window [5]
    float minThreshold; // Min threshold applied in sobel filtered image thresholding [0.01f]
    float maxThreshold; // Max threshold applied in sobel filtered image thresholding [0.1f]
    float matchThresholdFactor; // Factor used to reduce minEdge for objectness detection to improve occlusion/noise matching [30% -> 0.3f]
    float slidingWindowSizeFactor; // Reduces sliding window size to improve edge detection [1.0f]

    void filterSobel(cv::Mat &src, cv::Mat &dst);
    void thresholdMinMax(cv::Mat &src, cv::Mat &dst, float minThreshold, float maxThreshold);
public:
    // Constructors
    Objectness(unsigned int step = 5, float minThreshold = 0.01f, float maxThreshold = 0.1f, float matchThresholdFactor = 0.3f, float slidingWindowSizeFactor = 1.0f)
        : step(step), minThreshold(minThreshold), maxThreshold(maxThreshold), matchThresholdFactor(matchThresholdFactor), slidingWindowSizeFactor(slidingWindowSizeFactor) {}

    // Methods
    void extractMinEdgels(std::vector<TemplateGroup> &templateGroups, DatasetInfo &info);
    void objectness(cv::Mat &sceneGrayscale, cv::Mat &sceneColor, cv::Mat &sceneDepthNormalized, std::vector<Window> &windows, DatasetInfo &info);

    // Getters
    unsigned int getStep() const;
    float getMinThreshold() const;
    float getMaxThreshold() const;
    float getMatchThresholdFactor() const;
    float getSlidingWindowSizeFactor() const;

    // Setters
    void setStep(unsigned int step);
    void setMinThreshold(float minThreshold);
    void setMaxThreshold(float maxThreshold);
    void setMatchThresholdFactor(float matchThresholdFactor);
    void setSlidingWindowSizeFactor(float slidingWindowSizeFactor);
};

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
