#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

class Objectness {
private:
    float minThreshold; // Min threshold applied in sobel filtered image thresholding [0.01f]
    float maxThreshold; // Max threshold applied in sobel filtered image thresholding [0.1f]

    float matchThresholdFactor; // Factor used to reduce minEdge for objectness detection to improve occlusion/noise matching [30% -> 0.3f]
    float slidingWindowSizeFactor; // Reduces sliding window size to improve edge detection [1.0f]
    float slidingWindowStepFactor; // Reduces sliding window step (of already resized window) to improve edge detection [0.4f]

    void filterSobel(cv::Mat &src, cv::Mat &dst);
    void thresholdMinMax(cv::Mat &src, cv::Mat &dst, float minThreshold, float maxThreshold);
public:
    // Constructors
    Objectness();

    // Methods
    cv::Vec3f extractMinEdgels(std::vector<TemplateGroup> &templateGroups);
    cv::Rect objectness(cv::Mat &sceneGrayscale, cv::Mat &sceneColor, cv::Mat &sceneDepthNormalized, cv::Vec3f minEdgels);

    // Getters
    float getMinThreshold() const;
    float getMaxThreshold() const;
    float getMatchThresholdFactor() const;
    float getSlidingWindowSizeFactor() const;
    float getSlidingWindowStepFactor() const;

    // Setters
    void setMinThreshold(float minThreshold);
    void setMaxThreshold(float maxThreshold);
    void setMatchThresholdFactor(float matchThresholdFactor);
    void setSlidingWindowSizeFactor(float slidingWindowSizeFactor);
    void setSlidingWindowStepFactor(float slidingWindowStepFactor);
};

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
