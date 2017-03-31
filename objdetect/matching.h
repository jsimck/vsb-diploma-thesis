#ifndef VSB_SEMESTRAL_PROJECT_MATCHING_H
#define VSB_SEMESTRAL_PROJECT_MATCHING_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"

class Matching {
private:
    uint featurePointsCount;

    // Tests
    void objectSize(); // Test I
    void surfaceNormalOrientation(); // Test II
    void intensityGradients(); // Test III
    void depth(); // Test IV
    void color(); // Test V
public:
    // Constructor
    Matching(uint featurePointsCount = 100) : featurePointsCount(featurePointsCount) {}

    // Methods
    void match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth, const std::vector<Window> &windows);

    // Getters
    uint getFeaturePointsCount() const;

    // Setters
    void setFeaturePointsCount(uint featurePointsCount);
};

#endif //VSB_SEMESTRAL_PROJECT_MATCHING_H
