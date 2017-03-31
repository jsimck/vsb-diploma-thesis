#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"
#include "../core/template_match.h"

class TemplateMatcher {
private:
    uint featurePointsCount;

    // Tests
    inline bool testObjectSize(); // Test I
    inline float testSurfaceNormalOrientation(); // Test II
    inline float testIntensityGradients(); // Test III
    inline float testDepth(); // Test IV
    inline float testColor(); // Test V
public:
    // Constructor
    TemplateMatcher(uint featurePointsCount = 100) : featurePointsCount(featurePointsCount) {}

    // Methods
    void match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
               std::vector<Window> &windows, std::vector<TemplateMatch> &matches);

    // Getters
    uint getFeaturePointsCount() const;

    // Setters
    void setFeaturePointsCount(uint featurePointsCount);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
