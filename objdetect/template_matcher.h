#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"
#include "../core/template_match.h"
#include "../core/template_group.h"

class TemplateMatcher {
private:
    uint featurePointsCount;

    // Train thresholds
    uchar cannyThreshold1;
    uchar cannyThreshold2;
    uchar sobelMaxThreshold;
    uchar grayscaleMinThreshold;

    void generateFeaturePoints(std::vector<TemplateGroup> &groups);

    // Tests
    inline bool testObjectSize(float scale); // Test I
    inline float testSurfaceNormalOrientation(); // Test II
    inline float testIntensityGradients(); // Test III
    inline float testDepth(); // Test IV
    inline float testColor(); // Test V
public:
    // Constructor
    TemplateMatcher(uint featurePointsCount = 100, uchar cannyThreshold1 = 100, uchar cannyThreshold2 = 200,
                    uchar sobelMaxThreshold = 50, uchar grayscaleMinThreshold = 50)
        : featurePointsCount(featurePointsCount), cannyThreshold1(cannyThreshold1), cannyThreshold2(cannyThreshold2),
          sobelMaxThreshold(sobelMaxThreshold), grayscaleMinThreshold(grayscaleMinThreshold) {}

    // Methods
    void match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
               std::vector<Window> &windows, std::vector<TemplateMatch> &matches);
    void train(std::vector<TemplateGroup> &groups);

    // Getters
    uint getFeaturePointsCount() const;

    // Setters
    void setFeaturePointsCount(uint featurePointsCount);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
