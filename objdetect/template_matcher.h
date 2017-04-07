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

    // Methods
    inline void extractTemplateFeatures(std::vector<TemplateGroup> &groups);
    inline void generateFeaturePoints(std::vector<TemplateGroup> &groups);
    inline int quantizeOrientationGradients(float deg);

    // Tests
    inline bool testObjectSize(float scale); // Test I
    inline float testSurfaceNormalOrientation(); // Test II
    inline float testIntensityGradients(); // Test III
    inline float testDepth(); // Test IV
    inline float testColor(); // Test V
public:
    // Static methods
    static float extractGradientOrientation(cv::Mat &src, cv::Point &point);

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
    uchar getCannyThreshold1() const;
    uchar getCannyThreshold2() const;
    uchar getSobelMaxThreshold() const;
    uchar getGrayscaleMinThreshold() const;

    // Setters
    void setFeaturePointsCount(uint featurePointsCount);
    void setCannyThreshold1(uchar cannyThreshold1);
    void setCannyThreshold2(uchar cannyThreshold2);
    void setSobelMaxThreshold(uchar sobelMaxThreshold);
    void setGrayscaleMinThreshold(uchar grayscaleMinThreshold);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
