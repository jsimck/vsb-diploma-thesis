#ifndef VSB_SEMESTRAL_PROJECT_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/value_point.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Matcher {
private:

    // Methods
    cv::Vec3b normalizeHSV(const cv::Vec3b &hsv);
    uchar quantizeOrientationGradient(float deg);

    void extractFeatures(std::vector<Template> &templates);
    void generateFeaturePoints(std::vector<Template> &templates);
    void cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, uint pointsCount, std::vector<cv::Point> &out);

    void nonMaximaSuppression(std::vector<Match> &matches);

    // Tests
    bool testObjectSize(float scale); // Test I
    int testSurfaceNormal(uchar normal, Window &window, const cv::Mat &sceneDepth, const cv::Point &stable); // Test II
    int testGradients(uchar gradient, Window &window, const cv::Mat &sceneAngle, const cv::Mat &sceneMagnitude, const cv::Point &edge); // Test III
    int testDepth(int physicalDiameter, std::vector<int> &depths); // Test IV
    int testColor(cv::Vec3b HSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &stable); // Test V
public:
    // Params
    struct {
        uint pointsCount; // Number of feature points to extract for each template
        float tMatch; // [0.0 - 1.0] number indicating how many percentage of points should match
        float tOverlap; // [0.0 - 1.0] overlap threshold, how much should 2 windows overlap in order to calculate non-maxima suppression
        uchar tColorTest; // 0-180 HUE value max threshold to pass comparing colors between scene and template
        cv::Range neighbourhood; // (-2, 2) area to search around feature point to look for match
    } params;

    // Static methods
    static int median(std::vector<int> &values);

    // Constructor
    Matcher();

    // Methods
    void match(float scale, const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth, std::vector<Window> &windows, std::vector<Match> &matches);
    void train(std::vector<Template> &templates);
};

#endif //VSB_SEMESTRAL_PROJECT_MATCHER_H
