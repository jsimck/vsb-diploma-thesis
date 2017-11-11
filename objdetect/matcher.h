#ifndef VSB_SEMESTRAL_PROJECT_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <memory>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/value_point.h"
#include "../core/classifier_criteria.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Matcher {
private:
    std::shared_ptr<ClassifierCriteria> criteria;

    // Methods
    cv::Vec3b normalizeHSV(cv::Vec3b &hsv);

    void extractFeatures(std::vector<Template> &templates);
    void generateFeaturePoints(std::vector<Template> &templates);
    void cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, int pointsCount, std::vector<cv::Point> &out);
    void nonMaximaSuppression(std::vector<Match> &matches);

    // Tests
    int testObjectSize(float scale, float depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable); // Test I
    int testSurfaceNormal(uchar normal, Window &window, cv::Mat &sceneSurfaceNormalsQuantized, cv::Point &stable); // Test II
    int testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Mat &sceneMagnitudes, cv::Point &edge); // Test III
    int testDepth(float scale, float diameter, float depthMedian, Window &window, cv::Mat &sceneDepth, cv::Point &stable); // Test IV
    int testColor(cv::Vec3b HSV, Window &window, cv::Mat &sceneHSV, cv::Point &stable); // Test V
public:
    // Constructor
    Matcher() = default;

    // Methods
    void match(float scale, cv::Mat &sceneHSV, cv::Mat &sceneDepth, cv::Mat &sceneMagnitudes, cv::Mat &sceneAnglesQuantized, cv::Mat &sceneSurfaceNormalsQuantized, std::vector<Window> &windows, std::vector<Match> &matches);
    void train(std::vector<Template> &templates);

    void setCriteria(std::shared_ptr<ClassifierCriteria> criteria);
};

#endif //VSB_SEMESTRAL_PROJECT_MATCHER_H
