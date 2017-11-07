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
    cv::Vec3b normalizeHSV(const cv::Vec3b &hsv);
    uchar quantizeOrientationGradient(float deg);

    void extractFeatures(std::vector<Template> &templates);
    void generateFeaturePoints(std::vector<Template> &templates);
    void cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, int pointsCount,
                                 std::vector<cv::Point> &out);

    void nonMaximaSuppression(std::vector<Match> &matches);

    // Tests
    bool testObjectSize(float scale); // Test I
    int testSurfaceNormal(uchar normal, Window &window, const cv::Mat &sceneDepth, const cv::Point &stable); // Test II
    int testGradients(uchar gradient, Window &window, const cv::Mat &sceneAngle, const cv::Mat &sceneMagnitude, const cv::Point &edge); // Test III
    int testDepth(int physicalDiameter, std::vector<int> &depths); // Test IV
    int testColor(cv::Vec3b HSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &stable); // Test V
public:
    // Static methods
    static int median(std::vector<int> &values);

    // Constructor
    Matcher() = default;

    // Methods
    void match(float scale, const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth, std::vector<Window> &windows, std::vector<Match> &matches);
    void train(std::vector<Template> &templates);

    void setCriteria(std::shared_ptr<ClassifierCriteria> criteria);
};

#endif //VSB_SEMESTRAL_PROJECT_MATCHER_H
