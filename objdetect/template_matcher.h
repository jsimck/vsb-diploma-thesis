#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/group.h"

class TemplateMatcher {
private:
    uint featurePointsCount;
    float matchThreshold;
    cv::Range matchNeighbourhood;

    // Train thresholds
    uchar cannyThreshold1;
    uchar cannyThreshold2;
    uchar sobelMaxThreshold;
    uchar grayscaleMinThreshold;

    // Methods
    inline cv::Vec3b normalizeHSV(const cv::Vec3b &px);
    inline void extractTemplateFeatures(std::vector<Group> &groups);
    inline void generateFeaturePoints(std::vector<Group> &groups);
    inline int quantizeOrientationGradient(float deg);
    void nonMaximaSuppression(std::vector<Match> &matches);

    // Tests
    inline bool testObjectSize(float scale); // Test I
    inline int testSurfaceNormalOrientation(const int tNormal, Window &w, const cv::Mat &srcDepth, const cv::Point &featurePoint, const cv::Range &n); // Test II
    inline int testIntensityGradients(const int tOrientation, Window &w, const cv::Mat &srcGrayscale, const cv::Point &featurePoint, const cv::Range &n); // Test III
    inline int testDepth(int physicalDiameter, std::vector<int> &depths); // Test IV
    inline int testColor(const cv::Vec3b tHSV, Window &w, const cv::Mat &sceneHSV, const cv::Point &featurePoint, const cv::Range &n); // Test V
public:
    // Static methods
    static int median(std::vector<int> &values);
    static float extractOrientationGradient(const cv::Mat &src, cv::Point &point);

    // Constructor
    TemplateMatcher(uint featurePointsCount = 100, float matchThreshold = 0.6f, uchar cannyThreshold1 = 100, uchar cannyThreshold2 = 200,
                    uchar sobelMaxThreshold = 50, uchar grayscaleMinThreshold = 50, cv::Range matchNeighbourhood = cv::Range(5, 5))
        : featurePointsCount(featurePointsCount), matchThreshold(matchThreshold), cannyThreshold1(cannyThreshold1), cannyThreshold2(cannyThreshold2),
          sobelMaxThreshold(sobelMaxThreshold), grayscaleMinThreshold(grayscaleMinThreshold), matchNeighbourhood(matchNeighbourhood) {}

    // Methods
    void match(const cv::Mat &srcHSV, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
               std::vector<Window> &windows, std::vector<Match> &matches);
    void train(std::vector<Group> &groups);

    // Getters
    uint getFeaturePointsCount() const;
    uchar getCannyThreshold1() const;
    uchar getCannyThreshold2() const;
    uchar getSobelMaxThreshold() const;
    uchar getGrayscaleMinThreshold() const;
    float getMatchThreshold() const;
    const cv::Range &getMatchNeighbourhood() const;

    // Setters
    void setFeaturePointsCount(uint featurePointsCount);
    void setCannyThreshold1(uchar cannyThreshold1);
    void setCannyThreshold2(uchar cannyThreshold2);
    void setSobelMaxThreshold(uchar sobelMaxThreshold);
    void setGrayscaleMinThreshold(uchar grayscaleMinThreshold);
    void setMatchThreshold(float matchThreshold);
    void setMatchNeighbourhood(cv::Range matchNeighbourhood);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
