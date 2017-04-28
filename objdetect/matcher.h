#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/group.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Matcher {
private:
    float tMatch;
    uint pointsCount; // Number of feature points to extract for each template
    cv::Range neighbourhood; // Extent of local neighbourhood

    // Train thresholds
    uchar t1Canny;
    uchar t2Canny;
    uchar tSobel;
    uchar tGray;

    // Methods
    inline cv::Vec3b normalizeHSV(const cv::Vec3b &hsv);
    inline void extractFeatures(std::vector<Group> &groups);
    inline void generateFeaturePoints(std::vector<Group> &groups);
    inline int quantizeOrientationGradient(float deg);
    void nonMaximaSuppression(std::vector<Match> &matches);

    // Tests
    inline bool testObjectSize(float scale); // Test I
    inline int testSurfaceNormal(const uchar tNormal, Window &window, const cv::Mat &sceneDepth, const cv::Point &stable, const cv::Range &neighbourhood); // Test II
    inline int testGradients(const uchar tOrientation, Window &window, const cv::Mat &sceneGray, const cv::Point &edge, const cv::Range &neighbourhood); // Test III
    inline int testDepth(int physicalDiameter, std::vector<int> &depths); // Test IV
    inline int testColor(const cv::Vec3b tHSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &stable, const cv::Range &neighbourhood); // Test V
public:
    // Static methods
    static int median(std::vector<int> &values);
    static float orientationGradient(const cv::Mat &src, cv::Point &p);

    // Constructor
    Matcher(uint pointsCount = 100, float tMatch = 0.6f, uchar t1Canny = 100, uchar t2Canny = 200, uchar tSobel = 50, uchar tGray = 50, cv::Range neighbourhood = cv::Range(5, 5))
        : pointsCount(pointsCount), tMatch(tMatch), t1Canny(t1Canny), t2Canny(t2Canny), tSobel(tSobel), tGray(tGray), neighbourhood(neighbourhood) {}

    // Methods
    void match(const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth, std::vector<Window> &windows, std::vector<Match> &matches);
    void train(std::vector<Group> &groups);

    // Getters
    uint getPointsCount() const;
    uchar getT1Canny() const;
    uchar getT2Canny() const;
    uchar getTSobel() const;
    uchar getTGray() const;
    float getTMatch() const;
    const cv::Range &getNeighbourhood() const;

    // Setters
    void setPointsCount(uint count);
    void setT1Canny(uchar t1Canny);
    void setT2Canny(uchar t2Canny);
    void setTSobel(uchar tSobel);
    void setTGray(uchar tGray);
    void setTMatch(float tMatch);
    void setNeighbourhood(cv::Range neighbourhood);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_MATCHER_H
