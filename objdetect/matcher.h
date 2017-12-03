#ifndef VSB_SEMESTRAL_PROJECT_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <memory>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/classifier_criteria.h"

namespace tless {
    /**
     * class Hasher
     *
     * Used to train HashTables and quickly verify what templates should be matched
     * per each window passed form objectness detection
     */
    class Matcher {
    private:
        cv::Ptr<ClassifierCriteria> criteria;

        /**
         * @brief Selects scattered feature points, that are somehow uniformly distributed over the template
         *
         * @param[in]  points    Array of input points to extract scattered points from (pay attention to sorting)
         * @param[in]  count     Final number of points we want to extract from the points array
         * @param[out] scattered Output array containing extracted scattered points
         */
        void selectScatteredFeaturePoints(std::vector<std::pair<cv::Point, uchar>> &points, uint count, std::vector<cv::Point> &scattered);

        /**
         * @brief Applies non-maxima suppression to matched window, removing matches with large overlap and lower score
         *
         * This function calculates overlap between each window, if the overlap is > than [criteria.overlapFactor]
         * we only retain a match with higher score.
         *
         * @param[in,out] matches Input/output array of matches to apply non-maxima suppression on
         */
        void nonMaximaSuppression(std::vector<Match> &matches);

        // Tests
        int testObjectSize(float scale, ushort depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable); // Test I
        int testSurfaceNormal(uchar normal, Window &window, cv::Mat &sceneSurfaceNormalsQuantized, cv::Point &stable); // Test II
        int testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Mat &sceneMagnitudes, cv::Point &edge); // Test III
        int testDepth(float scale, float diameter, ushort depthMedian, Window &window, cv::Mat &sceneDepth, cv::Point &stable); // Test IV
        int testColor(cv::Vec3b HSV, Window &window, cv::Mat &sceneHSV, cv::Point &stable); // Test V

    public:
        // Constructor
        Matcher(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        // Methods
        void match(float scale, cv::Mat &sceneHSV, cv::Mat &sceneDepth, cv::Mat &sceneMagnitudes, cv::Mat &sceneAnglesQuantized,
                   cv::Mat &sceneSurfaceNormalsQuantized, std::vector<Window> &windows, std::vector<Match> &matches);

        /**
         * @brief Generates feature points and extract features for each template
         *
         * Edge points are detected by applying sobel operator on gray image (which was eroded and blurred) and
         * picking points with magnitude > minEdgeMag. Stable points are then selected as points with pixel value > minStableVal
         * and edge magnitude <= minEdgeMag. Both arrays of points are then filtered by selectScatteredFeaturePoints
         * and only [criteria.featurePointsCount] points are then retained. After we have [criteria.featurePointsCount]
         * stable and edge points, we perform feature extraction for depth median, gradients, normals, depths and colors for each template.
         *
         * @param[in,out] templates    Array of templates to extract features for
         * @param[in]     minStableVal Minimum gray pixel value of an extracted stablePoint
         * @param[in]     minEdgeMag   Minimum edge magnitude of an extracted edgePoint
         */
        void train(std::vector<Template> &templates, uchar minStableVal = 40, uchar minEdgeMag = 40);
    };
}

#endif
