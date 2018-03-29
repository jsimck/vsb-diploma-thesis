#ifndef VSB_SEMESTRAL_PROJECT_MATCHER_H
#define VSB_SEMESTRAL_PROJECT_MATCHER_H

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <memory>
#include "../core/window.h"
#include "../core/match.h"
#include "../core/classifier_criteria.h"
#include "../core/scene.h"

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
         * @brief Selects scattered feature points, that are somehow uniformly distributed over the template.
         *
         * @param[in]  points    Array of input points to extract scattered points from (pay attention to sorting)
         * @param[in]  count     Final number of points we want to extract from the points array
         * @param[out] scattered Output array containing extracted scattered points
         */
        void selectScatteredFeaturePoints(const std::vector<std::pair<cv::Point, uchar>> &points,
                                          uint count, std::vector<cv::Point> &scattered);

        /**
         * @brief Accumulates list of depth difference between scene and template across all stable points and computes median of depth differences.
         *
         * @param[in] sceneDepth   Input scene 16-bit depth image
         * @param[in] windowTl     Top left corner of currently processed window (used for offsets)
         * @param[in] stablePoints List of precomputed points on stable places of the object
         * @param[in] tplDepths    Precomputed template depths at stable points
         * @return                 Median value of depth differences across all stable points
         */
        int depthDiffMedian(const cv::Mat &sceneDepth, const cv::Point &windowTl, const std::vector<cv::Point> &stablePoints,
                            const std::vector<ushort> &tplDepths);


        /**
         * @brief Test IV in matching, it performs a depth test to se whether object depth differences are lower than median.
         *
         * @param sceneDepth  Input scene 16-bit depth image
         * @param windowTl    Top left corner of currently processed window (used for offsets)
         * @param diameter    Pre-calculated object diameter (candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK)
         * @param depthMedian Depth median computed from depth differences using depthDiffMedian() function
         * @param depth       Currently processed template depth, sampled at given stable feature point
         * @param stable      Currently processed stable feature point
         * @return            1 whether there was a match within defined boundaries around stable point (-patchOffset <-> patchOffset)
         */
        inline int testDepth(cv::Mat &sceneDepth, const cv::Point &windowTl, float diameter, int depthMedian,
                             ushort depth, const cv::Point &stable);

        // Tests
        inline int testObjectSize(ushort depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable); // Test I
        inline int testSurfaceNormal(uchar normal, Window &window, cv::Mat &sceneSurfaceNormalsQuantized, cv::Point &stable); // Test II
        inline int testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Point &edge); // Test III
        inline int testColor(uchar hue, Window &window, cv::Mat &sceneHSV, cv::Point &stable); // Test V

    public:
        Matcher(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {}

        /**
         * @brief Applies template matching for each template in candidate list of each window.
         *
         * Each template candidate for each window needs to pass 5 tests where we compare object size, surface normals, gradients
         * depth and color between template trained features and scene features on trained feature points. Feature point is matched
         * if there's a match inside small area around feature point (5x5) to compensate sliding window step. Each test is computed
         * in order of it's complexity, if candidate doesn't match at least [criteria.matchFactor] of feature points in each, no further
         * tests are computed and we continue with other candidates. Candidate that passes all tests gets final score of a fraction of
         * sum of matched points. After all windows have been tested, non-maxima suppression is applied to all matches to filter out the
         * best candidates which are than retained in the final matches vector.
         *
         * @param[in]  scene   Current scene in image scale pyramid
         * @param[in]  windows Windows array that passed objectness detection test with candidates filtered in hasher verification
         * @param[out] matches Final array foound matches
         */
        void match(ScenePyramid &scene, std::vector<Window> &windows, std::vector<Match> &matches);

        /**
         * @brief Generates feature points and extract features for each template.
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
