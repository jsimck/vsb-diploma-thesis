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
         * @param[in] stablePoints List of precomputed feature stable points shifted to current window
         * @param[in] tplDepths    Precomputed template depths at stable points
         * @return                 Median value of depth differences across all stable points
         */
        int depthDiffMedian(const cv::Mat &sceneDepth, const std::vector<cv::Point> &stablePoints, const std::vector<ushort> &tplDepths);

        /**
         * @brief Test I, perfmors check whether scene center depth lies within interval defined using avg template depth.
         *
         * @param sceneDepth Input 16-bit scene depth image
         * @param winCenter  Center point of currently processed window
         * @param avgDepth   Average depth across object in current tempate
         * @return           True whether depth is within defined interval, otherwise 0
         */
        inline bool testObjectSize(const cv::Mat &sceneDepth, const cv::Point winCenter, ushort avgDepth);

        /**
         * @brief Test II, performs check whether object quantized normals correspond to scene quantized normals.
         *
         * @param[in] sceneNormals Input 8-bit image of quantized scene normals
         * @param[in] stable       Currently processed stable feature point shifted to current window
         * @param[in] normal       Currently processed template normal (quantized), sampled at given stable feature point
         * @return                 1 whether both normals match, otherwise 0
         */
        inline int testNormals(const cv::Mat &sceneNormals, const cv::Point &stable, uchar normal);

        /**
         * @brief Test III, performs check whether object quantized gradients correspond to scene quantized gradients.
         *
         * @param[in] sceneGradients Input 8-bit image of quantized gradients
         * @param[in] edge           Currently processed edge feature point shifted to current window
         * @param[in] gradient       Currently processed template gradient (quantized), sampled at given edge feature point
         * @return                   1 whether both gradients match, otherwise 0
         */
        inline int testGradients(const cv::Mat &sceneGradients, const cv::Point &edge, uchar gradient);

        /**
         * @brief Test IV, performs a depth test to se whether object depth differences are lower than median.
         *
         * @param[in] sceneDepth  Input scene 16-bit depth image
         * @param[in] diameter    Pre-calculated object diameter (candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK)
         * @param[in] depthMedian Depth median computed from depth differences using depthDiffMedian() function
         * @param[in] depth       Currently processed template depth, sampled at given stable feature point
         * @param[in] stable      Currently processed stable feature point shifted to current window
         * @return                1 whether there was a match within small patch around stable point (-patchOffset <-> patchOffset), otherwise 0
         */
        inline int testDepth(const cv::Mat &sceneDepth, const cv::Point &stable, ushort depth, int depthMedian, float diameter);

        /**
         * @brief Test V, performs check whether object hue of HSV color space correspond to scene hue value (both are normalized).
         *
         * @param[in] sceneHSV Input 8-bit normalized scene hue values
         * @param[in] stable   Currently processed stable feature point shifted to current window
         * @param[in] hue      Currently processed hue, sampled at given stable feature point
         * @return             1 whether absolute differece of hue values is < criteria->maxHueDiff, otherwise 0
         */
        inline int testColor(const cv::Mat &sceneHue, const cv::Point &stable, uchar hue);

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
