#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_CRITERIA_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_CRITERIA_H

#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/persistence.hpp>
#include "template.h"

namespace tless {
    /**
     * @brief Defines parameters for classifier behaviour and holds info about dataset
     */
    struct ClassifierCriteria {
    public:
        // Train params
        cv::Size tripletGrid{12, 12}; //!< Relative size of the triplet grid, 12x12 yields 144 possible locations
        uint maxTripletDist = 3; //!< Maximum distance between p1, p2 from center in hash table triplets
        uint tablesCount = 100;  //!< Amount of tables to generate for hashing verification
        uint tablesTrainingMultiplier = 5;  //!< tablesTrainingMultiplier * tablesCount = yields amount of tables that are generated before only tablesCount containing most templates are picked
        uint featurePointsCount = 100; //!< Amount of points to generate for feature points matching
        float minMagnitude = 0.1f; //!< Minimal magnitude of edge gradient to classify it as valid
        ushort maxDepthDiff = 100; //!< When computing surface normals, contribution of pixel is ignored if the depth difference with central pixel is above this threshold
        // TODO fix deviation function based on the other paper
        std::vector<cv::Vec2f> depthDeviationFun{{10000, 0.14f}, {15000, 0.12f}, {20000, 0.1f}, {70000, 0.08f}}; //!< Depth error function, allowing depth values to be match within giveninterval

        // Detect Params
        int minVotes = 3; //!< Minimum amount of votes to classify template as a valid candidate for given window
        int windowStep = 5; //!< Objectness sliding window step
        int patchOffset = 2; //!< +-offset, defining neighbourhood to look for a feature point match
        float objectnessFactor = 0.3f; //!< Amount of edgels window must contain (30% of minimum) to classify as containing object in objectness detection
        float matchFactor = 0.6f; //!< Amount of feature points that needs to match to classify candidate as a match (at least 60%)
        float overlapFactor = 0.3f; //!< Permitted factor of which two templates can overlap
        float depthK = 0.05f; //!< Constant used in depth test in template matching phase

        struct {
            ushort minDepth = (ushort) -1; //!< Minimum depth found across all templates withing their bounding box
            ushort maxDepth = 0; //!< Maximum depth found across all templates withing their bounding box
            int minEdgels = INT_MAX; //!< Minimum number of edgels found in any template
            float depthScaleFactor = 10.0f; //!< Depth scaling factor to convert depth value to millimeters
            cv::Size smallestTemplate{500, 500}; //!< Size of the largest template found across all templates
            cv::Size largestArea{0, 0}; //!< Size of the largest area (largest width and largest height) found across all templates
        } info;

        friend void operator>>(const cv::FileNode &node, cv::Ptr<ClassifierCriteria> crit);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const ClassifierCriteria &crit);
        friend std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &crit);
    };
}

#endif