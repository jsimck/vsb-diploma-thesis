#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_CRITERIA_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_CRITERIA_H

#include <opencv2/core/types.hpp>
#include <ostream>
#include <memory>
#include "template.h"

/**
 * struct ClassifierCriteria
 *
 * Holds information about loaded data set and all thresholds for further computation
 */

struct ClassifierCriteria {
public:
    // Train params and thresholds
    struct {
        // Hasher
        struct {
            cv::Size grid; // Triplet grid size
            int tablesCount; // Number of tables to generate
            int maxDistance; // Max distance between each point in triplet
        } hasher;

        // Matcher
        struct {
            int pointsCount; // Number of feature points to extract for each template
        } matcher;

        // Objectness
        struct {
            float tEdgesMin; // Min threshold applied in sobel filtered image thresholding [0.01f]
            float tEdgesMax; // Max threshold applied in sobel filtered image thresholding [0.1f]
        } objectness;
    } train;

    // Detect params and thresholds
    struct {
        // Hasher
        struct {
            int minVotes; // Minimum number of votes to classify template as window candidate
        } hasher;

        // Matcher
        struct {
            float tMatch; // number indicating how many percentage of points should match [0.0 - 1.0]
            float tOverlap; // overlap threshold, how much should 2 windows overlap in order to calculate non-maxima suppression [0.0 - 1.0]
            uchar tColorTest; // HUE value max threshold to pass comparing colors between scene and template [0-180]
            cv::Range neighbourhood; // area to search around feature point to look for match [-num, num]
            std::vector<cv::Vec2f> depthDeviationFunction; // Correction function returning error in percentage for given depth
            float depthK; // Constant used in IV test (depth test)
            float tMinGradMag; // Minimal gradient magnitude to classify as gradient
            ushort maxDifference; // Max difference in surface normal quantization
        } matcher;

        // Objectness
        struct {
            int step; // Stepping for sliding window
            float tMatch; // Factor of minEdgels window should contain to be classified as valid [30% -> 0.3f]
        } objectness;
    } detect;

    // Data set info
    struct {
        int maxDepth; // Max depth within object bounding box
        int minEdgels;
        float depthScaleFactor; // in our cases 1 => 0.1mm so to get 1mm we need to multiply values by 10
        cv::Size smallestTemplate;
        cv::Size largestTemplate;
    } info;

    // Constructors
    ClassifierCriteria();

    // Persistence
    static void load(cv::FileStorage fsr, std::shared_ptr<ClassifierCriteria> criteria);
    void save(cv::FileStorage &fsw);

    friend std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &criteria);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFIER_CRITERIA_H
