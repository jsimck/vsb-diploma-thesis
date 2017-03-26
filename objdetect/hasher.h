#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"
#include "../core/template_group.h"

/**
 * class Hasher
 *
 * Class used to train templates and sliding windows, to prefilter
 * number of templates needed to be template matched in other stages of
 * template matching.
 */
class Hasher {
private:
    cv::Size featurePointsGrid;

    // Methods
    cv::Vec3d extractSurfaceNormal(const cv::Mat &src, cv::Point c);
    int quantizeSurfaceNormals(cv::Vec3f normal);
    int quantizeDepths(float depth);
public:
    // Constructors
    Hasher() {}
    Hasher(cv::Size referencePointsGrid) : featurePointsGrid(referencePointsGrid) {}

    // Methods
    void train(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables, unsigned int numberOfHashTables = 100);
    void verifyTemplateCandidates(const cv::Mat &scene, cv::Rect &objectnessROI, std::vector<HashTable> &hashTables, std::vector<TemplateGroup> &groups);

    // Getters
    const cv::Size getFeaturePointsGrid();

    // Setters
    void setFeaturePointsGrid(cv::Size featurePointsGrid);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
