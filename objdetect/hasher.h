#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"
#include "../core/template_group.h"
#include "../core/window.h"

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
    unsigned int hashTableCount;
    unsigned int histogramBinCount;
    std::vector<cv::Range> histogramBinRanges;

    // Methods
    cv::Vec3d extractSurfaceNormal(const cv::Mat &src, const cv::Point c);
    cv::Vec2i extractRelativeDepths(const cv::Mat &src, const cv::Point p1, const cv::Point p2, const cv::Point p3);
    int quantizeSurfaceNormals(cv::Vec3f normal);
    int quantizeDepths(float depth);
    void generateTriplets(std::vector<HashTable> &hashTables);
    void calculateDepthHistogramRanges(unsigned long histogramSum, unsigned long histogramValues[]);
    void calculateDepthBinRanges(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables);
public:
    // Statics
    static const int IMG_16BIT_VALUE_MAX;
    static const int IMG_16BIT_VALUES_RANGE;

    // Constructors
    Hasher() {}
    Hasher(cv::Size referencePointsGrid, unsigned int hashTableCount) : featurePointsGrid(referencePointsGrid), hashTableCount(hashTableCount) {}

    // Methods
    void initialize(const std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables);
    void train(std::vector<TemplateGroup> &groups, std::vector<HashTable> &hashTables);
    void verifyTemplateCandidates(const cv::Mat &scene, std::vector<Window> &windows, std::vector<HashTable> &hashTables, std::vector<TemplateGroup> &groups);

    // Getters
    const cv::Size getFeaturePointsGrid();
    unsigned int getHashTableCount() const;
    const std::vector<cv::Range> &getHistogramBinRanges() const;
    unsigned int getHistogramBinCount() const;

    // Setters
    void setFeaturePointsGrid(cv::Size featurePointsGrid);
    void setHashTableCount(unsigned int hashTableCount);
    void setHistogramBinRanges(const std::vector<cv::Range> &histogramBinRanges);
    void setHistogramBinCount(unsigned int histogramBinCount);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
